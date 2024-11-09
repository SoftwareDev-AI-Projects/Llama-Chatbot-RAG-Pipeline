import logging
import os
from utilities.data_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from templates.template import TEMPLATE
from fastapi import HTTPException

logger = logging.getLogger(__name__)
load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ['LANGCHAIN_TRACING_V2']= os.getenv("LANGCHAIN_TRACING_V2")
os.environ['LANGCHAIN_ENDPOINT'] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class RAGPipeline:
    def __init__(self, path_to_data):
        logger.log(logging.INFO, "Loading the CSV data to documents")
        self.documents = CSVLoader(path_to_data).load()

        logger.log(logging.INFO, "Generating splits for the documents")
        self.splits = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=64).split_documents(self.documents)

        logger.log(logging.INFO, "Loading the HuggingFace Embeddings model, OpenAI Chat model and PEFT Llama model")
        self.hf = self._load_huggingface_embeddings_model()
        self.multi_query_llm = self._load_openai_chat_llm()
        self.peft_model = self._load_peft_llama_model()

        logger.log(logging.INFO, "Creating FAISS vector store using HuggingFace embeddings. Setting up retriever on similarity and k=10")
        self.vectorstore = FAISS.from_documents(self.splits, self.hf)
        self.retriever = self.vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 10})

        logger.log(logging.INFO, "Creating Multi-query retriever using OpenAI Chat model")
        self.multi_query_retriever = MultiQueryRetriever.from_llm(self.retriever, self.multi_query_llm, include_original=True)

        logger.log(logging.INFO, "Creating RAG pipeline using PEFT Llama model and Chat template")
        self.chat_template = self._load_chat_template()
        self.rag_chain = {"context": self.multi_query_retriever | self.format_docs, "question": RunnablePassthrough()} | self.chat_template | self.peft_model | StrOutputParser()

    @staticmethod
    def _load_openai_chat_llm():
        """
        Load OpenAI Chat model
        :return: chat model isntance
        """
        return ChatOpenAI(temperature=0.7)

    @staticmethod
    def _load_huggingface_embeddings_model():
        """
        Load HuggingFace Embeddings model
        :return: model instance
        """
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': False}
        )

    @staticmethod
    def _load_peft_llama_model():
        """
        Load PEFT Llama model using qunatization config in 4bit and PEFT class
        :return: model instance
        """
        try:
            # Load final chat model
            peft_model_id = "Chryslerx10/Llama-3.2-1B-finetuned-amazon-reviews-QA-peft-4bit"
            config = PeftConfig.from_pretrained(peft_model_id, device_map='auto')
            # Quantization config for the model
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype='float16',
                bnb_4bit_use_double_quant=True
            )

            trained_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                device_map='auto',
                return_dict=True,
                quantization_config=bnb_config
            )
            # Tokenizer for the model
            tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
            tokenizer.pad_token = tokenizer.eos_token
            # Load the PEFT model
            peft_loaded_model_trained = PeftModel.from_pretrained(trained_model, peft_model_id, device_map='auto')
            llama_model_pipe = pipeline("text-generation", model=peft_loaded_model_trained, tokenizer=tokenizer,
                                        device_map='auto', max_new_tokens=64, temperature=0.5, top_k=5, top_p=0.95,
                                        repetition_penalty=1.2, do_sample=True, penalty_alpha=0.6)

            # Create huggingface pipeline based on LangChain specification
            llm = HuggingFacePipeline(pipeline=llama_model_pipe)

            return llm

        except Exception as e:
            logger.log(logging.ERROR, f"Error in loading PEFT Llama model: {e}")
            raise HTTPException(status_code=505, detail="Error in loading PEFT Llama model")

    @staticmethod
    def _load_chat_template():
        """
        Load Chat template
        :return: return rag template
        """
        return PromptTemplate.from_template(TEMPLATE)

    @staticmethod
    def format_docs(docs):
        """
        Format as single string and return documents retrieved from vector store
        :param docs: list of documents
        :return: String combining all documents
        """
        return " ".join(doc.page_content for doc in docs)

    def get_answer(self, product_name, question):
        """
        Get answer using RAG pipeline
        :param product_name: product name to filter
        :param question: query to get answer
        :return: answer from the LLM model
        """
        try:
            logger.log(logging.INFO, "Getting answer using RAG pipeline for the query: {question}")
            if product_name:
                logger.log(logging.INFO, f"Filtering for product: {product_name}")
                self.retriever.search_kwargs['filter'] = {'product_title': product_name}

            return self.rag_chain.invoke(question)

        except Exception as e:
            logger.log(logging.ERROR, f"Error in getting answer: {e}")
            raise HTTPException(status_code=600, detail="Error in getting answer")

import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollator, TrainingArguments, LlamaForCausalLM
from huggingface_hub import notebook_login
from datasets import load_dataset, Dataset
from transformers import GenerationConfig
from peft import LoraConfig, AutoPeftModelForCausalLM, PeftModel, PeftConfig
from trl import SFTTrainer
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_dataset(row):
    text = f"""[Instruction] You are a question-answering agent specialized in helping users with their queries about products based on relevant customer reviews. Your job is to analyze the reviews provided in the context and generate an accurate, helpful, and informative response to the question asked.

    1. Read the user's question carefully.
    2. Use the reviews given in the context to formulate your answer.
    3. If the product reviews don't contain enough information or is missing, inform the user that there aren't sufficient reviews to answer the question.
    4. If the question is unrelated to products, politely inform the user that you can only assist with product-related queries.
    5. Structure your response in a conversational and user-friendly manner. 

    Your goal is to provide helpful and contextually relevant answers to product-related questions.

    [Question]\n {row['question']}

    [Related Reviews]\n {row['review'] if row['review'] else ''}

    [Answer]\n {row['summary']}"""

    return {'text': text}


# Load the dataset
df = pd.read_csv(r"C:\Users\rajtu\OneDrive\Desktop\Raj\Datasets\llama_finetuning_reviews_qa_dataset.csv")
data = Dataset.from_pandas(df)
data = data.train_test_split(test_size=0.2)
data = data.map(prepare_dataset)

# Load the model and tokenizer in 4-bit precision
peft_model_id = "Chryslerx10/Llama-3.2-1B-finetuned-generalQA-peft-4bit"
config = PeftConfig.from_pretrained(peft_model_id, device_map='auto')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype='float16',
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map='auto',
    return_dict=True,
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
peft_loaded_model = PeftModel.from_pretrained(model, peft_model_id, device_map='auto')
model.config.use_cache = False
model.config.pretraining_tp=1
tokenizer.pad_token = tokenizer.eos_token

# PEFT Supervised fine-tuning for Llama model
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.5,
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="./model/Llama-3.2-1B-finetuned-amazonQA",
    logging_dir="./logs",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_steps=2000,
    learning_rate=2e-5,
    save_strategy="epoch",
    save_steps=2000,
    logging_steps=2000,
    save_total_limit=2,
    num_train_epochs=30,
    fp16=True,
    eval_strategy="epoch",
    load_best_model_at_end=True
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    dataset_text_field="text",
    peft_config=peft_config,
    args=training_args,
    train_dataset=data['train'],
    eval_dataset=data['test']
)

if __name__ == "__main__":
    trainer.train()
    # trainer.save_model(local path)


# _________________________________________________________________________________________________________ #
# Push the model to huggingface hub
# trainer.model.save_pretrained("./model/Llama-3.2-1B-finetuned-amazon-reviews-QA-peft-4bit")
# tokenizer.save_pretrained("./model/Llama-3.2-1B-finetuned-amazon-reviews-QA-peft-4bit")

# from huggingface_hub import HfApi, notebook_login
# notebook_login()

# api = HfApi()
# repo_name = "Llama-3.2-1B-finetuned-amazon-reviews-QA-peft-4bit"
# username = "Chryslerx10"

# api.create_repo(repo_id=f"{username}/{repo_name}", private=False)

# Push adapters to the Hub
# trainer.model.push_to_hub(repo_name)

# Push tokenizer to the Hub
# tokenizer.push_to_hub(repo_name)

# Define your quantization configuration
# quantization_config = {
#     "load_in_4bit": True,
#     "bnb_4bit_quant_type": "nf4",
#     "bnb_4bit_compute_dtype": "float16",
#     "bnb_4bit_use_double_quant": True
# }

# peft_model_id = "./model/Llama-3.2-1B-finetuned-amazon-reviews-QA-peft-4bit"

# Save the quantization configuration
# with open(f"{peft_model_id}/quant_config.json", "w") as f:
#     json.dump(quantization_config, f)

# Manually upload the quant_config file to the Hub

# _________________________________________________________________________________________________________ #
# Loading the fine-tuned model for inference
# peft_model_id = "Chryslerx10/Llama-3.2-1B-finetuned-amazon-reviews-QA-peft-4bit"
# config = PeftConfig.from_pretrained(peft_model_id, device_map='auto')
#
# trained_model = AutoModelForCausalLM.from_pretrained(
#     config.base_model_name_or_path,
#     device_map='auto',
#     return_dict=True
# )

# tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
# tokenizer.pad_token = tokenizer.eos_token

# peft_loaded_model_trained = PeftModel.from_pretrained(trained_model, peft_model_id, device_map='auto')

# config = GenerationConfig(max_new_tokens=128, temperature=0.5, top_k=5, top_p=0.95, repetition_penalty=1.2,
#                           do_sample=True, penalty_alpha=0.6)
# text = f"""[Instruction] You are a question-answering agent specialized in helping users with their queries about products based on relevant customer reviews. Your job is to analyze the reviews provided in the related reviews and generate an accurate, helpful, and informative response to the question asked.
#
#     1. Read the user's question carefully.
#     2. Use the reviews given in the Related reviews section to formulate your answer.
#     3. If the related reviews don't contain enough information or is missing, inform the user that there aren't sufficient reviews to answer the question.
#     4. If the question is unrelated to products, politely inform the user that you can only assist with product-related queries.
#     5. Structure your response in a conversational and user-friendly manner.
#
#     Your goal is to provide helpful and contextually relevant answers to product-related questions.
#
#     [Question]\n What is the diameter of earth?
#
#     [Related Reviews]\n
#
#     [Answer]\n"""
# out = peft_loaded_model.generate(**tokenizer([text], return_tensors='pt', padding=True, truncation=True).to(device),
#                                  generation_config=config)
# tokenizer.decode(out[0], skip_special_tokens=True)
import json
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollator, TrainingArguments, LlamaForCausalLM
from huggingface_hub import notebook_login
from datasets import load_dataset, Dataset
from transformers import GenerationConfig
from peft import LoraConfig, AutoPeftModelForCausalLM, PeftModel, PeftConfig
from trl import SFTTrainer
import evaluate
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
rouge = evaluate.load('rouge')


def validate_model(preds):
    """
    Validate the model using the ROUGE metric
    :param preds: token prediction and labels (encoded)
    :return:
    """
    predictions, labels = preds
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=predictions, references=labels, use_stemmer=True,
                           rouge_types=["rouge1", "rouge2", "rougeL"])
    return result


def prepare_dataset(row):
    """
    Template used to prepare the dataset for training
    :param row: one pair of question and answer
    :return: template text
    """
    text = f"""[Instruction] You are a question-answering agent which answers the question based on the related reviews. If related reviews are not provided, you can generate the answer based on the question. \n[Question] {row['instruction']}\n[Related Reviews] \n[Answer] {row['output']}"""
    return text


def create_chat_template(row):
    """
    Chat template used for training the conversational bot
    :param row: triplet of question-reviews-answer
    :return: chat template
    """
    text = f"""[Question] {row}
    [Related Reviews]
    [Answer] """

    return text


def generate_response(user_input):
    """
    Generate response based on the user inputs
    :param user_input: user input question
    :return: response of the llm model
    """
    text = create_chat_template(user_input)
    inputs = tokenizer([text], return_tensors='pt', padding=True, truncation=True).to(device)

    config = GenerationConfig(max_length=256, temperature=0.5, top_k=5, top_p=0.95, repetition_penalty=1.2,
                              do_sample=True, penalty_alpha=0.6)

    response = model.generate(**inputs, generation_config=config)
    return tokenizer.decode(response[0], skip_special_tokens=True)


# Load the dataset
ds = load_dataset("bergr7f/databricks-dolly-15k-subset-general_qa")
train = pd.DataFrame(ds['train'])
test = pd.DataFrame(ds['validation'])
train['text'] = train.apply(lambda row: prepare_dataset(row), axis=1)
test['text'] = test.apply(lambda row: prepare_dataset(row), axis=1)
dataset = Dataset.from_pandas(train)
test_dataset = Dataset.from_pandas(test)

# Load the tokenizer and pre-trained model in 4-bit precision
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype='float16',
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", quantization_config=bnb_config, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
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
    output_dir="./model/Llama-3.2-1B-finetuned-generalQA",
    logging_dir="./logs",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_steps=1000,
    learning_rate=2e-5,
    save_strategy="epoch",
    save_steps=1000,
    logging_steps=1000,
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
    train_dataset=dataset,
    eval_dataset=test_dataset
)

if __name__ == "__main__":
    trainer.train()
    # trainer.save_model(local path)

# _________________________________________________________________________________________________________ #
# Push the model to huggingface hub
# Optional code

# from huggingface_hub import HfApi
#
# api = HfApi()
# repo_name = "Repository name"
# username = "HuggingFace username"
#
# api.create_repo(repo_id=f"{username}/{repo_name}", private=False)
#
# Push adapters to the Hub
# trainer.model.push_to_hub(repo_name)
#
# Push tokenizer to the Hub
# tokenizer.push_to_hub(repo_name)
#
# Define your quantization configuration
# quantization_config = {
#     "load_in_4bit": True,
#     "bnb_4bit_quant_type": "nf4",
#     "bnb_4bit_compute_dtype": "float16",
#     "bnb_4bit_use_double_quant": True
# }
#
# peft_model_id = "local model path"
#
# Save the quantization configuration
# with open(f"{peft_model_id}/quant_config.json", "w") as f:
#     json.dump(quantization_config, f)

# Manually upload the quant_config file to the Hub

# _________________________________________________________________________________________________________ #
# Loading the fine-tuned model for inference
# Optional code

# peft_model_id = "username/repo_name"
# config = PeftConfig.from_pretrained(peft_model_id, device_map='auto')
#
# model = AutoModelForCausalLM.from_pretrained(
#     config.base_model_name_or_path,
#     device_map='auto',
#     return_dict=True
# )
#
# tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
# tokenizer.pad_token = tokenizer.eos_token
#
# peft_loaded_model = PeftModel.from_pretrained(model, peft_model_id, device_map='auto')
# peft_loaded_model.eval()
# peft_loaded_model.to(device)
# config = GenerationConfig(temperature=0.5, top_k=5, top_p=0.95, repetition_penalty=1.2, do_sample=True, penalty_alpha=0.6, max_new_tokens=128)
# text = f"""[Instruction] You are a question-answering agent which answers the question based on the related reviews. If related reviews are not provided, you can generate the answer based on the question. \n[Question] How is the phillips hair trimmer?\n[Related Reviews]  \n[Answer]"""
# out = peft_loaded_model.generate(**tokenizer([text], return_tensors='pt', padding=True, truncation=True).to(device), generation_config=config)
# tokenizer.decode(out[0], skip_special_tokens=True)
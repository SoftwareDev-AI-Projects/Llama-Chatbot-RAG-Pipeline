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
from huggingface_hub import login, HfApi
login(token="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")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

peft_model_id = "Chryslerx10/Llama-3.2-1B-finetuned-amazon-reviews-QA-peft-4bit"
config = PeftConfig.from_pretrained(peft_model_id, device_map='auto')

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map='auto',
    return_dict=True
)

tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
tokenizer.pad_token = tokenizer.eos_token

peft_loaded_model = PeftModel.from_pretrained(model, peft_model_id, device_map='auto')
peft_loaded_model.eval()
peft_loaded_model.to(device)
config = GenerationConfig(temperature=0.5, top_k=5, top_p=0.95, repetition_penalty=1.2, do_sample=True, penalty_alpha=0.6, max_new_tokens=128)

text = f"""[Instruction] You are a question-answering agent specialized in helping users with their queries about products based on relevant customer reviews. Your job is to analyze the reviews provided in the related reviews and generate an accurate, helpful, and informative response to the question asked.

    1. Read the user's question carefully.
    2. Use the reviews given in the Related reviews section to formulate your answer.
    3. If the related reviews don't contain enough information or is missing, inform the user that there aren't sufficient reviews to answer the question.
    4. If the question is unrelated to products, politely inform the user that you can only assist with product-related queries.
    5. Structure your response in a conversational and user-friendly manner.

    Your goal is to provide helpful and contextually relevant answers to product-related questions.

    [Question]\n What do customers love most about this product?

    [Related Reviews]\n User: l o v e this shampoo! I am not one to normally leave reviews, in fact, i think this is my first review on amazon. however, this product has earned it. i think I have bought maybe 5 or 6 bottles of this shampoo and I am hooked. i swear by this shampoo. my hair used to be so oily and i would get flakes non stop with any shampoo that had sulfur sulfates in it. not only did this product save my scalp, but my hair looks, feels and smells amazing! i refuse to use other shampoos now and hopefully will be using this product for years to come. love it! User: five stars great product User: scalp build up i do not have dandruff but i do get scalp build up pretty bad. i have a lot of long hair, not at all oily, and wash only twice a week. i have tried numerous shampoos to address the build up. this shampoo seems to be the ticket for me. I have used it twice now and for the first time in a long time, zero build up between washings. I am very satisfied and will continue to use this shampoo. User: worth the money have only used the shampoo 3 times, but so far so good. i have eczema and the shampoo eliviates the itching and flakes for 3 days. I am hoping it will be longer with prolonged use. highly recommend User: i really like this shampoo i really like this shampoo.the smell is not bother me at all, but for some people probably will.i love to use sage a lot in my live so i want to see how shampoo will work,and works good.i have short hair so i do not have to use conditioner but some people have to.just little drop and are big bubbles, 

    [Answer]\n"""

out = peft_loaded_model.generate(**tokenizer([text], return_tensors='pt', padding=True, truncation=True).to(device), generation_config=config)
print(tokenizer.decode(out[0], skip_special_tokens=True))
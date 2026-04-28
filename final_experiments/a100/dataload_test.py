from datasets import load_dataset
import sys
import os


from transformers import AutoModelForCausalLM, AutoTokenizer
# Before running this do an export HF_TOKEN=".." whatever is in the .env file
#model_name_or_path = "Qwen/Qwen2-7B"
model_name_or_path = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path
)
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

tokenizer_name = 'meta-llama/Llama-3.2-1B'
model_name_or_path = 'meta-llama/Llama-3.2-1B'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train[:5%]')

model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
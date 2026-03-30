from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import os

os.environ['HF_HOME'] = '/scratch/user/u.bg348806/.cache/huggingface'
os.environ['HF_HUB_CACHE'] = '/scratch/user/u.bg348806/.cache/huggingface'
os.environ['HF_DATASETS_OFFLINE'] = 1
os.environ['HF_HUB_OFFLINE'] = 1
os.environ['TRANSFORMERS_OFFLINE'] = 1



tokenizer_name = 'meta-llama/Llama-3.2-1B'
model_name_or_path = 'meta-llama/Llama-3.2-1B'

#tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train[:5%]')

#model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
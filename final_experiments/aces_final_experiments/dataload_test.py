from datasets import load_dataset
import sys
import os

project_dir = sys.argv[1] if len(sys.argv) > 1 else "./"
dataset_cache_path = os.path.join(project_dir, '.cache/huggingface/datasets')

dataset = load_dataset(
    'wikitext',
    'wikitext-103-raw-v1',
    split='train[:5%]',
    cache_dir=dataset_cache_path
)


from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "meta-llama/Llama-3.2-1B"
hub_cache_path = os.path.join(project_dir, '.cache/huggingface/hub')

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    cache_dir=hub_cache_path
)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    cache_dir=hub_cache_path
)
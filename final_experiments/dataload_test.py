# from datasets import load_dataset

# dataset = load_dataset(
#     'wikitext',
#     'wikitext-103-raw-v1',
#     split='train[:5%]',
#     cache_dir='/scratch/user/u.bg348806/.cache/huggingface/datasets'
# )


from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "meta-llama/Llama-3.2-1B"
cache_path = "/scratch/user/u.bg348806/.cache/huggingface/hub"

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    cache_dir=cache_path
)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    cache_dir=cache_path
)
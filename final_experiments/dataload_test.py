from datasets import load_dataset

dataset = load_dataset(
    'wikitext',
    'wikitext-103-raw-v1',
    split='train[:5%]',
    cache_dir='/scratch/user/u.bg348806/.cache/huggingface/datasets'
)
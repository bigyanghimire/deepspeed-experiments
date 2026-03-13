# train.py
from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF, UlyssesSPDataLoaderAdapter
from deepspeed.runtime.utils import move_to_device
from deepspeed.utils import groups
from torch import tensor
from transformers import AutoModelForCausalLM
import deepspeed
import deepspeed.comm as dist
import torch
from torch.utils.data import DataLoader
model_name_or_path = 'hf-internal-testing/tiny-random-LlamaForCausalLM'
seq_length = 64
sequence_parallel_size = 2
micro_batch_size = 4

from torch.utils.data.distributed import DistributedSampler

dtype = torch.bfloat16

# a simple Dataset
# replace with a real dataset but make sure `position_ids` are returned

torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

"""Load wikitext dataset for training."""
from datasets import load_dataset
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train[:5%]')

# Filter empty texts
dataset = dataset.filter(lambda x: len(x['text'].strip()) > 50)
print("dayaset",dataset[0])
# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        max_length=seq_length,
        truncation=True,
        return_tensors=None
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
print("tokensizeda datatset is",tokenized_dataset[0])
# Create distributed sampler and dataloader
sampler = DistributedSampler(tokenized_dataset)
dataloader = DataLoader(tokenized_dataset, batch_size=micro_batch_size, sampler=sampler, num_workers=2)




# if torch.distributed.get_rank()==1:
#     for i, batch in enumerate(dl):
#         print("i>>>>",i,batch)
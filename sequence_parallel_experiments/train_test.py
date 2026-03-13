# train.py
from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF, UlyssesSPDataLoaderAdapter
from deepspeed.runtime.utils import move_to_device
from deepspeed.utils import groups
from torch import tensor
from transformers import AutoModelForCausalLM
import deepspeed
import deepspeed.comm as dist
import torch

model_name_or_path = 'hf-internal-testing/tiny-random-LlamaForCausalLM'
seq_length = 64
sequence_parallel_size = 2
micro_batch_size = 4

from torch.utils.data.distributed import DistributedSampler

dtype = torch.bfloat16

# a simple Dataset
# replace with a real dataset but make sure `position_ids` are returned
input_ids = tensor([
    [1, 10, 10, 10, 2, 2],  # sequence 0
    [1, 20, 20, 20, 2, 2],  # sequence 1
    [1, 30, 30, 30, 2, 2],  # sequence 2
    [1, 40, 40, 40, 2, 2],  # sequence 3
    [1, 50, 50, 50, 2, 2],  # sequence 4
    [1, 60, 60, 60, 2, 2],  # sequence 5
    [1, 70, 70, 70, 2, 2],  # sequence 6
    [1, 80, 80, 80, 2, 2],  # sequence 7
])
position_ids = tensor([
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 3, 4, 5],
])
torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

ds = torch.utils.data.TensorDataset(input_ids, position_ids)
def collate_fn1(batch):
    input_ids_array=[]
    position_ids_array=[]
    labels_ids_array=[]
    #print("batch is in collate",batch)
    for i in range(0, len(batch)):
        input_ids, position_ids = batch[i]
        input_ids=input_ids.unsqueeze(0)
        position_ids=position_ids.unsqueeze(0)
        labels=input_ids.unsqueeze(0)
        
        input_ids_array.append(input_ids)
        position_ids_array.append(position_ids)
        labels_ids_array.append(labels)
    input_ids, position_ids = batch[0]
    return dict(input_ids=input_ids_array,
                position_ids=position_ids_array,
                labels=labels_ids_array)


def load_wikitext_data(tokenizer_name, seq_length, batch_size, world_size, rank):
    """Load wikitext dataset for training."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train[:5%]')

    # Filter empty texts
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 50)

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

    # Create distributed sampler and dataloader
    sampler = DistributedSampler(tokenized_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)

    print(f"[Rank {rank}] Dataset loaded: {len(tokenized_dataset)} examples, vocab_size={tokenizer.vocab_size}")
    return dataloader, tokenizer.vocab_size


def collate_fn(batch):
    # batch is a list of tuples: [(input_ids_0, position_ids_0), (input_ids_1, position_ids_1), ...]
    input_ids_list = [item[0] for item in batch]
    position_ids_list = [item[1] for item in batch]
    
    # Stack to create batch dimension
    input_ids_batch = torch.stack(input_ids_list, dim=0)  # [batch_size, seq_len]
    position_ids_batch = torch.stack(position_ids_list, dim=0)  # [batch_size, seq_len]
    
    return {
        'input_ids': input_ids_batch,
        'position_ids': position_ids_batch,
        'labels': input_ids_batch.clone()  # Labels same as input_ids for LM
    }
sampler = DistributedSampler(ds,shuffle=False)
dl = torch.utils.data.DataLoader(ds, batch_size=micro_batch_size, collate_fn=collate_fn, sampler=sampler)
print(f"len of dl is {len(dl)}")
if torch.distributed.get_rank()==1:
    for i, batch in enumerate(dl):
        print("i>>>>",i,batch)
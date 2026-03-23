# train.py
from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF, UlyssesSPDataLoaderAdapter
from deepspeed.runtime.utils import move_to_device
from deepspeed.utils import groups
from torch import tensor
from transformers import AutoModelForCausalLM
import deepspeed
import deepspeed.comm as dist
import torch
from collections import defaultdict
from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler
micro_batches = defaultdict(dict)
device = torch.device('cpu')
# Data for microbatch 0
micro_batches[0] = {
    'input_ids': torch.tensor([
        [1, 30, 30, 30, 3, 3, 3, 3],
        [1, 110, 110, 110, 11, 11, 11, 11]
    ], device=device),
    'position_ids': torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7]
    ], device=device),
    'labels': torch.tensor([
        [1, 30, 30, 30, 3, 3, 3, 3],
        [1, 110, 110, 110, 11, 11, 11, 11]
    ], device=device)
}

# Data for microbatch 1
micro_batches[1] = {
    'input_ids': torch.tensor([
        [1, 40, 40, 40, 4, 4, 4, 4],
        [1, 120, 120, 120, 12, 12, 12, 12]
    ], device=device),
    'position_ids': torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7]
    ], device=device),
    'labels': torch.tensor([
        [1, 40, 40, 40, 4, 4, 4, 4],
        [1, 120, 120, 120, 12, 12, 12, 12]
    ], device=device)
}

# Verification
print(f"Total micro_batches: {micro_batches}")
print(f"Device: {micro_batches[0]['input_ids'].device}")
sp_world_size=4
sp_rank=0
global_rank= 0
global_world_size = 8

def global_rank_to_group(rank:int, group_size : int):
    return int((rank // group_size))
def rank_within_group(global_rank: int, group_size: int) -> int:
    return global_rank % group_size
   
for batch in micro_batches.values():
    seq_length = len(batch["input_ids"][0])

    chunk_len = seq_length // sp_world_size

    # because we have to gather logits from all sp ranks we have to do the loss function ourselves
    # therefore remove labels to avoid an attempt to calculate loss by transformers
    labels = batch.pop("labels")
    labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
    batch["shift_labels"] = labels[..., 1:].contiguous()
    # free up temp memory
    del labels

    # batch sharding
    for k in batch.keys():
        # leave non-tensors alone
        if not torch.is_tensor(batch[k]):
            continue
        # at seqlen>10M and 32+ gpus this can take GBs of memory so keep the prefill buffer on cpu
        # In our case one group of GPU will always get one batch
        batch_size = 2
        group_size = global_world_size // batch_size # GPUs within a sequence
        my_group= global_rank_to_group(global_rank, group_size)
        print(f"my group os {my_group}")
        total_groups = global_world_size / group_size
        batches_per_group = 1
        sp_rank = rank_within_group(global_rank, group_size)
        batch[k] = batch[k][my_group:my_group+1, chunk_len * sp_rank:chunk_len * (sp_rank + 1)]
        print(f"batch {batch}")



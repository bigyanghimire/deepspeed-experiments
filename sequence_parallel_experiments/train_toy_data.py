# train.py with batch_size > 1

#from deepspeed.runtime.sequence_parallel.ulysses_sp2 import UlyssesSPAttentionHF, UlyssesSPDataLoaderAdapter
from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF, UlyssesSPDataLoaderAdapter, UlyssesSPDataLoaderAdapter2
from deepspeed.runtime.utils import move_to_device
from deepspeed.utils import groups
from torch import tensor
from transformers import AutoModelForCausalLM
import deepspeed
import deepspeed.comm as dist
import torch
from torch.utils.data.distributed import DistributedSampler
import time
# model_name_or_path = 'hf-internal-testing/tiny-random-LlamaForCausalLM'
model_name_or_path = 'meta-llama/Llama-3.2-1B'
seq_length = 64
sequence_parallel_size = 4
micro_batch_size = 2  # CHANGED: Now batch size = 4

config_dict = {
    "train_micro_batch_size_per_gpu": micro_batch_size,  # CHANGED
    "zero_optimization": {
        "stage": 2,
    },
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-3
        }
    },
    "sequence_parallel_size": sequence_parallel_size,
}

dtype = torch.bfloat16

# CHANGED: Create a dataset with multiple samples
input_ids = tensor([
    [1, 10, 10, 10, 1, 1,1,1],  # sequence 0
    [1, 20, 20, 20, 2, 2,2,2],  # sequence 1
    [1, 30, 30, 30, 3, 3,3,3],  # sequence 2
    [1, 40, 40, 40, 4, 4,4,4],  # sequence 3
    [1, 50, 50, 50, 5, 5,5,5],  # sequence 4
    [1, 60, 60, 60, 6, 6,6,6],  # sequence 5
    [1, 70, 70, 70, 7, 7,7,7],  # sequence 6
    [1, 80, 80, 80, 8, 8,8,8],  # sequence 7
    [1, 90, 90, 90, 9, 9, 9, 9],  # sequence 8
    [1, 100, 100, 100, 10, 10, 10, 10],  # sequence 9
    [1, 110, 110, 110, 11, 11, 11, 11],  # sequence 10
    [1, 120, 120, 120, 12, 12, 12, 12],  # sequence 11
    [1, 130, 130, 130, 13, 13, 13, 13],  # sequence 12
    [1, 140, 140, 140, 14, 14, 14, 14],  # sequence 13
    [1, 150, 150, 150, 15, 15, 15, 15],  # sequence 14
    [1, 160, 160, 160, 16, 16, 16, 16],  # sequence 15
])
position_ids = tensor([
    [0, 1, 2, 3, 4, 5,6,7],
    [0, 1, 2, 3, 4, 5,6,7],
    [0, 1, 2, 3, 4, 5,6,7],
    [0, 1, 2, 3, 4, 5,6,7],
    [0, 1, 2, 3, 4, 5,6,7],
    [0, 1, 2, 3, 4, 5,6,7],
    [0, 1, 2, 3, 4, 5,6,7],
    [0, 1, 2, 3, 4, 5,6,7],
    [0, 1, 2, 3, 4, 5,6,7],
    [0, 1, 2, 3, 4, 5,6,7],
    [0, 1, 2, 3, 4, 5,6,7],
    [0, 1, 2, 3, 4, 5,6,7],
    [0, 1, 2, 3, 4, 5,6,7],
    [0, 1, 2, 3, 4, 5,6,7],
    [0, 1, 2, 3, 4, 5,6,7],
    [0, 1, 2, 3, 4, 5,6,7],
])

ds = torch.utils.data.TensorDataset(input_ids, position_ids)

# CHANGED: New collate function that handles batches properly
# This is for every single batch
def collate_fn(batch):
    # one batch is a list of tuples: [(input_ids_0, position_ids_0), (input_ids_1, position_ids_1), ...]
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

dist.init_distributed(dist_backend='nccl', dist_init_required=True)

# Ulysses injection into HF Transformers
mpu = UlyssesSPAttentionHF.register_with_transformers(
    model_name_or_path=model_name_or_path,
    core_attn_implementation="sdpa",
    sequence_parallel_size=sequence_parallel_size,
    micro_batch_size=micro_batch_size,
    seq_length=seq_length,
    seq_length_is_variable=True,
)

# Deepspeed setup
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model, _, _, _ = deepspeed.initialize(
    config=config_dict,
    model=model,
    model_parameters=model.parameters(),
    mpu=mpu
)

# UlyssesSPDataLoaderAdapter injection
sp_group = groups._get_sequence_parallel_group()
# sp_group = deepspeed.runtime.sequence_parallel.parallel_state_sp.initialize_sequence_groups(2)
#sp_world_size = dist.get_world_size(sp_group)
#sp_rank = dist.get_rank(sp_group)

# sp_group = groups._get_sequence_parallel_group()
sp_world_size = groups._get_sequence_parallel_world_size()
sp_rank = groups._get_sequence_parallel_rank()
# print(f"sp rank is :{sp_rank} \n")
# print(f"sp group is :{sp_group}")
# print(f"sp world size is :{sp_world_size}")

# CHANGED: batch_size parameter
# Here whole data is shared by both meaning each GPU is getting full data on batches
sampler = DistributedSampler(ds,shuffle=False)
dl = torch.utils.data.DataLoader(
    ds, 
    batch_size=micro_batch_size,  # This controls how many sequences per batch
    collate_fn=collate_fn,
    sampler=sampler
)
# Normal training loop
# for iter, batch in enumerate(dl):
#     print(f"vanilla dl batch {batch} from rank {dist.get_rank()} at iter {iter}")
dl = UlyssesSPDataLoaderAdapter2(
    dl,
    sp_rank=sp_rank,
    sp_group=sp_group,
    sp_world_size=sp_world_size,
    device=model.device,
)

num_epochs=1
# Normal training loop
for i in range(0,num_epochs):
    iter_count=0
    for step, batch in enumerate(dl):
        if iter_count<30:
            start_time = time.time()
            batch = move_to_device(batch, model.device)
            print(f"in Iteration {step} Batch Rank : {int(deepspeed.comm.get_rank())} and batch is {batch} \n")
            # Verify shapes
            # if dist.get_rank() == 0 and iter == 0:
            #     print(f"Batch shapes:")
            #     print(f"  input_ids: {batch['input_ids'].shape}")  # Should be [batch_size, seq_len/sp_size]
            #     print(f"  position_ids: {batch['position_ids'].shape}")
            #     print(f"  labels: {batch.get('labels', 'N/A')}")
            #print(f"input_ids value rank: {dist.get_rank()} in iter {iter} : {batch['input_ids']}")  # Should be [batch_size, seq_len/sp_size]
            outputs = model(**batch)
            #print(f"outputs are {outputs} and logits are {outputs.logits}")
            # Loss calculation
            shift_labels = batch["shift_labels"]
            loss = model.module.loss_function(
                logits=outputs.logits,
                labels=None,
                shift_labels=shift_labels,
                vocab_size=model.module.config.vocab_size,
            )
            good_tokens       = (shift_labels != -100).view(-1).sum()
            losses_per_rank      = torch.distributed.nn.functional.all_gather(loss,        group=sp_group)
            good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=sp_group)

            total_loss        = sum(losses_per_rank[r] * good_tokens_per_rank[r] for r in range(sp_world_size))
            total_good_tokens = sum(good_tokens_per_rank)
            loss              = total_loss / max(total_good_tokens, 1)
            if dist.get_rank() == 0:
                print(f"Iteration {step}: loss={loss.item():.4f}")

            model.backward(loss)
            model.step()  # Don't forget optimizer step!
            step_time = time.time() - start_time
            if dist.get_rank() == 0:
                print(f"Step_time: {step_time}\n")
            iter_count=iter_count+1

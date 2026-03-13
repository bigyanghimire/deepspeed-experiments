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
seq_length = 64
sequence_parallel_size = 2
micro_batch_size = 4




class UlyssesSPDataLoaderAdapter:

    def __init__(
        self,
        dl: DataLoader,
        sp_rank: int,
        sp_group,
        sp_world_size,
        device,
    ):
        """
        This a DataLoader adapter which wraps around any existing DataLoader. It is used in conjunction with Ulysses to perform batch sharding on the sequence dimension.

        It gathers 1 sample(batch not sample) from each participating rank, using the DL it wraps, then shards each of them and sends back to the ranks. So that when dl->iter->next is called, we end up with:
        - rank 0: getting batch 0 shard 0
        - rank 1: getting batch 0 shard 1
        ...
        - rank n: getting batch 0 shard n
        which is used to compute the batch (from rank0) using all SP ranks.

        When the next iteration starts and dl->iter->next is called, we end up with:
        - rank 0: getting batch 1 shard 0
        - rank 1: getting batch 1 shard 1
        ...
        - rank n: getting batch 1 shard n
        which is used to compute a second batch (from rank1) using all SP ranks.

        This continues until SP iterations are performed. At this point we need to get more data and so the above repeats.

        The key thing to understand is that all SP ranks participate in processing a single DL sample. So instead of normal DataParallel we perform a sort of SP over DP.

        When SP number of iterations is completed it's an equivalent of performing a single iteration with normal DP.

        If more tokens need to be consumed per step use the gradient accumulation feature.

        Ulysses expects the following dict keys in each DL batch (`dl->iter->next`):
        - `input_ids`
        - `position_ids`
        - `labels`

        Additional entries can be present.

        The tensors are expected to be of shape: `[batch_size, seqlen, ...]`

        The sharding happens on the seqlen (1st) dimension for all tensors in the batch, any non-tensor entries get copied to all ranks.

        `attention_mask` isn't used by Ulysses, because it's typically too large when it's 4D, and position_ids is just 1D, therefore it's much much smaller and consumes little GPU memory.

        Arguments:
        - `dl`: an existing DataLoader object to wrap
        - `sp_rank`: SP rank
        - `sp_group`: SP group
        - `sp_world_size`: SP world size
        - `device`: cuda device

        Returns:
            Another DataLoader object
        """

        self.dl = dl
        self.sp_rank = sp_rank
        self.sp_group = sp_group
        self.sp_world_size = sp_world_size
        self.device = device

        self.iter = iter(dl)
        self.micro_batches: list[Any] = []

    def __len__(self):
        return len(self.dl) * self.sp_world_size

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.micro_batches) == 0:
            self.refill()

        return self.micro_batches.pop(0)

    def refill(self):
        # reset the iterator if StopIteration arrives, and re-raise it to allow multiple epochs to run
        try:
            batch = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dl)
            raise StopIteration
        micro_batches = defaultdict(dict)
        # XXX: replace with more efficient all-to-all?

        # we have batches of variable seqlen so in order to do all_gather on batches - we need to know the exact length of each tensor on each rank
        seqlen = torch.tensor(batch["input_ids"].shape[1], dtype=torch.int64, device=self.device)
        seqlens = [torch.zeros(1, dtype=torch.int64, device=self.device) for _ in range(self.sp_world_size)]
        dist.all_gather(seqlens, seqlen, group=self.sp_group)
        seqlens = [x[0].item() for x in seqlens] # [6,6] length of sequences in both gpus
        for k in batch.keys():
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(self.device)
                if seqlen != batch[k].shape[1]:
                    raise ValueError(
                        f"{k}'s shape {batch[k].shape} must match input_ids's shape {batch['input_ids'].shape}")
                with torch.no_grad():
                    tensor_list = [
                        torch.zeros((batch[k].shape[0], seqlens[i]), dtype=batch[k].dtype, device=batch[k].device)
                        for i in range(self.sp_world_size)
                    ]
                    # All ranks get the current batch that is being processed in each GPU
                    dist.all_gather(tensor_list, batch[k], group=self.sp_group)
            else:
                tensor_list = [None for _ in range(self.sp_world_size)]
                dist.all_gather_object(tensor_list, batch[k], group=self.sp_group)
            
            # We then arrange the batches by the ranks they are from
            for rank, tensor in enumerate(tensor_list):
                micro_batches[rank][k] = tensor

        del tensor_list
        del batch
        for batch in micro_batches.values():
            seq_length = len(batch["input_ids"][0])

            if seq_length % self.sp_world_size != 0:
                raise ValueError(f"batch's seqlen={seq_length} isn't divisible by sp-size={self.sp_world_size}")
            chunk_len = seq_length // self.sp_world_size

            # because we have to gather logits from all sp ranks we have to do the loss function ourselves
            # therefore remove labels to avoid an attempt to calculate loss by transformers
            # # Shift operation:
            # labels_original = [1, 10, 10, 10, 2, 2]
            # labels_padded   = [1, 10, 10, 10, 2, 2, -100]  # Pad with -100
            # shift_labels    = [10, 10, 10, 2, 2, -100]     # Skip first, this is the target
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
                #print(f"chunk {chunk_len * self.sp_rank:chunk_len * (self.sp_rank + 1)}")
                batch[k] = batch[k][:, chunk_len * self.sp_rank:chunk_len * (self.sp_rank + 1)].cpu()
            self.micro_batches.append(batch)

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


ds = torch.utils.data.TensorDataset(input_ids, position_ids)

sampler = DistributedSampler(ds,shuffle=False)
dl = torch.utils.data.DataLoader(ds, batch_size=micro_batch_size, collate_fn=collate_fn, sampler=sampler)

ranker_g = deepspeed.runtime.sequence_parallel.parallel_state_sp.initialize_sequence_groups(2, 4)
# sp_group = groups._get_sequence_parallel_group()
# sp_world_size = groups._get_sequence_parallel_world_size()
# sp_rank = groups._get_sequence_parallel_rank()

# print(f"sp rank is :{sp_rank}")
# print(f"sp group is :{sp_group}")
# print(f"sp world size is :{sp_world_size}")

# dl = UlyssesSPDataLoaderAdapter(
#     dl,
#     sp_rank=sp_rank,
#     sp_group=sp_group,
#     sp_world_size=sp_world_size,
#     device=model.device,
# )
num_groups=2
print(f"len of dl is {len(dl)}")
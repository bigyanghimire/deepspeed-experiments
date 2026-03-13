# train.py with batch_size > 1

from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF, UlyssesSPDataLoaderAdapter
from deepspeed.runtime.utils import move_to_device
from deepspeed.utils import groups
from torch import tensor
from transformers import AutoModelForCausalLM
import deepspeed
import deepspeed.comm as dist
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
model_name_or_path = 'Qwen/Qwen3-0.6B'
seq_length = 64
sequence_parallel_size = 2
micro_batch_size = 4  # CHANGED: Now batch size = 4

config_dict = {
    "train_micro_batch_size_per_gpu": micro_batch_size,  # CHANGED
    "zero_optimization": {
        "stage": 3,
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


def load_wikitext_data(tokenizer_name, seq_length, batch_size, world_size, rank):
    """Load wikitext dataset for training."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"[Rank {rank}] Loading wikitext dataset...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train[:5%]')

    # Filter empty texts
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 50)
    def ulysses_collate_fn(batch):
        input_ids = torch.stack([x["input_ids"] for x in batch])
        seq_len = input_ids.size(1)

        position_ids = torch.arange(seq_len).unsqueeze(0).expand_as(input_ids)

        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=input_ids.clone(),
        )
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
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, sampler=sampler, num_workers=2, collate_fn=ulysses_collate_fn)

    print(f"[Rank {rank}] Dataset loaded: {len(tokenized_dataset)} examples, vocab_size={tokenizer.vocab_size}")
    return dataloader, tokenizer.vocab_size


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
sp_world_size = groups._get_sequence_parallel_world_size()
sp_rank = groups._get_sequence_parallel_rank()

# Normal training loop
# for iter, batch in enumerate(dl):
#     print(f"vanilla dl batch {batch} from rank {dist.get_rank()} at iter {iter}")
local_rank = int(deepspeed.comm.get_rank())
world_size = deepspeed.comm.get_world_size()
dataloader_gpt, actual_vocab_size = load_wikitext_data(
    tokenizer_name=model_name_or_path,
    seq_length=seq_length,
    batch_size=micro_batch_size,
    world_size=world_size,
    rank=local_rank
)
dl = UlyssesSPDataLoaderAdapter(
    dataloader_gpt,
    sp_rank=sp_rank,
    sp_group=sp_group,
    sp_world_size=sp_world_size,
    device=model.device,
)

num_epochs=2
# Normal training loop
for i in range(0,num_epochs):
    for iter, batch in enumerate(dl):
        batch = move_to_device(batch, model.device)
        
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

        # Aggregate loss across SP ranks
        losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=sp_group)
        good_tokens = (shift_labels != -100).view(-1).sum()
        good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=sp_group)
        total_loss = sum(losses_per_rank[rank] * good_tokens_per_rank[rank] for rank in range(sp_world_size))
        total_good_tokens = sum(good_tokens_per_rank)
        loss = total_loss / max(total_good_tokens, 1)

        if dist.get_rank() == 0:
            print(f"Iteration {iter}: loss={loss.item():.4f}")

        model.backward(loss)
        model.step()  # Don't forget optimizer step!
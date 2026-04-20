"""
FSDP2 Training Script for Llama-3.1-8B on WikiText
Requires: PyTorch 2.4+, transformers, datasets
"""

import os
import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
from typing import Optional


def setup_distributed():
    """Initialize distributed training environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    torch.cuda.set_device(local_rank)
    
    return local_rank, world_size, rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


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

def get_wikitext_dataloader(
    tokenizer,
    split: str = "train",
    seq_length: int = 1024,
    batch_size: int = 2,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
):
    """Load and preprocess WikiText-103 dataset."""
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    
    def tokenize_function(examples):
        """Tokenize and concatenate text into fixed-length sequences."""
        # Tokenize
        tokenized = tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        
        # Concatenate all tokens
        concatenated = {
            "input_ids": sum(tokenized["input_ids"], [])
        }
        
        # Chunk into seq_length
        total_length = len(concatenated["input_ids"])
        # Drop remainder to get full sequences
        total_length = (total_length // seq_length) * seq_length
        
        result = {
            "input_ids": [
                concatenated["input_ids"][i : i + seq_length]
                for i in range(0, total_length, seq_length)
            ],
            "labels": [
                concatenated["input_ids"][i : i + seq_length]
                for i in range(0, total_length, seq_length)
            ],
        }
        
        return result
    
    # Process dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    
    # Set format for PyTorch
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    
    # Create distributed sampler
    sampler = DistributedSampler(
        tokenized_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=(split == "train"),
        drop_last=True,
    )
    
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


def apply_fsdp2(model, device_mesh: Optional[DeviceMesh] = None):
    """Apply FSDP2 to model using fully_shard API."""
    
    # Mixed precision policy
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    
    # Shard the model layer by layer for memory efficiency
    # For Llama, shard each transformer block
    for layer_id, layer in enumerate(model.model.layers):
        fully_shard(
            layer,
            mesh=device_mesh,
            mp_policy=mp_policy,
        )
    
    # Shard the full model
    fully_shard(
        model,
        mesh=device_mesh,
        mp_policy=mp_policy,
    )
    
    return model


def train_step(model, batch, optimizer, device):
    """Single training step."""
    input_ids = batch["input_ids"].to(device)
    #labels = batch["labels"].to(device)
    
    # Forward pass
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()


def main():
    # Configuration
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    SEQ_LENGTH = 512
    BATCH_SIZE = 1  # Per GPU
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 1
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_STEPS = 30  # Set to None to train full epoch
    CHECKPOINT_DIR = "./checkpoints"
    
    # Setup distributed
    local_rank, world_size, rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    # Create device mesh for FSDP2
    device_mesh = DeviceMesh("cuda", torch.arange(world_size))
    
    is_main = rank == 0
    
    if is_main:
        print(f"Starting FSDP2 training on {world_size} GPUs")
        print(f"Model: {MODEL_NAME}")
        print(f"Sequence Length: {SEQ_LENGTH}")
        print(f"Batch Size per GPU: {BATCH_SIZE}")
        print(f"Effective Batch Size: {BATCH_SIZE * world_size * GRADIENT_ACCUMULATION_STEPS}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if is_main:
        print("Loading model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        use_cache=False,  # Disable KV cache for training
    )
    
    model.to(device)
    
    # Apply FSDP2
    if is_main:
        print("Applying FSDP2...")
    
    model = apply_fsdp2(model, device_mesh)
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Load data
    if is_main:
        print("Loading dataset...")
    
    train_dataloader = get_wikitext_dataloader(
        tokenizer,
        split="train",
        seq_length=SEQ_LENGTH,
        batch_size=BATCH_SIZE,
        rank=rank,
        world_size=world_size,
    )

    # train_dataloader, actual_vocab_size = load_wikitext_data(
    #     tokenizer_name=MODEL_NAME,
    #     seq_length=SEQ_LENGTH,
    #     batch_size=BATCH_SIZE,
    #     world_size=world_size,
    #     rank=rank
    # )

    if is_main:
        print(f"Dataset loaded: {len(train_dataloader)} batches")
    
    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(NUM_EPOCHS):
        train_dataloader.sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        start_time = time.time()
        
        for step, batch in enumerate(train_dataloader):
            if MAX_STEPS and global_step >= MAX_STEPS:
                break
            
            # Training step
            loss = train_step(model, batch, optimizer, device)
            
            epoch_loss += loss
            global_step += 1
            
            # Logging
            if is_main == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (global_step * BATCH_SIZE * world_size * SEQ_LENGTH) / elapsed
                print(
                    f"Epoch {epoch} | Step {global_step} | "
                    f"Loss: {loss:.4f} | "
                    f"Tokens/sec: {tokens_per_sec:.0f}"
                )
            
            # Checkpoint saving (optional)
            if global_step % 500 == 0 and is_main:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"step_{global_step}")
                os.makedirs(checkpoint_path, exist_ok=True)
                
                # Save checkpoint (FSDP2 compatible)
                state_dict = get_model_state_dict(model)
                torch.save(
                    {
                        "model": state_dict,
                        "optimizer": optimizer.state_dict(),
                        "step": global_step,
                    },
                    os.path.join(checkpoint_path, "checkpoint.pt"),
                )
                print(f"Checkpoint saved at step {global_step}")
        
        if is_main:
            avg_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch} completed | Average Loss: {avg_loss:.4f}")
    
    # Final checkpoint
    if is_main:
        print("Saving final checkpoint...")
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "final")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        state_dict = get_model_state_dict(model)
        torch.save(
            {
                "model": state_dict,
                "optimizer": optimizer.state_dict(),
                "step": global_step,
            },
            os.path.join(checkpoint_path, "checkpoint.pt"),
        )
        print("Training completed!")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
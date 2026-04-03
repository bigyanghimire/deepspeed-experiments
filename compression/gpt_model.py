#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

"""
Training script demonstrating bf16_master_weights_and_grads and bf16_optimizer_states
options in DeepSpeed for reduced memory usage.

Usage:
    deepspeed --num_gpus=1 train.py --deepspeed_config configs/baseline.json
    deepspeed --num_gpus=1 train.py --deepspeed_config configs/bf16_master_wg.json
    deepspeed --num_gpus=1 train.py --deepspeed_config configs/bf16_full.json
"""

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import deepspeed
from deepspeed.accelerator import get_accelerator
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig
from transformers import AutoModelForCausalLM, GPT2Config
import os
from torch.profiler import profile, record_function, ProfilerActivity
import sys
model_name_or_path = 'meta-llama/Llama-3.1-8B'

# INCLUDE_LIST = ["deepspeed", "torch", "gpt_model.py"] 

# def tracer(frame, event, arg):
#     # Get the full path of the file currently being executed
#     filepath = frame.f_code.co_filename
    
#     # Check if the current file path contains any of our "interesting" keywords
#     if any(lib in filepath for lib in INCLUDE_LIST):
#         rank = os.environ.get("RANK", "0")
#         func = frame.f_code.co_name
#         # Just the file name, not the whole path, for cleaner logs
#         filename = filepath.split("/")[-1]

#         if event == "call":
#             print(f"[rank {rank}] CALL   {func} ({filename})")
#         elif event == "return":
#             print(f"[rank {rank}] RETURN {func}")

#     # We must return the tracer function to continue tracing the next line/call
#     return tracer

# # Only set the trace for the main rank to avoid log-spamming in distributed training
# if os.environ.get("RANK", "0") == "0":
#     sys.settrace(tracer)

def get_args():
    parser = argparse.ArgumentParser(description="BF16 Low-Precision Master Weights Demo")

    # Model configuration
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--vocab_size", type=int, default=50000, help="Vocabulary size")

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length")
    parser.add_argument("--num_steps", type=int, default=20, help="Number of training steps")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup steps before measuring")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")

    # DeepSpeed configuration
    parser.add_argument("--deepspeed_config", type=str, required=True, help="Path to DeepSpeed config")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    # Logging
    parser.add_argument("--log_interval", type=int, default=5, help="Log interval")

    # Activation checkpointing
    parser.add_argument("--activation_checkpointing", action="store_true", help="Enable activation checkpointing")

    # Loss logging
    parser.add_argument("--loss_log_file", type=str, default=None, help="File to save loss values for plotting")

    # Dataset
    parser.add_argument("--use_real_data", action="store_true", help="Use wikitext dataset instead of random data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()

def load_wikitext_data_packed(tokenizer_name, seq_length, batch_size, world_size, rank):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"[Rank {rank}] Loading wikitext dataset...")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train[:5%]')
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 50)

    # Tokenize without padding/truncation — get raw tokens
    def tokenize_function(examples):
        return tokenizer(examples['text'], return_tensors=None, truncation=False, padding=False)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    # ── Pack all tokens into chunks of exactly seq_length ──────────
    def pack_sequences(examples):
        # Flatten all token ids into one big list
        all_tokens = []
        for ids in examples['input_ids']:
            all_tokens.extend(ids)
            all_tokens.append(tokenizer.eos_token_id)  # separator between docs

        # Chunk into seq_length blocks, drop last incomplete chunk
        chunks = []
        for i in range(0, len(all_tokens) - seq_length, seq_length):
            chunks.append(all_tokens[i : i + seq_length])

        return {'input_ids': chunks}

    packed_dataset = tokenized_dataset.map(
        pack_sequences,
        batched=True,
        batch_size=1000,
        remove_columns=tokenized_dataset.column_names,
        desc=f"Packing into seq_length={seq_length}"
    )
    packed_dataset.set_format(type='torch', columns=['input_ids'])

    def ulysses_collate_fn(batch):
        input_ids = torch.stack([x["input_ids"] for x in batch])
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand_as(input_ids)
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=input_ids.clone(),
        )

    sampler = DistributedSampler(packed_dataset)
    dataloader = DataLoader(
        packed_dataset, batch_size=batch_size,
        sampler=sampler,
        collate_fn=ulysses_collate_fn
    )

    print(f"[Rank {rank}] Packed dataset: {len(packed_dataset)} chunks of {seq_length} tokens")
    return dataloader, tokenizer.vocab_size

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


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_memory_stats():
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "peak": 0}

    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak": torch.cuda.max_memory_allocated(),
    }


def format_memory(bytes_val):
    """Format bytes to human readable string."""
    gb = bytes_val / (1024 ** 3)
    return f"{gb:.2f} GB"


def main():
    args = get_args()

    # Initialize distributed
    deepspeed.init_distributed()
    local_rank = args.local_rank
    if local_rank == -1:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        # local_rank = int(deepspeed.comm.get_rank())

    world_size = deepspeed.comm.get_world_size()
    global_rank = deepspeed.comm.get_rank()
    device = get_accelerator().device_name(local_rank)
    print(f"local rank {local_rank}\n")
    torch.cuda.set_device(local_rank)

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # Load real data if requested
    dataloader = None
    dataloader, actual_vocab_size = load_wikitext_data_packed(
        tokenizer_name=model_name_or_path,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        world_size=world_size,
        rank=global_rank
    )

    # Create model
    print(f"[Rank {global_rank}] Creating model with hidden_dim={args.hidden_dim}, "
          f"num_layers={args.num_layers}, num_heads={args.num_heads}, vocab_size={actual_vocab_size}")

    # config = AutoConfig.from_pretrained(
    #     model_name_or_path
    # )

    # config.vocab_size = actual_vocab_size
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    total_params, trainable_params = count_parameters(model)
    print(f"[Rank {global_rank}] Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Enable activation checkpointing if requested
    if args.activation_checkpointing:
        #model.enable_activation_checkpointing()
        model.gradient_checkpointing_enable()
        print(f"[Rank {global_rank}] Activation checkpointing enabled")

    # Read config to check if torch_autocast is enabled
    import json
    with open(args.deepspeed_config, 'r') as f:
        ds_config = json.load(f)

    use_autocast = ds_config.get("torch_autocast", {}).get("enabled", False)
    autocast_dtype_str = ds_config.get("torch_autocast", {}).get("dtype", "torch.bfloat16")
    autocast_dtype = torch.bfloat16 if "bfloat16" in autocast_dtype_str else torch.float16

    # Initialize DeepSpeed - use config file path directly (not via args to avoid conflict)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=args.deepspeed_config,
    )

    print(f"[Rank {global_rank}] DeepSpeed initialized with config: {args.deepspeed_config}")

    mem_after_init = get_memory_stats()
    print(f"[Rank {global_rank}] Memory after init: allocated={format_memory(mem_after_init['allocated'])}, "
          f"reserved={format_memory(mem_after_init['reserved'])}")

    # Training loop
    model_engine.train()
    loss_fn = nn.CrossEntropyLoss()

    total_time = 0
    step_times = []
    loss_history = []

    num_steps = 30  # Set your desired total steps here
    global_step = 0   # Initialize a counter

    if global_rank == 0:
        profiler_context = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=6, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"./profiler_traces",
                worker_name="rank0"
            ),
            record_shapes=True
        )
    else:
        from contextlib import nullcontext
        profiler_context = nullcontext()

    with profiler_context as profiler:    
        for step, batch in enumerate(dataloader):
            step_start_time = time.time()
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            
            outputs = model_engine(**batch)
            loss = outputs.loss
            
            if global_rank == 0:
                print(f"Step {global_step}: loss={loss.item():.4f}")
                
            model_engine.backward(loss)
            model_engine.step()
            if profiler is not None and global_rank == 0:
                profiler.step()
            # 1. Increment the counter
            global_step += 1
            # 2. Check if we've reached the limit
            if global_step >= num_steps:
                break  # Exit the inner loop
                


    # Setup data iterator
    # if dataloader is not None:
    #     data_iter = iter(dataloader)

    # for step in range(args.num_steps):
    #     start_time = time.time()

    #     # Get input data
    #     if dataloader is not None:
    #         try:
    #             batch = next(data_iter)
    #         except StopIteration:
    #             data_iter = iter(dataloader)
    #             batch = next(data_iter)
    #         input_ids = batch['input_ids'].to(device)
    #         labels = input_ids.clone()  # For language modeling, labels = input_ids
    #     else:
    #         # Generate random input data
    #         input_ids = torch.randint(0, actual_vocab_size, (args.batch_size, args.seq_length), device=device)
    #         labels = torch.randint(0, actual_vocab_size, (args.batch_size, args.seq_length), device=device)

    #     # Forward pass with optional autocast
    #     if use_autocast:
    #         with torch.autocast(device_type="cuda", dtype=autocast_dtype):
    #             # logits = model_engine(input_ids)
    #             # loss = loss_fn(logits.view(-1, actual_vocab_size), labels.view(-1))
    #             outputs = model_engine(input_ids=input_ids, labels=labels)
    #             loss = outputs.loss
    #     else:
    #         # logits = model_engine(input_ids)
    #         # loss = loss_fn(logits.view(-1, actual_vocab_size), labels.view(-1))
    #         outputs = model_engine(input_ids=input_ids, labels=labels)
    #         loss = outputs.loss
    #     # Backward pass - use PyTorch-style backward
    #     backward_start_time = time.time()
    #     model_engine.backward(loss)
    #     backward_time = backward_start_time - time.time()

    #     # Optimizer step
    #     model_engine.step()

    #     step_time = time.time() - start_time

    #     if step >= args.warmup_steps:
    #         step_times.append(step_time)

    #     # Record loss for plotting
    #     loss_history.append((step, loss.item()))
    #     if global_rank == 0:
    #         print(f"Step {step}: loss={loss.item():.4f}")
        # if step % args.log_interval == 0 or step == args.num_steps - 1:
        #     mem_stats = get_memory_stats()
        #     print(f"[Rank {global_rank}] Step {step}: loss={loss.item():.4f}, "
        #           f"time={step_time:.3f}s, "
        #           f"alloc_mem={format_memory(mem_stats['allocated'])}, "
        #           f"peak_mem={format_memory(mem_stats['peak'])}")

    # # Final statistics
    # final_mem = get_memory_stats()
    # avg_step_time = sum(step_times) / len(step_times) if step_times else 0

    # print("\n" + "=" * 60)
    # print(f"[Rank {global_rank}] FINAL RESULTS")
    # print(f"  Config: {args.deepspeed_config}")
    # print(f"  Model: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}")
    # print(f"  Parameters: {total_params:,}")
    # print(f"  Batch size: {args.batch_size}, Seq length: {args.seq_length}")
    # print(f"  Average step time: {avg_step_time:.3f}s")
    # print(f"  Peak memory: {format_memory(final_mem['peak'])}")
    # print(f"  Final allocated memory: {format_memory(final_mem['allocated'])}")
    # print("=" * 60)

    # # Output machine-readable summary line for parsing
    # print(f"SUMMARY: config={args.deepspeed_config} batch_size={args.batch_size} params={total_params} "
    #       f"peak_mem_bytes={final_mem['peak']} alloc_mem_bytes={final_mem['allocated']} "
    #       f"avg_step_time={avg_step_time:.4f}")

    # # Save loss history to file if requested (only rank 0)
    # if args.loss_log_file and global_rank == 0:
    #     import csv
    #     with open(args.loss_log_file, 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(['step', 'loss'])
    #         writer.writerows(loss_history)
    #     print(f"Loss history saved to: {args.loss_log_file}")
    # #model_engine.save_checkpoint("model/")
    # model_engine.destroy()


if __name__ == "__main__":
    main()

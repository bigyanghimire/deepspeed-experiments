# train.py with batch_size > 1

from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF, UlyssesSPDataLoaderAdapter2
#from deepspeed.runtime.sequence_parallel.ulysses_sp2 import UlyssesSPAttentionHF, UlyssesSPDataLoaderAdapter

from deepspeed.runtime.utils import move_to_device
from deepspeed.utils import groups
from torch import tensor
from transformers import AutoModelForCausalLM
import deepspeed
import deepspeed.comm as dist
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import time
from torch.profiler import profile, record_function, ProfilerActivity
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_length', type=int, default=16000, help='Input sequence length')
    parser.add_argument('--seq_parallel_size', type=int, default=8, help='sequence parallel size')
    args = parser.parse_args()

    model_name_or_path = 'meta-llama/Llama-3.2-1B'
    
    # Use the value from the command line argument
    seq_length = args.seq_length
    sequence_parallel_size = args.seq_parallel_size
    
    print(f"Training with Sequence Length: {seq_length} Sequence parallel size: {sequence_parallel_size}")
    micro_batch_size = 2  # CHANGED: Now batch size = 4

    config_dict = {
        "train_micro_batch_size_per_gpu": micro_batch_size,  # CHANGED
        "zero_optimization": {
            "stage": 2
        },
        "bf16": {
            "enabled": True,
            "bf16_master_weights_and_grads": True,
            "bf16_optimizer_states": True
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

    def load_wikitext_data_packed(tokenizer_name, seq_length, batch_size, world_size, rank):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        print(f"[Rank {rank}] Loading wikitext dataset...")
        
        if dist.get_rank() == 0:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        dist.barrier()
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
        if dist.get_rank() == 0:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        dist.barrier()
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
        sampler = DistributedSampler(tokenized_dataset)
        dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, sampler=sampler, collate_fn=ulysses_collate_fn)

        print(f"[Rank {rank}] Dataset loaded: {len(tokenized_dataset)} examples, vocab_size={tokenizer.vocab_size}")
        return dataloader, tokenizer.vocab_size


    dist.init_distributed(dist_backend='nccl', dist_init_required=True)

    # Ulysses injection into HF Transformers
    mpu = UlyssesSPAttentionHF.register_with_transformers(
        model_name_or_path=model_name_or_path,
        core_attn_implementation="flash_attention_2",
        sequence_parallel_size=sequence_parallel_size,
        micro_batch_size=micro_batch_size,
        seq_length=seq_length,
        seq_length_is_variable=True,
    )

    # Deepspeed setup
    if dist.get_rank() == 0:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    dist.barrier()
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    print(f"gradient checkpointing enabled: {model.is_gradient_checkpointing}")  
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
    dataloader_gpt, actual_vocab_size = load_wikitext_data_packed(
        tokenizer_name=model_name_or_path,
        seq_length=seq_length,
        batch_size=micro_batch_size,
        world_size=world_size,
        rank=local_rank
    )
    dl = UlyssesSPDataLoaderAdapter2(
        dataloader_gpt,
        sp_rank=sp_rank,
        sp_group=sp_group,
        sp_world_size=sp_world_size,
        device=model.device,
    )

    if local_rank == 0:
        profiler_context = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=6, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"./grouped_profiler_traces_{sequence_parallel_size}",
                worker_name="rank0"
            ),
            record_shapes=True
        )
    else:
        from contextlib import nullcontext
        profiler_context = nullcontext()
    # Normal training loop
    num_epochs=1
    with profiler_context as profiler:
    # Normal training loop
        WARMUP = 10
        iter_count=0
        step_times = []
        for step, batch in enumerate(dl):
            if iter_count<30:
                torch.cuda.synchronize()
                start_time = time.time()
                batch = move_to_device(batch, model.device)
                
                outputs = model(**batch)
                # Loss calculation
                shift_labels = batch["shift_labels"]
                loss = model.module.loss_function(
                    logits=outputs.logits,
                    labels=None,
                    shift_labels=shift_labels,
                    vocab_size=model.module.config.vocab_size,
                )
                # Aggregate loss across SP ranks
                # losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=groups._get_data_parallel_group())
                # differentiable weighted per-shard-loss aggregation across ranks
                losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=sp_group)
                # special dealing with SFT that has prompt tokens that aren't used in loss computation
                good_tokens = (shift_labels != -100).view(-1).sum()
                good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=sp_group)
                total_loss = sum(losses_per_rank[rank] * good_tokens_per_rank[rank] for rank in range(sp_world_size))
                total_good_tokens = sum(good_tokens_per_rank)
                loss = total_loss / max(total_good_tokens, 1)

                if dist.get_rank() == 0:
                    print(f"Iteration {step}: loss={loss.item():.4f}")

                model.backward(loss)
                model.step()  # Don't forget optimizer step!
                torch.cuda.synchronize()
                step_time = time.time() - start_time
                if step >= WARMUP:
                    step_times.append(step_time)
                if local_rank == 0 and profiler is not None:
                    profiler.step()
                iter_count=iter_count+1
    if dist.get_rank() == 0:
        print(f"\n=== Summary over {len(step_times)} iterations ===")
        print(f"Step time: {np.mean(step_times):.2f}s ± {np.std(step_times):.2f}s (min={min(step_times):.2f}s, max={max(step_times):.2f}s)")
if __name__ == "__main__":
    main()
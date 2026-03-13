import os
import torch
import torch.distributed as dist

def run_all_to_all():
    # 1. Get environment variables set by Slurm
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 2. Initialize the process group
    # Note: MASTER_ADDR and MASTER_PORT must be set in the Slurm script or here
    
    # 3. Set the specific GPU for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # 4. Create tensors
    # Every GPU creates a tensor filled with its own Rank ID
    input_tensor = torch.full((world_size,), rank, dtype=torch.int64).to(device)
    output_tensor = torch.zeros((world_size,), dtype=torch.int64).to(device)
    
    if rank == 0:
        print(f"World Size: {world_size}")

    # 5. Perform All-to-All
    # This divides input_tensor into chunks and sends chunk 'j' to rank 'j'
    dist.all_to_all_single(output_tensor, input_tensor)
    
    # Synchronize so the prints don't overlap too messily
    dist.barrier()
    
    print(f"Rank {rank} (Local {local_rank}) - Received: {output_tensor.tolist()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    run_all_to_all()
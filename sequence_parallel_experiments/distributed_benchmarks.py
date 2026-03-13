import torch
import torch.distributed as dist
import time



def all_to_all_builtin(tensor, world_size):
    """Using PyTorch's built-in all-to-all"""
    output = torch.empty_like(tensor)
    
    input_list = list(tensor.chunk(world_size, dim=0))
    output_list = list(output.chunk(world_size, dim=0))
    
    dist.all_to_all(output_list, input_list)
    
    return output


def benchmark_scenario(scenario_name, participating_ranks, hidden_dim=4096, seq_len=16384, warmup=10, iters=100):
    """
    Benchmark all-to-all within a group of ranks
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Create process group for participating ranks
    if len(participating_ranks) < world_size:
        group = dist.new_group(participating_ranks)
    else:
        group = dist.group.WORLD
    
    # Only participating ranks do the benchmark
    if rank not in participating_ranks:
        return None
    
    group_size = len(participating_ranks)
    
    # Each rank has seq_len/group_size tokens, hidden_dim features
    chunk_size = seq_len // group_size
    tensor = torch.randn(group_size * chunk_size, hidden_dim, device='cuda')
    
    # Warmup
    for _ in range(warmup):
        output = torch.empty_like(tensor)
        input_list = list(tensor.chunk(group_size, dim=0))
        output_list = list(output.chunk(group_size, dim=0))
        dist.all_to_all(output_list, input_list, group=group)
        torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(iters):
        output = torch.empty_like(tensor)
        input_list = list(tensor.chunk(group_size, dim=0))
        output_list = list(output.chunk(group_size, dim=0))
        dist.all_to_all(output_list, input_list, group=group)
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iters * 1000
    
    # Calculate bandwidth
    # Each rank sends (group_size-1) chunks and receives (group_size-1) chunks
    data_per_rank = chunk_size * hidden_dim * 4 * (group_size - 1) * 2  # *4 for fp32, *2 for send+recv
    bandwidth_gbps = (data_per_rank / (avg_time_ms / 1000)) / 1e9
    
    if rank == participating_ranks[0]:  # Only first rank in group prints
        print(f"{scenario_name}:")
        print(f"  Time: {avg_time_ms:.3f} ms")
        print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
        print(f"  Group size: {group_size}, Seq len: {seq_len}, Hidden: {hidden_dim}")
        print()
    
    return avg_time_ms


def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print(f"Running on {world_size} GPUs")
        print("=" * 60)
        print()
    
    # Scenario 1: Ulysses-style - all 4 GPUs communicate together
    if world_size >= 4:
        benchmark_scenario(
            "Ulysses (4 GPUs global all-to-all)",
            participating_ranks=[0, 1, 2, 3],
            seq_len=32768  # Each GPU gets 8K tokens
        )
    
    # Scenario 2: Your approach - 2 groups of 2 GPUs each
    # if world_size >= 4:

    #     benchmark_scenario(
    #         "Group 0 (GPUs 0-1, local all-to-all)",
    #         participating_ranks=[0, 1, 2 ,3],
    #         seq_len=32768  # Each GPU gets 16K tokens
    #     )
 
    
    dist.barrier()
    
    if rank == 0:
        print("\nInterpretation:")
        print("- If Ulysses is faster: You have good global interconnect (NVSwitch/NVLink)")
        print("- If Groups are faster: You have locality benefits (e.g., PCIe topology, NUMA)")
        print("- Groups use 2x less communication per GPU but may have lower aggregate bandwidth")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
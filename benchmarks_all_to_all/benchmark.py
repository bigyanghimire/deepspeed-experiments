import torch
import torch.distributed as dist
import time
import json
import argparse
import os


def setup_distributed():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')

    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def benchmark_alltoall(world_size, message_size_mb, iterations=100):
    elements_per_pair = int(message_size_mb * 1024 * 1024 / 4)
    total_elements = world_size * elements_per_pair

    input_tensor = torch.randn(total_elements, device='cuda')
    output_tensor = torch.empty_like(input_tensor)

    # warmup
    for _ in range(10):
        dist.all_to_all_single(output_tensor, input_tensor)

    torch.cuda.synchronize()
    dist.barrier()

    start = time.perf_counter()

    for _ in range(iterations):
        dist.all_to_all_single(output_tensor, input_tensor)

    torch.cuda.synchronize()
    dist.barrier()
    end = time.perf_counter()

    avg_ms = (end - start) / iterations * 1000
    total_data_mb = message_size_mb * (world_size - 1)
    bandwidth = (total_data_mb / 1024) / (avg_ms / 1000)

    return {
        "type": "full",
        "world_size": world_size,
        "message_size_mb": message_size_mb,
        "time_ms": avg_ms,
        "bandwidth_gbps": bandwidth
    }


# def benchmark_grouped(world_size, group_size=4, message_size_mb=1, iterations=100):
#     rank = dist.get_rank()
#     assert world_size % group_size == 0

#     num_groups = world_size // group_size
#     group_id = rank // group_size

#     # create groups
#     groups = []
#     for g in range(num_groups):
#         ranks = list(range(g * group_size, (g + 1) * group_size))
#         groups.append(dist.new_group(ranks=ranks))

#     group = groups[group_id]

#     elements_per_pair = int(message_size_mb * 1024 * 1024 / 4)
#     total_elements = group_size * elements_per_pair

#     input_tensor = torch.randn(total_elements, device='cuda')
#     output_tensor = torch.empty_like(input_tensor)

#     # warmup
#     for _ in range(10):
#         dist.all_to_all_single(output_tensor, input_tensor, group=group)

#     torch.cuda.synchronize()
#     dist.barrier()

#     start = time.perf_counter()

#     for _ in range(iterations):
#         dist.all_to_all_single(output_tensor, input_tensor, group=group)

#     torch.cuda.synchronize()
#     dist.barrier()
#     end = time.perf_counter()

#     avg_ms = (end - start) / iterations * 1000
#     total_data_mb = message_size_mb * (group_size - 1)
#     bandwidth = (total_data_mb / 1024) / (avg_ms / 1000)

#     return {
#         "type": "grouped_4",
#         "world_size": world_size,
#         "group_size": group_size,
#         "message_size_mb": message_size_mb,
#         "time_ms": avg_ms,
#         "bandwidth_gbps": bandwidth
#     }


def main():
    print("in benchmarks")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results.json")
    args = parser.parse_args()

    rank, world_size, _ = setup_distributed()

    if world_size < 2:
        if rank == 0:
            print("Need at least 2 GPUs")
        return

    message_sizes = [1, 10, 50, 100, 200, 250]

    results = []

    if rank == 0:
        print(f"World size: {world_size}")

    for msg in message_sizes:
        if rank == 0:
            print(f"\nMessage size: {msg} MB")

        # full all-to-all
        res_full = benchmark_alltoall(world_size, msg)
        results.append(res_full)

        if rank == 0:
            print(f"Full: {res_full['time_ms']:.3f} ms | {res_full['bandwidth_gbps']:.2f} GB/s")

        # grouped (4 GPUs)
        # if world_size >= 4 and world_size % 4 == 0:
        #     res_grp = benchmark_grouped(world_size, 4, msg)
        #     results.append(res_grp)

        #     if rank == 0:
        #         print(f"Grouped(4): {res_grp['time_ms']:.3f} ms | {res_grp['bandwidth_gbps']:.2f} GB/s")

    if rank == 0:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
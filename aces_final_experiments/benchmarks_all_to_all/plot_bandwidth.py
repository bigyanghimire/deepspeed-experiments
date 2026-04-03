import matplotlib.pyplot as plt
import json
# Your data
with open('alltoall_benchmark_1304.json') as f:
    data = json.load(f)
    print(data)
# data=[
#   {
#     "type": "full",
#     "world_size": 8,
#     "message_size_mb": 1,
#     "time_ms": 0.8049968909472227,
#     "bandwidth_gbps": 8.49188062323607
#   },
#   {
#     "type": "grouped_4",
#     "world_size": 8,
#     "group_size": 4,
#     "message_size_mb": 1,
#     "time_ms": 0.05041363649070263,
#     "bandwidth_gbps": 58.11299687814225
#   },
#   {
#     "type": "full",
#     "world_size": 8,
#     "message_size_mb": 10,
#     "time_ms": 6.3673663046211,
#     "bandwidth_gbps": 10.735894831492317
#   },
#   {
#     "type": "grouped_4",
#     "world_size": 8,
#     "group_size": 4,
#     "message_size_mb": 10,
#     "time_ms": 0.21181223914027214,
#     "bandwidth_gbps": 138.31530755216752
#   },
#   {
#     "type": "full",
#     "world_size": 8,
#     "message_size_mb": 50,
#     "time_ms": 30.035947766155005,
#     "bandwidth_gbps": 11.379593467835974
#   },
#   {
#     "type": "grouped_4",
#     "world_size": 8,
#     "group_size": 4,
#     "message_size_mb": 50,
#     "time_ms": 0.8406891487538815,
#     "bandwidth_gbps": 174.24320894010313
#   }
# ]

# Extract data
msg_sizes = sorted(list(set(d["message_size_mb"] for d in data)))

full_bw = []
group4_bw = []

for m in msg_sizes:
    for d in data:
        if d["message_size_mb"] == m:
            if d["type"] == "full":
                full_bw.append(d["bandwidth_gbps"])
            elif d["type"] == "grouped_4":
                group4_bw.append(d["bandwidth_gbps"])

# Plot
plt.figure()

plt.plot(msg_sizes, full_bw, marker='o', label='Group size 8 (Full)')
plt.plot(msg_sizes, group4_bw, marker='o', label='Group size 4')

plt.xlabel("Message Size (MB per pair)")
plt.ylabel("Bandwidth (GB/s)")
plt.title("All-to-All Bandwidth vs Message Size")
plt.legend()
plt.grid(True)

# Save figure
plt.savefig("alltoall_bandwidth.png", dpi=300, bbox_inches='tight')

print("Saved to alltoall_bandwidth.png")
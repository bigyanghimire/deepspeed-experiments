import matplotlib.pyplot as plt
import json
# Your data
with open('alltoall_benchmark_1304.json') as f:
    data = json.load(f)
    print(data)

# Extract data
msg_sizes = sorted(list(set(d["message_size_mb"] for d in data)))

full_bw = []
group4_bw = []

for m in msg_sizes:
    for d in data:
        if d["message_size_mb"] == m:
            if d["type"] == "full":
                full_bw.append(d["time_ms"])
            elif d["type"] == "grouped_4":
                group4_bw.append(d["time_ms"])

# Plot
plt.figure()

plt.plot(msg_sizes, full_bw, marker='o', color='#1f77b4', label='Group size 8 (Full)')
plt.plot(msg_sizes, group4_bw, marker='s', color='#ff7f0e', label='Group size 4')

plt.xlabel("Message Size (MB per pair)")
plt.ylabel("Time (s)")
plt.title("All-to-All Time vs Message Size")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Save figure
plt.savefig("alltoall_latency.png", dpi=300, bbox_inches='tight')

print("Saved to alltoall_latency.png")
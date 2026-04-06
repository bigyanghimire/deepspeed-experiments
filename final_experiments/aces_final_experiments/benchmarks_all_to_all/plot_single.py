import matplotlib.pyplot as plt
import json
import numpy as np

# Load data
with open('alltoall_benchmark_1304.json') as f:
    data = json.load(f)

# Extract and sort message sizes
msg_sizes = sorted(list(set(d["message_size_mb"] for d in data)))

# Organize data (Mapping 'full' to Size 8 and 'grouped_4' to Size 4)
bw_8 = [d["bandwidth_gbps"] for m in msg_sizes for d in data if d["message_size_mb"] == m and d["type"] == "full"]
bw_4 = [d["bandwidth_gbps"] for m in msg_sizes for d in data if d["message_size_mb"] == m and d["type"] == "grouped_4"]
time_8 = [d["time_ms"] for m in msg_sizes for d in data if d["message_size_mb"] == m and d["type"] == "full"]
time_4 = [d["time_ms"] for m in msg_sizes for d in data if d["message_size_mb"] == m and d["type"] == "grouped_4"]

# --- Styling Configuration ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'grid.alpha': 0.3
})

fig, ax1 = plt.subplots(figsize=(8, 5))

# --- Plot Bandwidth (Primary Y-Axis) ---
line1, = ax1.plot(msg_sizes, bw_8, marker='o', markersize=7, linewidth=2.5, 
                  color='#1f77b4', linestyle='-', label='Bandwidth: World Size 8')
line2, = ax1.plot(msg_sizes, bw_4, marker='s', markersize=7, linewidth=2.5, 
                  color='#ff7f0e', linestyle='-', label='Bandwidth: World Size 4')

ax1.set_xlabel('Message Size (MB per pair)', fontsize=12, color='#1f77b4')
ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12, color='#1f77b4')
ax1.set_xscale('log') # Log scale for message sizes
ax1.set_xticks(msg_sizes)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.grid(True, linestyle='--', which='both')

# --- Plot Time (Secondary Y-Axis) ---
ax2 = ax1.twinx()
line3, = ax2.plot(msg_sizes, time_8, marker='o', markersize=7, linewidth=1.5, 
                  color='#1f77b4', linestyle='--', alpha=0.6, label='Time: World Size 8')
line4, = ax2.plot(msg_sizes, time_4, marker='s', markersize=7, linewidth=1.5, 
                  color='#ff7f0e', linestyle='--', alpha=0.6, label='Time: World Size 4')

ax2.set_ylabel('Execution Time (ms)', fontsize=12, color='#1f77b4')
ax2.set_yscale('log') # Log scale for execution time

# --- Legend and Layout ---
# Combine handles from both axes for a single legend
lines = [line1, line2, line3, line4]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True)

plt.title('All-to-All Performance: Grouped vs. Full Communication', pad=20)
fig.tight_layout()

# Save figure
plt.savefig("alltoall_performance_comparison.png", dpi=300, bbox_inches='tight')
print("Plot saved as alltoall_performance_comparison.png")
import matplotlib.pyplot as plt
import numpy as np

# 1. Convert your data into NumPy arrays
seq = np.array([1024, 2048, 4096, 8192, 16384, 32768])
seq_labels = ['1K', '2K', '4K', '8K', '16K', '32K']
bau_step = np.array([1.48, 1.51, 1.51, 1.55, 1.71, 2.14])
su_step = np.array([1.53, 1.55, 1.56, 1.59, 1.81, 2.38])

batch_size = 2
num_gpus = 8

# 2. Perform element-wise calculation (Fixed the TypeError)
su_total_throughput = (seq * batch_size) / su_step
bau_total_throughput = (seq * batch_size) / bau_step

# 3. Create the Plot
plt.rcParams.update({"font.family": "serif", "font.size": 11})
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(seq_labels, bau_total_throughput, marker='o', color='#1f77b4', 
        linewidth=2, label='BAU (Ours)')
ax.plot(seq_labels, su_total_throughput, marker='s', color='#ff7f0e', 
        linestyle='--', linewidth=2, label='SU (Baseline)')

ax.set_xlabel('Sequence Length')
ax.set_ylabel('Total Throughput (tokens/sec)')
ax.set_title(f'Total Throughput (Batch Size={batch_size}, {num_gpus} GPUs)')
ax.legend()
ax.grid(True, alpha=0.3)

# Optional: Add a secondary axis to show per-GPU throughput
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim()[0]/num_gpus, ax.get_ylim()[1]/num_gpus)
ax2.set_ylabel('Per-GPU Throughput (tokens/sec/GPU)')

plt.tight_layout()
plt.savefig('Images/throughput_comparison.png', bbox_inches='tight')
plt.show()
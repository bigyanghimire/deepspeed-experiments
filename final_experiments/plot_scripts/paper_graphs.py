import matplotlib.pyplot as plt
import numpy as np

# Global styling for a professional academic look
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "grid.alpha": 0.3,
    "savefig.dpi": 300  # High resolution for print
})

# --- DATA ---
seq_labels = ['1K', '2K', '4K', '8K', '16K', '32K']
bau_step = [1.48, 1.51, 1.51, 1.55, 1.71, 2.14]
su_step = [1.53, 1.55, 1.56, 1.59, 1.81, 2.38]

batch_labels = ['1', '2', '4', '8']
bau_batch = [1.51, 1.55, 1.68, 1.89]
su_batch = [1.50, 1.59, 1.78, 2.12]

comm_labels = ['1K', '2K', '4K', '8K', '16K', '32k']
bau_comm = [23, 28, 34, 48, 81, 132]
su_comm = [106, 176, 268, 507, 988, 1879]

# --- PLOT 1: END-TO-END STEP TIME (Line) ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(seq_labels, bau_step, marker='o', label='BAU (Ours)', linewidth=2)
ax.plot(seq_labels, su_step, marker='s', linestyle='--', label='SU (Baseline)', linewidth=2)
ax.set_xlabel('Sequence Length')
ax.set_ylabel('Step Time (seconds)')
ax.set_title('End-to-End Step Time (Batch Size=2)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('step_time_line.pdf',format='pdf')

# --- PLOT 2: SCALING WITH BATCH SIZE (Line) ---
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(batch_labels, bau_batch, marker='o', label='BAU (Ours)', linewidth=2)
ax.plot(batch_labels, su_batch, marker='s', linestyle='--', label='SU (Baseline)', linewidth=2)
ax.set_xlabel('Batch Size')
ax.set_ylabel('Step Time (seconds)')
ax.set_title('Step Time vs. Batch Size (Seq Len=8K)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('batch_scaling_line.pdf')

# --- PLOT 3: COMMUNICATION TIME (Grouped Bar) ---
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(comm_labels))
width = 0.35
ax.bar(x - width/2, bau_comm, width, label='BAU (Ours)', edgecolor='black', alpha=0.8)
ax.bar(x + width/2, su_comm, width, label='SU (Baseline)', edgecolor='black', alpha=0.8)
ax.set_xlabel('Sequence Length')
ax.set_ylabel('Time (ms)')
ax.set_title('All-to-All Communication Time')
ax.set_xticks(x)
ax.set_xticklabels(comm_labels)
ax.legend()
ax.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('comm_time_bar.pdf')
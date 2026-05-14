import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Sans"],
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    
    # Other styling
    "grid.alpha": 0.3,
    "savefig.dpi": 300
})

plt.rcParams.update({
    'legend.frameon': False,
    'lines.linewidth': 1,
})
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = False
# Data
batch_sizes = np.array([1, 2, 4, 8])

ulysses_time = np.array([1.04, 1.11, 1.31, 1.75])
ulysses_alltoall_percent = np.array([24.5, 26.9, 28.9, 33.8])

basp_time = np.array([1.03, 1.01, 1.14, 1.39])
basp_alltoall_percent = np.array([24.9, 15, 14.2, 0.5]) # <-- replace with real values

# Convert % → absolute times
ulysses_alltoall = ulysses_time * (ulysses_alltoall_percent / 100)
ulysses_compute = ulysses_time - ulysses_alltoall

basp_alltoall = basp_time * (basp_alltoall_percent / 100)
basp_compute = basp_time - basp_alltoall

# Plot
fig, ax = plt.subplots(figsize=(6, 4))

x = np.arange(len(batch_sizes))
width = 0.35

# Ulysses bars (left)
ax.bar(x - width/2, ulysses_compute, width, label='Others (Ulysses)', color='#FF4242')
bars_u = ax.bar(x - width/2, ulysses_alltoall, width,
                bottom=ulysses_compute, label='All-to-all (Ulysses)', color='#DD8452')

# BASP bars (right)
ax.bar(x + width/2, basp_compute, width, label='Others (BASP)', color='#645DD7')
bars_b = ax.bar(x + width/2, basp_alltoall, width,
                bottom=basp_compute, label='All-to-all (BASP)', color='#a29ee7')

# Labels
ax.set_xlabel('Batch Size', fontsize=13)
ax.set_ylabel('Average step time (s)', fontsize=13)

ax.set_xticks(x)
ax.set_xticklabels(batch_sizes)

# % labels inside all-to-all blocks
for i in range(len(batch_sizes)):
    # Ulysses
    ax.text(x[i] - width/2,
            ulysses_compute[i] + ulysses_alltoall[i]/2,
            f'{ulysses_alltoall_percent[i]:.1f}%',
            ha='center', va='center', fontsize=10, color='black', fontweight='bold')

    # BASP
    ax.text(x[i] + width/2,
            basp_compute[i] + basp_alltoall[i]/2,
            f'{basp_alltoall_percent[i]:.1f}%',
            ha='center', va='center', fontsize=10, color='black', fontweight='bold')

# Clean style (paper-friendly)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)

ax.legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig('batch_sizes_alltoall_comparison.pdf', dpi=300, bbox_inches='tight')
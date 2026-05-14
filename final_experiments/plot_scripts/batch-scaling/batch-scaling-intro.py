import matplotlib.pyplot as plt
import numpy as np

# Data
batch_sizes = np.array([1, 2, 4, 8])
ulysses_time = np.array([0.98, 1.11, 1.31, 1.75])  # seconds
alltoall_percent = np.array([24.5, 26.9, 28.9, 33.8])  # percentage

# Calculate absolute times
alltoall_time = ulysses_time * (alltoall_percent / 100)
compute_time = ulysses_time - alltoall_time

# Create figure with larger size for publication
fig, ax = plt.subplots(figsize=(8, 5))

# Stacked bar chart
width = 0.6
x = np.arange(len(batch_sizes))

bars1 = ax.bar(x, compute_time, width, label='Other', color='#2E86AB')
bars2 = ax.bar(x, alltoall_time, width, bottom=compute_time, 
               label='All-to-all', color='#E63946')

# Formatting
ax.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
ax.set_ylabel('Average time per Iteration (s)', fontsize=14, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(batch_sizes, fontsize=12)
ax.tick_params(axis='y', labelsize=12)

# Add percentage labels on all-to-all bars
for i, (bar, pct) in enumerate(zip(bars2, alltoall_percent)):
    height = bar.get_height()
    y_pos = compute_time[i] + height/2
    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
            f'{pct:.1f}%',
            ha='center', va='center', fontweight='bold', 
            color='white', fontsize=11)

# Legend
ax.legend(fontsize=12, loc='upper left', framealpha=0.95)

# Grid for readability
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)

# Add note

plt.tight_layout()
plt.savefig('alltoall_batch_intro.pdf', dpi=300, bbox_inches='tight')
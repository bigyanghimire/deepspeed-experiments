import numpy as np
import matplotlib.pyplot as plt

# Data
seq = np.array([2000, 4000, 8000, 16000, 32000])
baseline = np.array([1.5, 1.51, 1.55, 1.797405, 2.369966])
my_method = np.array([1.51, 1.52, 1.53, 1.676222, 2.123082])

# Calculate speedup and throughput
speedup = baseline / my_method
throughput_baseline = seq / baseline  # tokens/sec
throughput_my_method = seq / my_method

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Grouped Sequence Parallelism Performance Analysis', fontsize=16, fontweight='bold')

# Plot 1: Step Time vs Sequence Length
ax1 = axes[0, 0]
ax1.plot(seq, baseline, 'o-', label='Baseline (Ulysses)', linewidth=2, markersize=8)
ax1.plot(seq, my_method, 's-', label='Grouped SP', linewidth=2, markersize=8)
ax1.set_xlabel('Sequence Length', fontsize=12)
ax1.set_ylabel('Step Time (seconds)', fontsize=12)
ax1.set_title('Step Time vs Sequence Length', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)

# Plot 2: Speedup
ax2 = axes[0, 1]
ax2.plot(seq, speedup, 'D-', color='green', linewidth=2, markersize=8)
ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline')
ax2.set_xlabel('Sequence Length', fontsize=12)
ax2.set_ylabel('Speedup (Baseline / Grouped SP)', fontsize=12)
ax2.set_title('Speedup Over Baseline', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log', base=2)
# Add percentage labels
for i, (x, y) in enumerate(zip(seq, speedup)):
    ax2.annotate(f'{(y-1)*100:.1f}%', 
                xy=(x, y), 
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=9)

# Plot 3: Throughput
ax3 = axes[1, 0]
ax3.plot(seq, throughput_baseline, 'o-', label='Baseline (Ulysses)', linewidth=2, markersize=8)
ax3.plot(seq, throughput_my_method, 's-', label='Grouped SP', linewidth=2, markersize=8)
ax3.set_xlabel('Sequence Length', fontsize=12)
ax3.set_ylabel('Throughput (tokens/sec)', fontsize=12)
ax3.set_title('Throughput vs Sequence Length', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log', base=2)

# Plot 4: Overhead Analysis (absolute time difference)
ax4 = axes[1, 1]
time_saved = baseline - my_method
ax4.bar(range(len(seq)), time_saved, color=['green' if x > 0 else 'red' for x in time_saved])
ax4.axhline(y=0, color='black', linewidth=0.8)
ax4.set_xlabel('Sequence Length', fontsize=12)
ax4.set_ylabel('Time Saved (seconds)', fontsize=12)
ax4.set_title('Absolute Time Savings', fontsize=13, fontweight='bold')
ax4.set_xticks(range(len(seq)))
ax4.set_xticklabels(seq)
ax4.grid(True, alpha=0.3, axis='y')
# Add value labels on bars
for i, (x, y) in enumerate(zip(range(len(seq)), time_saved)):
    ax4.text(x, y + (0.01 if y > 0 else -0.01), 
            f'{y:.3f}s', 
            ha='center', 
            va='bottom' if y > 0 else 'top',
            fontsize=9)

plt.tight_layout()
plt.savefig('sp_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("=== Performance Summary ===")
print(f"\nSequence Length | Baseline (s) | Grouped SP (s) | Speedup | Time Saved (s)")
print("-" * 80)
for i in range(len(seq)):
    print(f"{seq[i]:>14} | {baseline[i]:>12.3f} | {my_method[i]:>14.3f} | {speedup[i]:>7.3f}x | {time_saved[i]:>14.3f}")

print(f"\n=== Overall Statistics ===")
print(f"Average speedup: {np.mean(speedup):.3f}x ({(np.mean(speedup)-1)*100:.1f}% improvement)")
print(f"Max speedup at seq_len={seq[np.argmax(speedup)]}: {np.max(speedup):.3f}x")
print(f"Total time saved (sum): {np.sum(time_saved):.3f}s")

# Scaling analysis
print(f"\n=== Scaling Behavior ===")
print(f"Baseline: {(baseline[-1]/baseline[0]):.2f}x slowdown from {seq[0]} to {seq[-1]}")
print(f"Grouped SP: {(my_method[-1]/my_method[0]):.2f}x slowdown from {seq[0]} to {seq[-1]}")
print(f"Better scaling: {((baseline[-1]/baseline[0]) / (my_method[-1]/my_method[0])):.2f}x improvement")
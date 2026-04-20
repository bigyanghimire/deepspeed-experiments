import matplotlib.pyplot as plt
import numpy as np

# Data
seq_sizes = [1024, 2048, 4096, 8192, 16384, 32768]
bau = [1.48, 1.51, 1.51, 1.55, 1.71, 2.14]
su = [1.53, 1.55, 1.56, 1.59, 1.81, 2.38]

# Calculate Speedup (Baseline / Method)
speedup = np.array(su) / np.array(bau)

# Plotting
plt.figure(figsize=(10, 6))
labels = [str(s) for s in seq_sizes]
bars = plt.bar(labels, speedup, color='skyblue', edgecolor='navy', alpha=0.8)

# Add a horizontal line at 1.0 (no speedup baseline)
plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Baseline (1.0x)')

# Formatting
plt.xlabel('Sequence Length', fontsize=12)
plt.ylabel(r'Speedup ($\frac{SU}{BAU}$)', fontsize=12)
plt.title('Speedup of BAU over SU Across Sequence Lengths', fontsize=14)
plt.ylim(0.9, 1.2)  # Focus range to see growth
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend()

# Adding data labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.3f}x', 
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('speedup_barchart.png')
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# 1. Data Setup
seq_labels = ['1K', '2K', '4K', '8K', '16K', '32K']
al_to_all_per = np.array([1.00, 1.6, 2.4, 4.5, 7.7, 11.3])
other_per = 100 - al_to_all_per # Remaining execution time

# --- PLOT 1: Bar Chart with Trend Line ---
plt.figure(figsize=(10, 6))
bars = plt.bar(seq_labels, al_to_all_per, color='skyblue', edgecolor='navy', alpha=0.7, label='All-to-All %')
plt.plot(seq_labels, al_to_all_per, marker='o', color='red', linestyle='--', linewidth=2, label='Growth Trend')

# Annotate values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.xlabel('Sequence Length', fontsize=12)
plt.ylabel('Percentage of Total Execution Time (%)', fontsize=12)
plt.title('Increase in All-to-All Communication Overhead', fontsize=14, fontweight='bold')
plt.ylim(0, 15) # Adjust limit to show labels clearly
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('Images/all_to_all_trend.png')
plt.show()


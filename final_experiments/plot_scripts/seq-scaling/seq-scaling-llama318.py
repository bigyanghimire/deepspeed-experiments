import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
# Global styling for a professional academic look

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    
    # Other styling
    "grid.alpha": 0.3,
    "savefig.dpi": 300
})
plt.rcParams.update({
    'legend.frameon': False,
    'lines.linewidth': 1,
    'axes.prop_cycle': cycler(color=['#645DD7', '#FF4242']),
})
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = False
# --- DATA ---
seq_labels = ['1K', '2K', '4K', '8K', '16K', '32K']
su_step = [2.62, 2.63, 2.72, 2.97, 3.68, 5.9]
bau_step = [2.47, 2.54, 2.67, 2.58, 3.01, 4.37]

# --- PLOT 1: END-TO-END STEP TIME (Line) ---
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(seq_labels, bau_step, marker='o', label='BASP (Ours)')
ax.plot(seq_labels, su_step, marker='s', linestyle='--', label='Ulysses-SP (Baseline)')
ax.set_xlabel('Sequence Length')
ax.set_ylabel('Average step time (s)')
ax.set_title('End-to-End Step Time (Batch Size=2)')
ax.legend()
plt.tight_layout()
plt.savefig('seq-scaling-llama318.pdf',format='pdf')

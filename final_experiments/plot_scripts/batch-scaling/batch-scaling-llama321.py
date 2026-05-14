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

# --- DATA ---

batch_labels = ['1', '2', '4', '8']
basp_batch = [0.99, 1.03, 1.14, 1.39]
su_batch = [0.98, 1.11, 1.78, 2.12]

# --- PLOT 2: SCALING WITH BATCH SIZE (Line) ---
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(batch_labels, basp_batch, marker='o', label='BASP (Ours)')
ax.plot(batch_labels, su_batch, marker='s', label='Ulysses-SP (Baseline)')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Step Time (seconds)')
ax.legend()
plt.tight_layout()
plt.savefig('batch_scaling.pdf')


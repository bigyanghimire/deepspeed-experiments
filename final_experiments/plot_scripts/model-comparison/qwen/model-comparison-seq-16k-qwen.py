import matplotlib.pyplot as plt
import numpy as np

# Data
models = ["Qwen 1.5-1.8B", "Qwen2.5-3B", "Qwen3-8B"]
ulysses = np.array([2.2, 3.62, 8.26])
ulysses_all_all_percent=np.array([37.7,35.2, 30.5])
basp = np.array([1.67, 2.95, 6.76])
basp_all_all_percent=np.array([16,19.1, 15.5])

plt.rcParams.update({
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

# Calculate actual all-to-all times
ulysses_alltoall = ulysses * (ulysses_all_all_percent / 100)
basp_alltoall = basp * (basp_all_all_percent / 100)

# Remaining time
ulysses_other = ulysses - ulysses_alltoall
basp_other = basp - basp_alltoall

# Plot setup
x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# Ulysses bars
bars1_other = ax.bar(
    x - width/2,
    ulysses_other,
    width,
    label='Others (Ulysses-SP)',
    color='#FF4242'
)

bars1_a2a = ax.bar(
    x - width/2,
    ulysses_alltoall,
    width,
    bottom=ulysses_other,
    label='All-to-All (Ulysses-SP) ',
    color='#DD8452'
)

# BASP bars
bars2_other = ax.bar(
    x + width/2,
    basp_other,
    width,
    label='Others (BASP)',
    color='#645DD7'
)

bars2_a2a = ax.bar(
    x + width/2,
    basp_alltoall,
    width,
    bottom=basp_other,
    label='All-to-All (BASP)',
    color='#a29ee7'
)

# Add percentage labels inside all-to-all section
for i in range(len(models)):
    ax.text(
        x[i] - width/2,
        ulysses_other[i] + ulysses_alltoall[i]/2,
        f"{ulysses_all_all_percent[i]}%",
        ha='center',
        va='center',
        fontsize=15,
        color='black',
        fontweight='bold'
    )

    ax.text(
        x[i] + width/2,
        basp_other[i] + basp_alltoall[i]/2,
        f"{basp_all_all_percent[i]}%",
        ha='center',
        va='center',
        fontsize=15,
        color='black',
        fontweight='bold'
    )

# Labels and formatting
ax.set_ylabel("Average step time (s)")
ax.set_title("Ulysses-SP vs BASP with All-to-All Communication Portion")
ax.set_xticks(x)
ax.set_xticklabels(models)
plt.xlabel("Model")
ax.legend()
ax.grid(False)

plt.tight_layout()
plt.savefig("model-comparison-seq-16k-qwen.pdf")
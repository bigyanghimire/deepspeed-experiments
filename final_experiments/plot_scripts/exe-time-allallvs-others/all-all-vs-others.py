import matplotlib.pyplot as plt
import numpy as np

# Data
cases = ["Llama 3.2-3B", "Qwen 2.5-3B"]
methods = ["Ulysses SP", "BASP"]

totals = np.array([
    [6.02, 4.56],
    [6.07, 4.77]
])

percent = np.array([
    [0.295,0.108],
    [0.272,0.112 ]
])

all_to_all = totals * percent
others = totals - all_to_all

x = np.arange(len(cases))
width = 0.28

fig, ax = plt.subplots(figsize=(6,4))

color_others = "#b0b0b0"
color_alltoall = "#4c4c6d"

# --- draw bars ---
bar_positions = []
bar_labels = []

for i in range(2):
    xpos = x + (i - 0.5) * width
    bar_positions.extend(xpos)
    bar_labels.extend([methods[i]] * len(cases))

    ax.bar(xpos, others[:, i], width,
           color=color_others, hatch='//',
           edgecolor='black', linewidth=0.8,
           label="Others" if i == 0 else "")

    ax.bar(xpos, all_to_all[:, i], width,
           bottom=others[:, i],
           color=color_alltoall,
           edgecolor='black', linewidth=0.8,
           label="All-to-All" if i == 0 else "")

    for j in range(len(cases)):
        ax.text(xpos[j],
                others[j, i] + all_to_all[j, i]/2,
                f"{percent[j, i]*100:.1f}%",
                ha='center', va='center',
                fontsize=9, color="white")

# --- first level ticks (methods) ---
ax.set_xticks(bar_positions)
ax.set_xticklabels(bar_labels, rotation=20)

# --- second level ticks (cases, centered) ---
for i, case in enumerate(cases):
    ax.text(x[i], -0.9, case, ha='center', va='top', fontsize=10)

ax.set_ylabel("Relative Time")
ax.set_ylim(0, 7)

ax.legend()

plt.tight_layout()
plt.savefig("all-all-vs-others.png", dpi=300)
plt.show()
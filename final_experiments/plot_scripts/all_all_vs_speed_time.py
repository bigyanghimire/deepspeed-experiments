import matplotlib.pyplot as plt
import numpy as np

# Data
seq_sizes = np.array([1024, 2048, 4096, 8192, 16384, 32768])

# Step times
bau_step = np.array([1.48, 1.51, 1.51, 1.55, 1.71, 2.14])
su_step  = np.array([1.53, 1.55, 1.56, 1.59, 1.81, 2.38])

# All-to-all times
bau_comm = np.array([23, 28, 34, 48, 81, 132])
su_comm  = np.array([106, 176, 268, 507, 988, 1879])

# Speedup ratios (BAU baseline)
step_speedup = su_step / bau_step
comm_speedup = su_comm / bau_comm

plt.figure(figsize=(8, 5))

plt.plot(
    seq_sizes,
    step_speedup,
    marker='o',
    linewidth=2,
    label='Step-time speedup ratio (BAU / SU)'
)

plt.plot(
    seq_sizes,
    comm_speedup,
    marker='s',
    linewidth=2,
    label='All-to-all speedup ratio (BAU / SU)'
)

plt.xscale('log', base=2)
plt.xticks(seq_sizes, labels=seq_sizes)

plt.xlabel('Sequence Length')
plt.ylabel('Speedup Ratio')
plt.title('Step-Time and Communication Speedup Ratios')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("speedup_ratio_comparison.png")
plt.show()
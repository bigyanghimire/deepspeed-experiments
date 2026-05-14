import matplotlib.pyplot as plt

gpus = [2, 4, 8]
seq_len = 8192
batch_size = 2  # replace with your real value

ulysses_time = [0.78, 0.48, 1.07]
basp_time = [0.72, 0.48, 1.0]

ulysses_tp = [(batch_size * seq_len) / t for t in ulysses_time]
basp_tp = [(batch_size * seq_len) / t for t in basp_time]

plt.plot(gpus, ulysses_tp, marker='o', label='Ulysses')
plt.plot(gpus, basp_tp, marker='s', label='BASP')

plt.xlabel("Number of GPUs")
plt.ylabel("Throughput (tokens/sec)")
plt.title("Throughput vs GPUs (Seq Len = 8192)")
plt.legend()
plt.grid()

plt.savefig('gpu-scaling.png',format='png')
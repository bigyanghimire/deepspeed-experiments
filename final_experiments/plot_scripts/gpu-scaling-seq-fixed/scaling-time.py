import matplotlib.pyplot as plt

gpus = [2, 4, 8]
seq_len = 8192
batch_size = 2  # replace with your real value

ulysses_time = [0.78, 0.48, 1.07]
basp_time = [0.72, 0.48, 1.0]


plt.plot(gpus, ulysses_time, marker='o', label='Ulysses')
plt.plot(gpus, basp_time, marker='s', label='BASP')

plt.xlabel("Number of GPUs")
plt.ylabel("Time (sec)")
plt.title("Time vs GPUs (Seq Len = 8192)")
plt.legend()
plt.grid()

plt.savefig('gpu-scaling-time.png',format='png')
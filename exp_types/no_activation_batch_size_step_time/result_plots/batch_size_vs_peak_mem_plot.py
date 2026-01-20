from pathlib import Path
from parse_results import parse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import re

def main():
    logs_dir = Path("../logs")
    batch_size_arr=[]
    baseline_step_times=[]
    bf16_step_times=[]

    for run_dir in sorted(logs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        results = parse(str(run_dir))
        print(f"results for {run_dir.name}:", results)
        batch_size_arr.append(results[0]["batch_size"])
        baseline_step_times.append(results[0]["peak_mem_bytes"] / (1024 ** 3))
        bf16_step_times.append(results[1]["peak_mem_bytes"] / (1024 ** 3))
    print("batch sizes",batch_size_arr)
    print("avsg",baseline_step_times)
    pdata={'batch_size':batch_size_arr,'baseline_peak_mem':baseline_step_times, 'bf16_peak_mem':bf16_step_times}
    df=pd.DataFrame.from_dict(pdata)
    sns.lineplot(x="batch_size",y='baseline_peak_mem',data=df, label="fp32")
    sns.lineplot(x="batch_size",y='bf16_peak_mem',data=df, label="bf16")
    #plt.yscale("log")
    # plt.axhline(y=n_b, linestyle="--",color="red",label="Network bandwidth")
    plt.legend()
    plt.title("Bf")
    plt.ylabel("Peak mem bytes")
    plt.xlabel("Batch size")

    plt.grid(True, which="both", linestyle=":")
    plt.savefig("batch_size_vs_peak_mem_plot.png")
    print(df)
if __name__ == "__main__":
    main()

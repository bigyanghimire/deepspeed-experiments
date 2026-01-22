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
        baseline_step_times.append(results[0]["avg_step_time"])
        bf16_step_times.append(results[1]["avg_step_time"])
    print("batch sizes",batch_size_arr)
    print("avsg",baseline_step_times)
    pdata={'batch_size':batch_size_arr,'baseline_avg_time':baseline_step_times, 'bf16_step_times':bf16_step_times}
    df=pd.DataFrame.from_dict(pdata)
    sns.lineplot(x="batch_size",y='baseline_avg_time',data=df, label="fp32")
    sns.lineplot(x="batch_size",y='bf16_step_times',data=df, label="bf16")
    #plt.yscale("log")
    # plt.axhline(y=n_b, linestyle="--",color="red",label="Network bandwidth")
    plt.legend()
    plt.title("Bf")
    plt.ylabel("Time")
    plt.xlabel("Batch size")

    plt.grid(True, which="both", linestyle=":")
    plt.savefig("batch_size_vs_time_plot.png")
    print(df)
if __name__ == "__main__":
    main()

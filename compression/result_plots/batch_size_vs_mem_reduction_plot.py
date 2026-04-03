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
    mem_reductions=[]
    for run_dir in sorted(logs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        results = parse(str(run_dir))
        print(f"results for {run_dir.name}:", results)
        batch_size_arr.append(results[0]["batch_size"])
        # baseline_step_times.append(results[0]["peak_mem_bytes"])
        # bf16_step_times.append(results[1]["peak_mem_bytes"])
        percent=((results[0]["peak_mem_bytes"] - results[1]["peak_mem_bytes"])/(results[0]["peak_mem_bytes"]))*100
        mem_reductions.append(percent)
    print("batch sizes",batch_size_arr)
    print("avsg",baseline_step_times)
    pdata={'batch_size':batch_size_arr,'percent':mem_reductions}
    df=pd.DataFrame.from_dict(pdata)
    sns.lineplot(x="batch_size",y='percent',data=df, label="%")
    #plt.yscale("log")
    # plt.axhline(y=n_b, linestyle="--",color="red",label="Network bandwidth")
    plt.legend()
    plt.title("Percent of mem reduction from baseline to bf16 as batch size increases")
    plt.ylabel("Percent of mem reduction")
    plt.xlabel("Batch size")

    plt.grid(True, which="both", linestyle=":")
    plt.savefig("batch_size_vs_mem_reduction_plot.png")
    print(df)
if __name__ == "__main__":
    main()

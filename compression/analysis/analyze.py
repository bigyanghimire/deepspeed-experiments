import torch
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np

def load_all_ranks(directory):
    """
    Finds all files matching 'flat_buffer_step_rank*.pt' and loads them.
    """
    # Adjust the pattern if your filenames look slightly different
    pattern = os.path.join(directory, "flat_buffer_step_rank*.pt")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No files found in {directory} matching 'flat_buffer_step_rank*.pt'")
        return {}

    data_by_rank = {}
    for f in files:
        try:
            checkpoint = torch.load(f, map_location="cpu", weights_only=True)
            # Use the rank saved in the dict, or fallback to parsing the filename
            rank = checkpoint.get("rank", f.split("rank")[-1].split(".")[0])
            data_by_rank[int(rank)] = checkpoint["tensor"][::1000000].numpy()
            print(f"Loaded Rank {rank} {data_by_rank[0].shape} from {os.path.basename(f)}")
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    return data_by_rank

def plot_gradients(data_by_rank):
    """
    Generates comparison plots for all loaded ranks.
    """
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Sort ranks to ensure legend and colors are consistent
    sorted_ranks = sorted(data_by_rank.keys())
    colors = sns.color_palette("husl", len(sorted_ranks))

    # 1. Distribution Plot (Gradients spread)
    plt.sca(axes[0])
    for i, rank in enumerate(sorted_ranks):
        grad = data_by_rank[rank]
        sns.kdeplot(grad, label=f"Rank {rank}", color=colors[i], bw_adjust=0.5)
    
    axes[0].set_title("Gradient Distribution Comparison", fontsize=15)
    axes[0].set_xlabel("Gradient Value")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # 2. Sequential Magnitude Plot (Along the flat buffer)
    plt.sca(axes[1])
    for i, rank in enumerate(sorted_ranks):
        grad = data_by_rank[rank]
        
        # Downsample or smooth for very large buffers
        # Here we use a simple decimation (every 100th point) for speed
        # or use a rolling mean for better pattern recognition
        if len(grad) > 10000:
            window = len(grad) // 500
            display_grad = np.convolve(grad, np.ones(window)/window, mode='valid')
        else:
            display_grad = grad

        plt.plot(display_grad, label=f"Rank {rank}", color=colors[i], alpha=0.6, linewidth=1)

    axes[1].set_title("Smoothed Gradient Values along Flat Buffer", fontsize=15)
    axes[1].set_xlabel("Buffer Index (Compressed)")
    axes[1].set_ylabel("Value")
    
    plt.tight_layout()
    plt.savefig("rank_gradient_analysis.png", dpi=300)
    print("\nPlot saved as 'rank_gradient_analysis.png'")
    plt.show()

if __name__ == "__main__":
    # Point this to the folder containing your .pt files
    PATH_TO_DATA = "./dump_dir" 
    
    ranks_data = load_all_ranks(PATH_TO_DATA)
    
    if ranks_data:
        # Quick summary stats
        print(f"\n{'Rank':<6} | {'Mean':<12} | {'Std':<12} | {'Max':<12}")
        print("-" * 50)
        for r in sorted(ranks_data.keys()):
            g = ranks_data[r]
            print(f"{r:<6} | {g.mean():<12.3e} | {g.std():<12.3e} | {g.max():<12.3e}")
        
        plot_gradients(ranks_data)
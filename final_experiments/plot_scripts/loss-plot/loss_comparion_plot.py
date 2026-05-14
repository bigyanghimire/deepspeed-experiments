import re
import argparse
import matplotlib.pyplot as plt
import os
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
plt.rcParams['axes.grid'] = False
def parse_and_plot(file_paths):
    plt.figure(figsize=(10, 6))
    
    # Define a pattern to capture iteration and loss
    pattern = r"Iteration (\d+): loss=([\d.]+)"
    
    # Keep track of if we actually found data to plot
    data_plotted = False

    for path in file_paths:
        iterations = []
        losses = []
        
        try:
            with open(path, 'r') as file:
                for line in file:
                    match = re.search(pattern, line)
                    if match:
                        iterations.append(int(match.group(1)))
                        losses.append(float(match.group(2)))

            if not iterations:
                print(f"Warning: No valid data found in '{path}'.")
                continue

            # Plot this file's data; use the filename as the label
            label_name = os.path.splitext(os.path.basename(path))[0]
            plt.plot(iterations, losses, marker='.', linestyle='-', label=f'Loss: {label_name}')
            data_plotted = True
            print(f"Parsed {len(iterations)} points from {path}.")

        except FileNotFoundError:
            print(f"Error: The file '{path}' does not exist.")
        except Exception as e:
            print(f"An unexpected error occurred with {path}: {e}")

    if data_plotted:
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Model convergence')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # Save and show
        save_name = "comparison_plot.pdf"
        plt.savefig(save_name, dpi=300)
        print(f"Plot saved as {save_name}. Opening window...")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse training logs and plot Iteration vs Loss for multiple files.")
    
    # nargs=2 ensures exactly two filenames are passed
    parser.add_argument("filenames", nargs=2, help="The paths to the two log files you want to compare.")
    
    args = parser.parse_args()
    
    parse_and_plot(args.filenames)
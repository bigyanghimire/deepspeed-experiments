import re
import argparse
import matplotlib.pyplot as plt
import os

def parse_and_plot(file_paths):
    plt.figure(figsize=(10, 6))
    
    # Define a pattern to capture iteration and loss
    pattern = r"Step (\d+): loss=([\d.]+)"
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
            label_name = os.path.basename(path)
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
        save_name = "comparison_plot.png"
        plt.savefig(save_name, dpi=150)
        print(f"Plot saved as {save_name}. Opening window...")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse training logs and plot Iteration vs Loss for multiple files.")
    
    # nargs=2 ensures exactly two filenames are passed
    parser.add_argument("filenames", nargs=2, help="The paths to the two log files you want to compare.")
    
    args = parser.parse_args()
    
    parse_and_plot(args.filenames)
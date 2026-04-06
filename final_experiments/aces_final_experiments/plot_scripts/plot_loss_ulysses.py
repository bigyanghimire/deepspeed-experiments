import re
import argparse
import matplotlib.pyplot as plt

def parse_and_plot(file_path):
    iterations = []
    losses = []

    # Regex to capture the iteration number and the loss float
    pattern = r"Iteration (\d+): loss=([\d.]+)"

    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    iterations.append(int(match.group(1)))
                    losses.append(float(match.group(2)))

        if not iterations:
            print(f"Warning: No valid data found in '{file_path}'.")
            return

        # Plotting logic
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, losses, marker='o', linestyle='-', color='#2c3e50', label='Training Loss')
        
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'Training Progress: {file_path}')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        print(f"Parsed {len(iterations)} points. Opening plot...")
        plt.show()
        plt.savefig(f"{file_path}.png", dpi=150)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Setting up the argument parser
    parser = argparse.ArgumentParser(description="Parse training logs and plot Iteration vs Loss.")
    
    # Adding the positional argument for the filename
    parser.add_argument("filename", help="The path to the log file you want to parse.")
    
    args = parser.parse_args()
    
    # Run the function with the provided argument
    parse_and_plot(args.filename)
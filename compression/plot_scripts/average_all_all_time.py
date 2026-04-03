import re
import argparse
import sys

def calculate_average_ata(file_path):
    times = []
    
    # Pattern looks for 'all-to-all time:', grabs the number, and stops at 'ms'
    pattern = r"all-to-all time:\s*([\d.]+)\s*ms"

    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    times.append(float(match.group(1)))

        if not times:
            print(f"No 'all-to-all' entries found in {file_path}.")
            return

        avg_time = sum(times) / len(times)
        
        print("-" * 40)
        print(f"File: {file_path}")
        print(f"Total entries: {len(times)}")
        print(f"Average All-to-All: {avg_time:.4f} ms")
        print("-" * 40)

    except FileNotFoundError:
        print(f"Error: Could not find file '{file_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average all-to-all communication time from logs.")
    parser.add_argument("filename", help="The log file to parse")
    
    args = parser.parse_args()
    calculate_average_ata(args.filename)
import argparse
import sys

def parse_step_times(filename):
    step_times = []
    
    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Expects format: "0: Step_time: 2.5403902530670166"
                if "Step_time:" in line:
                    try:
                        # Split by colon and take the last part
                        time_str = line.split(':')[-1].strip()
                        step_times.append(float(time_str))
                    except ValueError:
                        print(f"Skipping malformed line: {line.strip()}", file=sys.stderr)
                        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    
    warmup_steps = 5
    return step_times[warmup_steps:]

def main():
    parser = argparse.ArgumentParser(description="Calculate sum and average of Step_times from a log file.")
    parser.add_argument('filename', help="Path to the log file")
    args = parser.parse_args()

    times = parse_step_times(args.filename)

    if not times:
        print("No Step_time data found in file.")
        return

    total_sum = sum(times)
    count = len(times)
    average = total_sum / count

    print("-" * 30)
    print(f"File:    {args.filename}")
    print(f"Steps:   {count}")
    print(f"Total:   {total_sum:.6f} seconds")
    print(f"Average: {average:.6f} seconds")
    print("-" * 30)

if __name__ == "__main__":
    main()
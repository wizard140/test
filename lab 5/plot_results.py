import csv
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS = Path("results.csv")

def read_rows(path):
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "n": int(row["n"]),
                "cpu_us": float(row["cpu_us"]),
                "adjacent_us": float(row["adjacent_us"]),
                "blocks_us": float(row["blocks_us"]),
                "adjacent_speedup": float(row["adjacent_speedup"]),
                "blocks_speedup": float(row["blocks_speedup"]),
            })
    return rows

def find_break_even(rows, key):
    for row in rows:
        if row[key] >= 1.0:
            return row["n"]
    return None

def main():
    if not RESULTS.exists():
        print("results.csv was not found. Run the CUDA program first.")
        return

    rows = read_rows(RESULTS)
    n = [row["n"] for row in rows]
    cpu = [row["cpu_us"] for row in rows]
    adjacent = [row["adjacent_us"] for row in rows]
    blocks = [row["blocks_us"] for row in rows]
    adjacent_speedup = [row["adjacent_speedup"] for row in rows]
    blocks_speedup = [row["blocks_speedup"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(n, cpu, marker="o", label="CPU")
    plt.plot(n, adjacent, marker="o", label="GPU Adjacent")
    plt.plot(n, blocks, marker="o", label="GPU Blocks")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Vector Size (n)")
    plt.ylabel("Time (microseconds)")
    plt.title("Vector Scale-Add Timing")
    plt.legend()
    plt.tight_layout()
    plt.savefig("timing.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(n, adjacent_speedup, marker="o", label="Adjacent Speedup")
    plt.plot(n, blocks_speedup, marker="o", label="Blocks Speedup")
    plt.axhline(1.0, linestyle="--")
    plt.xscale("log", base=2)
    plt.xlabel("Vector Size (n)")
    plt.ylabel("Speedup vs CPU")
    plt.title("GPU Speedup over CPU")
    plt.legend()
    plt.tight_layout()
    plt.savefig("speedup.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(n, adjacent, marker="o", label="GPU Adjacent")
    plt.plot(n, blocks, marker="o", label="GPU Blocks")
    plt.xscale("log", base=2)
    plt.xlabel("Vector Size (n)")
    plt.ylabel("Time (microseconds)")
    plt.title("GPU Kernel Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=200)
    plt.close()

    adj_break = find_break_even(rows, "adjacent_speedup")
    blk_break = find_break_even(rows, "blocks_speedup")

    print("Saved timing.png, speedup.png, and comparison.png")
    print("Adjacent break-even n:", adj_break if adj_break is not None else "not reached")
    print("Blocks break-even n:", blk_break if blk_break is not None else "not reached")

if __name__ == "__main__":
    main()

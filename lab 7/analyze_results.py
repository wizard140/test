import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def read_rows(path):
    rows = []
    with open(path, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            row["width"] = int(row["width"])
            row["height"] = int(row["height"])
            row["pixels"] = int(row["pixels"])
            row["filter_width"] = int(row["filter_width"])
            row["filter_height"] = int(row["filter_height"])
            row["repeat"] = int(row["repeat"])
            row["time_ms"] = float(row["time_ms"])
            row["verified"] = int(row["verified"])
            rows.append(row)
    return rows


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def stddev(values):
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((x - avg) ** 2 for x in values) / (len(values) - 1))


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["image"],
            row["width"],
            row["height"],
            row["pixels"],
            row["filter"],
            row["filter_width"],
            row["filter_height"],
            row["version"],
        )
        grouped[key].append(row["time_ms"])

    summary = []
    for key, values in sorted(grouped.items()):
        summary.append({
            "image": key[0],
            "width": key[1],
            "height": key[2],
            "pixels": key[3],
            "filter": key[4],
            "filter_width": key[5],
            "filter_height": key[6],
            "version": key[7],
            "runs": len(values),
            "mean_ms": mean(values),
            "std_ms": stddev(values),
        })
    return summary


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_speedups(summary):
    by_case = defaultdict(dict)
    for row in summary:
        key = (row["image"], row["pixels"], row["filter"], row["filter_width"])
        by_case[key][row["version"]] = row

    speedup_rows = []
    for key, versions in sorted(by_case.items()):
        if "cpu_single" not in versions:
            continue
        cpu_mean = versions["cpu_single"]["mean_ms"]
        for version, row in versions.items():
            if version == "cpu_single":
                continue
            speedup_rows.append({
                "image": row["image"],
                "pixels": row["pixels"],
                "filter": row["filter"],
                "filter_width": row["filter_width"],
                "version": version,
                "cpu_mean_ms": cpu_mean,
                "gpu_mean_ms": row["mean_ms"],
                "speedup": cpu_mean / row["mean_ms"],
            })
    return speedup_rows


def plot_speedup_by_pixels(speedups, output_dir):
    grouped = defaultdict(list)
    for row in speedups:
        grouped[row["version"]].append((row["pixels"], row["speedup"]))

    plt.figure(figsize=(10, 6))
    for version, pairs in sorted(grouped.items()):
        pixels_to_values = defaultdict(list)
        for pixels, value in pairs:
            pixels_to_values[pixels].append(value)
        xs = sorted(pixels_to_values.keys())
        ys = [mean(pixels_to_values[x]) for x in xs]
        plt.plot(xs, ys, marker="o", label=version)

    plt.xlabel("Image size in pixels")
    plt.ylabel("Average speedup over cpu_single")
    plt.title("Speedup vs. image size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speedup_vs_pixels.png"), dpi=200)
    plt.close()


def plot_runtime_by_pixels(summary, output_dir):
    grouped = defaultdict(list)
    for row in summary:
        grouped[row["version"]].append((row["pixels"], row["mean_ms"]))

    plt.figure(figsize=(10, 6))
    for version, pairs in sorted(grouped.items()):
        pixels_to_values = defaultdict(list)
        for pixels, value in pairs:
            pixels_to_values[pixels].append(value)
        xs = sorted(pixels_to_values.keys())
        ys = [mean(pixels_to_values[x]) for x in xs]
        plt.plot(xs, ys, marker="o", label=version)

    plt.xlabel("Image size in pixels")
    plt.ylabel("Average runtime (ms)")
    plt.title("Runtime vs. image size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_vs_pixels.png"), dpi=200)
    plt.close()


def plot_filter_size_effect(summary, output_dir):
    grouped = defaultdict(list)
    for row in summary:
        key = (row["filter_width"], row["version"])
        grouped[key].append(row["mean_ms"])

    labels = []
    values = []
    for key in sorted(grouped.keys()):
        labels.append(f"{key[0]}x{key[0]}\n{key[1]}")
        values.append(mean(grouped[key]))

    x = np.arange(len(labels))
    plt.figure(figsize=(12, 6))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Average runtime (ms)")
    plt.title("Average runtime by filter size and version")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "filter_size_comparison.png"), dpi=200)
    plt.close()


def estimate_break_even(summary_rows):
    cases = defaultdict(dict)
    for row in summary_rows:
        key = (row["filter"], row["version"])
        cases[key].setdefault("points", []).append((row["pixels"], row["mean_ms"]))

    cpu_rows = defaultdict(list)
    for row in summary_rows:
        cpu_rows[row["filter"]].append((row["pixels"], row["mean_ms"])) if row["version"] == "cpu_single" else None

    out = []
    for (filter_name, version), data in sorted(cases.items()):
        if version == "cpu_single":
            continue
        gpu_points = sorted(data["points"])
        cpu_points = sorted(cpu_rows[filter_name])
        if len(gpu_points) < 2 or len(cpu_points) < 2:
            continue

        common_pixels = sorted(set(p for p, _ in gpu_points) & set(p for p, _ in cpu_points))
        gpu_map = {p: t for p, t in gpu_points}
        cpu_map = {p: t for p, t in cpu_points}
        diff_points = [(p, cpu_map[p] - gpu_map[p]) for p in common_pixels]
        estimated = "no crossover in tested range"

        for i in range(1, len(diff_points)):
            p0, d0 = diff_points[i - 1]
            p1, d1 = diff_points[i]
            if d0 == 0.0:
                estimated = f"{p0:.0f}"
                break
            if d0 < 0.0 and d1 > 0.0:
                fraction = (-d0) / (d1 - d0)
                estimated = f"{p0 + fraction * (p1 - p0):.2f}"
                break

        out.append({
            "filter": filter_name,
            "version": version,
            "estimated_break_even_pixels": estimated,
        })

    return out


def ttest_rows(rows, output_dir):
    by_case = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["image"], row["filter"])
        by_case[key][row["version"]].append(row["time_ms"])

    report_lines = []
    report_lines.append("Statistical comparison by image/filter\n")
    report_lines.append("====================================\n")

    for key, versions in sorted(by_case.items()):
        report_lines.append(f"\nCase: image={key[0]}, filter={key[1]}\n")
        cpu = versions.get("cpu_single", [])
        for version in ["gpu_global", "gpu_tiled", "gpu_constant", "gpu_tiled_constant"]:
            gpu = versions.get(version, [])
            if not cpu or not gpu:
                continue
            if HAVE_SCIPY:
                stat = stats.ttest_ind(cpu, gpu, equal_var=False)
                report_lines.append(
                    f"  {version}: mean cpu={mean(cpu):.6f} ms, mean gpu={mean(gpu):.6f} ms, p={stat.pvalue:.6g}\n"
                )
            else:
                report_lines.append(
                    f"  {version}: mean cpu={mean(cpu):.6f} ms, mean gpu={mean(gpu):.6f} ms, scipy not installed so no p-value computed\n"
                )

    with open(os.path.join(output_dir, "stat_tests.txt"), "w") as fp:
        fp.writelines(report_lines)


def write_summary_markdown(summary, speedups, break_even_rows, output_dir):
    lines = []
    lines.append("# Convolution Lab Summary\n\n")
    lines.append("## Main findings to fill into the report\n\n")
    lines.append("- Use `speedup_vs_pixels.png` for the required speedup figure.\n")
    lines.append("- Use `runtime_vs_pixels.png` to discuss raw runtimes.\n")
    lines.append("- Use `filter_size_comparison.png` to discuss 3x3 vs. 5x5 filter cost.\n\n")

    lines.append("## Best average speedups by version\n\n")
    best = defaultdict(float)
    for row in speedups:
        best[row["version"]] = max(best[row["version"]], row["speedup"])
    for version in sorted(best.keys()):
        lines.append(f"- {version}: best observed speedup = {best[version]:.4f}x\n")

    lines.append("\n## Estimated break-even points\n\n")
    for row in break_even_rows:
        lines.append(f"- {row['filter']} / {row['version']}: {row['estimated_break_even_pixels']} pixels\n")

    with open(os.path.join(output_dir, "analysis_summary.md"), "w") as fp:
        fp.writelines(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_results.py results.csv [output_dir]")
        sys.exit(1)

    results_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) >= 3 else os.path.dirname(results_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    rows = read_rows(results_path)
    summary = summarize(rows)
    speedups = build_speedups(summary)
    break_even_rows = estimate_break_even(summary)

    write_csv(
        os.path.join(output_dir, "timing_summary.csv"),
        summary,
        ["image", "width", "height", "pixels", "filter", "filter_width", "filter_height", "version", "runs", "mean_ms", "std_ms"],
    )

    write_csv(
        os.path.join(output_dir, "speedup_summary.csv"),
        speedups,
        ["image", "pixels", "filter", "filter_width", "version", "cpu_mean_ms", "gpu_mean_ms", "speedup"],
    )

    write_csv(
        os.path.join(output_dir, "break_even_summary.csv"),
        break_even_rows,
        ["filter", "version", "estimated_break_even_pixels"],
    )

    plot_speedup_by_pixels(speedups, output_dir)
    plot_runtime_by_pixels(summary, output_dir)
    plot_filter_size_effect(summary, output_dir)
    ttest_rows(rows, output_dir)
    write_summary_markdown(summary, speedups, break_even_rows, output_dir)

    print(f"Wrote analysis files to {output_dir}")


if __name__ == "__main__":
    main()

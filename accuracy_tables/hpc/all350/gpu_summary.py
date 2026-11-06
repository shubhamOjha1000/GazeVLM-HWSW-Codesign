"""A job's GPU use, from its once-a-minute log: the mean and the lowest stretch. Standard library only, so it also runs on a login node.

    python3 -m accuracy_tables.hpc.all350.gpu_summary /scratch/$USER/GazeVLM/logs/gpu_all350-pilot-*.csv

Torch's page gives no time window for cancelling jobs with low GPU use, so a mean alone can hide a long idle stretch.
"""
import os
import sys

WINDOW = 30      # samples, one a minute


def read_use(path):
    """GPU use (%) per sample of a log line like `2026/11/05 12:00:00.000, 45 %, 1234 MiB, 100.0 W`."""
    use = []
    with open(path) as f:
        for line in f:
            parts = line.split(",")
            if len(parts) > 1:
                value = parts[1].strip().split(" ")[0]
                if value.isdigit():
                    use.append(int(value))
    return use


def lowest_mean(use, window=WINDOW):
    """The lowest mean over `window` consecutive samples, or None if there are fewer samples than that."""
    if len(use) < window:
        return None
    total = low = sum(use[:window])
    for i in range(window, len(use)):
        total += use[i] - use[i - window]
        low = min(low, total)
    return low / window


def summary_line(path):
    if not os.path.exists(path):
        return f"{path}: no such log"
    use = read_use(path)
    if not use:
        return f"{os.path.basename(path)}: no samples"
    low = lowest_mean(use)
    stretch = f"lowest {WINDOW}-minute mean {low:.1f}%" if low is not None else f"under {WINDOW} samples, so no {WINDOW}-minute reading"
    return f"{os.path.basename(path)}: {len(use)} samples, mean {sum(use) / len(use):.1f}%, {stretch}"


def main(argv=None):
    for path in (sys.argv[1:] if argv is None else argv):
        print(summary_line(path), flush=True)


if __name__ == "__main__":
    main()

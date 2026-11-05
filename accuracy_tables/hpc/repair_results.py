"""Drops a half-written last line from a folder's result files, so a job killed mid-write can resume.

    python -m accuracy_tables.hpc.repair_results <folder>
"""
import glob
import os
import sys


def repair(folder):
    """Cuts every <folder>/*.jsonl back to its last complete line -> the files changed."""
    changed = []
    for path in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        with open(path, "rb+") as f:
            data = f.read()
            if data and not data.endswith(b"\n"):
                f.truncate(data.rfind(b"\n") + 1)
                changed.append(path)
    return changed


if __name__ == "__main__":
    for p in repair(sys.argv[1]):
        print("dropped a half-written last line:", p)

"""A rough estimate of a full run's GPU hours from a pilot's merged results. Reads only timings and counts, no answers.

    python -u -m accuracy_tables.hpc.all350.estimate_hours --root <pilot results folder> --model qwen3

In each condition, the median `sec` times the questions is the time spent answering, which grows with questions. The rest of its `sec` is
encoding and planning, which grows with videos. So each part is scaled on its own, from the pilot's videos and questions to the full run's.
`sec` leaves out startup and frame loading, so it prints how to correct for them from the pilot's real job lengths.
"""
import argparse
import json
import math
import os
import statistics

from accuracy_tables.hpc.all350 import prepare, shards

JOB_HOURS = 8.0
ROWS = ("uniform", "blind", "budget", "avgpool", "gazevlm", "gazevlm2", "streamingtom", "dycoke", "fastvid", "prunevid", "dtd")


def split_seconds(secs):
    """A condition's `sec` values -> (answering, encoding and planning), in seconds."""
    answering = statistics.median(secs) * len(secs)
    return answering, max(sum(secs) - answering, 0.0)


def estimate(records, pilot, full):
    """records: one row's records. pilot and full: (questions, videos) -> (pilot seconds, full-run seconds)."""
    by_condition = {}
    for r in records:
        if r.get("error") is None:
            by_condition.setdefault(r["condition"], []).append(r["sec"])
    measured = scaled = 0.0
    for secs in by_condition.values():
        answering, encoding = split_seconds(secs)
        measured += answering + encoding
        scaled += answering * full[0] / pilot[0] + encoding * full[1] / pilot[1]
    return measured, scaled


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    manifest = shards.read_manifest(a.root)
    pilot, full = (manifest["n_questions"], manifest["n_videos"]), prepare.TOTAL
    measured_total = scaled_total = 0.0
    for row in ROWS:
        path = os.path.join(a.root, a.model, f"{row}.jsonl")
        if os.path.exists(path):
            with open(path) as f:
                measured, scaled = estimate([json.loads(line) for line in f if line.strip()], pilot, full)
            measured_total += measured
            scaled_total += scaled
            print(f"   {row}: {measured / 3600:.2f} h in the pilot, about {scaled / 3600:.1f} h in the full run", flush=True)
    hours = scaled_total / 3600
    each = hours / manifest["n_shards"]
    print(f"recorded `sec` in the pilot: {measured_total / 3600:.2f} GPU hours", flush=True)
    print(f"full run: about {hours:.0f} GPU hours, {each:.1f} per shard, so {math.ceil(each / JOB_HOURS)} jobs per shard under the {JOB_HOURS:.0f}-hour limit", flush=True)
    print("to correct for startup and frame loading: multiply by the pilot's total job hours divided by its recorded `sec` hours", flush=True)


if __name__ == "__main__":
    main()

"""How far a sharded run has got, counted in answers. Standard library only, so it runs on a login node. It only counts lines, and reads no answers.

    python3 -m accuracy_tables.hpc.all350.shard_progress --root <results folder> --model qwen2.5

An answer is one question under one condition, so a row with 3 conditions has 3 answers per question. Each shard's total is its questions times
all its rows' conditions. A record with an error counts as written: the merge is what stops on those.
"""
import argparse
import csv
import json
import os

RUN_ORDER = ("uniform", "blind", "gazevlm", "streamingtom", "dycoke", "fastvid", "prunevid", "dtd", "avgpool", "gazevlm2", "budget")      # as run.py scores them


def conditions_per_row(targets):
    """{row: how many conditions it has}. Two budgets can share a condition, and then it's scored once."""
    rows = {}
    with open(targets, newline="") as f:
        for line in csv.DictReader(f):
            rows.setdefault(line["row"], set()).add(line["condition"])
    return {row: len(names) for row, names in rows.items()}


def count_records(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def progress(root, model, targets=None):
    """-> one dict per shard: shard, questions, videos, done, total, and rows = [(row, done, total)] in the order they run."""
    with open(os.path.join(root, "manifest", "manifest.json")) as f:
        manifest = json.load(f)
    conditions = conditions_per_row(targets or os.path.join(root, "calibrate", model, "targets.csv"))
    out = []
    for s in manifest["shards"]:
        folder = os.path.join(root, "shards", str(s["shard"]), model)
        rows = [(row, count_records(os.path.join(folder, f"{row}.jsonl")), conditions[row] * s["questions"]) for row in RUN_ORDER if row in conditions]
        out.append(dict(shard=s["shard"], questions=s["questions"], videos=len(s["videos"]), rows=rows,
                        done=sum(r[1] for r in rows), total=sum(r[2] for r in rows)))
    return out


def percent(done, total):
    return 100 * done / total if total else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", help="default: <root>/calibrate/<model>/targets.csv")
    a = ap.parse_args()
    shards = progress(a.root, a.model, a.targets)
    for s in shards:
        print(f"shard {s['shard']}: {s['done']} of {s['total']} answers ({percent(s['done'], s['total']):.1f}%), {s['questions']} questions on {s['videos']} videos")
        print("   " + ", ".join(f"{row} {done}/{total}" for row, done, total in s["rows"]))
    done, total = sum(s["done"] for s in shards), sum(s["total"] for s in shards)
    print(f"all shards: {done} of {total} answers ({percent(done, total):.1f}%)")


if __name__ == "__main__":
    main()

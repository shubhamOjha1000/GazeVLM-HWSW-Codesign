"""How far section 15's sharded FLOPs-matched run has got, counted in answers. Standard library only, so it runs on a login node. It only
counts lines, and reads no answers.

    python3 -m accuracy_tables.hpc.all350.flops_progress --root <results folder> --model qwen2.5

shard_progress.py's count, over the plan's 13 rows (its targets file) in the plan's order: an answer is one question under one condition,
and two budgets sharing a condition score it once.
"""
import argparse
import json
import os

from accuracy_tables.hpc.all350.shard_progress import conditions_per_row, count_records, percent


def progress(root, model, targets=None):
    """-> one dict per shard: shard, questions, videos, done, total, and rows = [(row, done, total)] in the plan's order."""
    with open(os.path.join(root, "manifest", "manifest.json")) as f:
        manifest = json.load(f)
    conditions = conditions_per_row(targets or os.path.join(root, "calibrate", model, "flops_targets.csv"))
    out = []
    for s in manifest["shards"]:
        folder = os.path.join(root, "shards", str(s["shard"]), model)
        rows = [(row, count_records(os.path.join(folder, f"{row}.jsonl")), n * s["questions"]) for row, n in conditions.items()]
        out.append(dict(shard=s["shard"], questions=s["questions"], videos=len(s["videos"]), rows=rows,
                        done=sum(r[1] for r in rows), total=sum(r[2] for r in rows)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", help="default: <root>/calibrate/<model>/flops_targets.csv")
    a = ap.parse_args(argv)
    shards = progress(a.root, a.model, a.targets)
    for s in shards:
        print(f"shard {s['shard']}: {s['done']} of {s['total']} answers ({percent(s['done'], s['total']):.1f}%), {s['questions']} questions on {s['videos']} videos")
        print("   " + ", ".join(f"{row} {done}/{total}" for row, done, total in s["rows"]))
    done, total = sum(s["done"] for s in shards), sum(s["total"] for s in shards)
    print(f"all shards: {done} of {total} answers ({percent(done, total):.1f}%)")


if __name__ == "__main__":
    main()

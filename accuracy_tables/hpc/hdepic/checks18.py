"""Runbook section 18's small checks and the shard merge. Prints no accuracy.

    python -u -m accuracy_tables.hpc.hdepic.checks18 pick (--questions 50 | --every 25 | --shard 0 --of 3)
    python -u -m accuracy_tables.hpc.hdepic.checks18 fovea --folder <records folder> [--tokens 17]
    python -u -m accuracy_tables.hpc.hdepic.checks18 totals
    python -u -m accuracy_tables.hpc.hdepic.checks18 merge --out-dir <results folder> --rows <comma list> [--model smolvlm] [--of 3]

pick    clips from s18/questions.json: the 6 groups taking turns, each in clip order, until --questions are covered; every --every-th
        clip (the calibrate job's timing sample); or shard --shard of --of, every --of-th clip from the --shard-th, so the shards hold
        the same number of questions to within 1. Prints a comma list for run.py's and flops_run.py's --videos.
fovea   section 17's check: every budget and no-gate record holds 17 image tokens per frame.
totals  questions and clips, for the run-time projection and the records check.
merge   the shards' records (<out-dir>/shard<s>/<model>/<row>.jsonl) into <out-dir>/<model>/<row>.jsonl: every record once, no
        (condition, question) twice, nothing overwritten; and the shards' GPU names into <out-dir>/<model>/gpu_types.txt. MERGED, or stops.
"""
import argparse
import json
import os
import sys

from accuracy_tables.hpc.hdepic import checks, groups, s18


def questions():
    with open(s18.questions_json()) as f:
        return json.load(f)


def pick_by_questions(qs, n):
    pools = [[q["video"] for q in sorted(qs, key=lambda q: q["video"]) if q["group"] == g] for g in groups.GROUPS]
    out = []
    while len(out) < n and any(pools):
        for pool in pools:
            if pool and len(out) < n:
                out.append(pool.pop(0))
    return out


def pick_every(qs, k):
    return sorted(q["video"] for q in qs)[:: int(k)]


def pick_shard(qs, s, of):
    if not 0 <= int(s) < int(of):
        raise SystemExit(f"shard {s} of {of}: need 0 <= shard < of")
    return sorted(q["video"] for q in qs)[int(s):: int(of)]


def merge(out_dir, rows, model="smolvlm", of=3):
    """-> {row: records written}. Stops on a missing shard folder, a (condition, qid) twice, or a merged file already there."""
    shards = [os.path.join(out_dir, f"shard{s}", model) for s in range(of)]
    missing = [p for p in shards if not os.path.isdir(p)]
    if missing:
        raise SystemExit(f"MERGE STOPPED: no {missing}. Run every shard first.")
    target = os.path.join(out_dir, model)
    os.makedirs(target, exist_ok=True)
    written = {}
    for row in rows:
        path = os.path.join(target, f"{row}.jsonl")
        if os.path.exists(path):
            raise SystemExit(f"MERGE STOPPED: {path} is there already; nothing is overwritten. Tell the user.")
        seen, lines = set(), []
        for sh in shards:
            p = os.path.join(sh, f"{row}.jsonl")
            if not os.path.exists(p):
                continue
            for line in open(p):
                if line.strip():
                    r = json.loads(line)
                    k = (r["condition"], r["qid"])
                    if k in seen:
                        raise SystemExit(f"MERGE STOPPED: {row} {k} in two shards. Tell the user.")
                    seen.add(k)
                    lines.append(line if line.endswith("\n") else line + "\n")
        with open(path + ".tmp", "w") as f:
            f.writelines(lines)
        os.replace(path + ".tmp", path)
        written[row] = len(lines)
    names = set()
    for sh in shards:
        p = os.path.join(sh, "gpu_types.txt")
        if os.path.exists(p):
            names |= {x.strip() for x in open(p) if x.strip()}
    with open(os.path.join(target, "gpu_types.txt"), "a") as f:
        f.writelines(n + "\n" for n in sorted(names))
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("pick")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--questions", type=int)
    g.add_argument("--every", type=int)
    g.add_argument("--shard", type=int)
    s.add_argument("--of", type=int, default=3)
    s = sub.add_parser("fovea")
    s.add_argument("--folder", required=True)
    s.add_argument("--tokens", type=int, default=17)
    sub.add_parser("totals")
    s = sub.add_parser("merge")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--rows", required=True)
    s.add_argument("--model", default="smolvlm")
    s.add_argument("--of", type=int, default=3)
    a = ap.parse_args(argv)
    if a.command == "pick":
        qs = questions()
        got = (pick_by_questions(qs, a.questions) if a.questions else pick_every(qs, a.every) if a.every else pick_shard(qs, a.shard, a.of))
        print(",".join(got))
    elif a.command == "fovea":
        n, bad = checks.fovea_records(a.folder, tokens=a.tokens)
        if not n or bad:
            sys.exit(f"FOVEATED TOKENS DIFFER: {bad} of {n} budget and no-gate records aren't {a.tokens} tokens a frame. Stop and tell the user.")
        print(f"FOVEATED TOKENS {a.tokens} ON EVERY KEPT FRAME: {n} budget and no-gate records", flush=True)
    elif a.command == "totals":
        qs = questions()
        print(f"{len(qs)} {len({q['video'] for q in qs})}")
    else:
        got = merge(os.path.abspath(a.out_dir), a.rows.split(","), a.model, a.of)
        print(f"MERGED: {sum(got.values())} records from {a.of} shards: {got}", flush=True)


if __name__ == "__main__":
    main()

"""Runbook section 17's small checks. Prints no accuracy.

    python -u -m accuracy_tables.hpc.hdepic.checks pick (--questions 50 | --every 25)
    python -u -m accuracy_tables.hpc.hdepic.checks fovea --folder <records folder> [--tokens 17]
    python -u -m accuracy_tables.hpc.hdepic.checks totals

pick    clips from questions.json: the 2 types taking turns, each in clip order, until --questions are covered; or every --every-th clip
        (the calibrate job's timing sample). Prints a comma list for run.py's --videos.
fovea   every budget and no-gate record in a folder must hold 17 image tokens per frame: FOVEATED TOKENS 17 ON EVERY KEPT FRAME.
totals  questions and clips in questions.json, for the run-time projection and the records check.
"""
import argparse
import json
import os
import sys

from accuracy_tables.hpc.hdepic import data


def questions():
    with open(data.questions_json()) as f:
        return json.load(f)


def pick_by_questions(qs, n):
    pools = [[q["video"] for q in qs if q["type"] == t] for t in data.GAZE_TYPES]
    out = []
    while len(out) < n and any(pools):
        for pool in pools:
            if pool and len(out) < n:
                out.append(pool.pop(0))
    return out


def pick_every(qs, k):
    return sorted(q["video"] for q in qs)[::int(k)]


def fovea_records(folder, rows=("budget", "nogate"), tokens=17):
    """(records read, records whose image tokens aren't tokens x frames)."""
    n = bad = 0
    for row in rows:
        path = os.path.join(folder, f"{row}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("error") is not None:
                        continue
                    n += 1
                    bad += r["n_image_tokens"] != tokens * r["n_frames"]
    return n, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("pick")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--questions", type=int)
    g.add_argument("--every", type=int)
    s = sub.add_parser("fovea")
    s.add_argument("--folder", required=True)
    s.add_argument("--tokens", type=int, default=17)
    sub.add_parser("totals")
    a = ap.parse_args(argv)
    if a.command == "pick":
        qs = questions()
        print(",".join(pick_by_questions(qs, a.questions) if a.questions else pick_every(qs, a.every)))
    elif a.command == "fovea":
        n, bad = fovea_records(a.folder, tokens=a.tokens)
        if not n or bad:
            sys.exit(f"FOVEATED TOKENS DIFFER: {bad} of {n} budget and no-gate records aren't {a.tokens} tokens a frame. Stop and tell the user.")
        print(f"FOVEATED TOKENS {a.tokens} ON EVERY KEPT FRAME: {n} budget and no-gate records", flush=True)
    else:
        qs = questions()
        print(f"{len(qs)} {len({q['video'] for q in qs})}")


if __name__ == "__main__":
    main()

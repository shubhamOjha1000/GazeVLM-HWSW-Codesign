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
merge   the shards' records (<out-dir>/shard<s>/<model>/<row>.jsonl) into <out-dir>/<model>/<row>.jsonl, shard by shard: every record
        once, no (condition, question) twice; each shard's gpu_ledger.jsonl carried into the merged folder's, every line's first_line
        moved by the records the earlier shards put before it, so each record keeps the GPU it ran on; and the shards' GPU names into
        gpu_types.txt. A file already there must be exactly what the merge would write (a rerun after a fix); anything else stops it, and
        nothing is overwritten. MERGED, or stops.
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


LEDGER = "gpu_ledger.jsonl"                 # flops_run.LEDGER, named here too so this module needs no torch


def read_lines(path):
    if not os.path.exists(path):
        return []
    return [line if line.endswith("\n") else line + "\n" for line in open(path) if line.strip()]


def write_once(path, content):
    """Writes content to path; a file already there must hold exactly content. -> 'written' or 'unchanged'."""
    if os.path.exists(path):
        with open(path) as f:
            if f.read() != content:
                raise SystemExit(f"MERGE STOPPED: {path} is there and isn't what the merge writes; nothing is overwritten. Tell the user.")
        return "unchanged"
    with open(path + ".tmp", "w") as f:
        f.write(content)
    os.replace(path + ".tmp", path)
    return "written"


def merge(out_dir, rows, model="smolvlm", of=3):
    """-> ({row: records}, {file: written or unchanged}). Stops on a missing shard folder, a (condition, qid) twice, or a merged file that
    isn't what this merge writes."""
    shards = [os.path.join(out_dir, f"shard{s}", model) for s in range(of)]
    missing = [p for p in shards if not os.path.isdir(p)]
    if missing:
        raise SystemExit(f"MERGE STOPPED: no {missing}. Run every shard first.")
    target = os.path.join(out_dir, model)
    os.makedirs(target, exist_ok=True)
    ledgers = [[json.loads(x) for x in read_lines(os.path.join(sh, LEDGER))] for sh in shards]
    written, state, ledger = {}, {}, []
    for row in rows:
        seen, lines = set(), []
        for s, sh in enumerate(shards):
            mine = read_lines(os.path.join(sh, f"{row}.jsonl"))
            for e in ledgers[s]:
                if e["row"] == row:
                    ledger.append(dict(e, first_line=int(e["first_line"]) + len(lines), shard=s))
            for line in mine:
                r = json.loads(line)
                k = (r["condition"], r["qid"])
                if k in seen:
                    raise SystemExit(f"MERGE STOPPED: {row} {k} in two shards. Tell the user.")
                seen.add(k)
            lines += mine
        state[f"{row}.jsonl"] = write_once(os.path.join(target, f"{row}.jsonl"), "".join(lines))
        written[row] = len(lines)
    state[LEDGER] = write_once(os.path.join(target, LEDGER), "".join(json.dumps(e) + "\n" for e in ledger))
    names = set()
    for sh in shards:
        names |= {x.strip() for x in read_lines(os.path.join(sh, "gpu_types.txt"))}
    path = os.path.join(target, "gpu_types.txt")
    have = {x.strip() for x in read_lines(path)}
    with open(path, "a") as f:
        f.writelines(n + "\n" for n in sorted(names - have))
    return written, state


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
        got, state = merge(os.path.abspath(a.out_dir), a.rows.split(","), a.model, a.of)
        new = sorted(k for k, v in state.items() if v == "written")
        print(f"MERGED: {sum(got.values())} records from {a.of} shards: {got}; the GPU ledger carried over; written {new or 'nothing new'}, "
              f"the rest already there and identical", flush=True)


if __name__ == "__main__":
    main()

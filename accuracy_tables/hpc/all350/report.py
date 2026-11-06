"""The results of a run: per model, for all its videos, AEA and Nymeria. CPU only.

    python -u -m accuracy_tables.hpc.all350.report --out-dir <results folder> [--list all350_pilot]

Gives n, accuracy, its 95% interval from resampling whole videos, frames per question, measured budget and gaze_missing_pct, and writes
<out_dir>/summary_by_group.csv. The job fails if any model or condition isn't complete, after writing the file so it can be read. It
declares no winner. compare_budget.py isn't called: this run has no no-gate row, and every line of the report says what that means.
"""
import argparse
import os
import sys

import pandas as pd

from accuracy_tables import common, run, summarize
from accuracy_tables.hpc.all350 import prepare, seed_results
from accuracy_tables.hpc.newvideos import paths

NOTE = "no no-gate control in this run, so these results can't show whether the gate itself helps"
CONDITIONS_PER_MODEL = 5         # uniform, blind and 3 budgets
ROW_FILES = ("uniform", "blind", "budget")


def video_groups(videos):
    """{group: its videos}. Names starting with `loc` are AEA, the others Nymeria."""
    videos = set(videos)
    return dict(all=videos, aea={v for v in videos if v.startswith("loc")}, nymeria={v for v in videos if not v.startswith("loc")})


def expected_questions(counts, groups):
    return {g: sum(counts[v] for v in vs) for g, vs in groups.items()}


def n_problems(table, expected):
    """What is incomplete in the report -> list of messages."""
    out = []
    for g, want in expected.items():
        if not want:
            continue
        rows = table[table.group == g]
        for model in seed_results.MODELS:
            m = rows[rows.model == model]
            if len(m) != CONDITIONS_PER_MODEL:
                out.append(f"{g} {model}: {len(m)} conditions, expected {CONDITIONS_PER_MODEL}")
            out += [f"{g} {model} {r.condition}: n = {r.n}, expected {want}" for r in m.itertuples() if r.n != want]
    return out


def error_records(out_dir):
    """{(model, row): records with an error}. They are left out of the summary, so they show up as a low n too."""
    found = {}
    for model in seed_results.MODELS:
        for row in ROW_FILES:
            bad = run._file_errors(os.path.join(out_dir, model, f"{row}.jsonl"))[1]
            if bad:
                found[(model, row)] = bad
    return found


def build_table(cfg, groups):
    tables = []
    for g, vs in groups.items():
        if vs:
            df = summarize.summarize(cfg, vs)
            if len(df):
                tables.append(df.assign(group=g, note=NOTE))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=["group", "model", "condition", "n"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--list", help="only these videos: a list written by the earlier steps (default: every video with questions)")
    a = ap.parse_args()
    cfg = common.load_config(prepare.CONFIG)
    out_dir = os.path.abspath(os.path.expanduser(a.out_dir))
    cfg["paths"]["out_dir"] = out_dir
    counts = prepare.question_counts()
    groups = video_groups(paths.read_list(a.list) if a.list else counts)
    table = build_table(cfg, groups)
    path = os.path.join(out_dir, "summary_by_group.csv")
    table.to_csv(path, index=False)

    print(NOTE, flush=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    show = ["model", "condition", "n", "accuracy", "ci_low", "ci_high", "budget_pct", "frames_per_q", "gaze_missing_pct", "videos"]
    for g in groups:
        rows = table[table.group == g]
        if len(rows):
            print(f"\n--- {g}", flush=True)
            print(rows[show].round(2).to_string(index=False), flush=True)
    print(f"\nwritten {path}", flush=True)

    errors = error_records(out_dir)
    if errors:
        print(f"records with an error: {errors}", flush=True)
    found = n_problems(table, expected_questions(counts, groups))
    for line in found[:40]:
        print("  ", line, flush=True)
    if found:
        sys.exit(f"INCOMPLETE: {len(found)} problems. Rerun the unfinished model, or tell the user.")
    print("EVERY n IS COMPLETE", flush=True)


if __name__ == "__main__":
    main()

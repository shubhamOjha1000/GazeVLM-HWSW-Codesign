"""The results of a run: per model, for all its videos, AEA and Nymeria. CPU only.

    python -u -m accuracy_tables.hpc.all350.report --out-dir <results folder> [--list all350_pilot] [--models smolvlm] [--targets <targets.csv>]

Gives n, accuracy, its 95% interval from resampling whole videos, frames per question, measured budget and gaze_missing_pct, and writes
<out_dir>/summary_by_group.csv. The job fails if any model or condition isn't complete, after writing the file so it can be read. It
declares no winner. compare_budget.py isn't called: this run has no no-gate row, and every line of the report says what that means.

With --targets (from calibrate.py --fractions), each row gets one line per budget with its target and its measured budget, in a fixed row
order that says nothing about accuracy, and the conditions expected are the ones in that file.
"""
import argparse
import os
import sys

import pandas as pd

from accuracy_tables import calibrate, common, run, summarize
from accuracy_tables.hpc.all350 import prepare, seed_results
from accuracy_tables.hpc.newvideos import paths

NOTE = "no no-gate control in this run, so these results can't show whether the gate itself helps"
CONDITIONS_PER_MODEL = 5         # uniform, blind and 3 budgets
ROW_FILES = ("uniform", "blind", "budget")
ROW_ORDER = (*ROW_FILES, "avgpool", "gazevlm", "gazevlm2", "streamingtom", "dycoke", "fastvid", "prunevid", "dtd")


def video_groups(videos):
    """{group: its videos}. Names starting with `loc` are AEA, the others Nymeria."""
    videos = set(videos)
    return dict(all=videos, aea={v for v in videos if v.startswith("loc")}, nymeria={v for v in videos if not v.startswith("loc")})


def expected_questions(counts, groups):
    return {g: sum(counts[v] for v in vs) for g, vs in groups.items()}


def n_problems(table, expected, models=seed_results.MODELS, conditions=CONDITIONS_PER_MODEL):
    """What is incomplete in the report -> list of messages."""
    out = []
    for g, want in expected.items():
        if not want:
            continue
        rows = table[table.group == g]
        for model in models:
            m = rows[rows.model == model]
            if len(m) != conditions:
                out.append(f"{g} {model}: {len(m)} conditions, expected {conditions}")
            out += [f"{g} {model} {r.condition}: n = {r.n}, expected {want}" for r in m.itertuples() if r.n != want]
    return out


def condition_problems(table, targets, expected):
    """Conditions the targets file wants and the results don't have, or the other way round -> list of messages."""
    out = []
    for g, want in expected.items():
        if not want:
            continue
        rows = table[table.group == g]
        for model, t in targets.groupby("model"):
            have, need = set(rows[rows.model == model].condition), set(t.condition)
            out += [f"{g} {model}: no results for {c}" for c in sorted(need - have)]
            out += [f"{g} {model}: results for a condition not in the targets: {c}" for c in sorted(have - need)]
    return out


def add_targets(table, targets):
    """One line per row and budget, in a fixed order: a condition two budgets share is under both."""
    out = table.merge(targets[["model", "row", "condition", "target_pct", "knob", "predicted_pct"]], on=["model", "row", "condition"], how="left")
    out["miss_pts"] = out["budget_pct"] - out["target_pct"]
    out["nearest_only"] = out["miss_pts"].abs() > calibrate.NEAREST_ONLY_PTS
    out["order"] = out["row"].map({row: i for i, row in enumerate(ROW_ORDER)})
    return out.sort_values(["group", "model", "order", "target_pct"]).drop(columns="order").reset_index(drop=True)


def gpu_types(out_dir, models):
    """{model: the GPU types its jobs recorded}. A folder whose jobs didn't record them gives no entry."""
    found = {}
    for model in models:
        path = os.path.join(out_dir, model, "gpu_types.txt")
        if os.path.exists(path):
            with open(path) as f:
                found[model] = sorted({line.strip() for line in f if line.strip()})
    return found


def error_records(out_dir, models=seed_results.MODELS, rows=ROW_FILES):
    """{(model, row): records with an error}. They are left out of the summary, so they show up as a low n too."""
    found = {}
    for model in models:
        for row in rows:
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
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=["group", "model", "row", "condition", "n", "budget_pct"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--list", help="only these videos: a list written by the earlier steps (default: every video with questions)")
    ap.add_argument("--models", default=",".join(seed_results.MODELS), help="comma list of the models this run covers")
    ap.add_argument("--targets", help="targets.csv from calibrate.py --fractions: each row's condition at each budget")
    a = ap.parse_args()
    cfg = common.load_config(prepare.CONFIG)
    out_dir = os.path.abspath(os.path.expanduser(a.out_dir))
    cfg["paths"]["out_dir"] = out_dir
    models = [m.strip() for m in a.models.split(",")]
    counts = prepare.question_counts()
    groups = video_groups(paths.read_list(a.list) if a.list else counts)
    table = build_table(cfg, groups)
    expected = expected_questions(counts, groups)
    targets = pd.read_csv(a.targets) if a.targets else None
    if targets is None:
        found = n_problems(table, expected, models)
        rows_scored = ROW_FILES
    else:
        found = n_problems(table, expected, models, targets.condition.nunique()) + condition_problems(table, targets, expected)
        rows_scored = ROW_ORDER
        table = add_targets(table, targets)
    path = os.path.join(out_dir, "summary_by_group.csv")
    table.to_csv(path, index=False)

    print(NOTE, flush=True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 60)
    show = ["model", "condition", "n", "accuracy", "ci_low", "ci_high", "budget_pct", "frames_per_q", "gaze_missing_pct", "videos"]
    if targets is not None:
        show = ["model", "row", "target_pct", "budget_pct", "miss_pts", "nearest_only", "n", "accuracy", "ci_low", "ci_high", "frames_per_q",
                "gaze_missing_pct", "prompt_tokens_per_q", "videos"]
    for g in groups:
        rows = table[table.group == g]
        if len(rows):
            print(f"\n--- {g}", flush=True)
            print(rows[show].round(2).to_string(index=False), flush=True)
    print(f"\nwritten {path}", flush=True)

    errors = error_records(out_dir, models, rows_scored)
    if errors:
        print(f"records with an error: {errors}", flush=True)
    types = gpu_types(out_dir, models)
    if types:
        print(f"GPU types recorded: {types}", flush=True)
    found += [f"{model}: results from more than one GPU type: {names}" for model, names in types.items() if len(names) > 1]
    for line in found[:40]:
        print("  ", line, flush=True)
    if found:
        sys.exit(f"INCOMPLETE: {len(found)} problems. Rerun the unfinished model, or tell the user.")
    print("EVERY n IS COMPLETE", flush=True)


if __name__ == "__main__":
    main()

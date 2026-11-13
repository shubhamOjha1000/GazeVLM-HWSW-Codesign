"""The FLOPs-matched table (runbook sections 14 and 15): per row and budget, for all videos, AEA and Nymeria. CPU only.

    python -u -m accuracy_tables.hpc.all350.flops_report --out-dir <results folder> [--model smolvlm] [--plan <flops_plan.json>] [--list all350_pilot]

Reads the plan (default <out-dir>/<model>/flops_plan.json; section 15 keeps it in <out-dir>/calibrate/<model>/, because the merge
rewrites <out-dir>/<model>/) and every row's results. Each question's FLOPs come from its own record (its frames and its
prompt's tokens) and the plan's measured units, as accuracy_tables/flops.py counts them; a baseline is charged every frame it encoded,
also those it left out of the prompt. Per row and budget: n, accuracy with its 95% interval from resampling whole videos, the FLOPs
share of Uniform K on the same questions against its target (on Qwen,
ours' and the no-gate control's vision is the plan's sum over each video's own frame layouts, and their image tokens must be the plan's), the token share, GFLOPs, frames and prompt tokens per question. Writes
<out-dir>/summary_flops_by_group.csv. Declares no winner. Fails if a planned condition isn't complete, after writing the file.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from accuracy_tables import flops, run
from accuracy_tables.hpc.all350 import prepare, report
from accuracy_tables.hpc.newvideos import paths

NOTE = ("matched on total FLOPs per question (gate + vision encoder + LLM prefill), with the baselines' own work counted as zero; "
        "nogate keeps ours' token budget; no winner is declared")
ROW_ORDER = ("uniform", "blind", "budget", "nogate", "uniform_flops") + run.BASELINE_ROWS
STATS = ["row", "condition", "n", "accuracy", "ci_low", "ci_high", "flops_pct", "token_pct", "gflops_per_q", "frames_per_q",
         "frames_encoded_per_q", "prompt_tokens_per_q", "videos"]


def load(out_dir, model):
    """(every row's records without errors, {row: records with an error})."""
    from vlmeval import egoev
    parts, errors = [], {}
    for row in ROW_ORDER:
        df = egoev.load_results(os.path.join(out_dir, model, f"{row}.jsonl"))
        if not len(df):
            continue
        if "error" in df:
            bad = int(df["error"].notna().sum())
            if bad:
                errors[row] = bad
            df = df[df["error"].isna()]
        parts.append(df.assign(row=row))
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), errors


def add_flops(res, plan):
    """res with frames_encoded and flops per record."""
    units, videos = plan["units"], plan["videos"]
    k_of = {(ln["row"], ln["condition"]): ln["K"] for ln in plan["lines"]}
    unknown = sorted(set(res.video) - set(videos))
    if unknown:
        sys.exit(f"results for videos the plan doesn't have: {unknown[:5]}. Stop and tell the user.")
    res = res.copy()
    k = np.array([k_of.get((r, c), 0) for r, c in zip(res.row, res.condition)], dtype=np.int64)
    video_frames = res.video.map({v: x["frames"] for v, x in videos.items()}).to_numpy().astype(np.int64)
    baseline = res.row.isin(run.BASELINE_ROWS).to_numpy()
    n_frames = res["n_frames"].fillna(0).to_numpy()
    res["frames_encoded"] = np.where(baseline, np.minimum(k, np.maximum(video_frames, 0)), n_frames)
    unit = np.where(res.row.isin(flops.FOVEA_ROWS), units["v_fovea"], units["v_stock"])
    views = np.where(res.row.isin(flops.TWO_VIEWS), 2, 1)
    decisions = res.video.map({v: x["decisions"] for v, x in videos.items()}).fillna(0).to_numpy()
    gate = np.where(res.row.isin(flops.GATE_ROWS), flops.GATE_MODELS * units["gate_per_decision"] * decisions, 0.0)
    vision = views * res["frames_encoded"].to_numpy() * unit
    fovea = plan.get("fovea")
    if fovea:
        planned = [fovea.get(c, {}).get(v, [np.nan])[0] if r in flops.FOVEA_ROWS else 0.0
                   for r, c, v in zip(res.row, res.condition, res.video)]
        vision = np.where(res.row.isin(flops.FOVEA_ROWS).to_numpy(), np.asarray(planned, dtype=np.float64), vision)
    res["flops"] = gate + vision + flops.prefill(units["prefill"], res["n_tokens"].to_numpy())
    return res


def fovea_problems(res, plan):
    """Ours' and the no-gate control's records whose image tokens aren't what the plan's layouts give (Qwen only) -> messages."""
    fovea = plan.get("fovea")
    if not fovea or not len(res):
        return []
    out = {}
    for r, c, v, n in zip(res.row, res.condition, res.video, res.n_image_tokens):
        if r in flops.FOVEA_ROWS and fovea.get(c, {}).get(v, [None, None])[1] != n:
            out[r, c] = out.get((r, c), 0) + 1
    return [f"{r} {c}: {n} records whose image tokens aren't the plan's" for (r, c), n in out.items()]


def condition_stats(res, ref_condition):
    """Per (row, condition): n, accuracy and its interval, FLOPs and token shares of the reference on the same questions, GFLOPs,
    frames and prompt tokens per question."""
    from vlmeval import egoev
    ref = res[(res.row == "uniform") & (res.condition == ref_condition)].set_index("qid")
    out = []
    for (row, cond), g in res.groupby(["row", "condition"], sort=False):
        g = g.copy()
        g["correct"] = g["correct"].astype(int)
        acc = egoev.accuracy_table(g).iloc[0]
        common_q = g.set_index("qid").index.intersection(ref.index)
        gq = g.set_index("qid").loc[common_q]
        flops_pct = 100 * gq["flops"].sum() / ref.loc[common_q, "flops"].sum() if len(common_q) else np.nan
        token_pct = 100 * gq["n_image_tokens"].sum() / ref.loc[common_q, "n_image_tokens"].sum() if len(common_q) else np.nan
        out.append(dict(row=row, condition=cond, n=int(acc.n), accuracy=acc.accuracy, ci_low=acc.ci_low, ci_high=acc.ci_high,
                        flops_pct=flops_pct, token_pct=token_pct, gflops_per_q=g["flops"].mean() / 1e9,
                        frames_per_q=g["n_frames"].mean(), frames_encoded_per_q=g["frames_encoded"].mean(),
                        prompt_tokens_per_q=g["n_tokens"].mean(), videos=g.video.nunique()))
    return pd.DataFrame(out, columns=STATS)


def table_for(stats, plan, group, model):
    """One line per planned row and budget, in a fixed order that says nothing about accuracy."""
    lines = pd.DataFrame([{k: v for k, v in ln.items() if k != "verified"} for ln in plan["lines"]])
    lines = lines.rename(columns={"predicted_pct": "planned_pct"})
    out = lines.merge(stats, on=["row", "condition"], how="left")
    out["miss_pts"] = out["flops_pct"] - out["target_pct"]
    out["nearest_only"] = out["miss_pts"].abs() > flops.NEAREST_ONLY_PTS
    out["order"] = out["row"].map({r: i for i, r in enumerate(ROW_ORDER)})
    out = out.sort_values(["order", "budget_pct"]).drop(columns="order")
    out.insert(0, "group", group)
    out.insert(1, "model", model)
    out["note"] = NOTE
    return out.reset_index(drop=True)


def problems(table, plan, expected, stats_by_group):
    """What is incomplete: a planned condition without all its questions, or results for a condition the plan doesn't have."""
    out = []
    want = {(ln["row"], ln["condition"]) for ln in plan["lines"]}
    for g, n_want in expected.items():
        if not n_want:
            continue
        t = table[table.group == g].drop_duplicates(["row", "condition"])
        out += [f"{g} {r.row} {r.condition}: n = {0 if pd.isna(r.n) else int(r.n)}, expected {n_want}"
                for r in t.itertuples() if pd.isna(r.n) or int(r.n) != n_want]
        s = stats_by_group.get(g)
        if s is not None and len(s):
            out += [f"{g}: results for a condition not in the plan: {r} {c}" for r, c in zip(s.row, s.condition) if (r, c) not in want]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="smolvlm")
    ap.add_argument("--plan", help="the plan, if not <out-dir>/<model>/flops_plan.json")
    ap.add_argument("--list", help="only these videos: a list written by the earlier steps (default: every video with questions)")
    a = ap.parse_args()
    out_dir = os.path.abspath(os.path.expanduser(a.out_dir))
    with open(a.plan or os.path.join(out_dir, a.model, "flops_plan.json")) as f:
        plan = json.load(f)
    counts = prepare.question_counts()
    groups = report.video_groups(paths.read_list(a.list) if a.list else counts)
    expected = report.expected_questions(counts, groups)
    res, errors = load(out_dir, a.model)
    if len(res):
        res = add_flops(res, plan)
    ref_condition = next(ln["condition"] for ln in plan["lines"] if ln["row"] == "uniform")
    tables, stats_by_group = [], {}
    for g, vs in groups.items():
        if not vs:
            continue
        sub = res[res.video.isin(vs)] if len(res) else res
        stats = condition_stats(sub, ref_condition) if len(sub) else pd.DataFrame(columns=STATS)
        stats_by_group[g] = stats
        tables.append(table_for(stats, plan, g, a.model))
    table = pd.concat(tables, ignore_index=True)
    path = os.path.join(out_dir, "summary_flops_by_group.csv")
    table.to_csv(path, index=False)

    print(NOTE, flush=True)
    print(f"units: one stock frame {plan['units']['v_stock'] / 1e9:,.2f} GFLOPs, one foveated frame {plan['units']['v_fovea'] / 1e9:,.2f} "
          f"GFLOPs, one gate decision {plan['units']['gate_per_decision']:,.0f} FLOPs x {flops.GATE_MODELS} models", flush=True)
    pd.set_option("display.width", 260)
    show = ["row", "budget_pct", "target_pct", "planned_pct", "flops_pct", "miss_pts", "nearest_only", "K", "knob", "token_pct", "n",
            "accuracy", "ci_low", "ci_high", "gflops_per_q", "frames_per_q", "frames_encoded_per_q", "prompt_tokens_per_q", "videos"]
    for g in groups:
        rows = table[table.group == g]
        if len(rows):
            print(f"\n--- {g}", flush=True)
            print(rows[show].round(3).to_string(index=False), flush=True)
    print(f"\nwritten {path}", flush=True)

    found = problems(table, plan, expected, stats_by_group) + fovea_problems(res, plan)
    if errors:
        print(f"records with an error: {errors}", flush=True)
    types = report.gpu_types(out_dir, [a.model])
    if types:
        print(f"GPU types recorded: {types}", flush=True)
    found += [f"{m}: results from more than one GPU type: {names}" for m, names in types.items() if len(names) > 1]
    for line in found[:40]:
        print("  ", line, flush=True)
    if found:
        sys.exit(f"INCOMPLETE: {len(found)} problems. Rerun the unfinished job, or tell the user.")
    print("EVERY n IS COMPLETE", flush=True)


if __name__ == "__main__":
    main()

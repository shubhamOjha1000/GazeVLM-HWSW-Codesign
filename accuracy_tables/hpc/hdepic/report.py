"""Runbook section 17, step 8: the FLOPs-matched table on HD-EPIC's gaze questions. CPU only. Declares no winner.

    python -u -m accuracy_tables.hpc.hdepic.report --out-dir <results folder> [--model smolvlm] [--reference <section 14 summary>]

Decided before any accuracy existed (2026-11-13):
  groups     both        both types together, every question weighted the same
             gaze_estimation, interaction_anticipation          each type alone
             mean_of_types   the mean of the 2 types' accuracies (and of their other columns)
  interval   section 14's: egoev.video_bootstrap, 95%, 1,000 draws. Each question is its own clip here, so questions are resampled by
             their real HD-EPIC video, as section 14 resamples questions by video. mean_of_types' interval resamples videos the same way
             and takes the mean of the 2 types' accuracies in each draw.
  also       chance per type (1 / choices), Uniform minus Blind per type with the same interval, and a side table against section 14's
             EgoEverything table (results/smolvlm_flops/summary_flops_by_group.csv, only read)
The gaze_estimation lines carry "gaze-driven rows see the wearer's gaze, which defines this type's answer". With about 1,000 questions a
type, intervals are about +-3 points: no gap is called a difference. Writes <out-dir>/summary_flops_by_group.csv (section 14's columns),
summary_chance_uniform_blind.csv and summary_vs_egoeverything.csv. Fails, after writing, if a planned condition isn't complete.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from accuracy_tables.hpc.all350 import flops_report as rep
from accuracy_tables.hpc.hdepic import data

GROUPS = ("both", "gaze_estimation", "interaction_anticipation")
LEAK = "gaze-driven rows see the wearer's gaze, which defines this type's answer"
B = 1000


def with_real_videos(res, real):
    """FLOPs stay per clip (add_flops reads each clip's frames); the interval and the video count use the real video."""
    return res.assign(video=res.video.map(real))


def group_table(res, plan, group, model, ref_condition):
    stats = rep.condition_stats(res, ref_condition) if len(res) else pd.DataFrame(columns=rep.STATS)
    t = rep.table_for(stats, plan, group, model)
    if group == "gaze_estimation":
        t["note"] = t["note"] + "; " + LEAK
    return t, stats


def mean_of_types(tables, res, plan, model):
    """The mean of the 2 types' lines, with a video-bootstrap interval of the mean of their accuracies."""
    from vlmeval import egoev
    a, b = (tables[t].set_index(["row", "budget_pct"]) for t in GROUPS[1:])
    numeric = [c for c in a.columns if c not in ("group", "model", "condition", "note", "nearest_only") and pd.api.types.is_numeric_dtype(a[c])]
    out = a.copy()
    out[numeric] = (a[numeric] + b[numeric]) / 2
    out["group"], out["note"] = "mean_of_types", a["note"].str.replace("; " + LEAK, "", regex=False)
    out["n"] = a["n"] + b["n"]
    for key, line in out.iterrows():
        cond = line["condition"]
        g = res[res.condition == cond]
        if not len(g) or g["type"].nunique() < 2:
            continue
        stat = lambda d: float(np.nanmean([100 * d.loc[d.type == t, "correct"].mean() for t in GROUPS[1:]]))
        lo, hi = egoev.video_bootstrap(g, stat, B)
        out.loc[key, ["ci_low", "ci_high"]] = lo, hi
        out.loc[key, "accuracy"] = stat(g)
    out["nearest_only"] = out["miss_pts"].abs() > rep.flops.NEAREST_ONLY_PTS
    return out.reset_index()[tables["both"].columns]


def chance_and_uniform_minus_blind(res, plan, n_choices):
    """Per type: chance (1 / choices), Uniform and Blind accuracy, and their difference with a video-bootstrap interval."""
    from vlmeval import egoev
    uni = next(ln["condition"] for ln in plan["lines"] if ln["row"] == "uniform")
    blind = next(ln["condition"] for ln in plan["lines"] if ln["row"] == "blind")
    rows = []
    for t in GROUPS:
        g = res if t == "both" else res[res.type == t]
        g = g[g.condition.isin([uni, blind])]
        if not len(g):
            continue
        diff = lambda d: 100 * (d.loc[d.condition == uni, "correct"].mean() - d.loc[d.condition == blind, "correct"].mean())
        lo, hi = egoev.video_bootstrap(g, diff, B)
        choices = n_choices[t]
        rows.append(dict(group=t, chance_pct=100 / choices, uniform_pct=100 * g.loc[g.condition == uni, "correct"].mean(),
                         blind_pct=100 * g.loc[g.condition == blind, "correct"].mean(), uniform_minus_blind=diff(g),
                         ci_low=lo, ci_high=hi, note=LEAK if t == "gaze_estimation" else ""))
    return pd.DataFrame(rows)


def vs_egoeverything(ours, reference):
    """Per row and budget: section 14's EgoEverything (all videos) next to each HD-EPIC group. Two datasets, two runs: no test."""
    keys = ["row", "budget_pct"]
    cols = ["accuracy", "ci_low", "ci_high", "flops_pct"]
    ego = reference[reference.group == "all"][keys + cols].rename(columns={c: f"{c}_egoeverything" for c in cols})
    parts = []
    for g in ("both", *GROUPS[1:], "mean_of_types"):
        x = ours[ours.group == g][keys + cols].rename(columns={c: f"{c}_hdepic" for c in cols}).assign(hdepic_group=g)
        parts.append(ego.merge(x, on=keys, how="outer"))
    out = pd.concat(parts, ignore_index=True)
    out["note"] = "two datasets, each matched on its own FLOPs; no paired test, no winner"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="smolvlm")
    ap.add_argument("--reference", help="section 14's summary_flops_by_group.csv (only read)")
    a = ap.parse_args(argv)
    out_dir = os.path.abspath(a.out_dir)
    with open(os.path.join(out_dir, a.model, "flops_plan.json")) as f:
        plan = json.load(f)
    with open(data.questions_json()) as f:
        qs = pd.DataFrame(json.load(f))
    type_of, real = dict(zip(qs.video, qs.type)), dict(zip(qs.video, qs.real_video))
    n_choices = {t: int(qs[qs.type == t].options.map(len).median()) if t != "both" else int(qs.options.map(len).median())
                 for t in GROUPS}
    res, errors = rep.load(out_dir, a.model)
    res = rep.add_flops(res, plan)
    res["type"] = res.video.map(type_of)
    res = with_real_videos(res, real)
    ref_condition = next(ln["condition"] for ln in plan["lines"] if ln["row"] == "uniform")
    tables, stats_by_group, expected = {}, {}, {}
    for g in GROUPS:
        sub = res if g == "both" else res[res.type == g]
        tables[g], stats_by_group[g] = group_table(sub, plan, g, a.model, ref_condition)
        expected[g] = int(len(qs) if g == "both" else (qs.type == g).sum())
    tables["mean_of_types"] = mean_of_types(tables, res, plan, a.model)
    table = pd.concat([tables[g] for g in (*GROUPS, "mean_of_types")], ignore_index=True)
    table.to_csv(os.path.join(out_dir, "summary_flops_by_group.csv"), index=False)
    extra = chance_and_uniform_minus_blind(res, plan, n_choices)
    extra.to_csv(os.path.join(out_dir, "summary_chance_uniform_blind.csv"), index=False)

    print(rep.NOTE, flush=True)
    pd.set_option("display.width", 260)
    show = ["row", "budget_pct", "target_pct", "planned_pct", "flops_pct", "miss_pts", "nearest_only", "K", "knob", "token_pct", "n",
            "accuracy", "ci_low", "ci_high", "gflops_per_q", "frames_per_q", "frames_encoded_per_q", "prompt_tokens_per_q", "videos"]
    for g in (*GROUPS, "mean_of_types"):
        print(f"\n--- {g}" + (f"  ({LEAK})" if g == "gaze_estimation" else ""), flush=True)
        print(tables[g][show].round(3).to_string(index=False), flush=True)
    print("\n--- chance and Uniform minus Blind per type (points)", flush=True)
    print(extra.round(2).to_string(index=False), flush=True)
    if a.reference and os.path.exists(a.reference):
        vs = vs_egoeverything(table, pd.read_csv(a.reference))
        vs.to_csv(os.path.join(out_dir, "summary_vs_egoeverything.csv"), index=False)
        print(f"\nwritten {out_dir}/summary_vs_egoeverything.csv ({len(vs)} lines)", flush=True)
    print(f"\nwritten {out_dir}/summary_flops_by_group.csv and summary_chance_uniform_blind.csv", flush=True)

    found = rep.problems(table[table.group.isin(GROUPS)], plan, expected, stats_by_group)
    if errors:
        print(f"records with an error: {errors}", flush=True)
    from accuracy_tables.hpc.all350 import report
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

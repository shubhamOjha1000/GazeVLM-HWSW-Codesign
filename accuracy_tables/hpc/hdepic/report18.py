"""Runbook section 18, step 8: the FLOPs-matched tables on HD-EPIC's 6 groups. CPU only. Declares no winner.

    python -u -m accuracy_tables.hpc.hdepic.report18 --out-dir <results folder> [--model smolvlm] [--reference <section 17 summary>]

Decided before any accuracy existed (2026-11-14):
  groups    all              every question, each weighted the same
            <group>          each of the 6 groups; fine_grained is labelled "2,000-question stratified sample of 21,000"
            <group>/<type>   each type
            mean_of_groups   the mean of the 6 groups' accuracies (and of their other columns)
  FLOPs     every table's flops_pct is measured on its own questions; the targets are matched over all questions, not per group
  interval  section 14's video bootstrap (95%, 1,000 draws), resampling real HD-EPIC videos (a question of several videos counts by its
            first); mean_of_groups resamples videos the same way and takes the mean of the 6 groups' accuracies in each draw
  also      chance (1 / choices) and Uniform minus Blind per group, and a side table against section 17's gaze group
            (results/hdepic_gaze_smolvlm_flops/summary_flops_by_group.csv, only read), labelled as a separate run
  notes     object_motion's and ingredient's leakage notes on their tables; on every table, D3's prompt deviation
Writes <out-dir>/summary_flops_by_group.csv, summary_chance_uniform_blind.csv and summary_vs_section17.csv. Fails, after writing, if a
planned condition isn't complete.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from accuracy_tables.hpc.all350 import flops_report as rep
from accuracy_tables.hpc.hdepic import groups, s18

B = 1000
D3_NOTE = "prompt: 'Frame at <TIME hh:mm:ss.sss video n>:' before every frame (D3, a deviation from section 14)"
SEPARATE = "separate run: section 17's gaze group on its own FLOPs targets; two runs, no paired test, no winner"


def label(group):
    return f"{group} ({s18.SAMPLE_LABEL})" if group == groups.SAMPLE_GROUP else group


def group_table(res, plan, name, model, ref_condition, note=""):
    stats = rep.condition_stats(res, ref_condition) if len(res) else pd.DataFrame(columns=rep.STATS)
    t = rep.table_for(stats, plan, name, model)
    t["note"] = t["note"] + "; " + D3_NOTE + (("; " + note) if note else "")
    return t, stats


def mean_of_groups(tables, res, names):
    """The mean of the groups' lines, with a video-bootstrap interval of the mean of their accuracies."""
    from vlmeval import egoev
    parts = [tables[g].set_index(["row", "budget_pct"]) for g in names]
    a = parts[0]
    numeric = [c for c in a.columns if c not in ("group", "model", "condition", "note", "nearest_only") and pd.api.types.is_numeric_dtype(a[c])]
    out = a.copy()
    out[numeric] = sum(p[numeric] for p in parts) / len(parts)
    out["n"] = sum(p["n"] for p in parts)
    out["group"], out["note"] = "mean_of_groups", rep.NOTE + "; " + D3_NOTE
    for key, line in out.iterrows():
        g = res[res.condition == line["condition"]]
        if not len(g) or g["group"].nunique() < len(names):
            continue
        stat = lambda d: float(np.nanmean([100 * d.loc[d.group == x, "correct"].mean() for x in names]))
        lo, hi = egoev.video_bootstrap(g, stat, B)
        out.loc[key, ["ci_low", "ci_high"]] = lo, hi
        out.loc[key, "accuracy"] = stat(g)
    out["nearest_only"] = out["miss_pts"].abs() > rep.flops.NEAREST_ONLY_PTS
    return out.reset_index()[tables[names[0]].columns]


def chance_and_uniform_minus_blind(res, plan, qs):
    """Per group and all: chance (1 / choices), Uniform and Blind accuracy, their difference with a video-bootstrap interval."""
    from vlmeval import egoev
    uni = next(ln["condition"] for ln in plan["lines"] if ln["row"] == "uniform")
    blind = next(ln["condition"] for ln in plan["lines"] if ln["row"] == "blind")
    rows = []
    for g in ("all", *groups.GROUPS):
        x = res if g == "all" else res[res.group == g]
        x = x[x.condition.isin([uni, blind])]
        q = qs if g == "all" else qs[qs.group == g]
        if not len(x):
            continue
        diff = lambda d: 100 * (d.loc[d.condition == uni, "correct"].mean() - d.loc[d.condition == blind, "correct"].mean())
        lo, hi = egoev.video_bootstrap(x, diff, B)
        rows.append(dict(group=label(g), chance_pct=100 * float(np.mean([1 / len(o) for o in q.options])),
                         uniform_pct=100 * x.loc[x.condition == uni, "correct"].mean(), blind_pct=100 * x.loc[x.condition == blind, "correct"].mean(),
                         uniform_minus_blind=diff(x), ci_low=lo, ci_high=hi, note=s18.NOTES.get(g, "")))
    return pd.DataFrame(rows)


def vs_section17(ours, reference):
    """Per row and budget: section 17's gaze group next to this run's all and mean_of_groups. Labelled a separate run."""
    keys = ["row", "budget_pct"]
    cols = ["accuracy", "ci_low", "ci_high", "flops_pct"]
    ref = reference[reference.group == "both"][keys + cols].rename(columns={c: f"{c}_section17_gaze" for c in cols})
    parts = []
    for g in ("all", "mean_of_groups"):
        x = ours[ours.group == g][keys + cols].rename(columns={c: f"{c}_section18" for c in cols}).assign(section18_group=g)
        parts.append(ref.merge(x, on=keys, how="outer"))
    out = pd.concat(parts, ignore_index=True)
    out["note"] = SEPARATE
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="smolvlm")
    ap.add_argument("--reference", help="section 17's summary_flops_by_group.csv (only read)")
    a = ap.parse_args(argv)
    from accuracy_tables.hpc.hdepic.load18 import first_real_video
    out_dir = os.path.abspath(a.out_dir)
    with open(os.path.join(out_dir, a.model, "flops_plan.json")) as f:
        plan = json.load(f)
    with open(s18.questions_json()) as f:
        qs = pd.DataFrame(json.load(f))
    res, errors = rep.load(out_dir, a.model)
    res = rep.add_flops(res, plan)
    res["group"] = res.video.map(dict(zip(qs.video, qs.group)))
    res["type"] = res.video.map(dict(zip(qs.video, qs.type)))
    res = res.assign(video=res.video.map(first_real_video(qs)))
    ref_condition = next(ln["condition"] for ln in plan["lines"] if ln["row"] == "uniform")
    tables, stats, expected = {}, {}, {}
    names = [("all", res, len(qs), "")]
    names += [(label(g), res[res.group == g], int((qs.group == g).sum()), s18.NOTES.get(g, "")) for g in groups.GROUPS]
    for (g, t), _ in qs.groupby(["group", "type"]):
        names.append((f"{label(g)}/{t}", res[(res.group == g) & (res.type == t)], int(((qs.group == g) & (qs.type == t)).sum()),
                      s18.NOTES.get(g, "")))
    for name, sub, n, note in names:
        tables[name], stats[name] = group_table(sub, plan, name, a.model, ref_condition, note)
        expected[name] = n
    tables["mean_of_groups"] = mean_of_groups(tables, res.assign(group=res.group.map(label)), [label(g) for g in groups.GROUPS])
    table = pd.concat([tables[n] for n in tables], ignore_index=True)
    table.to_csv(os.path.join(out_dir, "summary_flops_by_group.csv"), index=False)
    extra = chance_and_uniform_minus_blind(res, plan, qs)
    extra.to_csv(os.path.join(out_dir, "summary_chance_uniform_blind.csv"), index=False)

    print(rep.NOTE + "; " + D3_NOTE, flush=True)
    pd.set_option("display.width", 260)
    show = ["row", "budget_pct", "target_pct", "planned_pct", "flops_pct", "miss_pts", "nearest_only", "K", "knob", "token_pct", "n",
            "accuracy", "ci_low", "ci_high", "gflops_per_q", "frames_per_q", "frames_encoded_per_q", "prompt_tokens_per_q", "videos"]
    for name, t in tables.items():
        note = s18.NOTES.get(name.split("/")[0].split(" (")[0], "")
        print(f"\n--- {name}" + (f"  ({note})" if note else "") + ("  (measured FLOPs % on these questions; the targets are matched over "
              "all questions)" if name != "all" else ""), flush=True)
        print(t[show].round(3).to_string(index=False), flush=True)
    print("\n--- chance and Uniform minus Blind per group (points)", flush=True)
    print(extra.round(2).to_string(index=False), flush=True)
    if a.reference and os.path.exists(a.reference):
        vs = vs_section17(table, pd.read_csv(a.reference))
        vs.to_csv(os.path.join(out_dir, "summary_vs_section17.csv"), index=False)
        print(f"\n--- against section 17 ({SEPARATE}): written {out_dir}/summary_vs_section17.csv ({len(vs)} lines)", flush=True)
    print(f"\nwritten {out_dir}/summary_flops_by_group.csv and summary_chance_uniform_blind.csv", flush=True)

    found = rep.problems(table[table.group != "mean_of_groups"], plan, expected, stats)
    if errors:
        print(f"records with an error: {errors}", flush=True)
    from accuracy_tables.hpc.all350 import report
    types = report.gpu_types(out_dir, [a.model])
    if types:
        print(f"GPU types recorded: {types}", flush=True)
    found += [f"{m}: results from more than one GPU type: {n}" for m, n in types.items() if len(n) > 1]
    for line in found[:40]:
        print("  ", line, flush=True)
    if found:
        sys.exit(f"INCOMPLETE: {len(found)} problems. Rerun the unfinished job, or tell the user.")
    print("EVERY n IS COMPLETE", flush=True)


if __name__ == "__main__":
    main()

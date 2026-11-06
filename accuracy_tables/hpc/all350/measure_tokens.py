"""Tokens per frame of each model's smallest fovea, from real gaze on every frame of all 350 videos. CPU only. Reads no answers.

    python -u -m accuracy_tables.hpc.all350.measure_tokens

On SmolVLM2 the smallest fovea is always 17 tokens. On Qwen2.5, Qwen3 and Phi the layout's rings depend on where the gaze falls, so
the count varies by frame. The budget rule takes one number per model (budget.tokens_per_frame), and this proposes the mean over all
frames, rounded up. A frame with no gaze counts with the fovea at the image centre, as the run does. Qwen3's layout is Qwen2.5's scaled
up, so they share a count. On Phi a frame also has one row break per token row that has tokens, a separator and its 156-token global view.

It first checks the count against what the 12-video run recorded: on those videos, the tokens it predicts for Ours at this fovea setting
must equal every record of results/hpc_all/. Writes newvideos/all350/tokens_per_frame.csv. The proposed values go into config.all350.yaml
only with the user's OK.
"""
import math
import os
import sys

import numpy as np
import pandas as pd

from accuracy_tables import common, run
from accuracy_tables.hpc.all350 import prepare, seed_results
from accuracy_tables.hpc.newvideos import paths
from accuracy_tables.models import ADAPTERS
from vlmeval import budget_rule as br
from vlmeval import phi_fovea as ph
from vlmeval import qwen_fovea as qf

PHI_SEPARATOR, PHI_GLOBAL_VIEW = 1, 156
CENTRE = (0.5, 0.5)
MEASURED = ("qwen2.5", "qwen3", "phi")
NYMERIA_SECONDS = 1100                     # a typical Nymeria video, only to show the slot lengths


def _gaze(frame):
    return CENTRE if frame.get("gaze") is None else frame["gaze"]


def qwen_tokens(gaze, setting):
    return len(qf.foveated_cells(*gaze, a=setting["a"], img=ADAPTERS["qwen2.5"].SIDE, R_deg=setting["R"]))


def phi_parts(gaze, setting):
    """(crop tokens, row breaks) of a frame on Phi."""
    cells = ph.foveated_cells(*gaze, a=setting["a"], R_deg=setting["R"])
    return len(cells), len(np.unique(cells[:, 0] // ph.BASE))


def phi_tokens(gaze, setting):
    return sum(phi_parts(gaze, setting)) + PHI_SEPARATOR + PHI_GLOBAL_VIEW


def tokens_of(model, frame, setting):
    return phi_tokens(_gaze(frame), setting) if model == "phi" else qwen_tokens(_gaze(frame), setting)


def record_differences(data, setting, records_of):
    """Records of the 12-video Ours row whose tokens aren't what this count predicts -> {model: (records, differing)}."""
    out = {}
    for model in MEASURED:
        per_video = {}
        records = records_of(model)
        bad = 0
        for r in records:
            v = r["video"]
            if v not in per_video:
                per_video[v] = sum(tokens_of(model, fr, setting) for fr in data["gate"][v])
            bad += r["n_image_tokens"] != per_video[v]
        out[model] = (len(records), bad)
    return out


def ours_records(model, setting):
    """The hpc_all Ours records at this fovea setting. Only their tokens and videos are used."""
    ad = ADAPTERS[model]()
    name = run.ours_name(common.DEFAULTS["run"]["policy"], ad.setting_label(setting))
    path = os.path.join(seed_results.results_root(), "hpc_all", model, "ours.jsonl")
    return [r for r in seed_results.read_records(path) if r["condition"] == name and r.get("error") is None]


def spread(values):
    a = np.asarray(values, dtype=np.float64)
    p1, p50, p99 = np.percentile(a, [1, 50, 99])
    return dict(frames=len(a), mean=a.mean(), min=a.min(), p1=p1, p50=p50, p99=p99, max=a.max())


def proposed_value(mean):
    return int(math.ceil(mean))


def limits_table(means, cfg):
    """One row per model and budget, for a video with at least K frames: the limit, the share it should spend, the slot length."""
    rows = []
    for model in means:
        cap = cfg["run"]["max_frames"].get(model)
        full = (min(cfg["run"]["K"], cap) if cap else cfg["run"]["K"]) * ADAPTERS[model].stock_tokens
        mean = means[model]
        for f in cfg["budget"]["fractions"]:
            limit = br.frame_limit(float(f), full, proposed_value(mean), 10 ** 9, cap)
            rows.append(dict(model=model, budget_pct=100 * float(f), tokens_per_frame=proposed_value(mean), limit=limit,
                             expected_spend_pct=100 * limit * mean / full, slot_s=NYMERIA_SECONDS / limit))
    return pd.DataFrame(rows)


def main():
    cfg = common.load_config(prepare.CONFIG)
    data = common.load_data(cfg)
    setting = ADAPTERS["qwen2.5"]().fovea_settings(cfg["ours"])[0]
    print(f"smallest fovea on these 3 models: {setting}", flush=True)

    checked = record_differences(data, setting, lambda model: ours_records(model, setting))
    print(f"against the 12-video records (records, differing): {checked}", flush=True)
    if any(bad or not n for n, bad in checked.values()):
        sys.exit("TOKEN COUNT DIFFERS from the recorded run: stop and tell the user")
    print("TOKEN COUNT MATCHES RECORDS", flush=True)

    group_of = {v: g for g in prepare.GROUPS for v in paths.read_list(g)}
    values = {g: dict(qwen=[], phi=[], phi_crop=[], phi_breaks=[], no_gaze=[]) for g in prepare.GROUPS}
    for i, v in enumerate(data["videos"], 1):
        vals = values[group_of[v]]
        for f in range(data["metas"][v]["n"]):
            fr = data["frame"](v, f)
            g = _gaze(fr)
            crop, breaks = phi_parts(g, setting)
            vals["qwen"].append(qwen_tokens(g, setting))
            vals["phi"].append(crop + breaks + PHI_SEPARATOR + PHI_GLOBAL_VIEW)
            vals["phi_crop"].append(crop)
            vals["phi_breaks"].append(breaks)
            vals["no_gaze"].append(fr["gaze"] is None)
        if i % 50 == 0:
            print(f"   {i} of {len(data['videos'])} videos", flush=True)

    rows = []
    for g in (*prepare.GROUPS, "all"):
        parts = list(values) if g == "all" else [g]
        for what in ("qwen", "phi", "phi_crop", "phi_breaks"):
            rows.append(dict(group=g, count=what, **spread([x for p in parts for x in values[p][what]])))
        no_gaze = [x for p in parts for x in values[p]["no_gaze"]]
        print(f"{g}: {len(no_gaze)} frames, {100 * np.mean(no_gaze):.2f}% with no gaze", flush=True)
    table = pd.DataFrame(rows)
    prepare.write_csv(table, os.path.join(prepare.root(), "tokens_per_frame.csv"))
    pd.set_option("display.width", 200)
    print(table.round(2).to_string(index=False), flush=True)

    overall = table[table.group == "all"].set_index("count")["mean"]
    means = {"qwen2.5": overall["qwen"], "qwen3": overall["qwen"], "phi": overall["phi"], "smolvlm": 17.0}
    print("\nlimits for a video with at least K frames, at the proposed values:", flush=True)
    print(limits_table(means, cfg).round(2).to_string(index=False), flush=True)
    print(f"\nPROPOSED, needs the user's OK: qwen2.5 {proposed_value(means['qwen2.5'])}, qwen3 {proposed_value(means['qwen3'])}, "
          f"phi {proposed_value(means['phi'])}", flush=True)


if __name__ == "__main__":
    main()

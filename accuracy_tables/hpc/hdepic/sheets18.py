"""Runbook section 18, step 1c: gaze overlay sheets for the user to look at, one per group. CPU only, base environment.

    python -u -m accuracy_tables.hpc.hdepic.sheets18 [--per-group 12]

Per group, up to 12 questions (its types taking turns, each in plan order). Each tile is the MP4 frame at one of the question's candidates
(D2's query frame where there is one, else the middle candidate) with the projected gaze at that moment as a dot, its stamp, the correct
choice and the question. Then each planned video's share of projected gaze inside the frame (the lowest 10 printed, and the median).
Writes $GV_ROOT/data/hdepic/s18/sheets/ (HD-EPIC images and text: never commit it). Ends with SHEETS WRITTEN ... STOP.
"""
import argparse
import collections
import json
import os

import numpy as np

from accuracy_tables.hpc.hdepic import data, groups, s18
from accuracy_tables.hpc.hdepic import sheets as sheets17

LETTERS = "ABCDE"


def choose(plan, per_group):
    out = {}
    for g in groups.GROUPS:
        pools = collections.OrderedDict()
        for q in plan["questions"]:
            if q["group"] == g:
                pools.setdefault(q["type"], []).append(q)
        got = []
        while len(got) < per_group and any(pools.values()):
            for t in pools:
                if pools[t] and len(got) < per_group:
                    got.append(pools[t].pop(0))
        out[g] = got
    return out


def shown_frame(q):
    return q["forced"][0] if q["forced"] else q["K"] // 2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-group", type=int, default=12)
    a = ap.parse_args(argv)
    out = s18.sheets_dir()
    data.refuse_inside_raw(out)
    os.makedirs(out, exist_ok=True)
    with open(s18.plan_json()) as f:
        plan = json.load(f)
    folder = data.questions_dir()
    files, _ = groups.question_files(folder)
    raw, gaze = {}, {}
    for g, qs in choose(plan, a.per_group).items():
        tiles = []
        for q in qs:
            key = (q["group"], q["type"])
            if key not in raw:
                with open(os.path.join(folder, files[q["group"]][q["type"]])) as f:
                    raw[key] = json.load(f)
            r = raw[key][q["hdepic_qid"]] if isinstance(raw[key], dict) else raw[key][int(q["hdepic_qid"])]
            j = shown_frame(q)
            v = q["videos"][q["video_of"][j]]
            if v not in gaze:
                with np.load(s18.gaze_npz(v)) as z:
                    gaze[v] = {k: z[k] for k in ("t_us", "x", "y", "oof")}
            gz = sheets17.gaze_near(gaze[v], q["device_us"][j])
            lines = [f"{q['hdepic_qid']} ({q['type']}), {s18.stamp(q['mp4_s'][j], q['video_n'][j])}" + (" query frame" if q["forced"] else ""),
                     "no gaze within 60 ms" if gz is None else f"gaze ({gz[0]:.2f}, {gz[1]:.2f}){' out of view' if gz[2] else ''}",
                     f"answer {LETTERS[int(r['correct_idx'])]}: {r['choices'][int(r['correct_idx'])]}", f"Q: {r['question']}"]
            tiles.append(sheets17.tile(sheets17.read_frame(v, q["mp4_frames"][j]), gz, lines))
        if tiles:
            sheets17.sheet(tiles, os.path.join(out, f"group_{g}.jpg"))
        print(f"   {g}: {len(tiles)} tiles -> {out}/group_{g}.jpg", flush=True)
    shares = {}
    for v in plan["videos"]:
        with np.load(s18.gaze_npz(v)) as z:
            x, y, oof = z["x"], z["y"], z["oof"].astype(bool)
        inside = np.isfinite(x) & ~oof & (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
        shares[v] = 100 * float(inside.mean()) if len(x) else float("nan")
    low = sorted(shares, key=lambda v: shares[v])[:10]
    print(f"projected gaze inside the frame: median {np.nanmedian(list(shares.values())):.1f}% over {len(shares)} videos; lowest "
          + ", ".join(f"{v} {shares[v]:.1f}%" for v in low), flush=True)
    print(f"SHEETS WRITTEN: {len(raw)} types over {len(groups.GROUPS)} groups in {out}. STOP: the user looks at them.", flush=True)


if __name__ == "__main__":
    main()

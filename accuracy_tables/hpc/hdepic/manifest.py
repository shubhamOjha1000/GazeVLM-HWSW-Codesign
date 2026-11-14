"""Runbook section 17, step 4: the question manifest under D1 to D5, then HD-EPIC's own data check. CPU only, base environment.

    python -u -m accuracy_tables.hpc.hdepic.manifest --config accuracy_tables/hpc/config.hdepic.yaml

Writes $GV_ROOT/data/hdepic/questions.json (outside the repository: it holds HD-EPIC's question text, under its non-commercial licence):
one entry per planned question with qid, its clip as video, real_video, type, hdepic_qid, question, options (the 5 choices), answer
(the letter of correct_idx), t0 (its span's end, MP4 seconds), start_s, end_s and K. Then loads the data as every job will (common.load_data)
and checks it: every question once, every clip's frames, gate labels and Uniform K, and section 14's smallest SmolVLM fovea (17 tokens)
on every candidate. Prints the counts per type, every exclusion and its reason, the gaze missing per type, then data check OK.
EgoEverything's own checks (834 / 204 questions) aren't touched.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

from accuracy_tables.hpc.hdepic import data

LETTERS = "ABCDE"


def entries(plan, raw_by_type):
    """questions.json's entries from the plan and the 2 question files ({type: {hdepic_qid: question}})."""
    out = []
    for q in plan["questions"]:
        r = raw_by_type[q["type"]][q["hdepic_qid"]]
        choices = list(r["choices"])
        if len(choices) != q["n_choices"] or not 0 <= int(r["correct_idx"]) < len(choices) or len(choices) > len(LETTERS):
            raise ValueError(f"{q['hdepic_qid']}: {len(choices)} choices, correct_idx {r['correct_idx']}")
        out.append(dict(qid=q["qid"], video=q["pseudo"], real_video=q["video"], type=q["type"], hdepic_qid=q["hdepic_qid"],
                        question=r["question"], options=choices, answer=LETTERS[int(r["correct_idx"])], t0=q["end_s"],
                        start_s=q["start_s"], end_s=q["end_s"], K=q["K"]))
    return out


def read_raw():
    out = {}
    for qtype, name in data.GAZE_TYPES.items():
        with open(os.path.join(data.questions_dir(), name)) as f:
            out[qtype] = json.load(f)
    return out


def data_problems(d, plan, fovea_tokens=17):
    """What common.load_data's result gets wrong against the plan -> (problems, gaze missing per type)."""
    from vlmeval import smolvlm_fovea as sm
    found, missing = [], collections.Counter()
    want = {q["pseudo"]: q for q in plan["questions"]}
    if sorted(d["qs"].video) != sorted(want) or d["qs"].video.duplicated().any():
        found.append(f"{len(d['qs'])} questions loaded, {len(want)} planned")
    for v in d["videos"]:
        q = want.get(v)
        if q is None:
            found.append(f"{v}: not planned")
            continue
        if d["metas"][v]["n"] != q["K"] or len(d["uniform"][v]) != q["K"]:
            found.append(f"{v}: {d['metas'][v]['n']} frames and {len(d['uniform'][v])} Uniform frames, {q['K']} planned")
        if len(d["labels"][v]) != q["K"] - 1:
            found.append(f"{v}: {len(d['labels'][v])} gate labels for {q['K']} frames")
        for f in range(q["K"]):
            fr = d["frame"](v, f)
            g = fr["gaze"] if fr["gaze"] is not None else (0.5, 0.5)
            if fr["gaze"] is None:
                missing[q["type"]] += 1
            if len(sm.foveated_cells(*g, 1)) != fovea_tokens:
                found.append(f"{v} frame {f}: not {fovea_tokens} foveated tokens")
            if not os.path.exists(fr["path"]):
                found.append(f"{v} frame {f}: no {fr['path']}")
    return found, missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    from accuracy_tables import common
    out = data.questions_json()
    data.refuse_inside_raw(out)
    plan = data.read_plan()
    qs = entries(plan, read_raw())
    with open(out + ".tmp", "w") as f:
        json.dump(qs, f)
    os.replace(out + ".tmp", out)
    counts = collections.Counter(q["type"] for q in qs)
    total = {t: counts[t] + sum(e["type"] == t for e in plan["excluded"]) for t in data.GAZE_TYPES}
    for t in data.GAZE_TYPES:
        print(f"{t}: {counts[t]} of {total[t]} questions kept, K median {np.median([q['K'] for q in qs if q['type'] == t]):g}", flush=True)
    for e in plan["excluded"]:
        print(f"   excluded: {e['type']} {e['hdepic_qid']} on {e['video']}: {e['reason']}", flush=True)
    print(f"excluded: {len(plan['excluded'])}" + ("" if plan["excluded"] else " (none)"), flush=True)
    cfg = common.load_config(a.config)
    d = common.load_data(cfg)
    print(common.describe(d), flush=True)
    found, missing = data_problems(d, plan)
    frames = {t: sum(q["K"] for q in qs if q["type"] == t) for t in data.GAZE_TYPES}
    for t in data.GAZE_TYPES:
        print(f"{t}: {missing[t]} of {frames[t]} candidate frames have no gaze (fovea at the centre)", flush=True)
    for p in found[:40]:
        print("  ", p, flush=True)
    if found:
        sys.exit(f"DATA CHECK FAILED: {len(found)} problems. Stop and tell the user.")
    print(f"data check OK: {len(qs)} questions on {len(plan['videos'])} videos, every clip's frames, gate labels and Uniform K, and "
          "17 foveated tokens on every candidate", flush=True)


if __name__ == "__main__":
    main()

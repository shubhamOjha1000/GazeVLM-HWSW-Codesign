"""Runbook section 17, step 1a: every gaze question's candidate frames, under the user's D1 (2026-11-13). CPU only, base environment.

    python -u -m accuracy_tables.hpc.hdepic.plan

D1, as decided: each question is its own clip. Its candidates are K = min(64, the distinct MP4 frames in its span) frames, evenly over
the span by common.uniform_indices, the rule section 14 uses for a video's Uniform K frames. All are distinct, and all are inside the
span, so "what's next" never sees a frame after its span ends. A time-CSV row that repeats an earlier row's time counts once.
"interaction_anticipation" has 10 s spans, so 64 of about 300 frames. "gaze_estimation" has 15 to 64 or more frames, so K is often all
of them. Every frame maps to device time through the time CSV, never through a naive fps.

A question with fewer than 2 candidates (the gate needs a frame pair), or on a video without gaze, is excluded with its reason; none is
expected (step 0). The plan also reports every question with a candidate after its video's gaze ends (D5: P09-20240624-160737's gaze
ends 3.8 s early) or a 1 s gate window starting before its gaze begins. Writes $GV_ROOT/data/hdepic/plan.json: ids, times and frame
numbers, never question text. Ends with PLAN WRITTEN.
"""
import json
import os
import re
import sys

import numpy as np

from accuracy_tables import common
from accuracy_tables.hpc.hdepic import data

K = 64
WINDOW_US = 1_000_000                   # the gate's window before each candidate (build_rows.pair_rows window_us)
SHORT = {"gaze_estimation": "ge", "interaction_anticipation": "ia"}


def distinct_frames(mp4_ns, start_s, end_s):
    """MP4 frames inside [start, end], one per distinct time (the first of any repeated time-CSV rows), in time order."""
    idx = data.frames_between(mp4_ns, start_s, end_s)
    _, first = np.unique(mp4_ns[idx], return_index=True)
    return idx[np.sort(first)]


def candidates(mp4_ns, start_s, end_s, k=K):
    """K = min(k, distinct frames in the span) of them, evenly by common.uniform_indices (first and last included)."""
    idx = distinct_frames(mp4_ns, start_s, end_s)
    return idx[common.uniform_indices(len(idx), k)], len(idx)


def pseudo_name(qtype, hdepic_qid):
    m = re.search(r"(\d+)$", str(hdepic_qid))
    if not m:
        raise ValueError(f"{hdepic_qid}: no number at its end")
    return f"hd_{SHORT[qtype]}_{int(m.group(1)):05d}"


def plan_question(q, qid, video_info, times, gaze_us):
    """One question -> (its plan entry, or None and the reason it's excluded)."""
    inp = q["inputs"][0]
    v = inp["video"]
    if q["n_inputs"] != 1:
        return None, f"{q['n_inputs']} inputs (D2 expects 1)"
    if gaze_us is None:
        return None, "the video has no gaze"
    mp4_ns, vrs_ns = times
    idx, n_distinct = candidates(mp4_ns, inp["start_s"], inp["end_s"])
    if len(idx) < 2:
        return None, f"{len(idx)} distinct frames in its span"
    dev = vrs_ns[idx]
    entry = dict(qid=qid, hdepic_qid=q["qid"], type=q["type"], video=v, pseudo=pseudo_name(q["type"], q["qid"]),
                 start_s=inp["start_s"], end_s=inp["end_s"], n_distinct=int(n_distinct), K=int(len(idx)),
                 mp4_frames=[int(i) for i in idx], mp4_s=[float(t) / 1e9 for t in mp4_ns[idx]], device_us=[float(t) / 1e3 for t in dev],
                 n_choices=q["n_choices"], frames_after_gaze_end=int((dev / 1e3 > gaze_us[-1]).sum()),
                 windows_before_gaze_start=int((dev / 1e3 - WINDOW_US < gaze_us[0]).sum()),
                 past_video_end=bool(inp["end_s"] * 1e9 > mp4_ns[-1]))
    gaps = np.diff(np.asarray(entry["mp4_s"]))
    entry["gap_s_median"] = float(np.median(gaps))
    return entry, None


def video_info(video):
    import cv2
    cap = cv2.VideoCapture(os.path.join(data.video_dir(video), f"{video}.mp4"))
    info = dict(fps=cap.get(cv2.CAP_PROP_FPS), mp4_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    g = data.gaze_csv(video)
    info["gaze_csv"] = g
    return info


def make_plan(qs, info_of, times_of, gaze_of):
    """-> dict(videos, questions, excluded) over both types, qid numbered across them in the order of data.GAZE_TYPES."""
    kept, excluded, qid = [], [], 0
    for qtype in data.GAZE_TYPES:
        for q in qs[qtype]:
            v = q["inputs"][0]["video"]
            entry, why = plan_question(q, qid, info_of.get(v), times_of[v], gaze_of.get(v))
            if entry is None:
                excluded.append(dict(hdepic_qid=q["qid"], type=qtype, video=v, reason=why))
            else:
                kept.append(entry)
                qid += 1
    videos = sorted({e["video"] for e in kept})
    names = [e["pseudo"] for e in kept]
    if len(set(names)) != len(names):
        raise ValueError("two questions got the same clip name")
    return dict(d1="K = min(64, distinct MP4 frames in the span), evenly over it (common.uniform_indices); 1 s gate windows",
                videos=videos, video_info={v: info_of[v] for v in videos}, questions=kept, excluded=excluded)


def report(plan):
    import pandas as pd
    q = pd.DataFrame(plan["questions"])
    print(f"{len(q)} questions kept on {len(plan['videos'])} videos, {len(plan['excluded'])} excluded", flush=True)
    for e in plan["excluded"]:
        print(f"   excluded: {e['type']} {e['hdepic_qid']} on {e['video']}: {e['reason']}", flush=True)
    for qtype, g in q.groupby("type", sort=False):
        k = g.K.to_numpy()
        print(f"{qtype}: {len(g)} questions; K min {k.min()}, median {np.median(k):g}, max {k.max()}; K = 64 on {int((k == 64).sum())}; "
              f"distinct frames in the span median {g.n_distinct.median():g}; candidate gap median {g.gap_s_median.median():.3f} s",
              flush=True)
    late = q[q.frames_after_gaze_end > 0]
    print(f"D5: {len(late)} questions have a candidate after their video's gaze ends"
          + (": " + ", ".join(f"{r.hdepic_qid} ({r.video}, {r.frames_after_gaze_end} frames)" for r in late.itertuples()) if len(late) else ""),
          flush=True)
    early = q[q.windows_before_gaze_start > 0]
    print(f"{len(early)} questions have a 1 s gate window starting before their gaze does; {int(q.past_video_end.sum())} spans end past "
          "their video", flush=True)


def main():
    out = data.plan_json()
    data.refuse_inside_raw(out)
    qs = data.load_gaze_questions()
    videos = sorted({q["inputs"][0]["video"] for rows in qs.values() for q in rows})
    import pandas as pd
    info, times, gaze = {}, {}, {}
    for v in videos:
        info[v] = video_info(v)
        times[v] = data.read_time_csv(os.path.join(data.video_dir(v), f"{v}_mp4_to_vrs_time_ns.csv"))
        if info[v]["gaze_csv"]:
            gaze[v] = pd.read_csv(info[v]["gaze_csv"], usecols=["tracking_timestamp_us"])["tracking_timestamp_us"].to_numpy(np.int64)
            info[v]["gaze_hz"] = data.rate_hz(gaze[v])
    plan = make_plan(qs, info, times, gaze)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w") as f:
        json.dump(plan, f)
    os.replace(out + ".tmp", out)
    report(plan)
    print(f"PLAN WRITTEN: {out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)

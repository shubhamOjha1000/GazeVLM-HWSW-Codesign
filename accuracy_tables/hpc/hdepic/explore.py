"""Runbook section 17, step 0: what HD-EPIC's gaze questions and their videos hold, so the user can decide D1 to D5. CPU only, read-only.

    python -u -m accuracy_tables.hpc.hdepic.explore [--out <folder>] [--sample-frames 6]

Opens only the gaze group's 2 question files and the videos they use. Prints, and writes per_video.csv and per_question.csv (ids, times
and counts only, never question text) to $GV_ROOT/data/hdepic/explore/:

  questions   per type: count, choices per question, input form (one time or a span), inputs per question, videos, <TIME> / <BBOX> tags
  videos      per video: fps, frames, resolution, duration, time-CSV rows against MP4 frames (repeats, backward steps, the largest offset
              from i / fps), gaze file and rate, gaze coverage and its longest gap, IMU arrays, rates and magnitudes, calibration fields
  D1          per type: input span, MP4 frames in it, 1 fps frames in it, frames around the span's end, how each option would sit
              ((a) 64 candidates over the span, (b) 1 fps, (c) spans of 64 s or more): candidate spacing, 10 Hz gaze samples and IMU bins
              per gap, Uniform K, and the budget rule's frame limits at 5 / 10 / 20%
  D2          inputs and videos per question
  D3          tag counts in questions and choices
  D4          every touched video's gaze rate, and whether a whole step reaches the gate's 10 Hz
  D5          questions on the 2 videos without gaze
  leakage     for "where looking": whether gaze is recorded at the span's end (the answer's time) and the span holds it; where the
              projected point falls is step 1's overlay sheets
  disk        frames and bytes each D1 option needs, from JPEGs encoded as section 9 encodes them (long side 1344, quality 92)
Ends with STOP: D1 to D5 are the user's.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import pandas as pd

from accuracy_tables.hpc.hdepic import data

K, STOCK, FOVEA, CAP = 64, 81, 17, 64             # section 14's Uniform K, stock and foveated tokens per frame, frame cap
FRACTIONS = (0.05, 0.1, 0.2)
GATE_HZ = 10.0
FRAME_SIDE, JPEG_QUALITY = 1344, 92                 # vlmeval.frames.extract_1fps as section 9 calls it
AROUND_S = (1, 5, 10)


def quantiles(x, qs=(0, 0.1, 0.5, 0.9, 1)):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=np.float64)
    return {f"q{int(q * 100)}": (float(np.quantile(x, q)) if len(x) else float("nan")) for q in qs} | dict(n=int(len(x)))


def limit(fraction, k):
    """The budget rule's frame limit on k Uniform frames (vlmeval.budget_rule.frame_limit with section 14's numbers)."""
    from vlmeval import budget_rule as br
    return br.frame_limit(fraction, min(k, CAP) * STOCK, FOVEA, max(k, 1), CAP)


def frames_1fps(start_s, end_s):
    """1 fps frames (at whole seconds) inside [start, end]."""
    return max(0, int(np.floor(end_s)) - int(np.ceil(start_s)) + 1)


# ---------------------------------------------------------------------------- per video
def video_stats(video, cache):
    row = dict(video=video)
    d = data.video_dir(video)
    mp4 = os.path.join(d, f"{video}.mp4")
    fps = None
    if os.path.exists(mp4):
        import cv2
        cap = cv2.VideoCapture(mp4)
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        row.update(fps=fps, mp4_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        row["duration_s"] = row["mp4_frames"] / fps if fps else None
    else:
        row["note"] = "no mp4"
    tcsv = os.path.join(d, f"{video}_mp4_to_vrs_time_ns.csv")
    mp4_ns, vrs_ns = data.read_time_csv(tcsv)
    cache[video] = dict(mp4_ns=mp4_ns, vrs_ns=vrs_ns)
    ts = data.time_csv_stats(mp4_ns, fps)
    row.update({f"timecsv_{k}": v for k, v in ts.items()})
    row["timecsv_rows_minus_mp4_frames"] = ts["rows"] - row["mp4_frames"] if "mp4_frames" in row else None
    gpath = data.gaze_csv(video)
    row["gaze_file"] = os.path.basename(gpath) if gpath else None
    if gpath:
        g = pd.read_csv(gpath, usecols=["tracking_timestamp_us"])["tracking_timestamp_us"].to_numpy(np.int64)
        cache[video]["gaze_us"] = g
        hz = data.rate_hz(g)
        row.update(gaze_hz=round(hz, 2), gaze_gate_step=data.gate_step(hz), gaze_samples=int(len(g)),
                   gaze_starts_after_video_s=round((g[0] * 1e3 - vrs_ns[0]) / 1e9, 3),
                   gaze_ends_before_video_s=round((vrs_ns[-1] - g[-1] * 1e3) / 1e9, 3), gaze_longest_gap_s=round(data.longest_gap_s(g), 3))
    imu = os.path.join(d, "imu.npz")
    if os.path.exists(imu):
        s = data.imu_summary(imu)
        row["imu_arrays"] = json.dumps(s["arrays"])
        for side in ("left", "right"):
            for k, v in s.get(side, {}).items():
                row[f"imu_{side}_{k}"] = round(v, 4) if isinstance(v, float) else v
    cal = os.path.join(d, "calibration.json")
    if os.path.exists(cal):
        row["calibration"] = json.dumps(data.calibration_fields(cal), sort_keys=True)
    st = os.path.join(d, "status.json")
    if os.path.exists(st):
        with open(st) as f:
            row["status"] = json.dumps(json.load(f), sort_keys=True)[:400]
    return row


# ---------------------------------------------------------------------------- per question
def question_rows(qs, cache, gaze_rate):
    out = []
    for qtype, rows in qs.items():
        for q in rows:
            for inp in q["inputs"]:
                v, s, e = inp["video"], inp["start_s"], inp["end_s"]
                c = cache.get(v)
                r = dict(qid=q["qid"], type=qtype, video=v, input=inp["name"], form=inp["form"], start_s=s, end_s=e, span_s=e - s,
                         n_inputs=q["n_inputs"], n_videos=len(q["videos"]), n_choices=q["n_choices"],
                         frames_1fps=frames_1fps(s, e), q_time_tags=q["q_time_tags"], q_bbox_tags=q["q_bbox_tags"],
                         choice_time_tags=q["choice_time_tags"], choice_bbox_tags=q["choice_bbox_tags"],
                         no_gaze_video=v in data.NO_GAZE_VIDEOS)
                if c is not None:
                    mp4_ns = c["mp4_ns"]
                    r["frames_native"] = int(len(data.frames_between(mp4_ns, s, e)))
                    r["end_past_video_s"] = round(e - mp4_ns[-1] / 1e9, 3)
                    for w in AROUND_S:
                        r[f"native_within_{w}s_of_end"] = int(len(data.frames_between(mp4_ns, max(0.0, e - w), e + w)))
                    if "gaze_us" in c:
                        g = c["gaze_us"]
                        d0, d1 = data.device_ns(c["vrs_ns"], mp4_ns, s), data.device_ns(c["vrs_ns"], mp4_ns, e)
                        lo, hi = np.searchsorted(g, d0 // 1000), np.searchsorted(g, d1 // 1000, side="right")
                        r["gaze_in_span"] = int(hi - lo)
                        k = int(np.clip(np.searchsorted(g, d1 // 1000), 0, len(g) - 1))
                        near = min(abs(int(g[k]) - d1 // 1000), abs(int(g[max(k - 1, 0)]) - d1 // 1000))
                        r["gaze_nearest_end_ms"] = round(near / 1e3, 1)
                gap = (e - s) / (K - 1)
                r["option_a_gap_s"] = gap
                r["option_a_gaze_per_gap_10hz"] = GATE_HZ * gap
                r["option_a_imu_bins_per_gap"] = int(round(GATE_HZ * gap))
                if gaze_rate.get(v):
                    r["option_a_gaze_per_gap_native"] = gaze_rate[v] * gap
                out.append(r)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------- frame sizes on disk
def jpeg_bytes(videos, per_video):
    """Mean bytes of a frame encoded as section 9 encodes it, from per_video frames spread over each of these videos."""
    import cv2
    sizes = []
    for v in videos:
        cap = cv2.VideoCapture(os.path.join(data.video_dir(v), f"{v}.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in np.linspace(0, max(n - 1, 0), per_video).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok:
                continue
            h, w = fr.shape[:2]
            s = FRAME_SIDE / max(h, w)
            small = cv2.resize(fr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                sizes.append(len(buf))
        cap.release()
    return float(np.mean(sizes)) if sizes else float("nan"), len(sizes)


def option_frames(pq):
    """Frames each D1 option would extract: (a) 64 per question input, (b) the 1 fps frames the spans cover, once per video,
    (c) 64 per input of 64 s or more."""
    b = 0
    for v, g in pq.groupby("video"):
        secs = set()
        for s, e in zip(g.start_s, g.end_s):
            secs.update(range(int(np.ceil(s)), int(np.floor(e)) + 1))
        b += len(secs)
    return dict(a=int(K * len(pq)), b=int(b), c=int(K * (pq.span_s >= K).sum()))


# ---------------------------------------------------------------------------- the report
def show(title, frame):
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 60)
    print(f"\n--- {title}", flush=True)
    print(frame.to_string() if len(frame) else "(none)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="default: $GV_ROOT/data/hdepic/explore")
    ap.add_argument("--sample-frames", type=int, default=6, help="frames per video encoded to estimate the disk use")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(data.derived_root(), "explore")
    data.refuse_inside_raw(out)
    os.makedirs(out, exist_ok=True)

    qs = data.load_gaze_questions()
    all_videos = sorted(os.listdir(os.path.join(data.root(), "videos")))
    print(f"HD-EPIC: {len(all_videos)} video folders; gaze group only: {', '.join(f'{t} ({data.GAZE_TYPES[t]})' for t in qs)}", flush=True)
    summary = []
    for qtype, rows in qs.items():
        forms = collections.Counter(i["form"] for q in rows for i in q["inputs"])
        summary.append(dict(type=qtype, questions=len(rows), choices=dict(collections.Counter(q["n_choices"] for q in rows)),
                            input_forms=dict(forms), inputs_per_q=dict(collections.Counter(q["n_inputs"] for q in rows)),
                            multi_video=sum(len(q["videos"]) > 1 for q in rows), videos=len({v for q in rows for v in q["videos"]}),
                            q_time_tags=sum(q["q_time_tags"] > 0 for q in rows), q_bbox_tags=sum(q["q_bbox_tags"] > 0 for q in rows),
                            choice_time_tags=sum(q["choice_time_tags"] > 0 for q in rows),
                            choice_bbox_tags=sum(q["choice_bbox_tags"] > 0 for q in rows),
                            correct_idx=dict(sorted(collections.Counter(q["correct_idx"] for q in rows).items()))))
    show("questions per type (tag columns: questions with at least one tag)", pd.DataFrame(summary).set_index("type"))
    touched = sorted({v for rows in qs.values() for q in rows for v in q["videos"]})
    missing = [v for v in touched if v not in all_videos]
    print(f"\nthe gaze questions touch {len(touched)} of {len(all_videos)} videos" +
          (" (fewer than about 20: say so in the decisions)" if len(touched) < 20 else "") +
          (f"; NOT ON SCRATCH: {missing}" if missing else ""), flush=True)

    cache, rows = {}, []
    for i, v in enumerate(touched):
        if v in missing:
            continue
        rows.append(video_stats(v, cache))
        if i % 10 == 0:
            print(f"   {i} of {len(touched)} videos read", flush=True)
    pv = pd.DataFrame(rows)
    pv.to_csv(os.path.join(out, "per_video.csv"), index=False)
    cols = [c for c in ("fps", "mp4_frames", "width", "height", "duration_s", "timecsv_rows_minus_mp4_frames", "timecsv_repeated",
                        "timecsv_backwards", "timecsv_max_frame_offset", "gaze_file", "gaze_hz", "gaze_gate_step",
                        "gaze_starts_after_video_s", "gaze_ends_before_video_s", "gaze_longest_gap_s", "imu_left_hz", "imu_right_hz",
                        "imu_right_accel_mag", "imu_right_gyro_mag") if c in pv]
    show("per video", pv.set_index("video")[cols])
    if "calibration" in pv:
        kinds = pv.calibration.dropna().unique()
        print(f"\ncalibration: {len(kinds)} distinct structure(s); the first: {kinds[0] if len(kinds) else None}", flush=True)
    if "imu_arrays" in pv:
        print(f"imu.npz arrays (first video): {pv.imu_arrays.dropna().iloc[0] if pv.imu_arrays.notna().any() else None}", flush=True)
    if "status" in pv:
        print(f"status.json (first video, cut at 400 characters): {pv.status.dropna().iloc[0] if pv.status.notna().any() else None}",
              flush=True)

    gaze_rate = dict(zip(pv.video, pv.get("gaze_hz", pd.Series([None] * len(pv)))))
    pq = question_rows(qs, cache, gaze_rate)
    pq.to_csv(os.path.join(out, "per_question.csv"), index=False)

    d1 = []
    for qtype, g in pq.groupby("type"):
        for what in ("span_s", "frames_native", "frames_1fps", "native_within_1s_of_end", "native_within_5s_of_end",
                     "option_a_gap_s", "option_a_gaze_per_gap_10hz", "option_a_imu_bins_per_gap", "gaze_in_span", "end_past_video_s"):
            if what in g:
                d1.append(dict(type=qtype, measure=what, **quantiles(g[what])))
    show("D1: per type, quantiles over question inputs (q0 = min, q100 = max)", pd.DataFrame(d1).set_index(["type", "measure"]).round(3))
    d1b = []
    for qtype, g in pq.groupby("type"):
        kb = np.minimum(g.frames_1fps.to_numpy(), K)
        d1b.append(dict(type=qtype, inputs=len(g), zero_span=int((g.span_s == 0).sum()), span_ge_64s=int((g.span_s >= K).sum()),
                        option_b_K_below_64=int((kb < K).sum()), option_b_K_zero=int((kb == 0).sum()),
                        **{f"option_b_limit_{int(100 * f)}pct_median": float(np.median([limit(f, k) for k in kb])) for f in FRACTIONS},
                        **{f"option_b_limit_{int(100 * f)}pct_le1": int(sum(limit(f, k) <= 1 for k in kb)) for f in FRACTIONS},
                        option_a_gap_under_1s=int((g.option_a_gap_s < 1).sum()),
                        option_a_limits="/".join(str(limit(f, K)) for f in FRACTIONS)))
    show("D1: what each option does (b: Uniform K = 1 fps frames, capped at 64; a: K = 64 always; c: keeps span >= 64 s)",
         pd.DataFrame(d1b).set_index("type"))
    show("D2: inputs and videos per question", pq.drop_duplicates("qid").groupby("type")[["n_inputs", "n_videos"]].agg(["min", "max", "mean"]))
    show("D3: questions with tags", pq.drop_duplicates("qid").groupby("type")[["q_time_tags", "q_bbox_tags", "choice_time_tags",
                                                                               "choice_bbox_tags"]].agg(lambda s: int((s > 0).sum())))
    if "gaze_hz" in pv:
        rates = pv.groupby(pv.gaze_hz.round(0), dropna=False).agg(videos=("video", "size"), step=("gaze_gate_step", "first"))
        show("D4: gaze rate over the touched videos (step: every how many samples reach 10 Hz; empty: no whole step)", rates)
        none = pv[pv.gaze_hz.notna() & pv.gaze_gate_step.isna()].video.tolist()
        print(f"videos with no whole step to 10 Hz: {none if none else 'none'}", flush=True)
    d5 = pq[pq.no_gaze_video].drop_duplicates("qid").groupby(["type", "video"]).size()
    show("D5: gaze questions on the 2 videos without gaze", d5.to_frame("questions"))
    print(f"videos without a gaze file among the touched: {pv[pv.get('gaze_file').isna()].video.tolist() if 'gaze_file' in pv else 'n/a'}",
          flush=True)
    est = pq[pq.type == "gaze_estimation"]
    if len(est) and "gaze_nearest_end_ms" in est:
        print(f"\nleakage, where looking: {int((est.gaze_nearest_end_ms <= 60).sum())} of {len(est)} inputs have a gaze sample within 60 ms of "
              f"the span's end; the span ends inside the video on {int((est.end_past_video_s <= 0).sum())}. Under option (a) Uniform's "
              "64 candidates include the span's end; where the projected gaze lands is step 1's overlay sheets.", flush=True)

    sample = [v for v in pv.video if "mp4_frames" in pv and not pd.isna(pv.set_index("video").loc[v, "mp4_frames"])][:2]
    mean, n = jpeg_bytes(sample, a.sample_frames) if sample else (float("nan"), 0)
    frames = option_frames(pq)
    disk = pd.DataFrame([dict(option=k, frames=v, files=v, gigabytes=round(v * mean / 1e9, 2) if np.isfinite(mean) else None)
                         for k, v in frames.items()]).set_index("option")
    show(f"disk: frames per D1 option, at {mean / 1e3:,.0f} KB a frame (mean of {n} frames from {len(sample)} videos, long side "
         f"{FRAME_SIDE}, JPEG {JPEG_QUALITY}); compare with myquota above", disk)
    for qtype, ex in data.examples().items():
        print(f"\nexamples, {qtype} (log only: never commit HD-EPIC text):", flush=True)
        for e in ex:
            print(f"   {e['qid']}: inputs {e['inputs']}, correct {e['correct_idx']}", flush=True)
            print(f"      Q: {e['question']}", flush=True)
            for j, c in enumerate(e["choices"] or []):
                print(f"      {j}: {c}", flush=True)
    print(f"\nwritten {out}/per_video.csv and per_question.csv (ids, times and counts, no text)", flush=True)
    print("STOP: D1 to D5 are the user's. Send them this log.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)

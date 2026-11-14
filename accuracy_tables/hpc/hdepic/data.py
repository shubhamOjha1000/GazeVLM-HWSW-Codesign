"""HD-EPIC's gaze questions and per-video files, read-only (runbook section 17). Standard library and numpy / pandas only.

Where things are:
    $HDEPIC_ROOT/videos/<id>/<id>.mp4, <id>_mp4_to_vrs_time_ns.csv, mps/eye_gaze/*.csv, imu.npz, calibration.json, status.json
    $HDEPIC_ROOT/annotations/vqa-benchmark/<group>_<type>.json         only the gaze group's 2 files are ever opened
    $GV_ROOT/data/hdepic/                                               everything derived (outside the repository)

HD-EPIC's licence is non-commercial: nothing read here goes into the repository. Question and choice text is never written to a file;
only counts, ids, times and tag counts are.
"""
import json
import os
import re

import numpy as np

GAZE_TYPES = {"gaze_estimation": "gaze_gaze_estimation.json", "interaction_anticipation": "gaze_interaction_anticipation.json"}
NO_GAZE_VIDEOS = ("P02-20240210-113925", "P02-20240211-182630")
TIME_TAG = re.compile(r"<TIME[^>]*>")
BBOX_TAG = re.compile(r"<BBOX[^>]*>")
TIME_CSV_COLUMNS = ("mp4_time_ns", "relative_vrs_device_time_ns", "vrs_device_time_ns")


def root():
    return os.environ["HDEPIC_ROOT"]


def derived_root():
    return os.path.join(os.environ["GV_ROOT"], "data", "hdepic")


def video_dir(video):
    return os.path.join(root(), "videos", video)


def questions_dir():
    return os.path.join(root(), "annotations", "vqa-benchmark")


def refuse_inside_raw(path):
    """Stops if a path to write is inside $HDEPIC_ROOT, which is only read."""
    real, raw = os.path.realpath(path), os.path.realpath(root())
    if real == raw or real.startswith(raw + os.sep):
        raise SystemExit(f"{path} is inside {raw}, which is only read")


# ---------------------------------------------------------------------------- questions
def parse_time(text):
    """'HH:MM:SS.mmm' (or 'MM:SS.mmm') -> seconds."""
    if "-" in str(text):
        raise ValueError(f"not a time: {text!r}")
    parts = [float(p) for p in str(text).strip().split(":")]
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"not a time: {text!r}")
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def parse_inputs(inputs):
    """A question's `inputs` -> [dict(name, video, form, start_s, end_s)] in the file's order. form: 'time' (one point, start = end) or
    'span' (start_time / end_time); anything else stops, so no question is dropped silently."""
    out = []
    for name, item in inputs.items():
        if "time" in item:
            t = parse_time(item["time"])
            out.append(dict(name=name, video=item["id"], form="time", start_s=t, end_s=t))
        elif "start_time" in item and "end_time" in item:
            out.append(dict(name=name, video=item["id"], form="span", start_s=parse_time(item["start_time"]),
                            end_s=parse_time(item["end_time"])))
        else:
            raise ValueError(f"input {name!r} has neither time nor start_time / end_time: keys {sorted(item)}")
    return out


def tag_counts(text):
    text = str(text)
    return len(TIME_TAG.findall(text)), len(BBOX_TAG.findall(text))


def question_row(qtype, qid, q):
    """One question without its text: ids, inputs, number of choices, the answer's index, and the tags in the question and choices."""
    ins = parse_inputs(q["inputs"])
    choices = list(q.get("choices", []))
    qt, qb = tag_counts(q.get("question", ""))
    ct = sum(tag_counts(c)[0] for c in choices)
    cb = sum(tag_counts(c)[1] for c in choices)
    return dict(qid=qid, type=qtype, inputs=ins, n_inputs=len(ins), videos=sorted({i["video"] for i in ins}), n_choices=len(choices),
                correct_idx=q.get("correct_idx"), q_time_tags=qt, q_bbox_tags=qb, choice_time_tags=ct, choice_bbox_tags=cb)


def load_gaze_questions(folder=None):
    """{type: [question_row]} for the gaze group's 2 types only, in file order. Nothing else in vqa-benchmark is opened."""
    folder = folder or questions_dir()
    out = {}
    for qtype, name in GAZE_TYPES.items():
        with open(os.path.join(folder, name)) as f:
            raw = json.load(f)
        items = raw.items() if isinstance(raw, dict) else ((str(i), q) for i, q in enumerate(raw))
        out[qtype] = [question_row(qtype, qid, q) for qid, q in items]
    return out


def examples(folder=None, n=2):
    """The first n questions of each type with their choices, for the log only (never written to a file)."""
    folder = folder or questions_dir()
    out = {}
    for qtype, name in GAZE_TYPES.items():
        with open(os.path.join(folder, name)) as f:
            raw = json.load(f)
        items = list(raw.items())[:n] if isinstance(raw, dict) else list(enumerate(raw))[:n]
        out[qtype] = [dict(qid=k, question=q.get("question"), choices=q.get("choices"), correct_idx=q.get("correct_idx"),
                           inputs=q.get("inputs")) for k, q in items]
    return out


# ---------------------------------------------------------------------------- the time CSV: MP4 frame i <-> device time
def read_time_csv(path):
    """(mp4_time_ns, vrs_device_time_ns) as int64 arrays; row i is MP4 frame i."""
    import pandas as pd
    t = pd.read_csv(path)
    missing = [c for c in TIME_CSV_COLUMNS if c not in t.columns]
    if missing:
        raise ValueError(f"{path}: no columns {missing}; it has {list(t.columns)}")
    return t["mp4_time_ns"].to_numpy(np.int64), t["vrs_device_time_ns"].to_numpy(np.int64)


def time_csv_stats(mp4_ns, fps):
    """How the time CSV's rows sit against a naive i / fps: rows, repeated and backward steps, the largest gap in frames between
    frame i and round(mp4_time * fps)."""
    d = np.diff(mp4_ns)
    naive = np.round(mp4_ns / 1e9 * fps).astype(np.int64) if fps else np.zeros_like(mp4_ns)
    return dict(rows=int(len(mp4_ns)), repeated=int((d == 0).sum()), backwards=int((d < 0).sum()),
                max_frame_offset=int(np.abs(naive - np.arange(len(mp4_ns))).max()) if len(mp4_ns) and fps else None)


def frames_between(mp4_ns, start_s, end_s):
    """MP4 frame indices whose time is in [start, end] (at least the frame nearest a single time)."""
    lo = int(np.searchsorted(mp4_ns, int(round(start_s * 1e9)), side="left"))
    hi = int(np.searchsorted(mp4_ns, int(round(end_s * 1e9)), side="right"))
    if hi <= lo:
        k = int(np.clip(np.searchsorted(mp4_ns, int(round(start_s * 1e9))), 0, len(mp4_ns) - 1))
        if k > 0 and abs(mp4_ns[k - 1] - start_s * 1e9) <= abs(mp4_ns[k] - start_s * 1e9):
            k -= 1
        return np.array([k])
    return np.arange(lo, hi)


def device_ns(vrs_ns, mp4_ns, t_s):
    """The device time of an MP4 time, through the time CSV (the nearest frame's row)."""
    return int(vrs_ns[frames_between(mp4_ns, t_s, t_s)[0]])


# ---------------------------------------------------------------------------- gaze and IMU
def gaze_csv(video):
    """The MPS gaze CSV of a video (general first, as the AEA builder reads), or None."""
    d = os.path.join(video_dir(video), "mps", "eye_gaze")
    for name in ("general_eye_gaze.csv", "personalized_eye_gaze.csv"):
        if os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    return None


def rate_hz(t, unit=1e6):
    """Median sample rate of timestamps t (in 1 / unit seconds)."""
    t = np.asarray(t, dtype=np.float64)
    d = np.diff(t)
    d = d[d > 0]
    return float(unit / np.median(d)) if len(d) else float("nan")


def gate_step(hz, target=10.0, tolerance=1.0):
    """Every how many samples gives the gate its 10 Hz (as build_rows.gate_gaze_step), or None when no whole step does."""
    if not np.isfinite(hz) or hz <= 0:
        return None
    step = int(round(hz / target))
    return step if step >= 1 and abs(hz / step - target) <= tolerance else None


def longest_gap_s(t, unit=1e6):
    t = np.asarray(t, dtype=np.float64)
    return float(np.diff(t).max() / unit) if len(t) > 1 else float("nan")


def imu_summary(path):
    """{name: shape} of every array in imu.npz, and per side (left / right) its rate and mean accel and gyro magnitudes."""
    out = dict(arrays={})
    with np.load(path) as z:
        for k in z.files:
            out["arrays"][k] = list(z[k].shape)
        for side in ("left", "right"):
            t, a, g = (f"{side}_t_ns", f"{side}_accel", f"{side}_gyro")
            if t in z.files:
                tt = z[t]
                out[side] = dict(hz=rate_hz(tt, 1e9), samples=int(len(tt)),
                                 accel_mag=float(np.linalg.norm(z[a], axis=1).mean()) if a in z.files and len(z[a]) else None,
                                 gyro_mag=float(np.linalg.norm(z[g], axis=1).mean()) if g in z.files and len(z[g]) else None,
                                 span_s=float((tt[-1] - tt[0]) / 1e9) if len(tt) > 1 else 0.0)
    return out


def calibration_fields(path):
    """The calibration file's structure, two levels deep, and the RGB camera's entry if one is found by its label."""
    with open(path) as f:
        cal = json.load(f)
    shape = {k: (sorted(v) if isinstance(v, dict) else type(v).__name__ if not isinstance(v, list) else f"list of {len(v)}")
             for k, v in cal.items()} if isinstance(cal, dict) else type(cal).__name__
    rgb = None
    cams = cal.get("CameraCalibrations", []) if isinstance(cal, dict) else []
    for c in cams:
        if "rgb" in str(c.get("Label", "")).lower():
            rgb = {k: (v if not isinstance(v, (dict, list)) else (sorted(v) if isinstance(v, dict) else f"list of {len(v)}"))
                   for k, v in c.items()}
    return dict(top=shape, rgb=rgb)

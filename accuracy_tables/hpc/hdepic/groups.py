"""Runbook section 18: HD-EPIC's 6 non-gaze question groups, read-only. Standard library and numpy only (pandas inside functions).

Scope, set by the user on 2026-11-14 before any accuracy: fine_grained (a 2,000-question stratified sample of its about 21,000, seed 0),
and all of 3d_perception (about 1,500), object_motion (900), recipe (550), ingredient (400) and nutrition (200): about 5,550 questions,
minus D5's exclusions. The gaze group isn't in this run.

Question files are vqa-benchmark/<group>_<type>.json. A question's `inputs` hold one or more videos or clips and optionally one bounding
box. Everything here keeps ids, times, counts and tag shapes only; question and choice text is never written to a file.
"""
import json
import os
import re

import numpy as np

from accuracy_tables.hpc.hdepic import data

GROUPS = ("fine_grained", "3d_perception", "object_motion", "recipe", "ingredient", "nutrition")
ABOUT = {"fine_grained": 21000, "3d_perception": 1500, "object_motion": 900, "recipe": 550, "ingredient": 400, "nutrition": 200}
SAMPLE_GROUP, SAMPLE_N, SAMPLE_SEED = "fine_grained", 2000, 0
BOX_KEYS = ("bbox", "bounding_box", "box")
TAG = re.compile(r"<(TIME|BBOX)\b[^>]*>(?:[^<]{0,120}</\1>)?")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
CLOCK = re.compile(r"\d+:\d{2}(?::\d{2})?(?:\.\d+)?")


def sample_json():
    return os.path.join(data.derived_root(), "sample_fine_grained_2000.json")


# ---------------------------------------------------------------------------- files
def file_group(name):
    """(group, type) of a vqa-benchmark file name, or (None, None). The gaze group counts as 'gaze'; 'ingredients_' as ingredient."""
    stem = os.path.splitext(os.path.basename(name))[0]
    low = stem.lower()
    for g in sorted(GROUPS + ("gaze",), key=len, reverse=True):
        for prefix in (g + "_", g + "s_"):
            if low.startswith(prefix):
                return g, stem[len(prefix):]
    return None, None


def question_files(folder=None):
    """({group: {type: file name}} for the 6 groups, [other file names]): the gaze group's files and anything unknown."""
    folder = folder or data.questions_dir()
    groups, other = {g: {} for g in GROUPS}, []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        g, t = file_group(name)
        if g in groups:
            groups[g][t] = name
        else:
            other.append(name)
    return groups, other


# ---------------------------------------------------------------------------- one question
def as_box(x):
    """Four numbers [xmin, ymin, xmax, ymax] from a list, a one-item list of one, or a dict with those keys; else None."""
    if isinstance(x, dict) and all(k in x for k in ("xmin", "ymin", "xmax", "ymax")):
        x = [x["xmin"], x["ymin"], x["xmax"], x["ymax"]]
    if isinstance(x, (list, tuple)) and len(x) == 1 and isinstance(x[0], (list, tuple)):
        x = x[0]
    if isinstance(x, (list, tuple)) and len(x) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in x):
        return [float(v) for v in x]
    return None


def parse_inputs(inputs):
    """A question's `inputs` -> (segments, boxes, other), in the file's order.
    segments: dict(name, video, form, start_s, end_s); form 'time' (one instant), 'span' (start_time / end_time) or 'whole' (an id alone)
    boxes:    dict(name, values, segment): a box and the segment it belongs to (the one it sits in, else the one before it)
    other:    dict(name, shape) for anything else, so nothing is dropped silently"""
    segs, boxes, other = [], [], []
    items = inputs.items() if isinstance(inputs, dict) else enumerate(inputs or [])
    for name, item in items:
        box = as_box(item)
        if box is not None:
            boxes.append(dict(name=str(name), values=box, segment=len(segs) - 1 if segs else None))
            continue
        if isinstance(item, dict) and "id" in item:
            if "time" in item:
                t = data.parse_time(item["time"])
                seg = dict(form="time", start_s=t, end_s=t)
            elif "start_time" in item and "end_time" in item:
                seg = dict(form="span", start_s=data.parse_time(item["start_time"]), end_s=data.parse_time(item["end_time"]))
            else:
                seg = dict(form="whole", start_s=None, end_s=None)
            seg.update(name=str(name), video=str(item["id"]))
            for k in BOX_KEYS:
                b = as_box(item.get(k))
                if b is not None:
                    boxes.append(dict(name=f"{name}.{k}", values=b, segment=len(segs)))
            segs.append(seg)
            continue
        if isinstance(item, dict) and any(as_box(item.get(k)) is not None for k in BOX_KEYS):
            k = next(k for k in BOX_KEYS if as_box(item.get(k)) is not None)
            boxes.append(dict(name=f"{name}.{k}", values=as_box(item[k]), segment=len(segs) - 1 if segs else None))
            continue
        shape = type(item).__name__ + (":" + ",".join(sorted(map(str, item))) if isinstance(item, dict) else "")
        other.append(dict(name=str(name), shape=shape))
    return segs, boxes, other


def tags(text):
    """[(kind, shape, numbers)] of the <TIME ...> and <BBOX ...> tags in a text. shape: the tag with every digit as 9 and every letter after
    its name as a, so it shows the format without the text."""
    out = []
    for m in TAG.finditer(str(text or "")):
        inner = m.group(0)[len(m.group(1)) + 1:]
        shape = "<" + m.group(1) + re.sub(r"[A-Za-z]", "a", re.sub(r"\d", "9", inner))
        out.append((m.group(1), shape, [float(x) for x in NUMBER.findall(inner)]))
    return out


def tag_time_s(text):
    """The first clock time inside a <TIME ...> tag of a text, in seconds, or None."""
    for m in TAG.finditer(str(text or "")):
        c = CLOCK.search(m.group(0)) if m.group(1) == "TIME" else None
        if c:
            try:
                return data.parse_time(c.group(0))
            except ValueError:
                continue
    return None


def question_row(group, qtype, qid, q):
    """One question without its text."""
    try:
        segs, boxes, other = parse_inputs(q.get("inputs", {}))
        error = None
    except ValueError as e:
        segs, boxes, other, error = [], [], [], str(e)[:120]
    choices = list(q.get("choices", []))
    qt, ct = tags(q.get("question", "")), [t for c in choices for t in tags(c)]
    return dict(qid=str(qid), group=group, type=qtype, segments=segs, boxes=boxes, other=other, parse_error=error,
                n_segments=len(segs), videos=sorted({s["video"] for s in segs}), forms=sorted({s["form"] for s in segs}),
                n_choices=len(choices), correct_idx=q.get("correct_idx"),
                q_time_tags=sum(k == "TIME" for k, _, _ in qt), q_bbox_tags=sum(k == "BBOX" for k, _, _ in qt),
                c_time_tags=sum(k == "TIME" for k, _, _ in ct), c_bbox_tags=sum(k == "BBOX" for k, _, _ in ct),
                tag_shapes=sorted({s for _, s, _ in qt + ct}), bbox_tag_numbers=[n for k, _, n in qt + ct if k == "BBOX"],
                q_tag_time_s=tag_time_s(q.get("question", "")))


def load_groups(folder=None):
    """({group: {type: [question_row]}}, other file names). Opens only the 6 groups' files."""
    folder = folder or data.questions_dir()
    files, other = question_files(folder)
    out = {}
    for g, types in files.items():
        out[g] = {}
        for t, name in types.items():
            with open(os.path.join(folder, name)) as f:
                raw = json.load(f)
            items = raw.items() if isinstance(raw, dict) else ((str(i), q) for i, q in enumerate(raw))
            out[g][t] = [question_row(g, t, k, q) for k, q in items]
    return out, other


def raw_question(folder, group, qtype, qid, files=None):
    """One question's own entry (text included), for the log and the sheets only."""
    files = files or question_files(folder)[0]
    with open(os.path.join(folder, files[group][qtype])) as f:
        raw = json.load(f)
    return raw[qid] if isinstance(raw, dict) else raw[int(qid)]


# ---------------------------------------------------------------------------- the fine_grained sample
def apportion(sizes, n):
    """n split over keys in proportion to sizes, by largest remainder (ties by key)."""
    total = sum(sizes.values())
    raw = {k: n * v / total for k, v in sizes.items()}
    out = {k: int(np.floor(x)) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[: n - sum(out.values())]:
        out[k] += 1
    return out


def spread_over_videos(qids_by_video, n, rng):
    """n question ids taken one video at a time in turn, each video's questions and the videos in a seeded random order: as evenly over
    the videos as the type allows."""
    videos = sorted(qids_by_video)
    order = [videos[i] for i in rng.permutation(len(videos))]
    pools = {v: [sorted(qids_by_video[v])[i] for i in rng.permutation(len(qids_by_video[v]))] for v in order}
    out = []
    while len(out) < n and any(pools.values()):
        for v in order:
            if pools[v] and len(out) < n:
                out.append(pools[v].pop(0))
    return out


def stratified_sample(rows, n=SAMPLE_N, seed=SAMPLE_SEED):
    """({type: [qid]}, {type: quota}): n questions over the types in proportion to their sizes, then spread over the videos (a question's
    first video), one seeded generator over the types in name order."""
    by_type = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)
    quota = apportion({t: len(v) for t, v in by_type.items()}, n)
    rng = np.random.default_rng(seed)
    out = {}
    for t in sorted(by_type):
        byv = {}
        for r in by_type[t]:
            byv.setdefault(r["videos"][0] if r["videos"] else "", []).append(r["qid"])
        out[t] = spread_over_videos(byv, quota[t], rng)
    return out, quota


def write_or_check_sample(sample, quota, total, path=None):
    """Writes the sample's ids the first time; afterwards stops if a recomputed sample isn't the same, because it never changes."""
    path = path or sample_json()
    data.refuse_inside_raw(path)
    body = dict(group=SAMPLE_GROUP, n=sum(len(v) for v in sample.values()), seed=SAMPLE_SEED, of=int(total), by_type=quota, qids=sample,
                rule="proportional to each type's size (largest remainder), then one video at a time in turn, seeded order")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old.get("qids") != sample or old.get("seed") != SAMPLE_SEED:
            raise SystemExit(f"SAMPLE DIFFERS from {path}: it never changes after the explore step. Stop and tell the user.")
        return "unchanged"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(body, f)
    os.replace(path + ".tmp", path)
    return "written"


# ---------------------------------------------------------------------------- frames per question (D1 and D2 as proposed)
def split_k(durations, distinct, k=64):
    """K = min(k, all distinct frames) over a question's segments: 1 to each segment with a frame, the rest in proportion to duration
    (largest remainder), never more than a segment's own distinct frames."""
    m = len(durations)
    total = min(k, int(sum(distinct)))
    alloc = [min(1, int(d)) for d in distinct]
    left = total - sum(alloc)
    while left > 0:
        room = [i for i in range(m) if alloc[i] < distinct[i]]
        if not room:
            break
        w = np.array([max(float(durations[i]), 0.0) for i in room])
        w = w if w.sum() > 0 else np.ones(len(room))
        share = left * w / w.sum()
        add = np.floor(share).astype(int)
        add[np.argsort(-(share - add), kind="stable")[: left - int(add.sum())]] += 1
        for j, i in enumerate(room):
            alloc[i] += min(int(add[j]), int(distinct[i]) - alloc[i])
        if total - sum(alloc) == left:
            break
        left = total - sum(alloc)
    return alloc


def segment_bounds(seg, mp4_ns, around_s=0.0):
    """(start, end) in MP4 seconds: a whole video is its first to last frame; a single time is widened by around_s either side."""
    if seg["form"] == "whole":
        return 0.0, float(mp4_ns[-1]) / 1e9
    if seg["form"] == "time" and around_s:
        return max(0.0, seg["start_s"] - around_s), seg["end_s"] + around_s
    return seg["start_s"], seg["end_s"]


# ---------------------------------------------------------------------------- boxes
def rotate_box(box, k, size):
    """A box turned k x 90 degrees clockwise about the centre of a size x size image: what a box drawn on the native image becomes on an
    image turned that way."""
    x0, y0, x1, y1 = box
    pts = [(x0, y0), (x1, y1)]
    for _ in range(int(k) % 4):
        pts = [(size - y, x) for x, y in pts]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def box_scales(box, size=1408):
    """{name: box in MP4 px} for each resolution the numbers could be in, given their range."""
    hi = max(box)
    out = {"MP4 px": list(box)}
    if hi <= 1.0:
        out = {"normalized": [v * size for v in box]}
    else:
        out["2880 sensor"] = [v * 0.5 - 32 for v in box]
        if hi <= 1000:
            out["0-1000 grid"] = [v * size / 1000 for v in box]
    return out

"""Runbook section 18, step 1a: every question's candidate frames under the user's D1, D2 and D5 (2026-11-14). CPU only, base environment.

    python -u -m accuracy_tables.hpc.hdepic.plan18

The run's questions are the explore step's: fine_grained's sample (data/hdepic/sample_fine_grained_2000.json, recomputed and checked, never
changed) and all of the other 5 groups. Each question is its own clip, as in section 17.

  D1  each segment's frames are its distinct MP4 frames (a repeated time-CSV row counts once); a single instant is widened to the 1 s either
      side of it; a whole-video input is the video's first to last frame.
  D2  segments in time order (a video's recording start from its name, then the segment's start), a frame taken by an earlier segment of the
      same video isn't taken again, and K = min(64, all their distinct frames) split over the segments by duration, 1 at least each
      (groups.split_k), each segment's share spread evenly over it (common.uniform_indices). The positions where a segment begins are the
      plan's `starts`: the gate runs per segment and never across a gap (vlmeval.budget_rule's starts).
      object_motion's object_movement_counting and stationary_object_localization, when a question is a whole video plus one query instant
      inside it (an instant input on that video, or a <TIME> tag in its question): the frame at the instant is in every row (`forced`), and
      the rest of K is spread over the video's other frames, so no frame comes twice.
  D5  the questions on the 2 videos without gaze are excluded (the user counted 67: fine_grained 29, ingredient 20, recipe 18); the plan
      stops if its count differs. A question with frames outside its video's gaze stays; the plan counts those frames per group.
A question left with fewer than 2 candidates is excluded with its reason. Writes $GV_ROOT/data/hdepic/s18/plan.json: ids, times and frame
numbers, never question text. Ends with PLAN WRITTEN.
"""
import collections
import json
import os
import sys

import numpy as np

from accuracy_tables import common
from accuracy_tables.hpc.hdepic import data, groups, s18
from accuracy_tables.hpc.hdepic import plan as plan17


def numbers(segs):
    """Each segment's video number for its stamps: the number in its input's name, else its order among the question's inputs."""
    return [s18.input_number(s["name"], i + 1) for i, s in enumerate(segs)]


def time_order(segs):
    """Segment positions in time order: a video's recording start (from its name; the input order when a name has none), then the start."""
    first = {}
    for i, s in enumerate(segs):
        first.setdefault(s["video"], i)
    clocks = {v: s18.video_clock(v) for v in first}
    known = all(c is not None for c in clocks.values())
    key = lambda i: ((clocks[segs[i]["video"]] if known else "", first[segs[i]["video"]]), segs[i]["start_s"] or 0.0)
    return sorted(range(len(segs)), key=key)


def exception_query(q, cache):
    """(whole-video segment, query instant in MP4 s) for D2's object_motion exception, or None (any other shape or type)."""
    if (q["group"], q["type"]) not in s18.EXCEPTION_TYPES:
        return None
    wholes = [s for s in q["segments"] if s["form"] == "whole"]
    times = [s for s in q["segments"] if s["form"] == "time"]
    if len(wholes) != 1:
        return None
    w = wholes[0]
    if len(times) == 1 and times[0]["video"] == w["video"] and len(q["segments"]) == 2:
        t = times[0]["start_s"]
    elif len(q["segments"]) == 1 and q.get("q_tag_time_s") is not None:
        t = q["q_tag_time_s"]
    else:
        return None
    end = float(cache[w["video"]]["mp4_ns"][-1]) / 1e9
    return (w, float(t)) if 0.0 <= t <= end else None


def exception_frames(mp4_ns, t, k=s18.K):
    """(frames in time order, the query frame's position): the instant's frame and k - 1 more spread over the rest of the video."""
    allf = plan17.distinct_frames(mp4_ns, 0.0, float(mp4_ns[-1]) / 1e9)
    near = int(data.frames_between(mp4_ns, t, t)[0])
    qf = int(allf[np.searchsorted(mp4_ns[allf], mp4_ns[near])])          # the first frame with that time
    pool = allf[allf != qf]
    k = min(k, len(allf))
    rest = pool[common.uniform_indices(len(pool), k - 1)] if k > 1 and len(pool) else np.zeros(0, dtype=allf.dtype)
    idx = np.sort(np.concatenate([rest, [qf]]).astype(np.int64))
    return idx, int(np.flatnonzero(idx == qf)[0])


def plan_question(q, qid, cache):
    """-> (plan entry, None) or (None, why it's excluded)."""
    segs = q["segments"]
    if q["parse_error"]:
        return None, f"inputs didn't parse: {q['parse_error']}"
    if not segs:
        return None, "no video input"
    missing = sorted({s["video"] for s in segs if s["video"] not in cache})
    if missing:
        return None, f"no time CSV for {missing[0]}"
    if any(s["video"] in data.NO_GAZE_VIDEOS for s in segs):
        return None, "D5: on a video without gaze"
    nums = numbers(segs)
    exc = exception_query(q, cache)
    parts = []
    if exc is not None:
        w, t = exc
        idx, pos = exception_frames(cache[w["video"]]["mp4_ns"], t)
        parts.append(dict(video=w["video"], n=nums[segs.index(w)], cand=idx, a=0.0, b=float(cache[w["video"]]["mp4_ns"][-1]) / 1e9))
        forced = [pos]
    else:
        forced, taken = [], collections.defaultdict(set)
        for i in time_order(segs):
            s, c = segs[i], cache[segs[i]["video"]]
            a, b = groups.segment_bounds(s, c["mp4_ns"], s18.AROUND_S)
            idx = plan17.distinct_frames(c["mp4_ns"], a, b)
            idx = idx[~np.isin(idx, sorted(taken[s["video"]]))]
            taken[s["video"]].update(int(f) for f in idx)
            if len(idx):
                parts.append(dict(video=s["video"], n=nums[i], idx=idx, a=a, b=b))
        ks = groups.split_k([p["b"] - p["a"] for p in parts], [len(p["idx"]) for p in parts], s18.K) if parts else []
        for p, k in zip(parts, ks):
            p["cand"] = p["idx"][common.uniform_indices(len(p["idx"]), k)] if k else p["idx"][:0]
        parts = [p for p in parts if len(p["cand"])]
    K = int(sum(len(p["cand"]) for p in parts))
    if K < 2:
        return None, f"{K} candidate frame (the gate needs 2)"
    videos = list(dict.fromkeys(p["video"] for p in parts))
    e = dict(qid=qid, hdepic_qid=q["qid"], group=q["group"], type=q["type"], pseudo=s18.pseudo_name(qid), videos=videos, K=K,
             video_of=[], mp4_frames=[], mp4_s=[], device_us=[], video_n=[], starts=[], forced=forced, n_segments=len(parts),
             exception=exc is not None, n_choices=q["n_choices"], frames_outside_gaze=0, windows_into_previous_segment=0)
    prev = None
    for p in parts:
        c = cache[p["video"]]
        if e["mp4_frames"]:
            e["starts"].append(len(e["mp4_frames"]))
        dev = c["vrs_ns"][p["cand"]].astype(np.float64) / 1e3
        e["video_of"] += [videos.index(p["video"])] * len(p["cand"])
        e["mp4_frames"] += [int(f) for f in p["cand"]]
        e["mp4_s"] += [float(t) / 1e9 for t in c["mp4_ns"][p["cand"]]]
        e["device_us"] += [float(d) for d in dev]
        e["video_n"] += [int(p["n"])] * len(p["cand"])
        g = c.get("gaze_us")
        if g is not None and len(g):
            e["frames_outside_gaze"] += int(((dev < g[0]) | (dev > g[-1])).sum())
        if prev is not None and prev[0] == p["video"] and dev[0] - s18.WINDOW_US < prev[1]:
            e["windows_into_previous_segment"] += 1
        prev = (p["video"], float(dev[-1]))
    return e, None


def make_plan(scope, cache, info_of):
    """-> dict(videos, video_info, questions, excluded), qid numbered over the kept questions in scope order."""
    kept, excluded = [], []
    for q in scope:
        e, why = plan_question(q, len(kept), cache)
        if e is None:
            excluded.append(dict(hdepic_qid=q["qid"], group=q["group"], type=q["type"], videos=q["videos"], reason=why))
        else:
            kept.append(e)
    videos = sorted({v for e in kept for v in e["videos"]})
    return dict(rule="D1 to D5 of 2026-11-14 (runbook section 18)", videos=videos, video_info={v: info_of[v] for v in videos},
                questions=kept, excluded=excluded)


def d5_problems(plan):
    """The D5 count per group against the user's (67: fine_grained 29, ingredient 20, recipe 18)."""
    got = collections.Counter(e["group"] for e in plan["excluded"] if e["reason"].startswith("D5"))
    return [] if dict(got) == s18.D5_EXPECTED else [f"D5 excluded {dict(got)}, the user counted {s18.D5_EXPECTED}"]


def report(plan, n_exception_type):
    import pandas as pd
    q = pd.DataFrame(plan["questions"])
    ex = pd.DataFrame(plan["excluded"]) if plan["excluded"] else pd.DataFrame(columns=["group", "type", "reason", "hdepic_qid"])
    print(f"{len(q)} questions kept on {len(plan['videos'])} videos, {len(ex)} excluded", flush=True)
    for (g, t), x in q.groupby(["group", "type"]):
        k = x.K.to_numpy()
        n_ex = int(((ex.group == g) & (ex.type == t)).sum()) if len(ex) else 0
        print(f"   {g}/{t}: {len(x)} kept, {n_ex} excluded; K min {k.min()}, median "
              f"{np.median(k):g}, max {k.max()}; several segments {int((x.n_segments > 1).sum())}, several videos "
              f"{int((x.videos.map(len) > 1).sum())}, the query-frame exception {int(x.exception.sum())}", flush=True)
    for why, n in collections.Counter(e["reason"].split(":")[0] for e in plan["excluded"]).items():
        print(f"   excluded, {why}: {n}", flush=True)
    for e in [e for e in plan["excluded"] if not e["reason"].startswith("D5")][:20]:
        print(f"      {e['group']}/{e['type']} {e['hdepic_qid']}: {e['reason']}", flush=True)
    d5 = collections.Counter(e["group"] for e in plan["excluded"] if e["reason"].startswith("D5"))
    print(f"D5: {sum(d5.values())} questions excluded on the 2 videos without gaze: {dict(d5)}", flush=True)
    touched = int(q.exception.sum())
    print(f"D2's exception: the query frame is in every row on {touched} questions (of {n_exception_type} in object_movement_counting and "
          f"stationary_object_localization); K on them median {np.median(q[q.exception].K) if touched else float('nan'):g}", flush=True)
    for g, x in q.groupby("group"):
        print(f"   {g}: {int(x.frames_outside_gaze.sum())} of {int(x.K.sum())} candidate frames outside their video's gaze "
              f"(on {int((x.frames_outside_gaze > 0).sum())} questions); they get the centred fovea", flush=True)
    print(f"gate windows reaching into the previous segment (gaps under 1 s): {int(q.windows_into_previous_segment.sum())}", flush=True)


def main(argv=None):
    import pandas as pd
    from accuracy_tables.hpc.hdepic import explore18
    out = s18.plan_json()
    data.refuse_inside_raw(out)
    folder = data.questions_dir()
    loaded, _ = groups.load_groups(folder)
    fg = [q for qs in loaded[groups.SAMPLE_GROUP].values() for q in qs]
    if not os.path.exists(groups.sample_json()):
        sys.exit(f"no {groups.sample_json()}: the explore step writes it. Stop and tell the user.")
    sample, quota = groups.stratified_sample(fg)
    groups.write_or_check_sample(sample, quota, len(fg))
    print(f"fine_grained sample unchanged: {sum(len(v) for v in sample.values())} of {len(fg)} (seed {groups.SAMPLE_SEED})", flush=True)
    scope = explore18.in_scope(loaded, sample)
    videos = sorted({s["video"] for q in scope for s in q["segments"]})
    cache, info = {}, {}
    for i, v in enumerate(videos):
        tcsv = os.path.join(data.video_dir(v), f"{v}_mp4_to_vrs_time_ns.csv")
        if not os.path.exists(tcsv):
            continue
        mp4_ns, vrs_ns = data.read_time_csv(tcsv)
        cache[v] = dict(mp4_ns=mp4_ns, vrs_ns=vrs_ns)
        info[v] = plan17.video_info(v)
        if info[v]["gaze_csv"]:
            cache[v]["gaze_us"] = pd.read_csv(info[v]["gaze_csv"], usecols=["tracking_timestamp_us"])["tracking_timestamp_us"].to_numpy(np.float64)
            info[v]["gaze_hz"] = data.rate_hz(cache[v]["gaze_us"])
        if i % 25 == 0:
            print(f"   {i} of {len(videos)} videos read", flush=True)
    plan = make_plan(scope, cache, info)
    n_exc = sum((q["group"], q["type"]) in s18.EXCEPTION_TYPES for q in scope)
    report(plan, n_exc)
    found = d5_problems(plan)
    if found:
        sys.exit("D5 COUNT DIFFERS: " + "; ".join(found) + ". Stop and tell the user.")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w") as f:
        json.dump(plan, f)
    os.replace(out + ".tmp", out)
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

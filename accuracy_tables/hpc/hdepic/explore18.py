"""Runbook section 18, the explore step: what HD-EPIC's 6 non-gaze groups and their videos hold, so the user can decide D1 to D5. CPU only,
base environment, read-only on HD-EPIC. It also draws the fine_grained sample once (seed 0) and writes its ids.

    python -u -m accuracy_tables.hpc.hdepic.explore18 [--out <folder>] [--boxes 20] [--around 1.0]

Prints, and writes per_video.csv and per_question.csv (ids, times and counts only, never question text) to $GV_ROOT/data/hdepic/explore18/:
  files     the vqa-benchmark files per group and type, with their question counts against the user's numbers
  sample    fine_grained's 2,000: the count per type, and data/hdepic/sample_fine_grained_2000.json (ids only), written once and checked
            on every later run
  D1        per group and type: span lengths, native and 1 fps frames, and the questions under 64 distinct frames, under section 17's rule
            K = min(64, distinct MP4 frames in the span). A single instant has 1 frame, which the budget rule and the gate can't use, so
            the line for the proposed alternative (--around seconds either side) is printed too; a whole video is its first to last frame
  D2        segments and videos per question, and K split over the segments in proportion to their duration, 1 at least each
  D3        <TIME> and <BBOX> tags per type, their shapes (digits as 9, letters as a), the box numbers' ranges; then 20 boxed questions
            drawn at 0, 90, 180 and 270 degrees and at each resolution their numbers could be in (sheets in explore18/, never committed)
  D4        every touched video's gaze rate and step to 10 Hz; a rate with no whole step prints D4 STOP
  D5        questions on the 2 videos without gaze, per group and in the sample, and segments outside their video's gaze
  leakage   per group, whether gaze-driven rows see information the others don't
  disk      frames and files under D1 and D2 as proposed, how many section 17 already extracted, and the new gigabytes
Ends with STOP: D1 to D5 are the user's.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import pandas as pd

from accuracy_tables import common
from accuracy_tables.hpc.hdepic import data, explore, groups
from accuracy_tables.hpc.hdepic import plan as plan17

K = 64
LEAKAGE = {
    "fine_grained": "no answer comes from gaze. The fovea sits on the hands and the object in use most of the time, which helps an action "
                    "or object question the way the method intends; that's the method, not a leak",
    "3d_perception": "no: the answers come from the kitchen's digital twin (where a fixture is). The fovea is often away from the asked "
                     "fixture, so gaze-driven rows may see less of the layout, not more",
    "object_motion": "partly: people look at what they pick up and put down (HD-EPIC's own gaze priming), so the fovea often holds the "
                     "moving object. Not the answer itself, but information that follows from gaze, a milder form of gaze_estimation's",
    "recipe": "no: the answers come from the recipe steps over long spans; gaze isn't part of them",
    "ingredient": "mostly no. When the wearer reads a scale or a label they look at it, so the fovea can hold the number a weight question "
                  "asks for: information that follows from gaze, not the answer's definition",
    "nutrition": "no: the answers come from ingredient amounts and nutrition tables; gaze isn't part of them",
}


def show(title, frame):
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 60)
    print(f"\n--- {title}", flush=True)
    print(frame.to_string() if len(frame) else "(none)", flush=True)


def in_scope(loaded, sample):
    """The run's questions: every question of 5 groups, and fine_grained's sample."""
    rows = []
    for g, types in loaded.items():
        for t, qs in types.items():
            keep = set(sample.get(t, [])) if g == groups.SAMPLE_GROUP else None
            rows += [q for q in qs if keep is None or q["qid"] in keep]
    return rows


def question_measures(q, cache, around):
    """(one row of ids, times and counts; the (video, MP4 frame) candidates under D1 and D2 as proposed)."""
    out = dict(qid=q["qid"], group=q["group"], type=q["type"], n_segments=q["n_segments"], n_videos=len(q["videos"]),
               forms="+".join(q["forms"]), n_boxes=len(q["boxes"]), n_other=len(q["other"]), parse_error=q["parse_error"],
               no_gaze_video=any(v in data.NO_GAZE_VIDEOS for v in q["videos"]), q_time_tags=q["q_time_tags"],
               q_bbox_tags=q["q_bbox_tags"], c_time_tags=q["c_time_tags"], c_bbox_tags=q["c_bbox_tags"], n_choices=q["n_choices"])
    durations, wide_s, distinct, wide, idx_wide, f1, outside, missing, gaps = [], [], [], [], [], 0, 0, 0, []
    prev = None
    for s in q["segments"]:
        c = cache.get(s["video"])
        if c is None:
            missing += 1
            continue
        a, b = groups.segment_bounds(s, c["mp4_ns"])
        durations.append(b - a)
        distinct.append(len(plan17.distinct_frames(c["mp4_ns"], a, b)))
        f1 += explore.frames_1fps(a, b)
        a2, b2 = groups.segment_bounds(s, c["mp4_ns"], around)
        wide_s.append(b2 - a2)
        idx = plan17.distinct_frames(c["mp4_ns"], a2, b2)
        idx_wide.append((s["video"], idx))
        wide.append(len(idx))
        if "gaze_us" in c:
            d0, d1 = data.device_ns(c["vrs_ns"], c["mp4_ns"], a2) / 1e3, data.device_ns(c["vrs_ns"], c["mp4_ns"], b2) / 1e3
            outside += int(d0 - 1e6 < c["gaze_us"][0] or d1 > c["gaze_us"][-1])
        if prev is not None and prev[0] == s["video"]:
            gaps.append(a2 - prev[1])
        prev = (s["video"], b2)
    ks = groups.split_k(wide_s, wide) if wide else []
    cand = set()
    for (v, idx), k in zip(idx_wide, ks):
        if k:
            cand.update((v, int(f)) for f in idx[common.uniform_indices(len(idx), k)])
    out.update(span_s=float(sum(durations)), frames_native=int(sum(distinct)), frames_1fps=int(f1), K_rule=int(min(K, sum(distinct))),
               K_proposed=int(sum(ks)), segment_K_min=int(min(ks)) if ks else 0, segments_outside_gaze=outside, missing_videos=missing,
               gaps_under_1s=int(sum(g < 1.0 for g in gaps)), single_instant=int(sum(s["form"] == "time" for s in q["segments"])),
               whole_video=int(sum(s["form"] == "whole" for s in q["segments"])))
    return out, cand


def quantile_table(pq, cols):
    rows = []
    for (g, t), h in pq.groupby(["group", "type"]):
        for c in cols:
            x = h[c].astype(float)
            rows.append(dict(group=g, type=t, measure=c, n=len(x), **{f"q{int(100 * p)}": float(np.quantile(x, p)) for p in (0, .1, .5, .9, 1)}))
    return pd.DataFrame(rows).set_index(["group", "type", "measure"]).round(2)


def box_of(q):
    """(values, video, time in MP4 s, source) of a question's first box, or None."""
    for b in q["boxes"]:
        seg = q["segments"][b["segment"]] if b["segment"] is not None and b["segment"] < len(q["segments"]) else None
        seg = seg or (q["segments"][0] if q["segments"] else None)
        if seg is None:
            continue
        t = seg["start_s"] if seg["form"] != "whole" else q["q_tag_time_s"]
        return b["values"], seg["video"], t, "inputs"
    for nums in q["bbox_tag_numbers"]:
        if len(nums) == 4 and q["segments"]:
            seg = q["segments"][0]
            t = q["q_tag_time_s"] if q["q_tag_time_s"] is not None else seg["start_s"]
            return nums, seg["video"], t, "tag"
    return None


def pick_boxed(rows, n):
    """n boxed questions, the types taking turns, each in qid order."""
    pools = collections.OrderedDict()
    for q in sorted(rows, key=lambda r: (r["group"], r["type"], r["qid"])):
        if box_of(q) is not None:
            pools.setdefault((q["group"], q["type"]), []).append(q)
    out = []
    while len(out) < n and any(pools.values()):
        for k in pools:
            if pools[k] and len(out) < n:
                out.append(pools[k].pop(0))
    return out


COLOURS = {"MP4 px": (0, 0, 255), "2880 sensor": (255, 0, 0), "0-1000 grid": (0, 255, 255), "normalized": (0, 200, 0)}


def box_sheets(boxed, cache, out, folder, files, tile=336):
    """One row per boxed question: the frame at the box's time with the box at 0, 90, 180 and 270 degrees, every resolution in its colour."""
    import cv2
    from accuracy_tables.hpc.hdepic import sheets
    from accuracy_tables.hpc.hdepic.diagnose import text_pad
    rows, made = [], []
    for q in boxed:
        values, v, t, source = box_of(q)
        c = cache.get(v)
        if c is None or t is None:
            continue
        f = int(data.frames_between(c["mp4_ns"], t, t)[0])
        fr = sheets.read_frame(v, f)
        if fr is None:
            continue
        S = fr.shape[1]
        tiles = []
        for k in range(4):
            img = cv2.resize(fr, (tile, tile), interpolation=cv2.INTER_AREA)
            for name, b in groups.box_scales(values, S).items():
                x0, y0, x1, y1 = (u * tile / S for u in groups.rotate_box(b, k, S))
                cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), COLOURS[name], 2)
            cv2.putText(img, f"turned {90 * k}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(img)
        raw = groups.raw_question(folder, q["group"], q["type"], q["qid"], files)
        lines = [f"{q['qid']} ({q['group']}/{q['type']}), {v} frame {f} at {t:.2f} s, box from {source}: {[round(x, 1) for x in values]}",
                 f"Q: {raw.get('question', '')}"[:200], "colours: " + ", ".join(f"{k} {'red blue yellow green'.split()[i]}"
                                                                              for i, k in enumerate(COLOURS))]
        row = np.hstack(tiles)
        rows.append(np.vstack([row, text_pad(lines, row.shape[1])]))
    for i in range(0, len(rows), 10):
        path = os.path.join(out, f"bbox_{i // 10 + 1}.jpg")
        cv2.imwrite(path, np.vstack(rows[i:i + 10]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        made.append(path)
    return made


def kb_per_frame(n=400):
    """Mean size of section 17's extracted JPEGs (the same encoding), from up to n files."""
    root = os.path.join(data.derived_root(), "frames")
    sizes = []
    if os.path.isdir(root):
        for clip in sorted(os.listdir(root))[:: max(1, len(os.listdir(root)) // 50)]:
            d = os.path.join(root, clip)
            names = sorted(x for x in os.listdir(d) if x.endswith(".jpg"))[:: 4] if os.path.isdir(d) else []
            sizes += [os.path.getsize(os.path.join(d, x)) for x in names]
            if len(sizes) >= n:
                break
    return (float(np.mean(sizes)) / 1e3, len(sizes)) if sizes else (float("nan"), 0)


def section17_frames():
    """(video, MP4 frame) of every frame section 17 extracted, from its plan."""
    try:
        p = data.read_plan()
    except FileNotFoundError:
        return set()
    return {(q["video"], int(f)) for q in p["questions"] for f in q["mp4_frames"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="default: $GV_ROOT/data/hdepic/explore18")
    ap.add_argument("--boxes", type=int, default=20)
    ap.add_argument("--around", type=float, default=1.0, help="seconds either side of a single instant, for the proposed alternative")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(data.derived_root(), "explore18")
    data.refuse_inside_raw(out)
    os.makedirs(out, exist_ok=True)
    folder = data.questions_dir()

    files, other = groups.question_files(folder)
    loaded, _ = groups.load_groups(folder)
    rows = [dict(group=g, type=t, file=files[g][t], questions=len(loaded[g][t])) for g in groups.GROUPS for t in sorted(files[g])]
    show("question files per group (the gaze group isn't in this run)", pd.DataFrame(rows).set_index(["group", "type"]))
    for g in groups.GROUPS:
        n = sum(len(v) for v in loaded[g].values())
        print(f"   {g}: {n} questions in {len(files[g])} files (the user's number: about {groups.ABOUT[g]})"
              + ("  NO FILE: stop and tell the user" if not files[g] else ""), flush=True)
    print(f"other files, not opened: {other}", flush=True)
    if any(not files[g] for g in groups.GROUPS):
        sys.exit("a group has no question file. Stop and tell the user.")

    fg = [q for qs in loaded[groups.SAMPLE_GROUP].values() for q in qs]
    sample, quota = groups.stratified_sample(fg)
    state = groups.write_or_check_sample(sample, quota, len(fg))
    print(f"\nfine_grained sample: {sum(len(v) for v in sample.values())} of {len(fg)} (seed {groups.SAMPLE_SEED}), {state}: "
          f"{groups.sample_json()}", flush=True)
    for t in sorted(sample):
        first = {q["qid"]: (q["videos"][0] if q["videos"] else "") for q in loaded[groups.SAMPLE_GROUP][t]}
        vids = collections.Counter(first[i] for i in sample[t])
        print(f"   {t}: {len(sample[t])} of {len(loaded[groups.SAMPLE_GROUP][t])} on {len(vids)} videos, at most {max(vids.values())} a video",
              flush=True)

    scope = in_scope(loaded, sample)
    touched = sorted({v for q in scope for v in q["videos"]})
    all_videos = set(os.listdir(os.path.join(data.root(), "videos")))
    print(f"\nthe run's {len(scope)} questions touch {len(touched)} videos"
          + (f"; NOT ON SCRATCH: {[v for v in touched if v not in all_videos]}" if set(touched) - all_videos else ""), flush=True)
    cache, vrows = {}, []
    for i, v in enumerate(touched):
        if v not in all_videos:
            continue
        try:
            vrows.append(explore.video_stats(v, cache))
        except Exception as e:
            vrows.append(dict(video=v, note=f"{type(e).__name__}: {str(e)[:120]}"))
        if i % 20 == 0:
            print(f"   {i} of {len(touched)} videos read", flush=True)
    pv = pd.DataFrame(vrows)
    pv.to_csv(os.path.join(out, "per_video.csv"), index=False)
    if "note" in pv and pv.note.notna().any():
        show("videos with a note (no MP4, or a file that couldn't be read)", pv[pv.note.notna()][["video", "note"]].set_index("video"))

    measured, frames = [], {}
    for q in scope:
        m, cand = question_measures(q, cache, a.around)
        measured.append(m)
        frames.setdefault(q["group"], set()).update(cand)
        frames.setdefault("_files", collections.Counter())[q["group"]] += m["K_proposed"]
    pq = pd.DataFrame(measured)
    pq.to_csv(os.path.join(out, "per_question.csv"), index=False)
    errors = pq[pq.parse_error.notna()]
    print(f"\ninputs that didn't parse: {len(errors)}" + (f", for example {errors.parse_error.iloc[0]!r}" if len(errors) else ""), flush=True)
    odd = collections.Counter(o["shape"] for q in scope for o in q["other"])
    print(f"inputs that are neither a video, a clip nor a box: {dict(odd) if odd else 'none'}", flush=True)

    show("D1: per group and type, quantiles over questions (span in s; frames are distinct MP4 frames)",
         quantile_table(pq, ["span_s", "frames_native", "frames_1fps", "K_rule", "K_proposed"]))
    d1 = pq.groupby(["group", "type"]).agg(questions=("qid", "size"), single_instant=("single_instant", lambda s: int((s > 0).sum())),
                                           whole_video=("whole_video", lambda s: int((s > 0).sum())),
                                           K_rule_under_64=("K_rule", lambda s: int((s < K).sum())),
                                           K_rule_under_2=("K_rule", lambda s: int((s < 2).sum())),
                                           K_proposed_under_2=("K_proposed", lambda s: int((s < 2).sum())))
    show(f"D1: section 17's rule K = min(64, distinct frames); a single instant widened by {a.around:g} s either side as proposed", d1)
    d2 = pq.groupby(["group", "type"]).agg(segments_max=("n_segments", "max"), multi_segment=("n_segments", lambda s: int((s > 1).sum())),
                                           multi_video=("n_videos", lambda s: int((s > 1).sum())),
                                           a_segment_gets_1=("segment_K_min", lambda s: int((s == 1).sum())),
                                           gaps_under_1s=("gaps_under_1s", "sum"))
    show("D2: segments and videos per question; K split over segments by duration (1 at least each)", d2)
    d3 = pq.groupby(["group", "type"]).agg(q_time=("q_time_tags", lambda s: int((s > 0).sum())), q_bbox=("q_bbox_tags", lambda s: int((s > 0).sum())),
                                           choice_time=("c_time_tags", lambda s: int((s > 0).sum())),
                                           choice_bbox=("c_bbox_tags", lambda s: int((s > 0).sum())), input_box=("n_boxes", lambda s: int((s > 0).sum())))
    show("D3: questions with tags or an input box", d3)
    shapes = collections.Counter((q["group"], s) for q in scope for s in q["tag_shapes"])
    show("D3: the commonest tag shapes per group (digits as 9, letters as a)",
         pd.DataFrame([dict(group=g, shape=s, questions=n) for (g, s), n in shapes.most_common(30)]))
    nums = [("inputs", b["values"]) for q in scope for b in q["boxes"]] + [("tag", n) for q in scope for n in q["bbox_tag_numbers"]]
    if nums:
        bx = pd.DataFrame([dict(source=s, numbers=len(n), lo=min(n) if n else None, hi=max(n) if n else None) for s, n in nums])
        show("D3: box numbers by source (a box should hold 4)", bx.groupby("source").agg(
            boxes=("numbers", "size"), numbers=("numbers", lambda s: str(dict(sorted(collections.Counter(s).items())))), min=("lo", "min"), max=("hi", "max"),
            max_le_1=("hi", lambda s: int((s <= 1).sum())), max_le_1000=("hi", lambda s: int((s <= 1000).sum())),
            max_le_1408=("hi", lambda s: int((s <= 1408).sum())), max_le_2880=("hi", lambda s: int((s <= 2880).sum()))))
    boxed = pick_boxed(scope, a.boxes)
    made = box_sheets(boxed, cache, out, folder, files)
    print(f"D3: {len(boxed)} boxed questions drawn at 0 / 90 / 180 / 270 degrees and each resolution: {made} (HD-EPIC images and text: "
          f"never commit them); boxed questions in the run: {int(((pq.n_boxes > 0) | (pq.q_bbox_tags > 0) | (pq.c_bbox_tags > 0)).sum())}",
          flush=True)

    if "gaze_hz" in pv:
        rates = pv.groupby(pv.gaze_hz.round(0), dropna=False).agg(videos=("video", "size"), step=("gaze_gate_step", "first"))
        show("D4: gaze rate over the touched videos (step: every how many samples reach 10 Hz; empty: no whole step)", rates)
        none = pv[pv.gaze_hz.notna() & pv.gaze_gate_step.isna()].video.tolist()
        print(f"D4 STOP: no whole step to 10 Hz on {none}" if none else "D4: every touched video's gaze has a whole step to 10 Hz", flush=True)
        print(f"videos without a gaze file: {pv[pv.gaze_file.isna()].video.tolist() if 'gaze_file' in pv else 'n/a'}", flush=True)
    sample_ids = {(t, i) for t, ids in sample.items() for i in ids}
    d5 = pq.assign(in_sample=[r.group == groups.SAMPLE_GROUP and (r.type, r.qid) in sample_ids for r in pq.itertuples()])
    show("D5: questions on the 2 videos without gaze, per group (fine_grained: in the sample)",
         d5[d5.no_gaze_video].groupby("group").agg(questions=("qid", "size")))
    print(f"D5: of the sample's 2,000, {int(d5[d5.no_gaze_video & d5.in_sample].shape[0])} fall on those 2 videos; questions with a segment "
          f"outside its video's gaze: {int((pq.segments_outside_gaze > 0).sum())}", flush=True)
    print("\n--- leakage, per group", flush=True)
    for g in groups.GROUPS:
        print(f"   {g}: {LEAKAGE[g]}", flush=True)

    kb, nk = kb_per_frame()
    old = section17_frames()
    rows = []
    for g in groups.GROUPS:
        s = frames.get(g, set())
        reused = len(s & old)
        rows.append(dict(group=g, questions=int((pq.group == g).sum()), files=int(frames["_files"][g]), unique_frames=len(s),
                         already_extracted=reused, new_frames=len(s) - reused, new_gb=round((len(s) - reused) * kb / 1e6, 2),
                         gb_if_every_question_has_its_own=round(frames["_files"][g] * kb / 1e6, 2)))
    every = set().union(*(frames.get(g, set()) for g in groups.GROUPS))
    rows.append(dict(group="all", questions=len(pq), files=int(sum(frames["_files"].values())), unique_frames=len(every),
                     already_extracted=len(every & old), new_frames=len(every - old), new_gb=round(len(every - old) * kb / 1e6, 2),
                     gb_if_every_question_has_its_own=round(sum(frames["_files"].values()) * kb / 1e6, 2)))
    show(f"disk: D1 and D2 as proposed, at {kb:,.0f} KB a frame (mean of {nk} of section 17's frames); section 17 extracted {len(old)} distinct "
         "frames. files = one per frame per question (hard links to one copy per video frame cost no space); compare with myquota above",
         pd.DataFrame(rows).set_index("group"))

    print("\n--- 2 example questions per group (log only: never commit HD-EPIC text)", flush=True)
    for g in groups.GROUPS:
        ex = [q for q in scope if q["group"] == g][:2]
        for q in ex:
            raw = groups.raw_question(folder, g, q["type"], q["qid"], files)
            print(f"   {g}/{q['type']} {q['qid']}: inputs {raw.get('inputs')}, correct {raw.get('correct_idx')}", flush=True)
            print(f"      Q: {raw.get('question')}", flush=True)
            for j, c in enumerate(raw.get("choices") or []):
                print(f"      {j}: {c}", flush=True)
    print(f"\nwritten {out}/per_video.csv and per_question.csv (ids, times and counts, no text)", flush=True)
    print("STOP: D1 to D5 are the user's. Send them this log and the box sheets.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)

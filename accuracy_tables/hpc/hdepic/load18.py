"""HD-EPIC's 6 non-gaze groups as the accuracy tables load data (runbook section 18). common.load_data calls this for a config with
`dataset: hdepic18`; every other config loads as before.

The returned dict is common.load_data's (qs, videos, test_videos, metas, gate, uniform, clips, labels, frame) plus two keys only this
dataset sets, which every row reads through common.budget_frames and flops_calibrate.with_uniform:
  segment_starts  {clip: positions where a segment begins}: the budget rule treats each as a frame 0 of its own, so the gate runs per
                  segment and never across a gap (D2)
  forced          {clip: [position]}: D2's query frame, put in every row's frames in place of the nearest one (common.include_forced)
Each question is its own clip, as in section 17. A frame: dict(key=(clip, position), path, t, gaze, gaze_trace), where t is the frame's
MP4 time as a vlmeval.qwen_mcq.Stamp carrying "<TIME hh:mm:ss.sss video n>" (D3), so every row's prompt reads "Frame at <TIME ...>:".
gaze is section 17's: the fixation over the second before the frame on its own video's projected gaze, or None (the fovea at the centre,
section 9's rule) when that second has no sample or one out of view.
"""
import json
import os

import pandas as pd


def load_data(cfg):
    from accuracy_tables import common
    from accuracy_tables.hpc.hdepic import s18
    from accuracy_tables.hpc.hdepic.load import gaze_at, gaze_track, read_questions
    from vlmeval import budget_rule as br
    from vlmeval import egoev
    from vlmeval.frames import frame_path
    from vlmeval.qwen_mcq import Stamp
    p = cfg["paths"]
    qs = read_questions(p["questions_json"], cfg["run"]["videos"])
    clips_all, _ = egoev.gate_clips(p["folds_dir"], test_only=False)
    keep = set(qs.video)
    clips = [c for c in clips_all if c["sequence"] in keep]
    missing = keep - {c["sequence"] for c in clips}
    if missing:
        raise ValueError(f"{len(missing)} clips have no gate rows, for example {sorted(missing)[:5]}")
    videos = [c["sequence"] for c in clips]
    labels = egoev.video_labels(clips, pd.read_csv(p["oof_cache"]))
    metas = {}
    for v in videos:
        with open(os.path.join(p["frames_local_dir"], v, "meta.json")) as f:
            metas[v] = json.load(f)
    starts = {v: [int(s) for s in metas[v].get("starts", [])] for v in videos}
    forced = {v: [int(f) for f in metas[v]["forced"]] for v in videos if metas[v].get("forced")}
    tracks = {r: gaze_track(s18.gaze_npz(r)) for r in sorted({r for m in metas.values() for r in m["videos"]})}

    def frame(v, f):
        m = metas[v]
        g, trace = gaze_at(tracks[m["videos"][m["video_of"][f]]], m["device_us"][f] / 1e6)
        t = Stamp(m["ts"][f], s18.stamp(m["ts"][f], m["video_n"][f]))
        return dict(key=(v, int(f)), path=frame_path(m, f), t=t, gaze=g, gaze_trace=trace)

    gate = {}
    for c in clips:
        v, fn, n = c["sequence"], c["frame_no"], metas[c["sequence"]]["n"]
        reasons, _ = br.send_reasons(labels[v][:n - 1], starts[v])
        marks = sorted({int(fn[t]) for t, r in enumerate(reasons) if r} | set(forced.get(v, ())))
        gate[v] = [frame(v, f) for f in marks if f < n]
    uniform = {v: [frame(v, f) for f in common.include_forced(common.uniform_indices(metas[v]["n"], cfg["run"]["K"]), forced.get(v, ()))]
               for v in videos}
    return dict(qs=qs, videos=videos, test_videos=set(videos), metas=metas, gate=gate, uniform=uniform, clips=clips, labels=labels,
                frame=frame, segment_starts=starts, forced=forced)


def first_real_video(qs):
    """{clip: the real video its interval is clustered by}: a question's first video (all of them, for a question of several videos)."""
    return dict(zip(qs.video, [r[0] if isinstance(r, (list, tuple)) else r for r in qs.real_videos]))


def stamps(metas):
    """Every planned frame's stamp, for the token count."""
    from accuracy_tables.hpc.hdepic import s18
    return [s18.stamp(t, n) for m in metas.values() for t, n in zip(m["ts"], m["video_n"])]

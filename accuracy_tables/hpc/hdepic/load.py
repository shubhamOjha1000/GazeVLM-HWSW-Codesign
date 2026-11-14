"""HD-EPIC's gaze questions as the accuracy tables load data (runbook section 17). common.load_data calls this for a config with
`dataset: hdepic`; every other config loads as before.

The returned dict is exactly common.load_data's: qs, videos, test_videos, metas, gate, uniform, clips, labels and frame(video, f).
Each question is its own clip, named in plan.json ("video" everywhere below means that clip), so section 14's rules apply per question:
Uniform K is the clip's candidates (K = min(64, its frames)), the budget rule's token budget is its own Uniform tokens, and the gate's
labels come from its own rows. Clips and labels are read exactly as common.load_data reads them (egoev.gate_clips on the folds folder,
egoev.video_labels on paths.oof_cache); the gate's frames are the same D+P selection.

A frame: dict(key=(clip, f), path, t = its MP4 seconds (the prompt's "Frame at" time), gaze, gaze_trace). gaze is
vlmeval.spatial.gaze_fixation on the samples of the second before the frame, in device time, from the projected gaze; None (the fovea
then sits at the centre, counted in n_gaze_missing) if that second has no sample or any sample out of view, as gaze_tracks drops a 1 s
window with one out-of-view sample on the other datasets.
"""
import json
import os

import numpy as np
import pandas as pd

GAZE_WINDOW_S = 1.0


def gaze_track(npz):
    """(device seconds, xy in [0, 1], out of view) of one video's projected gaze, samples with a finite position only."""
    with np.load(npz) as z:
        t, x, y, oof = z["t_us"].astype(np.float64) / 1e6, z["x"], z["y"], z["oof"].astype(bool)
    ok = np.isfinite(x) & np.isfinite(y)
    return t[ok], np.stack([x[ok], y[ok]], 1), oof[ok]


def gaze_at(track, t_s, window=GAZE_WINDOW_S):
    """(gaze point or None, trace) for a frame at device time t_s."""
    from vlmeval import spatial as sp
    ts, xy, oof = track
    m = (ts > t_s - window) & (ts <= t_s)
    trace = [tuple(map(float, p)) for p in xy[m]]
    if not m.any() or oof[m].any():
        return None, trace
    g = sp.gaze_fixation((ts[m], xy[m]), t_s, window=window)
    return (tuple(map(float, g)) if g is not None else None), trace


def read_questions(path, sel):
    with open(path) as f:
        qs = pd.DataFrame(json.load(f))
    if isinstance(sel, (list, tuple)):
        unknown = set(sel) - set(qs.video)
        if unknown:
            raise ValueError(f"run.videos: {sorted(unknown)[:5]} are not HD-EPIC clips")
        qs = qs[qs.video.isin(set(sel))]
    elif sel not in ("all", "test"):
        raise ValueError(f"run.videos must be all, test or a list of clip names, got {sel!r}")
    return qs.reset_index(drop=True)


def load_data(cfg):
    from accuracy_tables import common
    from accuracy_tables.hpc.hdepic import data as hd
    from vlmeval import egoev
    from vlmeval.frames import frame_path
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
    sent = egoev.select_conditions(clips, {"full": labels}, main="full")[f"{cfg['run']['policy']}: full"]
    metas = {}
    for v in videos:
        with open(os.path.join(p["frames_local_dir"], v, "meta.json")) as f:
            metas[v] = json.load(f)
    real = dict(zip(qs.video, qs.real_video))
    tracks = {r: gaze_track(hd.gaze_npz(r)) for r in sorted(set(real.values()))}

    def frame(v, f):
        m = metas[v]
        g, trace = gaze_at(tracks[real[v]], m["device_us"][f] / 1e6)
        return dict(key=(v, int(f)), path=frame_path(m, f), t=float(m["ts"][f]), gaze=g, gaze_trace=trace)

    gate = {v: [frame(v, f) for f in sent[v] if f < metas[v]["n"]] for v in videos}
    uniform = {v: [frame(v, f) for f in common.uniform_indices(metas[v]["n"], cfg["run"]["K"])] for v in videos}
    return dict(qs=qs, videos=videos, test_videos=set(videos), metas=metas, gate=gate, uniform=uniform, clips=clips, labels=labels,
                frame=frame)

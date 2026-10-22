"""Config and data for the accuracy-table runs.

load_config reads the YAML config (paths resolved, defaults filled in); load_data builds, once
for all models, what every row is scored on: the EgoEverything questions on the chosen videos,
the gate's frames (policy D+P by default) with one gaze point each, and the Uniform K frames.
Items are the per-question dicts the vlmeval/*_fovea.py run_questions functions take.
"""

import copy
import json
import os
import re

import numpy as np

DEFAULTS = {
    "paths": {"questions_json": None, "links_json": None, "oof_cache": None,
              "frames_local_dir": "frames_1344", "raw_download_dir": "raw", "out_dir": "results"},
    "run": {"videos": "test", "policy": "D+P", "K": 96, "max_frames": {"smolvlm": 64},
            "parity": True, "phi_remote_parity": False},
    "models": {},
    "ours": {"rings": [{"R": 70, "a": 160}, {"R": 180, "a": 40}, {"R": 180, "a": 160}, {"R": 240, "a": 160}],
             "smolvlm_k": [1, 3, 5, 7]},
    "gazevlm": {"rho": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "sigma": 0.1, "r_min": 0.125, "thumb": 2,
                "gaze": "fixation"},
    "streamingtom": {"ratio": [0.05, 0.1], "tau": 0.9, "knn": 7, "beta": 0.6, "released_dpc": False},
    "dycoke": {"settings": [{"window": 32, "prune": 0.98}, {"window": 32, "prune": 0.93}]},
    "fastvid": {"r": [0.05, 0.1], "c": 8, "tau": 0.9, "ignore": 0.95, "d": 0.4, "p": 4, "beta": 0.6},
    "prunevid": {"settings": [{"tau": 0.8, "beta": 0.07}, {"tau": 0.8, "beta": 0.14}], "gamma": 0.25, "knn": 7,
                 "min_tokens": 14},
    "dtd": {"settings": [{"threshold": 0.96, "absolute": False}, {"threshold": 0.91, "absolute": False}]},
}
GAZE_WINDOW = 1.0       # seconds of gaze before a frame: its fixation estimate and its raw trace
REQUIRED_PATHS = ("folds_dir", "gate_ckpt_dir", "frames_cache_dir")
POLICIES = ("D", "D+P", "D+P all")
_VAR = re.compile(r"\$\{([^}]+)\}")


# ---------------------------------------------------------------------------- config
def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def resolve_paths(paths, base_dir):
    """${key} -> another entry of paths (or an environment variable), ~ expanded, relative paths
    taken from base_dir (the config file's folder). None stays None."""
    def sub(v, depth=0):
        if depth > 20:
            raise ValueError(f"circular ${{...}} reference in paths: {v}")

        def one(m):
            k = m.group(1)
            if k in paths and paths[k] is not None:
                return sub(str(paths[k]), depth + 1)
            if k in os.environ:
                return os.environ[k]
            raise KeyError(f"${{{k}}} is neither a key of `paths` nor an environment variable")
        return _VAR.sub(one, v)

    out = {}
    for k, v in paths.items():
        if v is None:
            out[k] = None
            continue
        v = os.path.expanduser(sub(str(v)))
        out[k] = v if os.path.isabs(v) else os.path.normpath(os.path.join(base_dir, v))
    return out


def load_config(path):
    import yaml
    with open(path) as f:
        cfg = _merge(DEFAULTS, yaml.safe_load(f) or {})
    cfg["paths"] = resolve_paths(cfg["paths"], os.path.dirname(os.path.abspath(path)))
    missing = [k for k in REQUIRED_PATHS if not cfg["paths"].get(k)]
    if missing:
        raise KeyError(f"{path}: set paths.{', paths.'.join(missing)}")
    if cfg["run"]["policy"] not in POLICIES:
        raise ValueError(f"run.policy must be one of {POLICIES}, got {cfg['run']['policy']!r}")
    if cfg["gazevlm"]["gaze"] not in ("fixation", "trace"):
        raise ValueError(f"gazevlm.gaze must be fixation or trace, got {cfg['gazevlm']['gaze']!r}")
    for s in cfg["dycoke"]["settings"]:
        if int(s["window"]) < 2 or not 0 <= float(s["prune"]) < 1:
            raise ValueError(f"dycoke.settings: window >= 2 and 0 <= prune < 1, got {s}")
    for s in cfg["prunevid"]["settings"]:
        if not 0 < float(s["beta"]) <= 1 or not -1 <= float(s["tau"]) <= 1:
            raise ValueError(f"prunevid.settings: 0 < beta <= 1 and -1 <= tau <= 1, got {s}")
    if not 0 < float(cfg["prunevid"]["gamma"]) <= 1:
        raise ValueError(f"prunevid.gamma must be in (0, 1], got {cfg['prunevid']['gamma']}")
    for s in cfg["dtd"]["settings"]:
        lo = -1.0 if s["absolute"] else 0.0
        if not lo <= float(s["threshold"]) <= 1:
            raise ValueError(f"dtd.settings: threshold in [{lo:g}, 1] (absolute: cosine; relative: share dropped), got {s}")
    return cfg


# ---------------------------------------------------------------------------- frames
def uniform_indices(n, k):
    """k frame indices evenly spaced over n frames, first and last included (all n if n <= k)."""
    if n <= 0 or k <= 0:
        return []
    return sorted({int(i) for i in np.round(np.linspace(0, n - 1, min(k, n)))})


def cap_frames(frames, max_frames):
    """At most max_frames of a time-ordered frame list, evenly subsampled (None = no cap)."""
    if not max_frames or len(frames) <= max_frames:
        return list(frames)
    return [frames[i] for i in uniform_indices(len(frames), max_frames)]


def make_items(qs, frames_of):
    """One dict per question (sorted by video, so each video's frames are encoded once):
    qid, video, question, options, answer, frames = frames_of[video]."""
    return [dict(qid=int(q.qid), video=q.video, question=q.question, options=list(q.options),
                 answer=q.answer, frames=frames_of.get(q.video, []))
            for q in qs.sort_values(["video", "qid"]).itertuples()]


# ---------------------------------------------------------------------------- data
def _questions_and_clips(cfg):
    from vlmeval import egoev
    p, sel = cfg["paths"], cfg["run"]["videos"]
    qs_all = egoev.load_questions(p["questions_json"])
    clips_all, _ = egoev.gate_clips(p["folds_dir"], test_only=False)
    gate_videos = {c["sequence"] for c in clips_all}
    test_videos = {c["sequence"] for c in clips_all if c["is_test"]}
    qs = qs_all[qs_all.video.isin(gate_videos)]
    if sel == "test":
        qs = qs[qs.video.isin(test_videos)]
    elif isinstance(sel, (list, tuple)):
        unknown = set(sel) - gate_videos
        if unknown:
            raise ValueError(f"run.videos: {sorted(unknown)} are not gate videos")
        qs = qs[qs.video.isin(set(sel))]
    elif sel != "all":
        raise ValueError(f"run.videos must be test, all or a list of video names, got {sel!r}")
    qs = qs.reset_index(drop=True)
    clips = [c for c in clips_all if c["sequence"] in set(qs.video)]
    return qs, clips, test_videos


def _gate_selection(cfg, clips):
    """{video: frame numbers} sent by the gate under run.policy (held-out fold predictions for
    training videos, the 5-fold ensemble for test videos; cached in paths.oof_cache)."""
    import pandas as pd
    from vlmeval import egoev
    p = cfg["paths"]
    if p["oof_cache"] and os.path.exists(p["oof_cache"]):
        oof = pd.read_csv(p["oof_cache"])
    else:
        from eyehead.oof import oof_predictions
        oof = oof_predictions(p["folds_dir"], p["gate_ckpt_dir"])
        if p["oof_cache"]:
            os.makedirs(os.path.dirname(p["oof_cache"]), exist_ok=True)
            oof.to_csv(p["oof_cache"], index=False)
    sel = egoev.select_conditions(clips, {"full": egoev.video_labels(clips, oof)}, main="full")
    return sel[f"{cfg['run']['policy']}: full"]


def _frame_metas(cfg, videos):
    from vlmeval import qwen_fovea as qf
    from vlmeval.frames import ensure_frames
    p = cfg["paths"]
    links = {"sequences": {}}
    if p["links_json"] and os.path.exists(p["links_json"]):
        links = json.load(open(p["links_json"]))
    metas = {}
    for v in videos:
        try:
            metas[v] = ensure_frames(v, links, p["frames_local_dir"], p["frames_cache_dir"],
                                     raw_root=p["raw_download_dir"], size=qf.IMG)
        except KeyError:
            raise FileNotFoundError(f"{v}: no {v}_px{qf.IMG}.tar in {p['frames_cache_dir']} and no download "
                                    f"link (paths.links_json = {p['links_json']})") from None
    return metas


def load_data(cfg):
    """dict(qs, videos, test_videos, metas, gate = {video: [frame]}, uniform = {video: [frame]})
    with frame = dict(key=(video, frame number), path, t seconds, gaze=(u, v) or None,
    gaze_trace=[(u, v), ...] the raw gaze samples of the GAZE_WINDOW seconds before t)."""
    from vlmeval import spatial as sp
    from vlmeval.frames import frame_path
    qs, clips, test_videos = _questions_and_clips(cfg)
    videos = [c["sequence"] for c in clips]
    sent = _gate_selection(cfg, clips)
    metas = _frame_metas(cfg, videos)

    def frame_time(v, k):
        return metas[v]["ts"][k] if k < metas[v]["n"] else k

    tracks, _ = sp.gaze_tracks(cfg["paths"]["folds_dir"], videos, frame_time)

    def frame(v, f):
        t = float(metas[v]["ts"][f])
        g = sp.gaze_fixation(tracks[v], t, window=GAZE_WINDOW) if v in tracks else None
        trace = []
        if v in tracks:
            ts_, xy_ = tracks[v]
            trace = [tuple(map(float, p)) for p in xy_[(ts_ >= t - GAZE_WINDOW) & (ts_ <= t)]]
        return dict(key=(v, int(f)), path=frame_path(metas[v], f), t=t,
                    gaze=tuple(map(float, g)) if g is not None else None, gaze_trace=trace)

    gate = {v: [frame(v, f) for f in sent[v] if f < metas[v]["n"]] for v in videos}
    uniform = {v: [frame(v, f) for f in uniform_indices(metas[v]["n"], cfg["run"]["K"])] for v in videos}
    return dict(qs=qs, videos=videos, test_videos=test_videos, metas=metas, gate=gate, uniform=uniform)


def describe(data):
    """A short text summary of what load_data found."""
    qs, gate, uni = data["qs"], data["gate"], data["uniform"]
    n_test = sum(v in data["test_videos"] for v in data["videos"])
    miss = sum(fr["gaze"] is None for frs in gate.values() for fr in frs)
    lines = [f"{len(qs)} questions on {len(data['videos'])} videos ({n_test} test videos)",
             f"gate frames per video: mean {np.mean([len(f) for f in gate.values()]):.1f}, "
             f"gaze missing on {miss} of {sum(len(f) for f in gate.values())}",
             f"uniform frames per video: mean {np.mean([len(f) for f in uni.values()]):.1f}"]
    return "\n".join(lines)

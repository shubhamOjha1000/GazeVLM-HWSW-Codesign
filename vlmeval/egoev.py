"""EgoEverything questions on the gate's videos, which frames each policy sends, and scoring.

EgoEverything (arXiv 2604.08342, hf: roxqtang/EgoEverything) is a 5-option multiple-choice
benchmark built from real gaze traces on Aria Everyday Activities (video names `loc...`)
and Nymeria. Each item has a `keyframes[0].timestamp` (seconds from the video start) where
the target object appears; questions are about something seen at least 3 minutes earlier.

Frame conditions, all drawn from the same 1 fps frames the gate's labels were built on:

    D: <model>              policy D (send when the view settles + on each REFIXATION)
                            driven by that model's predicted labels
    D: ground truth         policy D on the DINOv2 labels (the policy's ceiling)
    uniform (same budget)   the same number of frames per video as the main D, evenly spaced
    random (same budget)    the same number, drawn at random (seeded)
    blind (no frames)       question and options only
"""

import json
import os
import re

import numpy as np
import pandas as pd

from eyehead import policy as pol

HF_REPO = "roxqtang/EgoEverything"
HF_FILE = "Egoeverything_VQA.json"
LETTERS = "ABCDE"
_OPT_PREFIX = re.compile(r"^\s*\(?([A-Ea-e])[\.\):]\s*")


# ---------------------------------------------------------------------------- questions
def load_questions(path=None):
    """-> DataFrame, one row per question: qid, video, question, options (5 strings, any
    'A.' prefix removed), answer (letter), t0 (keyframe seconds), object, bbox, dist2gaze."""
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset")
    raw = json.load(open(path))
    if isinstance(raw, dict):                      # tolerate {"data": [...]} style files
        raw = next(v for v in raw.values() if isinstance(v, list))
    rows = []
    for i, it in enumerate(raw):
        o = it["options"]
        o = [o[k] for k in sorted(o)] if isinstance(o, dict) else list(o)
        opts = [_OPT_PREFIX.sub("", str(x)).strip() for x in o]
        ans = str(it["answer"]).strip()
        if ans[:1].upper() in LETTERS and (len(ans) == 1 or not ans[1:2].isalpha()):
            ans = ans[0].upper()
        else:                                      # answer given as option text
            ans = LETTERS[[o.lower() for o in opts].index(_OPT_PREFIX.sub("", ans).lower())]
        kf = (it.get("keyframes") or [{}])[0] or {}
        bb = str(kf.get("bbox") or "").strip()
        bbox = [float(v) for v in bb.split(",")] if bb.count(",") == 3 else None
        rows.append(dict(qid=i, video=it["video_name"], question=it["question"], options=opts,
                         answer=ans, t0=float(kf["timestamp"]) if kf.get("timestamp") is not None
                         else np.nan, object=kf.get("object_name"), bbox=bbox,
                         dist2gaze=it.get("AveDist2Gaze")))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- selection
def gate_clips(folds_dir, test_only=True):
    """The gate's per-video label sequences (one clip per video), with frame numbers."""
    df, test = pol.load_labels(folds_dir)
    clips = [c for c in pol.make_clips(df) if c["is_test"] or not test_only]
    for c in clips:
        c["frame_no"] = np.array([pol.frame_no(p) for p in c["frames"]])
    return clips, test


def _policy_d(lab, pursuit=None):
    """Policy D (send when the view settles + on each REFIXATION). pursuit='first' / 'all' /
    int also sends PURSUIT frames (see eyehead.policy.send_settle)."""
    lab = np.asarray(lab)
    return pol.send_settle(np.isin(lab, pol.MOVING), lab == pol.R, refix=True,
                           pu=lab == pol.P, pursuit=pursuit)


def select_frames(clips, test, preds, main="full", seed=0):
    """-> {condition: {video: sorted list of frame numbers}}.

    `preds` maps a model name to its test_preds.csv. The uniform and random conditions get,
    per video, exactly as many frames as 'D: <main>' sends on that video.
    """
    loaded = {k: pol.load_predictions(v, test) for k, v in preds.items()
              if v and os.path.exists(v)}
    if main not in loaded:
        raise ValueError(f"main model {main!r} not among loaded predictions {list(loaded)}")
    rng = np.random.default_rng(seed)
    out = {}
    for c in clips:
        v, fn = c["sequence"], c["frame_no"]
        n = len(fn)
        out.setdefault("D: ground truth", {})[v] = fn[_policy_d(c["labels"])].tolist()
        for name, (hard, _) in loaded.items():
            out.setdefault(f"D: {name}", {})[v] = fn[_policy_d(hard[c["test_rows"]])].tolist()
        k = len(out[f"D: {main}"][v])
        out.setdefault("uniform (same budget)", {})[v] = \
            fn[np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))].tolist()
        out.setdefault("random (same budget)", {})[v] = \
            sorted(fn[rng.choice(n, k, replace=False)].tolist())
        out.setdefault("blind (no frames)", {})[v] = []
    return {c: {v: [int(f) for f in fr] for v, fr in per.items()} for c, per in out.items()}


def _uniform(fn, k):
    return fn[np.unique(np.round(np.linspace(0, len(fn) - 1, max(1, k))).astype(int))]


def _hybrid(fn, d_frames, k, frac=0.5):
    """k frames: round(k*frac) evenly spaced, then the policy's frames that sit farthest
    from anything already chosen (so the gate fills the gaps the even spacing leaves
    rather than duplicating it), topped up with the farthest remaining frames if the
    policy runs out."""
    chosen = set(_uniform(fn, int(round(k * frac))).tolist()) if k * frac >= 0.5 else set()
    for pool in ([f for f in d_frames if f not in chosen],
                 [f for f in fn.tolist() if f not in chosen]):
        pool = [f for f in pool if f not in chosen]
        while len(chosen) < k and pool:
            if chosen:
                arr = np.array(sorted(chosen))
                j = int(np.argmax([np.abs(arr - f).min() for f in pool]))
            else:
                j = 0
            chosen.add(pool.pop(j))
    return sorted(int(f) for f in chosen)


def video_labels(clips, preds):
    """Per-video predicted label sequences from a per-pair prediction table (columns
    feat_frame_2 and pred, e.g. eyehead.oof.oof_predictions)."""
    m = dict(zip(preds["feat_frame_2"], preds["pred"]))
    out = {}
    for c in clips:
        tail = c["frames"][1:]                       # feat_frame_2 of each pair, in order
        miss = sum(p not in m for p in tail)
        if miss:
            raise KeyError(f"{c['sequence']}: {miss} frame pairs have no prediction")
        out[c["sequence"]] = np.array([m[p] for p in tail], dtype=int)
    return out


def cap_gaps(sent, max_gap):
    """Causal gap cap: also send a frame whenever `max_gap` seconds (frames, at 1 fps) have
    passed since the last one sent. Deployable: it only looks backwards."""
    out = np.asarray(sent, bool).copy()
    last = 0
    for t in range(1, len(out)):
        if out[t]:
            last = t
        elif t - last >= max_gap:
            out[t] = True
            last = t
    return out


def fixed_rate(n_frames, every):
    """Indices of one frame every `every` seconds (may be fractional), starting at frame 0.
    Rounding can push the last time past the final frame, so indices are clipped."""
    idx = np.round(np.arange(0, n_frames, every)).astype(int)
    return np.unique(np.clip(idx, 0, n_frames - 1))


def select_conditions(clips, model_labels, main="full", timeout=5, hybrid_frac=0.5, seed=0,
                      fixed_every=(3, 4, 5, 6, 8), max_gaps=(5, 6, 8, 10)):
    """-> {condition: {video: sorted frame numbers}} for the full-scale run.

    model_labels: {model name: {video: predicted labels, one per frame pair}}.
    'fixed every N s' is the deployable baseline: one interval chosen in advance and used
    on every video. The '(D budget)' / '(E budget)' conditions instead copy, per video,
    how many frames D / E sent there, which a real system could not know beforehand.
    """
    rng = np.random.default_rng(seed)
    out = {}
    put = lambda cond, v, fr: out.setdefault(cond, {}).__setitem__(v, [int(f) for f in fr])
    for c in clips:
        v, fn = c["sequence"], c["frame_no"]
        n = len(fn)
        put("D: ground truth", v, fn[_policy_d(c["labels"])])
        for name, labs in model_labels.items():
            put(f"D: {name}", v, fn[_policy_d(labs[v])])
            put(f"D+P: {name}", v, fn[_policy_d(labs[v], pursuit='first')])
            put(f"D+P all: {name}", v, fn[_policy_d(labs[v], pursuit='all')])
        lab = np.asarray(model_labels[main][v])
        e = pol.send_settle(np.isin(lab, pol.MOVING), lab == pol.R, refix=True, timeout=timeout)
        put(f"E: {main}", v, fn[e])
        kd, ke = len(out[f"D: {main}"][v]), len(out[f"E: {main}"][v])
        put("uniform (D budget)", v, _uniform(fn, kd))
        put(f"hybrid {int(100*hybrid_frac)}/{int(100*(1-hybrid_frac))} (D budget)", v,
            _hybrid(fn, out[f"D: {main}"][v], kd, hybrid_frac))
        put("random (D budget)", v, sorted(fn[rng.choice(n, kd, replace=False)]))
        put("uniform (E budget)", v, _uniform(fn, ke))
        for e in fixed_every:
            put(f"fixed every {e} s", v, fn[pol.send_every(len(c["labels"]), e)])
        d_mask = _policy_d(lab)
        for g in max_gaps:
            put(f"D + max gap {g} s: {main}", v, fn[cap_gaps(d_mask, g)])
            put(f"D + max gap {g} s: ground truth", v, fn[cap_gaps(_policy_d(c["labels"]), g)])
        put("blind (no frames)", v, [])

    # For each gap-capped policy, a fixed interval chosen as ONE number for all videos so
    # that the total frame count matches: the deployable equal-cost baseline.
    total_s = sum(len(c["frame_no"]) for c in clips)
    for g in max_gaps:
        name = f"D + max gap {g} s: {main}"
        every = total_s / max(1, sum(len(fr) for fr in out[name].values()))
        for c in clips:
            put(f"fixed, rate of D + max gap {g} s", c["sequence"],
                c["frame_no"][fixed_rate(len(c["frame_no"]), every)])
    return out


def gap_screen(sel, cov, max_gaps=(5, 6, 8, 10), main="full", n_frames=None):
    """Coverage of each gap-capped policy against the fixed interval at the same total rate.
    `cov` is the output of coverage(). Positive advantage = the policy covers more."""
    hit = cov.groupby("condition")[["hit_1s", "hit_2s", "hit_5s"]].mean() * 100
    rows = []
    for g in max_gaps:
        p, f = f"D + max gap {g} s: {main}", f"fixed, rate of D + max gap {g} s"
        if p not in hit.index or f not in hit.index:
            continue
        fr = sum(len(x) for x in sel[p].values())
        secs = sum(n_frames.values()) if n_frames else np.nan
        rows.append(dict(max_gap_s=g, frames_per_min=60 * fr / secs,
                         policy_within_2s=hit.loc[p, "hit_2s"], fixed_within_2s=hit.loc[f, "hit_2s"],
                         advantage_2s=hit.loc[p, "hit_2s"] - hit.loc[f, "hit_2s"],
                         policy_within_1s=hit.loc[p, "hit_1s"], fixed_within_1s=hit.loc[f, "hit_1s"],
                         advantage_1s=hit.loc[p, "hit_1s"] - hit.loc[f, "hit_1s"]))
    return pd.DataFrame(rows).set_index("max_gap_s")


def budget_table(sel, n_frames):
    """Frames sent per video and overall, per condition."""
    rows = []
    for cond, per in sel.items():
        k = np.array([len(per[v]) for v in per])
        n = np.array([n_frames[v] for v in per])
        rows.append(dict(condition=cond, frames_per_video=k.mean(), min=int(k.min()),
                         max=int(k.max()), sent_pct=100 * k.sum() / n.sum(),
                         sends_per_min=60 * k.sum() / n.sum()))
    return pd.DataFrame(rows).set_index("condition")


# ---------------------------------------------------------------------------- coverage
def coverage(qs, sel, frame_time, tols=(1.0, 2.0, 5.0)):
    """Per question and condition: distance (s) from the keyframe to the nearest sent frame.

    A proxy needing no VLM: if a policy rarely sends a frame near the moment a question is
    about, the VLM cannot see the answer however good it is. `frame_time(video, frame_no)`
    returns seconds.
    """
    rows = []
    for cond, per in sel.items():
        if not any(per.values()):
            continue
        for q in qs.itertuples():
            fr = per.get(q.video, [])
            d = (min(abs(frame_time(q.video, f) - q.t0) for f in fr)
                 if fr and np.isfinite(q.t0) else np.inf)
            rows.append(dict(condition=cond, qid=q.qid, video=q.video, dist=d,
                             **{f"hit_{t:g}s": d <= t for t in tols}))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- results
def load_results(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return pd.DataFrame(rows)


def video_bootstrap(df, stat, B=1000, seed=0):
    """95% interval of stat(df) by resampling whole videos (rows within a video are
    correlated, so resampling questions would make the interval too narrow)."""
    rng = np.random.default_rng(seed)
    vids = df["video"].unique()
    groups = {v: df[df["video"] == v] for v in vids}
    vals = [stat(pd.concat([groups[v] for v in rng.choice(vids, len(vids))]))
            for _ in range(B)]
    return np.percentile(vals, [2.5, 97.5])


def accuracy_table(res, B=1000):
    rows = []
    for cond, g in res.groupby("condition", sort=False):
        lo, hi = video_bootstrap(g, lambda d: 100 * d["correct"].mean(), B)
        rows.append(dict(condition=cond, n=len(g), accuracy=100 * g["correct"].mean(),
                         ci_low=lo, ci_high=hi, frames_per_q=g["n_frames"].mean(),
                         sec_per_q=g["sec"].mean() if "sec" in g else np.nan))
    return pd.DataFrame(rows).set_index("condition")


def paired_test(res, a, b, B=1000):
    """Same questions under conditions a and b: accuracy difference with a video-bootstrap
    interval, and an exact two-sided sign test on the discordant questions (McNemar)."""
    x = res[res["condition"] == a].set_index("qid")
    y = res[res["condition"] == b].set_index("qid")
    common = x.index.intersection(y.index)
    x, y = x.loc[common], y.loc[common]
    n01 = int(((x["correct"] == 0) & (y["correct"] == 1)).sum())
    n10 = int(((x["correct"] == 1) & (y["correct"] == 0)).sum())
    try:
        from scipy.stats import binomtest
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    except ImportError:
        p = float("nan")
    d = pd.DataFrame({"video": x["video"], "diff": x["correct"] - y["correct"]})
    lo, hi = video_bootstrap(d, lambda t: 100 * t["diff"].mean(), B)
    return dict(a=a, b=b, n=len(common), diff=100 * d["diff"].mean(), ci_low=lo, ci_high=hi,
                a_only_right=n10, b_only_right=n01, p_mcnemar=p)

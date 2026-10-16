"""The four labels as a time series, and a "send when the view settles" policy.

    python -m eyehead.policy --folds_dir $DRIVE/folds --thresholds $DRIVE/thresholds.json \
        --feat_cache $DRIVE/feat_cache.npz \
        --pred "full=$DRIVE/runs/eyehead/test_preds.csv" \
        --baseline_preds $DRIVE/release/gate_v1/test_preds.csv \
        --out_dir $DRIVE/runs/policy

Each CSV row is one consecutive frame pair: label L[k] describes the change from frame
f[k] to f[k+1]. Sorting a video's rows by frame index gives its label sequence; rows whose
frames do not chain (a gap) start a new clip.

Two questions, answered on the DINOv2 labels themselves before any model is involved:

1. How long are the runs? If TRANSITION and PURSUIT mostly last one frame, sending only
   when the view settles saves little over sending every non-STABLE frame. If they often
   last 3+ frames, it saves a lot.
2. What ends a moving run: STABLE, REFIXATION, or the end of the video?

Then the policies are replayed on the ground-truth labels and on each model's predicted
labels for the test videos:

    A  send every frame whose label is not STABLE           (the design PDF's gate)
    B  send TRANSITION + REFIXATION                         (discard PURSUIT + STABLE)
    C  settle: send the first frame of each settled stretch after a moving one
    D  C + send each REFIXATION inside a settled stretch    (new gaze target)
    E  D + a timeout: during a long moving stretch, send every `timeout` frames
    F  E with hysteresis on P(moving) (models with probabilities only)

moving = {TRANSITION, PURSUIT} (the scene changed), settled = {REFIXATION, STABLE}.

Scored against the ground-truth labels:

    sends / min          compute: frames sent per minute of video (1 fps)
    settles caught %     of the true moving -> settled boundaries, how many got a sent frame
                         within one frame of the boundary
    send offset (s)      when caught, how late: -1 = the frame before the settle, 0 = at it
    refix caught %       true REFIXATIONs inside a settled stretch that got a send at or
                         one frame after
    mid-motion sends %   sent frames in the middle of a moving stretch (likely blurred,
                         not attended)
    stale frames %       frames whose whole-frame DINOv2 CLS cosine to the LAST SENT frame
                         is below tau_frame: the VLM's latest view no longer matches what
                         the wearer sees (same test the labels use, but against the last
                         sent frame instead of the previous frame)
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from eyehead.features import NAMES

T, P, R, S = 0, 1, 2, 3
MOVING = (T, P)
INV = {v: k for k, v in NAMES.items()}
FPS = 1.0
_FRAME = re.compile(r"feat_(\d+)\.npz$")


# ---------------------------------------------------------------------------- loading
def frame_no(path):
    m = _FRAME.search(str(path).replace("\\", "/"))
    if not m:
        raise ValueError(f"cannot read a frame index from {path!r}")
    return int(m.group(1))


def load_labels(folds_dir):
    """All labelled rows (the fold file holds every non-test row) plus the test rows,
    sorted by video and frame. `test_row` is the row's position in test.csv, -1 otherwise."""
    cols = ["sequence", "feat_frame_1", "feat_frame_2", "quad"]
    pool = pd.read_csv(os.path.join(folds_dir, "train_val_fold1.csv"), usecols=cols)
    test = pd.read_csv(os.path.join(folds_dir, "test.csv"), usecols=cols)
    pool["is_test"], pool["test_row"] = 0, -1
    test["is_test"], test["test_row"] = 1, np.arange(len(test))
    df = pd.concat([pool, test], ignore_index=True)
    df["f1"] = df["feat_frame_1"].map(frame_no)
    df["f2"] = df["feat_frame_2"].map(frame_no)
    return df.sort_values(["sequence", "f2"]).reset_index(drop=True), test


def make_clips(df):
    """Split each video into clips of chained pairs (f1 of a row == f2 of the previous)."""
    clips = []
    for seq, g in df.groupby("sequence", sort=False):
        g = g.reset_index(drop=True)
        brk = np.r_[True, g["f1"].to_numpy()[1:] != g["f2"].to_numpy()[:-1]]
        for _, c in g.groupby(np.cumsum(brk)):
            clips.append(dict(
                sequence=seq, is_test=int(c["is_test"].iloc[0]),
                labels=c["quad"].to_numpy().astype(int),
                frames=[c["feat_frame_1"].iloc[0]] + c["feat_frame_2"].tolist(),
                test_rows=c["test_row"].to_numpy()))
    return clips


def load_cls(feat_cache, clips):
    """CLS vector per frame of every clip, from the training feature cache; None if the
    cache is missing or does not cover a clip."""
    if not feat_cache or not os.path.exists(feat_cache):
        return [None] * len(clips)
    z = np.load(feat_cache, allow_pickle=True)
    F, index = z["F"].astype(np.float32), {p: i for i, p in enumerate(z["paths"])}
    F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-9)
    out = []
    for c in clips:
        rows = [index.get(p) for p in c["frames"]]
        out.append(None if any(r is None for r in rows) else F[rows])
    return out


def load_predictions(path, test):
    """A test_preds.csv -> (hard labels, probabilities or None), in test.csv row order."""
    p = pd.read_csv(path)
    if len(p) != len(test) or (p["sequence"].to_numpy() != test["sequence"].to_numpy()).any():
        raise ValueError(f"{path} is not row-aligned with test.csv")
    if "true_quad" in p.columns and (p["true_quad"].map(INV).to_numpy()
                                     != test["quad"].to_numpy()).any():
        raise ValueError(f"{path} was scored against different labels")
    hard = p["pred_quad"].map(INV).to_numpy().astype(int)
    pcols = [f"p_{NAMES[c]}" for c in range(4)]
    probs = p[pcols].to_numpy() if all(c in p.columns for c in pcols) else None
    return hard, probs


# ---------------------------------------------------------------------------- run lengths
def runs(x):
    """[(value, start, length)] for each stretch of equal consecutive values."""
    x = np.asarray(x)
    if not len(x):
        return []
    start = np.flatnonzero(np.r_[True, x[1:] != x[:-1]])
    return list(zip(x[start], start, np.diff(np.r_[start, len(x)])))


def run_table(clips):
    rows = []
    for ci, c in enumerate(clips):
        lab = c["labels"]
        base = dict(clip=ci, sequence=c["sequence"], is_test=c["is_test"])
        for v, s, n in runs(lab):
            nxt = lab[s + n] if s + n < len(lab) else -1
            rows.append(dict(base, kind="class", run=NAMES[int(v)], length=int(n),
                             next=NAMES.get(int(nxt), "END"), mix=""))
        for v, s, n in runs(np.isin(lab, MOVING)):
            nxt = lab[s + n] if s + n < len(lab) else -1
            part = lab[s:s + n]
            mix = ("" if not v else "TRANSITION only" if (part == T).all()
                   else "PURSUIT only" if (part == P).all() else "mixed")
            rows.append(dict(base, kind="state", run="moving" if v else "settled",
                             length=int(n), next=NAMES.get(int(nxt), "END"), mix=mix))
    return pd.DataFrame(rows)


def length_summary(rt):
    out = []
    for (kind, run), g in rt.groupby(["kind", "run"], sort=False):
        L = g["length"].to_numpy()
        out.append(dict(kind=kind, run=run, runs=len(L), frames=int(L.sum()),
                        mean=L.mean(), median=float(np.median(L)),
                        p90=float(np.percentile(L, 90)), max=int(L.max()),
                        pct_runs_len1=100 * (L == 1).mean(),
                        pct_runs_ge3=100 * (L >= 3).mean(),
                        pct_frames_in_runs_ge3=100 * L[L >= 3].sum() / L.sum()))
    return pd.DataFrame(out)


def transition_matrix(clips):
    m = np.zeros((4, 4), int)
    for c in clips:
        lab = c["labels"]
        np.add.at(m, (lab[:-1], lab[1:]), 1)
    return pd.DataFrame(m, index=list(NAMES.values()), columns=list(NAMES.values()))


# ---------------------------------------------------------------------------- policies
def send_all_but_stable(lab):
    return np.r_[True, lab != S]


def send_transition_refix(lab):
    return np.r_[True, np.isin(lab, (T, R))]


def send_settle(mv, rf, refix=False, timeout=None):
    """Frame 0 is always sent. Frame k+1 is sent when pair k is the first settled pair
    after a moving one; with `refix`, also when pair k is a REFIXATION inside a settled
    stretch; with `timeout`, every `timeout` frames during a long moving stretch."""
    n = len(mv)
    sent = np.zeros(n + 1, bool)
    sent[0] = True
    last, prev_mv, run_start = 0, False, 0
    for k in range(n):
        t = k + 1
        if mv[k]:
            if not prev_mv:
                run_start = t
            if timeout and t - max(last, run_start - 1) >= timeout:
                sent[t] = True
        elif prev_mv or (refix and rf[k]):
            sent[t] = True
        if sent[t]:
            last = t
        prev_mv = bool(mv[k])
    return sent


def moving_hysteresis(p_move, hi, lo):
    """Enter 'moving' only above `hi`, leave it only below `lo`."""
    mv = np.zeros(len(p_move), bool)
    state = False
    for k, p in enumerate(p_move):
        state = p > lo if state else p > hi
        mv[k] = state
    return mv


def policies(lab, probs=None, timeout=5, hyst=(0.6, 0.4)):
    """-> {policy name: sent mask of length n+1} for one clip."""
    mv, rf = np.isin(lab, MOVING), lab == R
    out = {
        "A: all but STABLE": send_all_but_stable(lab),
        "B: TRANSITION + REFIX": send_transition_refix(lab),
        "C: settle": send_settle(mv, rf),
        "D: settle + refix": send_settle(mv, rf, refix=True),
        f"E: D + {timeout}s timeout": send_settle(mv, rf, refix=True, timeout=timeout),
    }
    if probs is not None and hyst:
        hi, lo = hyst
        out[f"F: E + hysteresis {hi:g}/{lo:g}"] = send_settle(
            moving_hysteresis(probs[:, T] + probs[:, P], hi, lo),
            probs[:, R] > probs[:, S], refix=True, timeout=timeout)
    return out


# ---------------------------------------------------------------------------- scoring
def score_clip(sent, gt, F=None, tau_f=None):
    """Counts for one clip, judged against the ground-truth labels `gt`."""
    n = len(gt)
    gmv = np.isin(gt, MOVING)
    settles = [k + 1 for k in range(1, n) if gmv[k - 1] and not gmv[k]]
    caught, offsets = 0, []
    for t in settles:
        w = np.flatnonzero(sent[t - 1:min(n, t + 1) + 1])
        if len(w):
            caught += 1
            offsets.append(int(w[0]) - 1)
    refix = [k + 1 for k in range(1, n) if gt[k] == R and not gmv[k - 1]]
    refix_caught = sum(bool(sent[t:min(n, t + 1) + 1].any()) for t in refix)
    mid = sum(1 for t in range(1, n) if sent[t] and gmv[t - 1] and gmv[t])
    out = dict(frames=n + 1, n=n, sends=int(sent.sum()), settles=len(settles),
               settles_caught=caught, offset_sum=sum(offsets), refix=len(refix),
               refix_caught=refix_caught, mid_motion=mid)
    if F is not None and tau_f is not None:
        last = np.maximum.accumulate(np.where(sent, np.arange(n + 1), 0))
        cos = (F * F[last]).sum(1)[1:]
        out.update(stale=int((cos < tau_f).sum()), gap=float((1 - cos).sum()), cls_n=n)
    return out


def aggregate(rows):
    d = pd.DataFrame(rows).sum(numeric_only=True)
    res = {
        "sends / min": d["sends"] / (d["frames"] / FPS / 60),
        "send %": 100 * d["sends"] / d["frames"],
        "settles caught %": 100 * d["settles_caught"] / max(1, d["settles"]),
        "send offset (s)": d["offset_sum"] / max(1, d["settles_caught"]) / FPS,
        "refix caught %": 100 * d["refix_caught"] / max(1, d["refix"]),
        "mid-motion sends %": 100 * d["mid_motion"] / max(1, d["sends"]),
    }
    if "stale" in d and d.get("cls_n", 0) > 0:
        res["stale frames %"] = 100 * d["stale"] / d["cls_n"]
        res["mean 1 - cos to last sent"] = d["gap"] / d["cls_n"]
    res.update(frames=int(d["frames"]), settles=int(d["settles"]), refix=int(d["refix"]))
    return res


# ---------------------------------------------------------------------------- everything
def run_all(folds_dir, thresholds=None, feat_cache=None, preds=None, baseline_preds=None,
            timeout=5, hyst=(0.6, 0.4), verbose=True):
    """Run-length statistics plus every policy on every source. Returns a dict of tables
    and the sent masks, so a notebook can plot timelines from it."""
    df, test = load_labels(folds_dir)
    clips = make_clips(df)
    th_path = thresholds or os.path.join(os.path.dirname(folds_dir.rstrip("/")),
                                         "thresholds.json")
    tau_f = json.load(open(th_path))["tau_frame"] if os.path.exists(th_path) else None
    cls = load_cls(feat_cache, clips)
    n_cls = sum(c is not None for c in cls)
    if verbose:
        print(f"{df['sequence'].nunique()} videos, {len(df):,} labelled pairs -> "
              f"{len(clips)} clips ({sum(c['is_test'] for c in clips)} test)")
        print(f"CLS features for staleness: {n_cls}/{len(clips)} clips"
              + ("" if n_cls else "  (no --feat_cache: the stale-frames column is skipped)"))

    rt = run_table(clips)
    out = dict(clips=clips, runs=rt, tau_frame=tau_f,
               lengths_all=length_summary(rt),
               lengths_test=length_summary(rt[rt["is_test"] == 1]),
               transitions_all=transition_matrix(clips),
               moving_next_all=pd.crosstab(rt[rt["run"] == "moving"]["mix"],
                                           rt[rt["run"] == "moving"]["next"], margins=True))

    test_ids = [i for i, c in enumerate(clips) if c["is_test"]]
    sources = {"ground truth, all videos": (list(range(len(clips))), None),
               "ground truth, test": (test_ids, None)}
    for name, path in ([("old gate", baseline_preds)] if baseline_preds else []) + \
                      list((preds or {}).items()):
        if path and os.path.exists(path):
            sources[name] = (test_ids, load_predictions(path, test))
        elif verbose:
            print(f"   skipping {name}: {path} not found")

    results, sent = [], {}
    for src, (ids, pred) in sources.items():
        per_policy = {}
        for i in ids:
            c = clips[i]
            if pred is None:
                lab, probs = c["labels"], None
            else:
                lab = pred[0][c["test_rows"]]
                probs = None if pred[1] is None else pred[1][c["test_rows"]]
            for pol, mask in policies(lab, probs, timeout, hyst).items():
                per_policy.setdefault(pol, []).append(
                    score_clip(mask, c["labels"], cls[i], tau_f))
                sent[(src, pol, i)] = mask
        for pol, rows in per_policy.items():
            results.append(dict(source=src, policy=pol, **aggregate(rows)))
    out["results"] = pd.DataFrame(results)
    out["sent"] = sent
    return out


def main():
    ap = argparse.ArgumentParser("Label run lengths and the settle-based send policy")
    ap.add_argument("--folds_dir", required=True)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--feat_cache", default=None,
                    help="feat_cache.npz written by gatetrain (CLS per frame); enables the "
                         "stale-frames metric")
    ap.add_argument("--pred", action="append", default=[],
                    help="NAME=path/to/test_preds.csv from eyehead.infer; repeatable")
    ap.add_argument("--baseline_preds", default=None,
                    help="the old gaze-only gate's test_preds.csv")
    ap.add_argument("--timeout", type=int, default=5)
    ap.add_argument("--hyst", default="0.6,0.4", help="hi,lo for P(moving); '' to disable")
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()

    preds = dict(p.split("=", 1) for p in a.pred)
    hyst = tuple(float(v) for v in a.hyst.split(",")) if a.hyst else None
    o = run_all(a.folds_dir, a.thresholds, a.feat_cache, preds, a.baseline_preds,
                a.timeout, hyst)

    pd.set_option("display.width", 200)
    print("\nRUN LENGTHS, all videos (frames = seconds at 1 fps)")
    print(o["lengths_all"].round(2).to_string(index=False))
    print("\nRUN LENGTHS, test videos")
    print(o["lengths_test"].round(2).to_string(index=False))
    print("\nWHAT ENDS A MOVING RUN (all videos)")
    print(o["moving_next_all"].to_string())
    print("\nLABEL -> NEXT LABEL (all videos; rows = current)")
    print(o["transitions_all"].to_string())
    print("\nPOLICIES (scored against the ground-truth labels)")
    cols = ["sends / min", "send %", "settles caught %", "send offset (s)",
            "refix caught %", "mid-motion sends %", "stale frames %"]
    r = o["results"]
    print(r[["source", "policy"] + [c for c in cols if c in r.columns]]
          .round(2).to_string(index=False))

    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        o["runs"].to_csv(os.path.join(a.out_dir, "runs.csv"), index=False)
        o["lengths_all"].to_csv(os.path.join(a.out_dir, "run_lengths_all.csv"), index=False)
        o["lengths_test"].to_csv(os.path.join(a.out_dir, "run_lengths_test.csv"), index=False)
        o["moving_next_all"].to_csv(os.path.join(a.out_dir, "moving_run_next.csv"))
        o["transitions_all"].to_csv(os.path.join(a.out_dir, "transitions.csv"))
        r.to_csv(os.path.join(a.out_dir, "policy_results.csv"), index=False)
        print(f"\nwrote {a.out_dir}/")


if __name__ == "__main__":
    main()

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
    G  E, but a settle only counts once it has lasted `min_settle` frames (ignores
       one-second settled blips; costs that much extra delay)
    U  uniform sampling, one frame every N seconds, no sensors at all -- the baseline
       every gaze/IMU policy has to beat at the same send rate

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
    stale settled %      the same, counted only on frames where the wearer has settled
                         (true label REFIXATION or STABLE). The settle policies skip moving
                         frames on purpose, so THIS is the staleness they should be judged on
    stale moving %       the same, on moving frames only (what skipping motion costs)
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


def send_every(n, every):
    """Uniform sampling: frames 0, every, 2*every, ... of a clip with n pairs."""
    sent = np.zeros(n + 1, bool)
    sent[::every] = True
    return sent


def send_settle(mv, rf, refix=False, timeout=None, min_settle=1):
    """Frame 0 is always sent. After a moving stretch, the frame on which the view has
    been settled for `min_settle` consecutive pairs is sent (1 = the first settled frame).
    With `refix`, a REFIXATION inside an already-answered settled stretch is sent too.
    With `timeout`, a frame is sent every `timeout` frames while a moving stretch is still
    unanswered. Settled blips shorter than `min_settle` do not end the stretch."""
    n = len(mv)
    sent = np.zeros(n + 1, bool)
    sent[0] = True
    last, pending, settled_run, moving_since = 0, False, 0, 0
    for k in range(n):
        t = k + 1
        if mv[k]:
            settled_run = 0
            if not pending:
                pending, moving_since = True, t
            if timeout and t - max(last, moving_since - 1) >= timeout:
                sent[t] = True
        else:
            settled_run += 1
            if pending:
                if settled_run >= min_settle:
                    sent[t], pending = True, False
            elif refix and rf[k]:
                sent[t] = True
        if sent[t]:
            last = t
    return sent


def moving_hysteresis(p_move, hi, lo):
    """Enter 'moving' only above `hi`, leave it only below `lo`."""
    mv = np.zeros(len(p_move), bool)
    state = False
    for k, p in enumerate(p_move):
        state = p > lo if state else p > hi
        mv[k] = state
    return mv


UNIFORM_EVERY = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15)


def uniform_policies(n, every=UNIFORM_EVERY):
    return {f"U: every {e} s": send_every(n, e) for e in every}


def policies(lab, probs=None, timeout=5, hyst=(0.6, 0.4), min_settle=2):
    """-> {policy name: sent mask of length n+1} for one clip."""
    mv, rf = np.isin(lab, MOVING), lab == R
    out = {
        "A: all but STABLE": send_all_but_stable(lab),
        "B: TRANSITION + REFIX": send_transition_refix(lab),
        "C: settle": send_settle(mv, rf),
        "D: settle + refix": send_settle(mv, rf, refix=True),
        f"E: D + {timeout}s timeout": send_settle(mv, rf, refix=True, timeout=timeout),
        f"G: E, settle >= {min_settle}s": send_settle(mv, rf, refix=True, timeout=timeout,
                                                      min_settle=min_settle),
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
        cos = (F * F[last]).sum(1)[1:]            # frames 1..n; frame t ends pair t-1
        stale = cos < tau_f
        out.update(stale=int(stale.sum()), gap=float((1 - cos).sum()), cls_n=n,
                   stale_settled=int(stale[~gmv].sum()), n_settled=int((~gmv).sum()),
                   stale_moving=int(stale[gmv].sum()), n_moving=int(gmv.sum()))
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
        res["stale settled %"] = 100 * d["stale_settled"] / max(1, d["n_settled"])
        res["stale moving %"] = 100 * d["stale_moving"] / max(1, d["n_moving"])
        res["stale frames %"] = 100 * d["stale"] / d["cls_n"]
        res["mean 1 - cos to last sent"] = d["gap"] / d["cls_n"]
    res.update(frames=int(d["frames"]), settles=int(d["settles"]), refix=int(d["refix"]))
    return res


# ---------------------------------------------------------------------------- everything
def run_all(folds_dir, thresholds=None, feat_cache=None, preds=None, baseline_preds=None,
            timeout=5, hyst=(0.6, 0.4), min_settle=2, verbose=True):
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
    all_ids = list(range(len(clips)))
    sources = {"ground truth, all videos": (all_ids, None),
               "ground truth, test": (test_ids, None),
               "uniform, all videos": (all_ids, "uniform"),
               "uniform, test": (test_ids, "uniform")}
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
            if pred is None or isinstance(pred, str):
                lab, probs = c["labels"], None
            else:
                lab = pred[0][c["test_rows"]]
                probs = None if pred[1] is None else pred[1][c["test_rows"]]
            masks = (uniform_policies(len(lab)) if pred == "uniform"
                     else policies(lab, probs, timeout, hyst, min_settle))
            for pol, mask in masks.items():
                per_policy.setdefault(pol, []).append(
                    score_clip(mask, c["labels"], cls[i], tau_f))
                sent[(src, pol, i)] = mask
        for pol, rows in per_policy.items():
            results.append(dict(source=src, policy=pol, **aggregate(rows)))
    out["results"] = pd.DataFrame(results)
    out["sent"] = sent
    out["vs_uniform"] = vs_uniform(out["results"])
    return out


VS_UNIFORM = (("stale settled %", True), ("settles caught %", False),
              ("refix caught %", False), ("stale frames %", True))


def vs_uniform(results, metrics=VS_UNIFORM):
    """Every sensor-driven policy against uniform sampling at the SAME send rate.

    Uniform sampling's metric is linearly interpolated at the policy's sends/min, on the
    same videos (all-videos ground truth against all-videos uniform, everything else
    against the test videos). `advantage` is signed so that positive = the policy is
    better than uniform sampling at equal compute.
    """
    rows = []
    for _, r in results[~results["source"].str.startswith("uniform")].iterrows():
        usrc = "uniform, all videos" if r["source"].endswith("all videos") else "uniform, test"
        u = results[results["source"] == usrc].sort_values("sends / min")
        row = dict(source=r["source"], policy=r["policy"], **{"sends / min": r["sends / min"]})
        for m, lower_better in metrics:
            if m not in results.columns:
                continue
            base = float(np.interp(r["sends / min"], u["sends / min"], u[m]))
            row[f"{m} policy"] = r[m]
            row[f"{m} uniform"] = base
            row[f"{m} advantage"] = (base - r[m]) if lower_better else (r[m] - base)
        rows.append(row)
    return pd.DataFrame(rows)


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
    ap.add_argument("--min_settle", type=int, default=2,
                    help="policy G: a settle counts only after this many settled frames")
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()

    preds = dict(p.split("=", 1) for p in a.pred)
    hyst = tuple(float(v) for v in a.hyst.split(",")) if a.hyst else None
    o = run_all(a.folds_dir, a.thresholds, a.feat_cache, preds, a.baseline_preds,
                a.timeout, hyst, a.min_settle)

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
            "refix caught %", "mid-motion sends %", "stale settled %", "stale moving %",
            "stale frames %"]
    r = o["results"]
    print(r[["source", "policy"] + [c for c in cols if c in r.columns]]
          .round(2).to_string(index=False))
    v = o["vs_uniform"]
    print("\nAGAINST UNIFORM SAMPLING AT THE SAME SEND RATE (advantage > 0 = policy better)")
    print(v[["source", "policy", "sends / min"] + [c for c in v.columns if c.endswith("advantage")]]
          .round(2).to_string(index=False))

    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        o["runs"].to_csv(os.path.join(a.out_dir, "runs.csv"), index=False)
        o["lengths_all"].to_csv(os.path.join(a.out_dir, "run_lengths_all.csv"), index=False)
        o["lengths_test"].to_csv(os.path.join(a.out_dir, "run_lengths_test.csv"), index=False)
        o["moving_next_all"].to_csv(os.path.join(a.out_dir, "moving_run_next.csv"))
        o["transitions_all"].to_csv(os.path.join(a.out_dir, "transitions.csv"))
        r.to_csv(os.path.join(a.out_dir, "policy_results.csv"), index=False)
        v.to_csv(os.path.join(a.out_dir, "vs_uniform.csv"), index=False)
        print(f"\nwrote {a.out_dir}/")


if __name__ == "__main__":
    main()

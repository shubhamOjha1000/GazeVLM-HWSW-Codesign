"""Spatial screen (no GPU): does gaze put the fine patches on the question's target?

The spatial module (repo apt-vlm, `src/models/foveation.py`) gives each frame a quadtree of
patch sizes from the gaze point: 14 px where the eccentricity from gaze is below the first
ring, 28 px out to the second ring, 56 px beyond (FOVI rings, evenly spaced in log(r + a)).
This file re-implements that layout in numpy, at the same defaults (336 px image, p = 14,
3 sizes, 110 deg pinhole), so it can be scored against EgoEverything's target boxes before
any VLM run.

Gaze comes from the fold CSVs' `gaze_xy_window`: the MPS gaze already projected into the
upright preview frame with the factory calibration (CPF -> camera -> fisheye project ->
rotate), ~10 samples per 1 s window. EgoEverything boxes are x1, y1, x2, y2 in the same
1408 px upright frame. Both are scaled to the 336 px grid here.

Measures, per question:
    inside        gaze point inside the target box
    d_deg         angle from gaze to the nearest point of the box (0 when inside)
    in_fovea      d_deg below the first ring: some of the target gets 14 px patches
    tokens        tokens in the frame's layout (depends on a and on where the gaze is)
    tok_box       tokens spent on the target box (area-weighted)
    uni_box       tokens a uniform layout with the SAME total would spend on the box
    ratio         tok_box / uni_box: > 1 means the target gets more detail than uniform
                  resizing at equal cost

Limits: the box is drawn on the keyframe only; the eccentricity uses a pinhole camera,
whereas Aria RGB is fisheye (Step 4 of the spatial module, not built yet).
"""

import math
import os

import numpy as np
import pandas as pd

from eyehead import policy as pol

P, S, IMG, FOV = 14, 3, 336, 110.0


# ---------------------------------------------------------------------------- geometry
def ring_radii_deg(R_deg, a, num_scales=S):
    """FOVI: num_scales - 1 ring radii, evenly spaced in log(r + a) on [log a, log(R + a)]."""
    lo, hi = math.log(a), math.log(R_deg + a)
    return [math.exp(lo + (hi - lo) * k / num_scales) - a for k in range(1, num_scales)]


def _rays(x, y, img, fov):
    f = (img / 2) / math.tan(math.radians(fov) / 2)
    x, y = np.asarray(x, float), np.asarray(y, float)
    v = np.stack([x - img / 2, y - img / 2, np.full_like(x, f)], axis=-1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def ecc_deg(x, y, gx, gy, img=IMG, fov=FOV):
    """Angle (deg) between the camera rays through pixels (x, y) (any shape) and through
    the gaze pixel (gx, gy). atan2(|a x b|, a . b) stays exact near 0."""
    a = _rays(x, y, img, fov)
    g = _rays(gx, gy, img, fov)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, g), axis=-1), (a * g).sum(-1)))


def density_map(gu, gv, a, num_scales=S, img=IMG, p=P, fov=FOV):
    """Tokens per 14 px cell (1, 1/4, 1/16 ...) of the foveated layout for gaze (gu, gv) in
    [0, 1]; its sum is the token count. Same rule as APT's select_patches_by_threshold on
    foveation.py's importance maps: a cell of size 2^i p is kept when the eccentricity of
    its centre is >= ring i-1 and no larger kept cell covers it."""
    rings = ring_radii_deg(fov / 2, a, num_scales)
    gx, gy = gu * img, gv * img
    n = img // p
    covered = np.zeros((n, n), bool)
    dens = np.ones((n, n))
    for i in range(num_scales - 1, 0, -1):
        k = 2 ** i
        g = n // k
        c = (np.arange(g) + 0.5) * p * k
        X, Y = np.meshgrid(c, c)                       # rows = y, columns = x
        keep = ecc_deg(X, Y, gx, gy, img, fov) >= rings[i - 1]
        keep &= ~covered.reshape(g, k, g, k).any(axis=(1, 3))
        up = np.repeat(np.repeat(keep, k, 0), k, 1)
        dens[up] = 1.0 / (k * k)
        covered |= up
    return dens


def _cell_overlap(lo, hi, n, p):
    e = np.arange(n) * p
    return np.clip(np.minimum(hi, e + p) - np.maximum(lo, e), 0, p) / p


def box_weights(box, img=IMG, p=P):
    """Fraction of each 14 px cell covered by box (x1, y1, x2, y2) in img pixels."""
    n = img // p
    x1, y1, x2, y2 = box
    return np.outer(_cell_overlap(y1, y2, n, p), _cell_overlap(x1, x2, n, p))


def box_metrics(gu, gv, box, a_values, img=IMG, p=P, fov=FOV, num_scales=S):
    """Gaze-vs-box measures for one gaze point and one box (img pixels), per a."""
    gx, gy = gu * img, gv * img
    x1, y1, x2, y2 = box
    d = float(ecc_deg(np.clip(gx, x1, x2), np.clip(gy, y1, y2), gx, gy, img, fov))
    W = box_weights(box, img, p)
    cells = W.sum()
    out = []
    for a in a_values:
        dens = density_map(gu, gv, a, num_scales, img, p, fov)
        tokens = dens.sum()
        tok_box = (dens * W).sum()
        uni_box = tokens * cells / dens.size
        out.append(dict(a=a, inside=d == 0.0, d_deg=d,
                        in_fovea=d < ring_radii_deg(fov / 2, a, num_scales)[0],
                        tokens=tokens, tok_box=tok_box, uni_box=uni_box,
                        ratio=tok_box / uni_box if uni_box > 0 else np.nan))
    return out


def ring_table(a_values, num_scales=S, fov=FOV, img=IMG):
    """Ring radii and token counts for a gaze at the image centre, per a."""
    rows = []
    for a in a_values:
        r = ring_radii_deg(fov / 2, a, num_scales)
        rows.append(dict(a=a, **{f"ring{i + 1}_deg": x for i, x in enumerate(r)},
                         tokens_centre_gaze=density_map(0.5, 0.5, a, num_scales).sum(),
                         tokens_uniform=(img // P) ** 2))
    return pd.DataFrame(rows).set_index("a")


# ---------------------------------------------------------------------------- gaze tracks
def gaze_tracks(folds_dir, videos, frame_time):
    """{video: (t seconds, xy in [0, 1] upright)} from the fold CSVs' gaze_xy_window.
    The n samples of a window [t(f1), t(f2)) are placed evenly inside it (10 Hz, so the
    timing error is under 50 ms). Windows with any out-of-FOV filler are dropped."""
    cols = ["sequence", "feat_frame_1", "feat_frame_2", "gaze_xy_window", "n_oof_in_gap"]
    files = [os.path.join(folds_dir, f) for f in ("train_val_fold1.csv", "test.csv")]
    have = set(pd.read_csv(files[1], nrows=0).columns)
    miss = [c for c in cols if c not in have]
    if miss:
        raise KeyError(f"{files[1]} has no column(s) {miss}: rebuild the folds from the "
                       f"full 16-column feature CSV")
    df = pd.concat([pd.read_csv(f, usecols=cols) for f in files], ignore_index=True)
    df = df[df["sequence"].isin(set(videos))]
    tracks, dropped = {}, 0
    for v, g in df.groupby("sequence"):
        ts, xy = [], []
        for r in g.itertuples():
            if r.n_oof_in_gap > 0 or not isinstance(r.gaze_xy_window, str):
                dropped += 1
                continue
            pts = np.array([[float(u) for u in s.split(",")]
                            for s in r.gaze_xy_window.split(";") if s])
            t1 = frame_time(v, pol.frame_no(r.feat_frame_1))
            t2 = frame_time(v, pol.frame_no(r.feat_frame_2))
            ts.append(t1 + (np.arange(len(pts)) + 0.5) * (t2 - t1) / len(pts))
            xy.append(pts)
        if not ts:
            continue
        t = np.concatenate(ts)
        o = np.argsort(t, kind="stable")
        tracks[v] = (t[o], np.concatenate(xy)[o])
    return tracks, dropped


def gaze_at(track, t, max_dt=0.15):
    """Gaze sample nearest time t, or None if none within max_dt seconds."""
    ts, xy = track
    j = int(np.searchsorted(ts, t))
    cand = [i for i in (j - 1, j) if 0 <= i < len(ts)]
    if not cand:
        return None
    i = min(cand, key=lambda i: abs(ts[i] - t))
    return tuple(xy[i]) if abs(ts[i] - t) <= max_dt else None


def gaze_fixation(track, t, window=1.0, lead=0.0, radius_deg=3.0, tau=0.5):
    """One gaze point for the frame at time t, from the ~10 samples of the second before it.

    A plain (weighted) mean can land between two fixations when the eyes jumped during the
    window, on a spot nobody looked at. Instead: weight each sample by recency,
    exp(-(t - t_i) / tau), find the sample whose radius_deg neighbourhood holds the most
    weight (the densest fixation cluster), and return the weighted centroid of that
    neighbourhood. None if the window has no samples. lead > 0 would also admit samples after
    the frame; the default 0 keeps it causal, like the gate."""
    ts, xy = track
    m = (ts > t - window) & (ts <= t + lead)
    if not m.any():
        return None
    pts, w = xy[m], np.exp(-np.clip(t - ts[m], 0, None) / tau)
    if len(pts) == 1:
        return float(pts[0, 0]), float(pts[0, 1])
    p = pts * IMG
    i, j = np.meshgrid(np.arange(len(p)), np.arange(len(p)), indexing="ij")
    near = ecc_deg(p[i, 0], p[i, 1], p[j, 0], p[j, 1]) <= radius_deg
    c = near[int(np.argmax(near @ w))]
    g = (w[c, None] * pts[c]).sum(0) / w[c].sum()
    return float(g[0]), float(g[1])


def spread_at(track, t, half=0.2):
    """Largest angle (deg) between any two gaze samples within +-half s of t: small = the
    eyes are fixating at that moment, large = mid-saccade. NaN if under 2 samples."""
    ts, xy = track
    m = np.abs(ts - t) <= half
    if m.sum() < 2:
        return np.nan
    p = xy[m] * IMG
    i, j = np.triu_indices(len(p), 1)
    return float(ecc_deg(p[i, 0], p[i, 1], p[j, 0], p[j, 1]).max())


# ---------------------------------------------------------------------------- screens
def _box336(q, meta):
    s = IMG / meta["orig_w"]
    return [b * s for b in q.bbox]


def keyframe_screen(qs, tracks, metas, a_values=(10, 20, 40, 80, 160), n_other=5,
                    min_shift=30.0, seed=0, gaze_fn=None):
    """Spatial module alone: on the 1 fps frame nearest each question's keyframe, where does
    the fine region land for each gaze source?

        gaze               the wearer's gaze at that frame
        centre             fovea fixed at the image centre (head direction only)
        gaze, other time   the same wearer's gaze at a random other moment of the video
                           (>= min_shift s away; mean of n_other draws): keeps the wearer's
                           typical gaze distribution, breaks the link to this view
    """
    gaze_fn = gaze_fn or gaze_fixation
    rng = np.random.default_rng(seed)
    rows = []
    for q in qs.itertuples():
        if q.bbox is None or not np.isfinite(q.t0) or q.video not in tracks:
            continue
        m, tr = metas[q.video], tracks[q.video]
        ts_f = np.asarray(m["ts"])
        k = int(np.argmin(np.abs(ts_f - q.t0)))
        t, L = ts_f[k], ts_f[-1]
        box = _box336(q, m)
        g = gaze_fn(tr, t)
        shifts = rng.uniform(min_shift, max(min_shift, L - min_shift), n_other)
        sources = {"gaze": [g] if g is not None else [],
                   "centre": [(0.5, 0.5)],
                   "gaze, other time": [x for x in (gaze_fn(tr, (t + s) % L) for s in shifts)
                                        if x is not None]}
        # gaze to box-centre distance in source pixels, to check against the dataset's own
        # AveDist2Gaze (it should rise with it if the gaze projection is right)
        cx, cy = (q.bbox[0] + q.bbox[2]) / 2, (q.bbox[1] + q.bbox[3]) / 2
        px = np.hypot(g[0] * m["orig_w"] - cx, g[1] * m["orig_h"] - cy) if g is not None else np.nan
        for src, pts in sources.items():
            if not pts:
                continue
            per = [box_metrics(u, v, box, a_values) for u, v in pts]
            for ia in range(len(a_values)):
                d = pd.DataFrame([p[ia] for p in per]).mean(numeric_only=True).to_dict()
                rows.append(dict(qid=q.qid, video=q.video, source=src, frame=k, **d,
                                 dist2gaze=q.dist2gaze,
                                 gaze_box_centre_px=px if src == "gaze" else np.nan))
    return pd.DataFrame(rows)


def combined_screen(qs, sel, conditions, tracks, metas, frame_time, a=40, tol=1.0, gaze_fn=None):
    """Temporal + spatial: for each frame-selection condition, take the sent frame nearest
    each question's keyframe; a hit needs that frame within `tol` s of the keyframe AND the
    target inside the fine region of that frame's foveated layout. The keyframe's box is
    used for the sent frame, so only small `tol` makes sense."""
    rows = []
    gaze_fn = gaze_fn or gaze_fixation
    for c in conditions:
        per = sel[c]
        for q in qs.itertuples():
            if q.bbox is None or not np.isfinite(q.t0) or q.video not in tracks:
                continue
            fr = per.get(q.video, [])
            if fr:
                times = np.array([frame_time(q.video, f) for f in fr])
                j = int(np.argmin(np.abs(times - q.t0)))
                dt, t = abs(times[j] - q.t0), times[j]
            else:
                dt, t = np.inf, np.nan
            hit_t = dt <= tol
            box = _box336(q, metas[q.video])
            for src in ("gaze", "centre"):
                g = (gaze_fn(tracks[q.video], t) if src == "gaze" else (0.5, 0.5)) if hit_t else None
                r = box_metrics(g[0], g[1], box, [a])[0] if g is not None else {}
                rows.append(dict(condition=c, source=src, qid=q.qid, video=q.video, dt=dt,
                                 hit_time=hit_t, hit_fovea=bool(r.get("in_fovea", False)),
                                 ratio=r.get("ratio", np.nan)))
    return pd.DataFrame(rows)


def sent_frame_stability(sel, conditions, tracks, frame_time, half=0.2):
    """Gaze spread (deg) around every frame each condition sends: is the gaze point a
    fixation (reliable fovea) or mid-saccade at the moment the frame is taken?"""
    rows = []
    for c in conditions:
        for v, fr in sel[c].items():
            if v not in tracks:
                continue
            for f in fr:
                rows.append(dict(condition=c, video=v, frame=f,
                                 spread=spread_at(tracks[v], frame_time(v, f), half)))
    return pd.DataFrame(rows)

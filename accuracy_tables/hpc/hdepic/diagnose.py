"""Runbook section 18, step 0b (base environment, CPU only): why section 17's foveated rows scored below Blind on "where looking". Read-only
on HD-EPIC and on results/hdepic_gaze_smolvlm_flops/; writes only $GV_ROOT/data/hdepic/diagnose/. Changes no setting.

    python -u -m accuracy_tables.hpc.hdepic.diagnose files
    python -u -m accuracy_tables.hpc.hdepic.diagnose answers [--results <folder>]
    python -u -m accuracy_tables.hpc.hdepic.diagnose motion
    python -u -m accuracy_tables.hpc.hdepic.diagnose render [--results <folder>] [--config <yaml>]

files    what $HDEPIC_ROOT holds beyond what section 17 read: its top folders, the annotation folders, every file of 3 planned videos'
         folders, and any other file with "gaze" in its name. HD-EPIC's intermediate data has a per-frame gaze centre in image space; if
         it's on scratch, its structure is printed so it can be compared with ours.
answers  from section 17's saved records, per type and row: the options chosen (A to E) and the correct ones, how often a row picks
         Blind's or Uniform's option, and for the foveated rows how often a wrong pick is the same option at every budget and the same as
         nogate's, and their accuracy with and without frames that had no gaze.
motion   on step 0a's 10 videos, 60 s each: the image motion between consecutive MP4 frames (phase correlation, its sign checked on a
         known shift first) against the motion the right IMU's gyro predicts through our calibration and our 90-degree turn (step 0a's
         motion matrix). If our coordinates are the MP4's, "as ours" fits best, at a scale near 1 and a lag near 0 frames; a wrong turn,
         a mirror, a wrong scale and a clock offset each show as a different best fit.
render   5 "where looking" frames exactly as the model sees them at 17 tokens (SmolVLM's processor, then the cells its encoder gets, each
         shown over the area it covers) next to the stock 384 px frame; and step 0a's 20 questions with our gaze, Meta's, the fovea's cell
         and the other orientations drawn on the frame, with the correct choice and the rows' picks. The sheets and the log hold HD-EPIC
         images and text: never commit them.
Each command prints FINDING lines.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

from accuracy_tables.hpc.hdepic import data
from accuracy_tables.hpc.hdepic import diagnose_aria as da

LETTERS = "ABCDE"
MODEL = "smolvlm"
FOVEA_ROWS = ("budget", "nogate")
HYPOTHESES = {                                     # what the MP4's coordinates would be, given ours, about the image centre
    "as ours": ((1, 0), (0, 1)),
    "turned 90 cw": ((0, -1), (1, 0)),
    "turned 180": ((-1, 0), (0, -1)),
    "turned 90 ccw": ((0, 1), (-1, 0)),
    "mirrored left-right": ((-1, 0), (0, 1)),
    "mirrored top-bottom": ((1, 0), (0, -1)),
    "transposed": ((0, 1), (1, 0)),
    "anti-transposed": ((0, -1), (-1, 0)),
}
TAGS = {"turned 90 cw": "R90", "turned 180": "R180", "turned 90 ccw": "R270", "mirrored left-right": "MX", "mirrored top-bottom": "MY",
        "transposed": "TR", "anti-transposed": "AT"}
MOTION_FRAMES, MOTION_START = 1800, 900            # 60 s of a 30 fps video, from 30 s in
CROP, SMALL = 0.5, 176                             # the centre half of the frame, at 176 px, for phase correlation
MOVING = 0.002                                     # predicted motion (units of the image side) for a pair to count as moving
LAGS = 90                                          # frames either way in the lag scan
MOTION_OK = dict(r=0.4, slope=(0.7, 1.4), lag=2)


def default_results():
    return os.path.join(os.environ["GV_ROOT"], "results", "hdepic_gaze_smolvlm_flops")


def write_json(name, obj):
    path = os.path.join(da.diagnose_dir(), name)
    data.refuse_inside_raw(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, default=float)
    os.replace(path + ".tmp", path)
    return path


# ---------------------------------------------------------------------------- files
def mask(value):
    """A CSV cell for the log: numbers as they are, any text as <text>."""
    try:
        float(value)
        return value
    except (TypeError, ValueError):
        return "<text>" if value else value


def describe_file(path, rows=2):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".csv":
            with open(path, newline="") as f:
                r = csv.reader(f)
                head = next(r, [])
                first = [row for row in (next(r, None) for _ in range(rows)) if row]
            return dict(columns=head, first_rows=[[mask(v) for v in row] for row in first])
        if ext == ".npz":
            with np.load(path, allow_pickle=False) as z:
                return {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files}
        if ext == ".npy":
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            return dict(shape=list(a.shape), dtype=str(a.dtype))
        if ext == ".json":
            with open(path) as f:
                j = json.load(f)
            return dict(keys=sorted(j)[:20]) if isinstance(j, dict) else dict(list=len(j))
    except Exception as e:
        return dict(error=repr(e)[:160])
    return dict(bytes=os.path.getsize(path))


def gaze_like(rel):
    """A file with "gaze" in its name that section 17 didn't read (it read mps/eye_gaze/*.csv only)."""
    rel = rel.replace(os.sep, "/").lower()
    return "gaze" in os.path.basename(rel) and "/mps/eye_gaze/" not in "/" + rel


def files_cmd(a):
    root = data.root()
    plan = data.read_plan()
    print(f"--- {root}", flush=True)
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        print(f"   {name}/  {len(os.listdir(p))} entries" if os.path.isdir(p) else f"   {name}  {os.path.getsize(p)} bytes", flush=True)
    ann = os.path.join(root, "annotations")
    if os.path.isdir(ann):
        print("--- annotations", flush=True)
        for d, sub, names in sorted(os.walk(ann)):
            if d.count(os.sep) - ann.count(os.sep) <= 2:
                print(f"   {os.path.relpath(d, ann)}/  {len(names)} files: {sorted(names)[:8]}{' ...' if len(names) > 8 else ''}", flush=True)
    found = []
    for v in plan["videos"][:3]:
        print(f"--- videos/{v}", flush=True)
        for d, _, names in sorted(os.walk(data.video_dir(v))):
            for n in sorted(names):
                p = os.path.join(d, n)
                rel = os.path.relpath(p, root)
                print(f"   {os.path.relpath(p, data.video_dir(v))}  {os.path.getsize(p)} bytes", flush=True)
                if gaze_like(rel):
                    found.append(rel)
    for top in sorted(os.listdir(root)):
        if top == "videos" or not os.path.isdir(os.path.join(root, top)):
            continue
        for k, (d, _, names) in enumerate(os.walk(os.path.join(root, top))):
            if k > 5000:
                break
            found += [os.path.relpath(os.path.join(d, n), root) for n in names if gaze_like(os.path.relpath(os.path.join(d, n), root))]
    for rel in found[:12]:
        print(f"   gaze file {rel}: {describe_file(os.path.join(root, rel))}", flush=True)
    print(f"FINDING files: {len(found)} gaze files besides MPS's eye_gaze CSVs" + (f", for example {found[:3]}" if found else
          " (HD-EPIC's intermediate data, with its per-frame gaze centre, isn't on scratch)"), flush=True)
    write_json("files.json", dict(gaze_files=found))


# ---------------------------------------------------------------------------- answers
def label(row, budget_pct):
    return row if row in ("uniform", "blind") else f"{row} {float(budget_pct):g}%"


def load_answers(results, model=MODEL):
    """Section 17's records (no errors) with type, label, budget and the real video; and the questions."""
    import pandas as pd
    from accuracy_tables.hpc.all350 import flops_report as rep
    with open(os.path.join(results, model, "flops_plan.json")) as f:
        plan = json.load(f)
    with open(data.questions_json()) as f:
        qs = pd.DataFrame(json.load(f))
    res, _ = rep.load(results, model)
    lines = pd.DataFrame(plan["lines"])[["row", "condition", "budget_pct"]]
    res = res.merge(lines, on=["row", "condition"], how="left")
    res["label"] = [label(r, b) for r, b in zip(res.row, res.budget_pct)]
    res["type"] = res.video.map(dict(zip(qs.video, qs.type)))
    return res, qs


def pick_table(res, qs):
    """Per type and label: n, accuracy, the share of each letter picked, the share equal to Blind's and to Uniform's pick, the mean top
    probability, and (foveated rows) the share of frames that had no gaze."""
    import pandas as pd
    out = []
    for t, g in res.groupby("type"):
        blind = g[g.label == "blind"].set_index("video")["pred"] if "pred" in g else None
        uni = g[g.label == "uniform"].set_index("video")["pred"] if "pred" in g else None
        for lab, h in g.groupby("label", sort=False):
            r = dict(type=t, label=lab, n=len(h), accuracy=100 * h.correct.mean())
            if "pred" in h:
                for L in LETTERS:
                    r[f"pick_{L}"] = 100 * (h.pred == L).mean()
                if blind is not None and lab != "blind":
                    r["same_as_blind"] = 100 * (h.pred.to_numpy() == h.video.map(blind).to_numpy()).mean()
                if uni is not None and lab != "uniform":
                    r["same_as_uniform"] = 100 * (h.pred.to_numpy() == h.video.map(uni).to_numpy()).mean()
            if "p" in h and h.p.notna().all():
                r["mean_top_p"] = float(np.mean([max(p) for p in h.p]))
            if "n_gaze_missing" in h and h.row.iloc[0] in FOVEA_ROWS:
                r["frames_without_gaze"] = 100 * h.n_gaze_missing.sum() / max(1, h.n_frames.sum())
            out.append(r)
        q = qs[qs.type == t]
        out.append(dict(type=t, label="correct answers", n=len(q), **{f"pick_{L}": 100 * (q.answer == L).mean() for L in LETTERS}))
    return pd.DataFrame(out)


def same_wrong(res, row, budgets):
    """For one foveated row: on questions it gets wrong at every budget, the share where all its wrong picks are the same option; and per
    budget, on questions both it and nogate get wrong, the share where they picked the same option."""
    g = res[res.row == row].pivot_table(index="video", columns="budget_pct", values="pred", aggfunc="first")
    c = res[res.row == row].pivot_table(index="video", columns="budget_pct", values="correct", aggfunc="first")
    cols = [b for b in budgets if b in g.columns]
    wrong_all = (c[cols] == 0).all(axis=1)
    out = dict(row=row, wrong_at_every_budget=int(wrong_all.sum()),
               same_option_at_every_budget=100 * float((g.loc[wrong_all, cols].nunique(axis=1) == 1).mean()) if wrong_all.any() else None)
    if row != "nogate":
        ng = res[res.row == "nogate"].pivot_table(index="video", columns="budget_pct", values="pred", aggfunc="first")
        nc = res[res.row == "nogate"].pivot_table(index="video", columns="budget_pct", values="correct", aggfunc="first")
        for b in cols:
            if b in ng.columns:
                both = (c[b] == 0) & (nc[b].reindex(c.index) == 0)
                out[f"same_as_nogate_{b:g}%"] = 100 * float((g.loc[both, b] == ng[b].reindex(c.index)[both]).mean()) if both.any() else None
    return out


def gaze_split(res, row):
    """Accuracy of a foveated row on questions whose frames all had gaze, against those with at least one frame without."""
    g = res[res.row == row]
    if "n_gaze_missing" not in g:
        return None
    full, part = g[g.n_gaze_missing == 0], g[g.n_gaze_missing > 0]
    return dict(row=row, with_gaze_n=len(full), with_gaze_acc=100 * full.correct.mean() if len(full) else None,
                missing_n=len(part), missing_acc=100 * part.correct.mean() if len(part) else None)


def answers_cmd(a):
    import pandas as pd
    res, qs = load_answers(a.results)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    table = pick_table(res, qs)
    print("--- per type and row: n, accuracy and the options picked (%), against Blind's and Uniform's picks", flush=True)
    print(table.round(1).to_string(index=False), flush=True)
    findings, summary = [], dict(table=table.to_dict("records"))
    for t in ("gaze_estimation", "interaction_anticipation"):
        sub = res[res.type == t]
        budgets = sorted(sub[sub.row == "budget"].budget_pct.dropna().unique())
        rows = [same_wrong(sub, r, budgets) for r in FOVEA_ROWS if (sub.row == r).any()]
        splits = [s for s in (gaze_split(sub, r) for r in FOVEA_ROWS if (sub.row == r).any()) if s]
        print(f"\n--- {t}: the foveated rows' wrong picks", flush=True)
        print(pd.DataFrame(rows).round(1).to_string(index=False), flush=True)
        print(f"--- {t}: the foveated rows with every frame's gaze against with some frames without", flush=True)
        print(pd.DataFrame(splits).round(1).to_string(index=False), flush=True)
        summary[t] = dict(same_wrong=rows, gaze_split=splits)
        tt = table[table.type == t].set_index("label")
        for lab in [l for l in tt.index if l.split()[0] in FOVEA_ROWS]:
            r = tt.loc[lab]
            top = max(LETTERS, key=lambda L: r.get(f"pick_{L}", 0) or 0)
            findings.append(f"FINDING answers {t} {lab}: accuracy {r.accuracy:.1f}%, picks {top} on {r[f'pick_{top}']:.0f}% (correct answers are "
                            f"{top} on {tt.loc['correct answers', f'pick_{top}']:.0f}%), Blind's pick on {r.get('same_as_blind', float('nan')):.0f}%, "
                            f"Uniform's on {r.get('same_as_uniform', float('nan')):.0f}%, frames without gaze {r.get('frames_without_gaze', float('nan')):.1f}%")
    for line in findings:
        print(line, flush=True)
    print(f"written {write_json('answers.json', summary)}", flush=True)


# ---------------------------------------------------------------------------- motion
def phase_shift(a, b):
    """cv2's phase correlation of two float32 images, with a Hann window: ((dx, dy), response)."""
    import cv2
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), win)
    return (float(dx), float(dy)), float(resp)


def shift_sign(size=SMALL, seed=0):
    """(sx, sy) so that sx * dx, sy * dy are the content's motion from the first image to the second, right and down positive. Found on a
    known shift, so it doesn't rest on cv2's sign convention."""
    import cv2
    rng = np.random.default_rng(seed)
    a = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), 3)
    b = np.roll(a, (2, 3), axis=(0, 1))                                         # content moves 3 right and 2 down
    (dx, dy), _ = phase_shift(a, b)
    if abs(abs(dx) - 3) > 0.5 or abs(abs(dy) - 2) > 0.5:
        raise SystemExit(f"phase correlation found ({dx:.2f}, {dy:.2f}) for a (3, 2) shift. Stop and tell the user.")
    return float(np.sign(dx) * 1.0), float(np.sign(dy) * 1.0)


def gyro_rotation(t_us, gyro, t0_us, t1_us):
    """The rotation vector (rad, IMU frame) the gyro integrates between each t0 and t1."""
    t_us = np.asarray(t_us, np.float64)
    dt = np.diff(t_us) / 1e6
    cum = np.concatenate([np.zeros((1, 3)), np.cumsum(np.asarray(gyro, np.float64)[:-1] * dt[:, None], axis=0)])
    at = lambda t: np.stack([np.interp(t, t_us, cum[:, k]) for k in range(3)], -1)
    return at(np.asarray(t1_us, np.float64)) - at(np.asarray(t0_us, np.float64))


def vector_r(meas, pred):
    """How well two sets of 2D motions agree in direction and sign, -1 to 1 (an uncentred correlation over both axes)."""
    den = np.sqrt((meas ** 2).sum() * (pred ** 2).sum())
    return float((meas * pred).sum() / den) if den > 0 else float("nan")


def motion_fit(meas, pred, moving):
    """-> ({hypothesis: r}, best, slope of meas on pred under 'as ours', {lag: r}, best lag) on the moving pairs."""
    m, p = meas[moving], pred[moving]
    rs = {name: vector_r(m, p @ np.asarray(H, float).T) for name, H in HYPOTHESES.items()}
    best = max(rs, key=lambda k: -np.inf if not np.isfinite(rs[k]) else rs[k])
    slope = float((m * p).sum() / (p ** 2).sum()) if (p ** 2).sum() > 0 else float("nan")
    lags = {}
    n = len(meas)
    for L in range(-LAGS, LAGS + 1):
        i = np.arange(max(0, -L), min(n, n - L))
        lags[L] = vector_r(meas[i], pred[i + L])
    best_lag = max(lags, key=lambda k: -np.inf if not np.isfinite(lags[k]) else lags[k])
    return rs, best, slope, lags, best_lag


def motion_ok(best, r, slope, lag):
    return best == "as ours" and r >= MOTION_OK["r"] and MOTION_OK["slope"][0] <= slope <= MOTION_OK["slope"][1] and abs(lag) <= MOTION_OK["lag"]


def decode_window(video, f0, n):
    """Grey centre crops of MP4 frames f0 .. f0 + n - 1 at SMALL px, read in order (exact frame numbers), the decoder's time of each, and
    the crop's scale (units of the image side per small pixel)."""
    import cv2
    cap = cv2.VideoCapture(os.path.join(data.video_dir(video), f"{video}.mp4"))
    crops, dec, i, scale = [], [], 0, None
    while i < f0 + n:
        if i < f0:
            if not cap.grab():
                break
        else:
            ok, fr = cap.read()
            if not ok:
                break
            dec.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1e3)
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            h, w = g.shape
            c = int(round(min(h, w) * CROP))
            y0, x0 = (h - c) // 2, (w - c) // 2
            crops.append(cv2.resize(g[y0:y0 + c, x0:x0 + c], (SMALL, SMALL), interpolation=cv2.INTER_AREA).astype(np.float32))
            scale = (c / SMALL / w, c / SMALL / h)
        i += 1
    cap.release()
    return crops, np.asarray(dec), scale


def motion_video(video, M, sign):
    mp4_ns, vrs_ns = data.read_time_csv(os.path.join(data.video_dir(video), f"{video}_mp4_to_vrs_time_ns.csv"))
    f0 = int(min(MOTION_START, max(0, len(mp4_ns) - MOTION_FRAMES - 1)))
    crops, dec, scale = decode_window(video, f0, MOTION_FRAMES)
    n = len(crops)
    if n < 100:
        return dict(video=video, error=f"only {n} frames decoded")
    frames = np.arange(f0, f0 + n)
    decoder_off = float(np.abs(dec - mp4_ns[frames] / 1e9).max())
    meas = np.zeros((n - 1, 2))
    resp = np.zeros(n - 1)
    for k in range(n - 1):
        (dx, dy), resp[k] = phase_shift(crops[k], crops[k + 1])
        meas[k] = sign[0] * dx * scale[0], sign[1] * dy * scale[1]
    with np.load(os.path.join(data.video_dir(video), "imu.npz")) as z:
        t_us, gyro = z["right_t_ns"].astype(np.float64) / 1e3, z["right_gyro"].astype(np.float64)
    dev = vrs_ns[frames].astype(np.float64) / 1e3
    pred = gyro_rotation(t_us, gyro, dev[:-1], dev[1:]) @ np.asarray(M, float).T
    moving = (np.linalg.norm(pred, axis=1) > MOVING) & (resp > 0.05)
    rs, best, slope, lags, lag = motion_fit(meas, pred, moving)
    return dict(video=video, frames=n, start_frame=f0, moving_pairs=int(moving.sum()), decoder_offset_s=decoder_off, r=rs, best=best,
                r_as_ours=rs["as ours"], slope=slope, best_lag_frames=int(lag), r_at_best_lag=lags[lag],
                ok=bool(motion_ok(best, rs["as ours"], slope, lag)))


def motion_cmd(a):
    with open(da.aria_json()) as f:
        aria = json.load(f)
    sign = shift_sign()
    print(f"phase correlation's sign, from a known shift: x {sign[0]:+.0f}, y {sign[1]:+.0f}", flush=True)
    out = []
    for v in aria["motion_videos"]:
        r = motion_video(v, aria["videos"][v]["motion_M"], sign)
        out.append(r)
        if "error" in r:
            print(f"   {v}: {r['error']}", flush=True)
            continue
        others = ", ".join(f"{k} {x:+.2f}" for k, x in sorted(r["r"].items(), key=lambda kv: -np.nan_to_num(kv[1], nan=-9)) if k != "as ours")
        print(f"   {v}: {r['moving_pairs']} moving pairs of {r['frames'] - 1}; as ours r {r['r_as_ours']:+.2f}, scale {r['slope']:.2f}, best "
              f"lag {r['best_lag_frames']:+d} frames (r {r['r_at_best_lag']:+.2f}); best fit '{r['best']}'; others: {others}; decoder "
              f"within {r['decoder_offset_s']:.3f} s of the time CSV: {'OK' if r['ok'] else 'NOT AS OURS'}", flush=True)
    good = [r for r in out if r.get("ok")]
    bests = sorted({r["best"] for r in out if "best" in r})
    scales = [r["slope"] for r in out if "slope" in r]
    lags = [r["best_lag_frames"] for r in out if "best_lag_frames" in r]
    print(f"FINDING motion: {len(good)} of {len(out)} videos move as our coordinates predict (best fits {bests}; scale median "
          f"{np.median(scales) if scales else float('nan'):.2f}; lag median {np.median(lags) if lags else float('nan'):+.0f} frames): "
          + ("MOTION MATCHES" if out and len(good) == len(out) else "MOTION DIFFERS"), flush=True)
    print(f"written {write_json('motion.json', dict(sign=sign, videos=out))}", flush=True)


# ---------------------------------------------------------------------------- render
def model_view(sm, x01, cells, mean, std):
    """What SmolVLM's encoder gets for each cell (sm.cells_to_patches, the run's own function), put back over the area the cell covers:
    (3, 384, 384) in [0, 1]. A 42 px cell is the stock pixels; a 126 px cell is its 42 px version, shown 3 times larger."""
    import torch
    import torch.nn.functional as F
    n = len(cells)
    p = sm.cells_to_patches(x01, cells, mean, std)
    crops = p.reshape(n, 3, 3, 3, sm.PATCH, sm.PATCH).permute(0, 3, 1, 4, 2, 5).reshape(n, 3, sm.BASE, sm.BASE)
    crops = crops * torch.as_tensor(std, dtype=torch.float32).view(1, 3, 1, 1) + torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    view = torch.zeros(3, sm.IMG, sm.IMG)
    for (y, x, s), c in zip(np.asarray(cells).tolist(), crops):
        view[:, y:y + s, x:x + s] = F.interpolate(c[None], size=(s, s), mode="nearest")[0]
    return view.clamp(0, 1)


def fine_cells_match(sm, x01, cells, view):
    """The largest difference between the model's view and the stock pixels on the fovea's 42 px cells (0 up to float error)."""
    worst = 0.0
    for y, x, s in np.asarray(cells).tolist():
        if s == sm.BASE:
            worst = max(worst, float((view[:, y:y + s, x:x + s] - x01[:, y:y + s, x:x + s]).abs().max()))
    return worst


def to_bgr(img01):
    a = (np.asarray(img01.permute(1, 2, 0) if hasattr(img01, "permute") else img01) * 255).round().clip(0, 255).astype(np.uint8)
    return a[:, :, ::-1].copy()


def alternatives(g):
    """The gaze point under each orientation hypothesis but ours."""
    c = np.asarray(g, float) - 0.5
    return {k: tuple(0.5 + np.asarray(H, float) @ c) for k, H in HYPOTHESES.items() if k != "as ours"}


def text_pad(lines, width):
    import cv2
    pad = np.full((18 * len(lines) + 6, width, 3), 255, np.uint8)
    for i, t in enumerate(lines):
        safe = "".join(ch if ord(ch) < 128 else "?" for ch in str(t))[: width // 7]
        cv2.putText(pad, safe, (4, 16 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return pad


def stack_rows(rows, path):
    import cv2
    w = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], w - r.shape[1], 3), 255, np.uint8)]) for r in rows]
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def picks_of(res, video):
    """{label: letter} of every row's pick on one clip."""
    g = res[res.video == video]
    return dict(zip(g.label, g.pred)) if "pred" in g else {}


def choice_text(q, letter):
    i = LETTERS.find(str(letter))
    return f"{letter}: {q['options'][i]}" if 0 <= i < len(q["options"]) else str(letter)


def render_cmd(a):
    import cv2
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from accuracy_tables import common
    from vlmeval import smolvlm_fovea as sm
    from vlmeval import spatial as sp
    res, qs = load_answers(a.results)
    qmap = {r["video"]: r for r in qs.to_dict("records")}
    cfg = common.load_config(a.config)
    d = common.load_data(cfg)
    ip = AutoProcessor.from_pretrained(sm.MODEL_CANDIDATES[0]).image_processor
    mean, std = ip.image_mean, ip.image_std
    out_dir = da.diagnose_dir()
    data.refuse_inside_raw(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 5 frames as the model sees them: "where looking" questions Uniform gets right and nogate at its smallest budget gets wrong, one per
    # real video (sorted by video, then clip), the span's last frame
    small = res[res.row == "nogate"].budget_pct.min()
    right = set(res[(res.label == "uniform") & (res.correct == 1)].video)
    wrong = set(res[(res.row == "nogate") & (res.budget_pct == small) & (res.correct == 0)].video)
    pool = sorted((qmap[v]["real_video"], v) for v in right & wrong if qmap.get(v, {}).get("type") == "gaze_estimation")
    chosen, seen = [], set()
    for real, v in pool:
        if real not in seen:
            chosen.append(v)
            seen.add(real)
    rows, worst = [], 0.0
    for v in chosen[:5]:
        fr = d["uniform"][v][-1]
        rec = sm.stock_input(ip, Image.open(fr["path"]).convert("RGB"))
        g = fr["gaze"] if fr["gaze"] is not None else (0.5, 0.5)
        cells = sm.foveated_cells(*g, 1)
        view = model_view(sm, rec["x01"], cells, mean, std)
        worst = max(worst, fine_cells_match(sm, rec["x01"], cells, view))
        marked = to_bgr(view)
        for y, x, s in np.asarray(cells).tolist():
            cv2.rectangle(marked, (x, y), (x + s - 1, y + s - 1), (0, 200, 0) if s == sm.BASE else (160, 160, 160), 1)
        for px, py in fr["gaze_trace"]:
            cv2.circle(marked, (int(px * sm.IMG), int(py * sm.IMG)), 2, (0, 140, 255), -1)
        cv2.circle(marked, (int(g[0] * sm.IMG), int(g[1] * sm.IMG)), 7, (0, 0, 255), 2)
        q, pk = qmap[v], picks_of(res, v)
        lines = [f"{q['hdepic_qid']} ({v}, {q['real_video']}), frame at {fr['t']:.2f} s; gaze {'none: centre' if fr['gaze'] is None else tuple(round(c, 3) for c in g)}",
                 f"Q: {q['question']}"[:160], "correct " + choice_text(q, q["answer"]),
                 *[f"{lab}: {choice_text(q, pk[lab])}" for lab in sorted(pk) if lab.split()[0] in ("uniform", "blind", *FOVEA_ROWS)]]
        panel = np.hstack([to_bgr(rec["x01"]), to_bgr(view), marked])
        rows.append(np.vstack([panel, text_pad(lines, panel.shape[1])]))
    if rows:
        stack_rows(rows, os.path.join(out_dir, "fovea_as_the_model_sees_it.jpg"))
    print(f"FINDING render: {len(rows)} frames rendered as the model sees them (stock 384 px | the 17 cells the encoder gets | the same with "
          f"the cells, the gaze trace and the fixation); the fovea's cells differ from the stock pixels by at most {worst:.2e}", flush=True)

    # step 0a's 20 questions: our fixation (the fovea's), the raw sample, Meta's point, the fovea's cell and the other orientations
    with open(da.aria_json()) as f:
        aria = json.load(f)
    tiles, table = [], []
    T = 448
    for r in aria["questions"]:
        v = r["pseudo"]
        fr = d["uniform"][v][-1]
        img = cv2.resize(cv2.imread(fr["path"]), (T, T), interpolation=cv2.INTER_AREA)
        g = fr["gaze"]
        W, H = r["size"]
        meta = r.get("meta_upright_px")
        meta = None if meta is None else (meta[0] / W, meta[1] / H)
        raw = tuple(r["saved"])
        if g is not None:
            cells = sm.foveated_cells(*g, 1)
            fine = np.asarray([c for c in np.asarray(cells).tolist() if c[2] == sm.BASE])
            y0, x0 = fine[:, 0].min(), fine[:, 1].min()
            y1, x1 = (fine[:, 0] + fine[:, 2]).max(), (fine[:, 1] + fine[:, 2]).max()
            f_ = T / sm.IMG
            cv2.rectangle(img, (int(x0 * f_), int(y0 * f_)), (int(x1 * f_), int(y1 * f_)), (0, 200, 0), 2)
            for k, (ax, ay) in alternatives(g).items():
                cv2.putText(img, TAGS[k], (int(ax * T) - 8, int(ay * T) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(img, (int(g[0] * T), int(g[1] * T)), 10, (0, 0, 255), 2)
        cv2.circle(img, (int(raw[0] * T), int(raw[1] * T)), 3, (0, 0, 255), -1)
        if meta is not None:
            cv2.drawMarker(img, (int(meta[0] * T), int(meta[1] * T)), (255, 0, 0), cv2.MARKER_CROSS, 14, 2)
        ecc = float(sp.ecc_deg(g[0] * sp.IMG, g[1] * sp.IMG, raw[0] * sp.IMG, raw[1] * sp.IMG)) if g is not None else float("nan")
        q, pk = qmap[v], picks_of(res, v)
        d_meta = da.dist([raw[0] * W, raw[1] * H], meta and [meta[0] * W, meta[1] * H])
        table.append(dict(clip=v, video=r["video"], fixation=None if g is None else tuple(round(c, 3) for c in g),
                          sample=tuple(round(c, 3) for c in raw), meta=None if meta is None else tuple(round(c, 3) for c in meta),
                          meta_px=round(d_meta, 2), fixation_to_sample_deg=round(ecc, 1),
                          frames_without_gaze=sum(f_["gaze"] is None for f_ in d["uniform"][v]), K=len(d["uniform"][v])))
        lines = [f"{q['hdepic_qid']}  {v}", f"fixation {table[-1]['fixation']}, sample {table[-1]['sample']}, Meta {table[-1]['meta']}",
                 "correct " + choice_text(q, q["answer"]), "uniform " + choice_text(q, pk.get("uniform", "?")),
                 f"nogate {small:g}% " + choice_text(q, pk.get(f"nogate {small:g}%", "?"))]
        tiles.append(np.vstack([img, text_pad(lines, T)]))
    rows = []
    for i in range(0, len(tiles), 4):
        part = tiles[i:i + 4]
        h = max(t.shape[0] for t in part)
        rows.append(np.hstack([np.vstack([t, np.full((h - t.shape[0], T, 3), 255, np.uint8)]) for t in part]))
    if rows:
        stack_rows(rows, os.path.join(out_dir, "projection_20_questions.jpg"))
    import pandas as pd
    t = pd.DataFrame(table)
    print("--- step 0a's questions: the fovea's fixation, the gaze sample at the span's end, Meta's point (all in [0, 1]), and how far apart",
          flush=True)
    print(t.to_string(index=False), flush=True)
    if len(t):
        print(f"FINDING gaze: the fovea's fixation is a median {t.fixation_to_sample_deg.median():.1f} deg from the gaze sample at the span's "
              f"end ({int((t.fixation_to_sample_deg > 10).sum())} of {len(t)} over 10 deg); Meta's point is at most {t.meta_px.max():.2f} px "
              f"from ours; {int(t.frames_without_gaze.sum())} of {int(t.K.sum())} of their frames have no gaze (fovea at the centre)", flush=True)
    print("legend: red ring = the fixation the fovea used, red dot = the gaze sample at the span's end, blue cross = Meta's, green box = "
          "the fovea's cell, cyan tags = where the fixation would be under each other orientation: " + ", ".join(f"{v} {k}" for k, v in TAGS.items()), flush=True)
    print(f"written {out_dir}/fovea_as_the_model_sees_it.jpg and projection_20_questions.jpg (HD-EPIC images and text: never commit them)",
          flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("files")
    for name in ("answers", "render"):
        s = sub.add_parser(name)
        s.add_argument("--results", default=None, help="default: $GV_ROOT/results/hdepic_gaze_smolvlm_flops (only read)")
        if name == "render":
            s.add_argument("--config", default="accuracy_tables/hpc/config.hdepic.yaml")
    sub.add_parser("motion")
    a = ap.parse_args(argv)
    if getattr(a, "results", "") is None:
        a.results = default_results()
    dict(files=files_cmd, answers=answers_cmd, motion=motion_cmd, render=render_cmd)[a.command](a)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)

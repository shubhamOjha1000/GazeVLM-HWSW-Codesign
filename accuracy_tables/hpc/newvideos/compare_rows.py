"""Check 1: the new builder on the 12 known videos against the existing fold CSVs and gate predictions. CPU only.

    python -u -m accuracy_tables.hpc.newvideos.compare_rows

Reads data/folds/ and oof_preds_full.csv and never writes there. Each video's predictions come from the same model as the
existing ones (its `source` column): the held-out fold's model for a training video, the 5 models averaged for a test video.
A 5-model average on a training video would include models that trained on it.

Ends with REBUILD MATCHES, or REBUILD DIFFERS and a failed job. The tolerances allow for the last printed digit only.
"""
import os
import sys

import numpy as np
import pandas as pd

from accuracy_tables.hpc.newvideos import paths
from accuracy_tables.hpc.newvideos.score_gate import load_models
from eyehead import policy as pol
from eyehead.oof import _score

COUNT_COLS = ("n_velocities", "n_gaze_in_gap", "n_oof_in_gap", "n_imu_in_gap")
WINDOW_TOL = {"gaze_rates_window": 2e-5, "imu_window": 2e-5, "gaze_xy_window": 2e-4}


def parse_window(s):
    """'a,b;c,d' -> float64 array; an empty cell -> shape (0, 0). The result keeps NaN."""
    if not isinstance(s, str) or not s.strip():
        return np.zeros((0, 0))
    return np.array([[float(v) for v in row.split(",")] for row in s.split(";")], dtype=np.float64)


def compare_column(new, old):
    """Two lists of packed strings -> rows with the same text, rows with the same shape, and the largest absolute
    difference over same-shaped rows. NaN on one side only makes that difference infinite."""
    same_text = same_shape = 0
    worst = 0.0
    for a, b in zip(new, old):
        a, b = a if isinstance(a, str) else "", b if isinstance(b, str) else ""
        if a == b:
            same_text += 1
            same_shape += 1
            continue
        x, y = parse_window(a), parse_window(b)
        if x.shape != y.shape:
            continue
        same_shape += 1
        if (np.isnan(x) != np.isnan(y)).any():
            worst = float("inf")
        d = np.abs(x - y)
        d = d[np.isfinite(d)]
        if d.size:
            worst = max(worst, float(d.max()))
    return dict(same_text=same_text, same_shape=same_shape, max_abs_diff=worst)


def compare_video(video, new, old):
    """new, old: one video's rows sorted by frame -> (one result row, whether everything matched)."""
    row = dict(video=video, new_pairs=len(new), old_pairs=len(old))
    ok = len(new) == len(old) and (new["frame"].to_numpy() == old["frame"].to_numpy()).all()
    row["same_frames"] = bool(ok)
    if not ok:
        return row, False
    for c in COUNT_COLS:
        same = int((new[c].to_numpy() == old[c].to_numpy()).sum())
        row[f"{c}_same"] = same
        ok = ok and same == len(new)
    for c, tol in WINDOW_TOL.items():
        r = compare_column(new[c].tolist(), old[c].tolist())
        row[f"{c}_same_text"], row[f"{c}_same_shape"], row[f"{c}_max_diff"] = r["same_text"], r["same_shape"], r["max_abs_diff"]
        ok = ok and r["same_shape"] == len(new) and r["max_abs_diff"] <= tol
    return row, bool(ok)


def models_for(source, cks):
    """The models behind a video's existing predictions: 'fold<i>' -> that fold's, 'test ensemble' -> all of them."""
    return [cks[int(source[len("fold"):]) - 1]] if source.startswith("fold") else cks


def compare_predictions(new, old_preds, cks, device="cpu"):
    """new and old_preds sorted by frame, same frames -> classes that agree, largest probability difference."""
    sources = old_preds["source"].unique()
    if len(sources) != 1:
        raise ValueError(f"expected one source per video, got {list(sources)}")
    p = np.mean([_score(ck, new, device) for ck in models_for(sources[0], cks)], axis=0)
    old_p = old_preds[[f"p_{n}" for n in ("TRANSITION", "PURSUIT", "REFIXATION", "STABLE")]].to_numpy()
    return dict(source=sources[0], classes_same=int((p.argmax(1) == old_preds["pred"].to_numpy()).sum()),
                max_prob_diff=float(np.abs(p - old_p).max()))


def with_frame(df):
    return df.assign(frame=df["feat_frame_2"].map(pol.frame_no)).sort_values("frame").reset_index(drop=True)


def main():
    videos = paths.read_list("aea12")
    cols = ["sequence", "feat_frame_2", *COUNT_COLS, *WINDOW_TOL]
    d = paths.known_folds_dir()
    old = pd.concat([pd.read_csv(os.path.join(d, f), usecols=cols) for f in ("train_val_fold1.csv", "test.csv")],
                    ignore_index=True)
    preds = pd.read_csv(paths.known_preds_csv())
    cks = load_models(paths.ckpt_dir())
    rows, all_ok = [], True
    for v in videos:
        new = with_frame(pd.read_csv(paths.rows_csv(v)))
        row, ok = compare_video(v, new, with_frame(old[old["sequence"] == v]))
        if row["same_frames"]:
            r = compare_predictions(new, with_frame(preds[preds["sequence"] == v]), cks)
            row.update(r)
            ok = ok and r["classes_same"] == len(new)
        rows.append(row)
        all_ok = all_ok and ok
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(table.set_index("video").T.to_string())
    path = os.path.join(paths.root(), "compare_rows.csv")
    table.to_csv(path, index=False)
    print(f"\nwritten {path}")
    if not all_ok:
        sys.exit("REBUILD DIFFERS: stop and tell the user")
    print("REBUILD MATCHES")


if __name__ == "__main__":
    main()

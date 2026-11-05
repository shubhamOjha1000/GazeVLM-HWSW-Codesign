"""Check 3: the gate's behaviour on each group, side by side, for a person to judge. CPU only, no answers.

    python -u -m accuracy_tables.hpc.newvideos.check_gate --groups aea12,aea13,nymeria

Prints the predicted class mix, the real marks per minute and per video (policy D+P), the reasons for them, how many videos
the budget limits thin, the share of frames with no gaze, and the gate's inputs. An odd pattern on Nymeria means stop and tell
the user. Writes newvideos/check_gate_<groups>.csv, and one row per video in marks_<groups>.csv.

aea12 reads data/folds/ and oof_preds_full.csv, only reading them. It covers the videos with questions, so it is the reference
the VLM runs are compared on.
"""
import argparse
import os

import numpy as np
import pandas as pd

from accuracy_tables.hpc.newvideos import paths
from eyehead import policy as pol
from eyehead.features import EYE_COL, IMU_COL, NAMES, build, vor_check
from eyehead.infer import load_ckpt
from vlmeval import budget_rule as br

REASONS = (br.FRAME0, br.SETTLE, br.REFIX, br.PURSUIT)
BUDGETS = (0.05, 0.1, 0.2)
STOCK_TOKENS, TOKENS_PER_FRAME, FRAME_CAP = 81, 17, 64        # SmolVLM2: stock frame, smallest fovea, K


def video_summary(video, lab, n_gaze, n_oof, n_imu):
    """One video's rows in frame order -> one result row. A frame has no gaze if the second before it has no sample or any
    sample out of view (vlmeval.spatial.gaze_tracks), and frame 0 never has one."""
    reasons, _ = br.send_reasons(lab)
    frames = len(lab) + 1
    counts = {r: sum(x == r for x in reasons) for r in REASONS}
    marks = sum(counts.values())
    no_gaze = (np.asarray(n_oof) > 0) | (np.asarray(n_gaze) == 0)
    row = dict(video=video, frames=frames, marks=marks, marks_per_min=60 * marks / frames, **counts,
               no_gaze_frames=1 + int(no_gaze.sum()), windows_out_of_view=int((np.asarray(n_oof) > 0).sum()),
               windows_no_gaze=int((np.asarray(n_gaze) == 0).sum()), windows_no_imu=int((np.asarray(n_imu) == 0).sum()))
    for b in BUDGETS:
        limit = br.frame_limit(b, min(frames, FRAME_CAP) * STOCK_TOKENS, TOKENS_PER_FRAME, frames, FRAME_CAP)
        row[f"limit_{int(100 * b)}"] = limit
        row[f"thinned_{int(100 * b)}"] = marks > limit
    return row


def load_group(group):
    """-> rows and predictions of a group's videos, joined by frame pair and sorted by video and frame."""
    cols = ["sequence", "feat_frame_2", "n_gaze_in_gap", "n_oof_in_gap", "n_imu_in_gap", EYE_COL, IMU_COL]
    if group == "aea12":
        keep = set(paths.read_list("aea12"))
        d = paths.known_folds_dir()
        rows = pd.concat([pd.read_csv(os.path.join(d, f), usecols=cols) for f in ("train_val_fold1.csv", "test.csv")],
                         ignore_index=True)
        rows = rows[rows["sequence"].isin(keep)]
        preds = pd.read_csv(paths.known_preds_csv(), usecols=["feat_frame_2", "pred"])
    else:
        rows = pd.read_csv(os.path.join(paths.folds_dir(group), "test.csv"), usecols=cols)
        preds = pd.read_csv(paths.preds_csv(group), usecols=["feat_frame_2", "pred"])
    out = rows.merge(preds, on="feat_frame_2", how="left", validate="one_to_one")
    if out["pred"].isna().any():
        raise ValueError(f"{group}: {int(out['pred'].isna().sum())} frame pairs have no prediction")
    out["frame"] = out["feat_frame_2"].map(pol.frame_no)
    return out.sort_values(["sequence", "frame"]).reset_index(drop=True)


def input_stats(df, ck):
    """The gate's inputs on these rows, with one fold's head map, and its sign check on that map."""
    f = build(df, np.asarray(ck["head_map"]), ck["seq_len"], ck["eye_min"], ck["head_min"])
    eye = f["X"][:, 0][f["M"]]
    head = f["X"][:, 1][f["M"] & f["imu_ok"][:, None]]
    vor = vor_check(f["E"], f["H"], f["M"], f["imu_ok"])
    return dict(eye_speed_p50=np.percentile(eye, 50), eye_speed_p90=np.percentile(eye, 90),
                head_speed_p50=np.percentile(head, 50) if head.size else np.nan,
                head_speed_p90=np.percentile(head, 90) if head.size else np.nan,
                vor_median_cos=vor["median_cos"], vor_frac_negative=vor["frac_negative"])


def group_summary(per_video, labels):
    """The per-video rows and all predicted labels of one group -> {metric: value}."""
    v = per_video
    frames, marks, pairs = v["frames"].sum(), v["marks"].sum(), len(labels)
    out = dict(videos=len(v), frames=frames, seconds_p50=v["frames"].median(), seconds_max=v["frames"].max())
    for c, name in NAMES.items():
        out[f"class_{name}_pct"] = 100 * (labels == c).mean()
    out.update(marks_per_min=60 * marks / frames, marks_per_video_min=v["marks"].min(),
               marks_per_video_p50=v["marks"].median(), marks_per_video_p90=v["marks"].quantile(0.9),
               marks_per_video_max=v["marks"].max())
    for r in REASONS:
        out[f"marks_{r}_pct"] = 100 * v[r].sum() / marks
    for b in BUDGETS:
        out[f"videos_thinned_{int(100 * b)}pct"] = 100 * v[f"thinned_{int(100 * b)}"].mean()
    out.update(frames_no_gaze_pct=100 * v["no_gaze_frames"].sum() / frames,
               windows_out_of_view_pct=100 * v["windows_out_of_view"].sum() / pairs,
               windows_no_gaze_pct=100 * v["windows_no_gaze"].sum() / pairs,
               windows_no_imu_pct=100 * v["windows_no_imu"].sum() / pairs)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", required=True, help="comma list of aea12 and lists scored by score_gate.py")
    a = ap.parse_args()
    groups = [g.strip() for g in a.groups.split(",")]
    ck = load_ckpt(os.path.join(paths.ckpt_dir(), "fold1.pt"), "cpu")
    table, per_video = {}, []
    for g in groups:
        df = load_group(g)
        rows = [video_summary(v, x["pred"].to_numpy().astype(int), x["n_gaze_in_gap"].to_numpy(), x["n_oof_in_gap"].to_numpy(),
                              x["n_imu_in_gap"].to_numpy()) for v, x in df.groupby("sequence", sort=False)]
        v = pd.DataFrame(rows).assign(group=g)
        per_video.append(v)
        table[g] = dict(group_summary(v, df["pred"].to_numpy().astype(int)), **input_stats(df, ck))
    table = pd.DataFrame(table)
    pd.set_option("display.width", 200)
    print(table.round(2).to_string())
    name = "_".join(groups)
    table.to_csv(os.path.join(paths.root(), f"check_gate_{name}.csv"))
    pd.concat(per_video, ignore_index=True).to_csv(os.path.join(paths.root(), f"marks_{name}.csv"), index=False)
    print(f"\nwritten {paths.root()}/check_gate_{name}.csv and marks_{name}.csv")


if __name__ == "__main__":
    main()

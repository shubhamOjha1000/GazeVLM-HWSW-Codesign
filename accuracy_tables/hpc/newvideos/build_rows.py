"""One CSV of frame pairs per video, with the columns the gate and the fovea read. CPU only, base environment.

    python -u -m accuracy_tables.hpc.newvideos.build_rows --video <name>

It follows src.dataprep.build_feature_csv and calls the same window functions. What it can't do the same way:

- **No DINOv2.** frame_similarity and gaze_patch_token_sim are training targets. eyehead.features.load_rows requires the
  columns, so they are written empty. quad is the same: a placeholder that nothing may score against.
- **No VRS.** The gaze is projected in advance by project_gaze.py and the IMU comes from imu.npz.
- **Feature names are only names.** No .npz exists. Predictions are matched by this exact string, so it holds the video.

Each stream is rebased to its own first sample and frames sit at i / native fps, as the gate's training data was built.
timing.json isn't used.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from accuracy_tables.hpc.newvideos import paths
from src.common import aria_io as io
from src.common import gaze_geometry as gg

IMU_HZ = 10.0
PLACEHOLDER_QUAD = 3


def feature_name(video, k):
    return f"{video}/feat_{k:05d}.npz"


def lookup_projector(yaw, pitch, x, y, oof):
    """The projection saved by project_gaze.py as a fn(yaw, pitch) -> (x, y, oof), for gg.window_gaze_samples."""
    good = np.isfinite(yaw) & np.isfinite(pitch)
    table = {(a, b): (c, d, bool(e)) for a, b, c, d, e in zip(yaw[good].tolist(), pitch[good].tolist(), x[good].tolist(),
                                                              y[good].tolist(), oof[good].tolist())}
    return lambda a, b: table[(float(a), float(b))]


def load_imu(path):
    """The right IMU, as src.common.aria_io.load_imu_from_vrs returns it: (ts_us rebased, [accel xyz, gyro xyz])."""
    with np.load(path) as z:
        t_ns, accel, gyro = z["right_t_ns"], z["right_accel"], z["right_gyro"]
    if not len(t_ns):
        return None, None
    ts = t_ns.astype(np.float64) / 1e3
    return ts - ts[0], np.concatenate([accel, gyro], axis=1).astype(np.float32)


def pair_rows(video, f_ts, g_ts, g_yaw, g_pit, project, imu_ts, imu_arr, imu_hz=IMU_HZ):
    """One dict per consecutive frame pair, formatted as in src.dataprep.build_feature_csv."""
    rows = []
    for i in range(1, len(f_ts)):
        t0, t1 = f_ts[i - 1], f_ts[i]
        rat = gg.window_gaze_rates(g_ts, g_yaw, g_pit, t0, t1)
        win = gg.window_gaze_samples(g_ts, g_yaw, g_pit, t0, t1, project=project)
        n_bins = int(round(imu_hz * (t1 - t0) / 1e6))
        iw = io.window_imu_binned(imu_ts, imu_arr, t0, t1, n_bins)
        imu_rows, n_imu = iw["rows"], iw["n_raw"]
        a_mag = g_mag = float("nan")
        if imu_ts is not None and n_imu:
            raw = imu_arr[(imu_ts >= t0) & (imu_ts < t1)]
            a_mag = float(np.linalg.norm(raw[:, :3], axis=1).mean())
            g_mag = float(np.linalg.norm(raw[:, 3:], axis=1).mean())
        rows.append(dict(
            sequence=video,
            feat_frame_1=feature_name(video, i - 1),
            feat_frame_2=feature_name(video, i),
            gaze_patch_token_sim=float("nan"),
            frame_similarity=float("nan"),
            n_velocities=rat["n_velocities"],
            gaze_rates_window=";".join(f"{r[0]:.5f},{r[1]:.5f},{r[2]:.5f}" for r in rat["rates"]),
            n_gaze_in_gap=win["n"],
            n_oof_in_gap=win["n_oof"],
            gaze_xy_window=";".join(f"{x:.4f},{y:.4f}" for x, y in zip(win["x"], win["y"])),
            n_imu_in_gap=n_imu,
            imu_window=";".join(",".join(f"{v:.5f}" for v in r) for r in imu_rows),
            imu_accel_mag_mean=round(a_mag, 5),
            imu_gyro_mag_mean=round(g_mag, 5),
            quad=PLACEHOLDER_QUAD,
        ))
    return rows


def build_video(video, force=False):
    out = paths.rows_csv(video)
    if os.path.exists(out) and not force:
        print(f"   {video}: rows already built", flush=True)
        return
    meta = paths.frame_meta(video)
    f_ts = np.asarray(meta["ts"], dtype=np.float64) * 1e6
    if len(f_ts) < 2:
        raise ValueError(f"{video}: {len(f_ts)} frames, so no pairs")
    with np.load(paths.gaze_npz(video)) as z:
        g = {k: z[k] for k in ("t_us", "yaw", "pitch", "x", "y", "oof")}
    imu_ts, imu_arr = load_imu(os.path.join(paths.video_dir(video), "imu.npz"))
    project = lookup_projector(g["yaw"], g["pitch"], g["x"], g["y"], g["oof"])
    rows = pair_rows(video, f_ts, g["t_us"], g["yaw"], g["pitch"], project, imu_ts, imu_arr)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, out)
    stats = dict(video=video, frames=len(f_ts), frames_span_s=float(f_ts[-1] / 1e6), gaze_span_s=float(g["t_us"][-1] / 1e6),
                 imu_span_s=float(imu_ts[-1] / 1e6) if imu_ts is not None else None)
    with open(paths.rows_stats(video), "w") as f:
        json.dump(stats, f)
    print(f"   {video}: {len(rows)} pairs; spans in s: frames {stats['frames_span_s']:.0f}, gaze {stats['gaze_span_s']:.0f}, "
          f"IMU {stats['imu_span_s'] if stats['imu_span_s'] is None else round(stats['imu_span_s'])}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_video_args(ap)
    ap.add_argument("--force", action="store_true", help="build again even if the video's result exists")
    a = ap.parse_args()
    for v in paths.chosen_videos(a):
        build_video(v, a.force)


if __name__ == "__main__":
    main()

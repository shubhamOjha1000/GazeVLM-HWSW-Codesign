"""Each video's gaze samples projected onto its frames. Runs in the `aria` environment, which has no pandas or torch.

    python -u -m accuracy_tables.hpc.newvideos.project_gaze --video <name>

Writes newvideos/gaze/<video>.npz, which build_rows.py reads in the base environment. The rules are the existing builder's
(src.dataprep.build_feature_csv and src.common.aria_io.load_gaze_raw): depth 1 m, the upright frame, coordinates in [0, 1],
times rebased to the stream's first sample. No VRS was kept, so the calibration comes from calibration.json.

calibration.json describes the full 2880 px sensor, while the videos are 1408 px. The builder never saw this, because Aria's VRS
loader crops and scales the camera to the stream's resolution before returning it. The JSON loader doesn't, so video_calibration
applies the same rule. Skipping it leaves gaze about 2.3% too close to the image centre.

AEA's general_eye_gaze.csv has one yaw. Nymeria's comes from a newer gaze model with a left and a right yaw, so its combined
yaw is taken from Aria's own reader. The plain mean is saved next to it, only so that the two can be compared.
"""
import argparse
import csv
import os

import numpy as np

from accuracy_tables.hpc.newvideos import paths
from src.common import gaze_geometry as gg

DEPTH_M, ROTATE_CW90 = 1.0, True          # configs/temporal_analysis.yaml, which a test checks
SENSOR_SIZE, VIDEO_SIZE = 2880, 1408      # Aria's only rule to 1408 px on these glasses, from 2880 px:
CROP_SCALE, CROP_OFFSET = 0.5, 32.0       # core/calibration/loader/AriaCalibRescaleAndCrop.cpp in projectaria_tools


def video_size(meta):
    """The source video's side in px, from its frames' meta.json. Only square videos exist in this data."""
    w, h = meta.get("orig_w"), meta.get("orig_h")
    if not w or w != h:
        raise ValueError(f"expected a square video in meta.json, got orig_w={w}, orig_h={h}")
    return int(w)


def video_calibration(cam, size):
    """The RGB camera matched to a video of this size, as Aria's VRS loader returns it. Stops on any other size pair,
    because a wrong rule would move the gaze with no error. Aria's rescale also applies its half-pixel convention, so
    this calls it rather than repeating the arithmetic."""
    have = tuple(int(v) for v in cam.get_image_size())
    if have == (size, size):
        return cam
    if have != (SENSOR_SIZE, SENSOR_SIZE) or size != VIDEO_SIZE:
        raise ValueError(f"no crop rule from a {have} calibration to a {size} px video")
    return cam.rescale(np.array([VIDEO_SIZE, VIDEO_SIZE], dtype=int), CROP_SCALE, np.array([CROP_OFFSET, CROP_OFFSET]))


def read_gaze_csv(path):
    """-> dict(t_us, pitch, yaw, left_yaw, right_yaw) as read, with None for the yaw columns this format doesn't use.
    Empty cells become NaN. A yaw_rads_cpf column holding only zeros or NaN counts as absent."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows")

    def col(name):
        return np.array([float(r[name]) if r[name] not in ("", None) else np.nan for r in rows], dtype=np.float64)

    out = dict(t_us=col("tracking_timestamp_us"), pitch=col("pitch_rads_cpf"), yaw=None, left_yaw=None, right_yaw=None)
    if "yaw_rads_cpf" in rows[0]:
        yaw = col("yaw_rads_cpf")
        if np.isfinite(yaw).any() and np.nanmax(np.abs(yaw)) > 0:
            out["yaw"] = yaw
    if out["yaw"] is None:
        out["left_yaw"], out["right_yaw"] = col("left_yaw_rads_cpf"), col("right_yaw_rads_cpf")
    return out


def align_yaw(t_us, gazes):
    """The reader's yaw for each CSV row, matched by timestamp; NaN for a row the reader doesn't return."""
    by_time = {int(round(g.tracking_timestamp.total_seconds() * 1e6)): float(g.yaw) for g in gazes}
    return np.array([by_time.get(int(t), np.nan) for t in t_us], dtype=np.float64)


def project_all(yaw, pitch, project):
    """-> x, y, oof per sample; NaN and False where a sample has no finite yaw and pitch."""
    x, y, oof = np.full(len(yaw), np.nan), np.full(len(yaw), np.nan), np.zeros(len(yaw), dtype=bool)
    for i in np.flatnonzero(np.isfinite(yaw) & np.isfinite(pitch)):
        out = project(yaw[i], pitch[i])
        x[i], y[i], oof[i] = out[0], out[1], out[2]
    return x, y, oof


def project_video(video, force=False):
    from projectaria_tools.core import calibration, mps
    out = paths.gaze_npz(video)
    if os.path.exists(out) and not force:
        print(f"   {video}: gaze already projected", flush=True)
        return
    d = paths.video_dir(video)
    path = os.path.join(d, "general_eye_gaze.csv")
    g = read_gaze_csv(path)
    if g["yaw"] is not None:
        gaze_format, yaw, yaw_mean = "one yaw", g["yaw"], np.full(len(g["yaw"]), np.nan)
    else:
        gaze_format = "left and right yaw"
        yaw, yaw_mean = align_yaw(g["t_us"], mps.read_eyegaze(path)), (g["left_yaw"] + g["right_yaw"]) / 2
    with open(os.path.join(d, "calibration.json")) as f:
        device_calib = calibration.device_calibration_from_json_string(f.read())
    if device_calib is None:
        raise ValueError(f"{video}: calibration.json did not parse")
    cam = video_calibration(device_calib.get_camera_calib("camera-rgb"), video_size(paths.frame_meta(video)))
    project = gg.calibration_projector(device_calib, depth_m=DEPTH_M, rotate_cw90=ROTATE_CW90, cam_calib=cam)
    x, y, oof = project_all(yaw, g["pitch"], project)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp.npz"
    np.savez(tmp, t_us=g["t_us"] - g["t_us"][0], yaw=yaw, pitch=g["pitch"], x=x, y=y, oof=oof, yaw_mean=yaw_mean,
             gaze_format=np.array(gaze_format))
    os.replace(tmp, out)
    usable = np.isfinite(x)
    print(f"   {video}: {gaze_format}, {len(yaw)} samples, {int(usable.sum())} usable, {100 * oof.mean():.1f}% out of view",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_video_args(ap)
    ap.add_argument("--force", action="store_true", help="project again even if the video's result exists")
    a = ap.parse_args()
    for v in paths.chosen_videos(a):
        project_video(v, a.force)


if __name__ == "__main__":
    main()

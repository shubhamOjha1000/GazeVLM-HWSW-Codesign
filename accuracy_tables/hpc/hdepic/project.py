"""Runbook section 17, step 1b: each planned video's gaze projected onto its upright MP4 frames. Runs in the `aria` environment.

    python -u -m accuracy_tables.hpc.hdepic.project

The same projection as section 9 (accuracy_tables.hpc.newvideos.project_gaze): depth 1 m, the VRS image turned 90 degrees clockwise to
the upright MP4 (HD-EPIC's "270 degrees counter-clockwise"), coordinates in [0, 1], the camera matched to the MP4's size by Aria's own
rule, which stops on any size pair it doesn't know. One difference: times stay in device time (MPS tracking_timestamp_us), not rebased,
because HD-EPIC's frames, IMU and gaze all meet through the time CSV's device clock. Writes $GV_ROOT/data/hdepic/gaze/<video>.npz.
"""
import json
import os

import numpy as np

from accuracy_tables.hpc.hdepic import data
from accuracy_tables.hpc.newvideos import project_gaze as pg
from src.common import gaze_geometry as gg


def project_video(video, info, force=False):
    from projectaria_tools.core import calibration, mps
    out = data.gaze_npz(video)
    if os.path.exists(out) and not force:
        print(f"   {video}: gaze already projected", flush=True)
        return
    if info["width"] != info["height"]:
        raise ValueError(f"{video}: a {info['width']} x {info['height']} MP4; the projection expects a square one")
    path = info["gaze_csv"]
    g = pg.read_gaze_csv(path)
    if g["yaw"] is not None:
        gaze_format, yaw = "one yaw", g["yaw"]
    else:
        gaze_format, yaw = "left and right yaw", pg.align_yaw(g["t_us"], mps.read_eyegaze(path))
    with open(os.path.join(data.video_dir(video), "calibration.json")) as f:
        device_calib = calibration.device_calibration_from_json_string(f.read())
    if device_calib is None:
        raise ValueError(f"{video}: calibration.json did not parse")
    cam = pg.video_calibration(device_calib.get_camera_calib("camera-rgb"), int(info["width"]))
    project = gg.calibration_projector(device_calib, depth_m=pg.DEPTH_M, rotate_cw90=pg.ROTATE_CW90, cam_calib=cam)
    x, y, oof = pg.project_all(yaw, g["pitch"], project)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp.npz"
    np.savez(tmp, t_us=g["t_us"], yaw=yaw, pitch=g["pitch"], x=x, y=y, oof=oof, gaze_format=np.array(gaze_format))
    os.replace(tmp, out)
    usable = np.isfinite(x)
    inside = usable & ~oof & (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
    print(f"   {video}: {gaze_format}, {len(yaw)} samples, {int(usable.sum())} usable, {100 * inside.sum() / max(1, len(yaw)):.1f}% inside "
          "the frame", flush=True)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default=None, help="a plan.json (default: section 17's); section 18 passes its own")
    a = ap.parse_args(argv)
    with open(a.plan or data.plan_json()) as f:
        plan = json.load(f)
    for v in plan["videos"]:
        project_video(v, plan["video_info"][v])
    print(f"PROJECTED: {len(plan['videos'])} videos", flush=True)


if __name__ == "__main__":
    main()

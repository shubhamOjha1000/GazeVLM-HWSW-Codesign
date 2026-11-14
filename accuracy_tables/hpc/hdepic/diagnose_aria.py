"""Runbook section 18, step 0a, in the `aria` environment (projectaria-tools and numpy; no pandas, no torch): section 17's gaze projection
checked against calibration.json and against Meta's own code. Read-only on HD-EPIC and on section 17's derived data; writes only
$GV_ROOT/data/hdepic/diagnose/aria.json, which step 0b reads. Changes no setting.

    python -u -m accuracy_tables.hpc.hdepic.diagnose_aria [--questions 20] [--motion-videos 10]

Section 17's gaze (project.py) is depth 1 m; Aria's 2880 -> 1408 crop and scale when calibration.json describes the full sensor; and
the native image turned 90 degrees clockwise to the upright MP4. Each check tests one part of that:

  calibration  per video: the RGB camera's model, image size, focal length and principal point, as calibration.json holds them and as
               matched to the MP4; and the camera's and the right IMU's rotation against the device, in degrees from AEA's (section 9's
               12 videos). A calibration already turned to the upright image would sit about 90 degrees from AEA's
  meta         20 "where looking" questions, at the gaze sample nearest the span's end: section 17's saved point against Meta's own
               reprojection (projectaria_tools.core.mps.utils.get_gaze_vector_reprojection) in the native image, and our 90-degree turn
               against Meta's own upright camera (calibration.rotate_camera_calib_cw90deg), in MP4 pixels
  gravity      per video: the direction of "up" (the right IMU's mean acceleration) in our upright image, in degrees from straight up.
               A wrong turn shows as about 90 or 180 degrees; a head's roll is far smaller
  motion       per video: how the image centre moves in our upright image per radian of rotation about each right-IMU axis. Step 0b
               compares it with the motion it measures in the MP4 itself
"""
import argparse
import json
import os
import sys

import numpy as np

from accuracy_tables.hpc.hdepic import data

DEPTH_M = 1.0                      # section 9's and section 17's depth
EPS = 1e-3                         # radians, for the motion matrix
UP_OK_DEG = 45.0                   # "up" further than this from the image's up means the turn is wrong
MATCH_PX = 1.0                     # ours and Meta's within this many MP4 pixels means the same projection


def diagnose_dir():
    return os.path.join(data.derived_root(), "diagnose")


def aria_json():
    return os.path.join(diagnose_dir(), "aria.json")


# ---------------------------------------------------------------------------- choices (step 0b uses the same ones)
def choose_questions(plan, n):
    """n "where looking" questions spread over videos: the videos ranked by how many such questions they hold (then by name), one question
    of each in plan order, then a second round, and so on."""
    by_video = {}
    for q in plan["questions"]:
        if q["type"] == "gaze_estimation":
            by_video.setdefault(q["video"], []).append(q)
    ranked = sorted(by_video, key=lambda v: (-len(by_video[v]), v))
    out, r = [], 0
    while len(out) < n and any(len(by_video[v]) > r for v in ranked):
        out += [by_video[v][r] for v in ranked if len(by_video[v]) > r][: n - len(out)]
        r += 1
    return out


def choose_motion_videos(plan, n):
    count = {}
    for q in plan["questions"]:
        if q["type"] == "gaze_estimation":
            count[q["video"]] = count.get(q["video"], 0) + 1
    return sorted(plan["videos"], key=lambda v: (-count.get(v, 0), v))[:n]


# ---------------------------------------------------------------------------- geometry, any projector
def rotation_angle_deg(Ra, Rb):
    """The angle of the rotation between two rotation matrices."""
    c = (np.trace(np.asarray(Ra, float).T @ np.asarray(Rb, float)) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def up_angle_deg(project_upright, up_cam, step=0.05):
    """Degrees between the image direction of up_cam (a vector in the camera frame) at the image centre and straight up in the upright
    image: 0 = up, +90 = right, -90 = left, 180 = down. project_upright: camera-frame point -> (x, y) in [0, 1], or None."""
    p0 = np.array([0.0, 0.0, 1.0])
    a, b = project_upright(p0), project_upright(p0 + step * np.asarray(up_cam, float) / np.linalg.norm(up_cam))
    if a is None or b is None:
        return float("nan")
    return float(np.degrees(np.arctan2(b[0] - a[0], -(b[1] - a[1]))))


def motion_matrix(project_upright, R_cam_imu, eps=EPS):
    """(2, 3): the upright image centre's motion (in units of the image side) per radian of rotation about each IMU axis. A point fixed in
    the world moves in the camera frame by -omega x p while the camera turns at omega."""
    p = np.array([0.0, 0.0, 1.0])
    a = np.asarray(project_upright(p), float)
    M = np.zeros((2, 3))
    for i in range(3):
        w = np.asarray(R_cam_imu, float)[:, i]
        M[:, i] = (np.asarray(project_upright(p - eps * np.cross(w, p)), float) - a) / eps
    return M


# ---------------------------------------------------------------------------- projectaria-tools
def load_calibration(folder):
    from projectaria_tools.core import calibration
    with open(os.path.join(folder, "calibration.json")) as f:
        dc = calibration.device_calibration_from_json_string(f.read())
    if dc is None:
        raise ValueError(f"{folder}/calibration.json did not parse")
    return dc


def rotations(dc):
    """(device <- RGB camera, device <- right IMU) rotation matrices, as calibration.json gives them."""
    cam = dc.get_camera_calib("camera-rgb")
    return (np.asarray(cam.get_transform_device_camera().rotation().to_matrix(), float),
            np.asarray(dc.get_imu_calib("imu-right").get_transform_device_imu().rotation().to_matrix(), float))


def upright_projector(cam):
    """Section 17's chain without its clip to [0, 1]: camera point -> native pixel (cam.project) -> our 90-degree turn."""
    from src.common import gaze_geometry as gg
    W, H = (int(v) for v in cam.get_image_size())

    def proj(p):
        px = cam.project(np.asarray(p, float).reshape(3, 1))
        if px is None:
            return None
        u, v = np.asarray(px, float).ravel()[:2]
        return gg.native_to_upright_cw90(u / W, v / H)
    return proj


def aea_reference():
    """AEA's camera and IMU rotations (section 9's 12 videos), and how far AEA's own videos sit from the first."""
    from accuracy_tables.hpc.newvideos import paths
    cams, imus = [], []
    for v in paths.read_list("aea12"):
        try:
            c, i = rotations(load_calibration(paths.video_dir(v)))
        except Exception:
            continue
        cams.append(c)
        imus.append(i)
    if not cams:
        return None
    return dict(cam=cams[0], imu=imus[0], videos=len(cams),
                cam_spread_deg=max(rotation_angle_deg(cams[0], c) for c in cams),
                imu_spread_deg=max(rotation_angle_deg(imus[0], i) for i in imus))


def video_checks(video, info, aea, want_motion):
    from accuracy_tables.hpc.newvideos import project_gaze as pg
    dc = load_calibration(data.video_dir(video))
    raw = dc.get_camera_calib("camera-rgb")
    cam = pg.video_calibration(raw, int(info["width"]))
    R_dev_cam, R_dev_imu = rotations(dc)
    proj = upright_projector(cam)
    out = dict(video=video, model=str(raw.get_model_name()), json_size=[int(v) for v in raw.get_image_size()],
               mp4_size=[int(info["width"]), int(info["height"])], matched_size=[int(v) for v in cam.get_image_size()],
               focal=[round(float(v), 2) for v in np.ravel(cam.get_focal_lengths())],
               principal=[round(float(v), 2) for v in np.ravel(cam.get_principal_point())],
               cam_vs_aea_deg=rotation_angle_deg(R_dev_cam, aea["cam"]) if aea else None,
               imu_vs_aea_deg=rotation_angle_deg(R_dev_imu, aea["imu"]) if aea else None)
    with np.load(os.path.join(data.video_dir(video), "imu.npz")) as z:
        acc = z["right_accel"].astype(np.float64)
    a = acc.mean(0)
    out["up_deg"] = up_angle_deg(proj, R_dev_cam.T @ (R_dev_imu @ (a / np.linalg.norm(a))))
    if want_motion:
        out["motion_M"] = motion_matrix(proj, R_dev_cam.T @ R_dev_imu).tolist()
    return out, dc, cam


def meta_points(video, qs, dc, cam):
    """Per question: the gaze sample nearest the span's end, section 17's saved point for it, ours recomputed, and Meta's own."""
    from projectaria_tools.core import calibration, mps
    from projectaria_tools.core.mps import utils as mu
    from src.common import gaze_geometry as gg
    gaze = mps.read_eyegaze(data.gaze_csv(video))
    t = np.array([g.tracking_timestamp.total_seconds() * 1e6 for g in gaze])
    with np.load(data.gaze_npz(video)) as z:
        saved_t, saved_x, saved_y = z["t_us"].astype(np.float64), z["x"], z["y"]
    W, H = (int(v) for v in cam.get_image_size())
    T_dev_cpf, T_dev_cam = dc.get_transform_device_cpf(), cam.get_transform_device_camera()
    try:
        cam_up, up_error = calibration.rotate_camera_calib_cw90deg(cam), None
    except Exception as e:                                                   # older projectaria-tools
        cam_up, up_error = None, repr(e)[:160]
    px_list = lambda px: None if px is None else [float(v) for v in np.asarray(px, float).ravel()[:2]]
    rows = []
    for q in qs:
        target = float(q["device_us"][-1])
        k = int(np.argmin(np.abs(t - target)))
        g = gaze[k]
        j = int(np.argmin(np.abs(saved_t - t[k])))
        row = dict(qid=q["qid"], pseudo=q["pseudo"], hdepic_qid=q["hdepic_qid"], video=video, sample_ms_from_end=round((t[k] - target) / 1e3, 1),
                   saved_matches_sample=bool(abs(saved_t[j] - t[k]) < 1.0), saved=[float(saved_x[j]), float(saved_y[j])], size=[W, H])
        p_dev = T_dev_cpf @ mps.get_eyegaze_point_at_depth(g.yaw, g.pitch, DEPTH_M)
        native = px_list(cam.project(T_dev_cam.inverse() @ p_dev))
        row["ours_native_px"] = native
        row["ours_upright_px"] = None if native is None else [v * s for v, s in zip(gg.native_to_upright_cw90(native[0] / W, native[1] / H), (W, H))]
        try:
            row["meta_native_px"] = px_list(mu.get_gaze_vector_reprojection(g, "camera-rgb", dc, cam, DEPTH_M))
        except Exception as e:
            row["meta_native_error"] = repr(e)[:160]
        if cam_up is not None:
            row["meta_upright_px"] = px_list(cam_up.project(cam_up.get_transform_device_camera().inverse() @ p_dev))
        else:
            row["meta_upright_error"] = up_error
        rows.append(row)
    return rows


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1])) if a is not None and b is not None else float("nan")


def summarize(questions, videos):
    """FINDING lines from the checks."""
    lines = []
    nat = [dist(q.get("ours_native_px"), q.get("meta_native_px")) for q in questions]
    up = [dist(q.get("ours_upright_px"), q.get("meta_upright_px")) for q in questions]
    saved = [dist(q.get("ours_upright_px"), [q["saved"][0] * q["size"][0], q["saved"][1] * q["size"][1]]) for q in questions
             if q.get("saved_matches_sample")]
    worst = lambda x: float(np.nanmax(x)) if np.isfinite(x).any() else float("nan")
    up_known = bool(np.isfinite(up).any())
    ok = np.isfinite(nat).any() and worst(nat) <= MATCH_PX and (not up_known or worst(up) <= MATCH_PX)
    lines.append(f"FINDING projection: on {len(questions)} questions ours is at most {worst(nat):.2f} px from Meta's own reprojection in "
                 "the native image and " + (f"at most {worst(up):.2f} px from Meta's own upright camera" if up_known else
                                            "Meta's upright camera isn't in this projectaria-tools") +
                 f"; section 17's saved points are at most {worst(saved):.2f} px from ours recomputed (0 unless clipped at the edge): "
                 + ("SAME AS META'S" if ok else "DIFFERS FROM META'S"))
    ups = np.array([v["up_deg"] for v in videos.values()], float)
    fine = np.isfinite(ups)
    if fine.any():
        bad = [k for k, v in videos.items() if not np.isfinite(v["up_deg"]) or abs(v["up_deg"]) > UP_OK_DEG]
        lines.append(f"FINDING gravity: up is a median {np.median(ups[fine]):+.1f} deg from the upright image's up (worst "
                     f"{ups[fine][np.argmax(np.abs(ups[fine]))]:+.1f}) over {int(fine.sum())} videos: "
                     + ("TURN OK" if not bad else f"TURN WRONG ON {len(bad)}: {bad[:8]}"))
    cams = np.array([v["cam_vs_aea_deg"] for v in videos.values() if v.get("cam_vs_aea_deg") is not None], float)
    if len(cams):
        lines.append(f"FINDING calibration: the RGB camera's rotation is a median {np.median(cams):.1f} deg (max {cams.max():.1f}) from AEA's; "
                     f"sizes in calibration.json {sorted({tuple(v['json_size']) for v in videos.values()})}, matched to "
                     f"{sorted({tuple(v['matched_size']) for v in videos.values()})}, models {sorted({v['model'] for v in videos.values()})}")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=int, default=20)
    ap.add_argument("--motion-videos", type=int, default=10)
    a = ap.parse_args(argv)
    out = aria_json()
    data.refuse_inside_raw(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plan = data.read_plan()
    qs = choose_questions(plan, a.questions)
    motion = set(choose_motion_videos(plan, a.motion_videos))
    aea = aea_reference()
    if aea:
        print(f"AEA reference: {aea['videos']} videos; their cameras sit up to {aea['cam_spread_deg']:.2f} deg and their right IMUs up to "
              f"{aea['imu_spread_deg']:.2f} deg from the first's", flush=True)
    else:
        print("AEA reference: no AEA calibration.json found; the rotations aren't compared", flush=True)
    videos, questions = {}, []
    for v in plan["videos"]:
        try:
            row, dc, cam = video_checks(v, plan["video_info"][v], aea, v in motion)
        except Exception as e:
            print(f"   {v}: {e!r}", flush=True)
            continue
        videos[v] = row
        mine = [q for q in qs if q["video"] == v]
        if mine:
            questions += meta_points(v, mine, dc, cam)
        print(f"   {v}: {row['model']}, json {row['json_size']} -> {row['matched_size']}, f {row['focal']}, c {row['principal']}, camera "
              f"{row['cam_vs_aea_deg'] if row['cam_vs_aea_deg'] is None else round(row['cam_vs_aea_deg'], 2)} deg from AEA's, up "
              f"{row['up_deg']:+.1f} deg" + (", motion matrix" if "motion_M" in row else ""), flush=True)
    print("\n--- the 20 questions: our point and Meta's, in MP4 pixels (native = before the turn; upright = after)", flush=True)
    for q in questions:
        print(f"   {q['pseudo']} {q['video']}: sample {q['sample_ms_from_end']:+.1f} ms from the span's end; ours native "
              f"{q.get('ours_native_px')}, Meta's {q.get('meta_native_px', q.get('meta_native_error'))}; ours upright "
              f"{q.get('ours_upright_px')}, Meta's {q.get('meta_upright_px', q.get('meta_upright_error'))}; saved {q['saved']}", flush=True)
    findings = summarize(questions, videos)
    with open(out + ".tmp", "w") as f:
        json.dump(dict(questions=questions, videos=videos, motion_videos=sorted(motion & set(videos)), aea=None if not aea else
                       {k: v for k, v in aea.items() if k not in ("cam", "imu")}, findings=findings), f)
    os.replace(out + ".tmp", out)
    for line in findings:
        print(line, flush=True)
    print(f"written {out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)

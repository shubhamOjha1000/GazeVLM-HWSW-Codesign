import argparse, os, json, yaml
import numpy as np, pandas as pd, torch, cv2
from src.common import aria_io as io
from src.common import gaze_geometry as gg
from src.common import encoders as enc


def parse_args():
    ap = argparse.ArgumentParser("Temporal frame-vs-gaze similarity CSV builder")
    ap.add_argument("--urls_json", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--config", default="configs/temporal_analysis.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--projection", choices=["pinhole", "calibration"], default=None)
    ap.add_argument("--n_viz", type=int, default=3,
                    help="how many consecutive-pair visualizations to save per sequence")
    ap.add_argument("--viz_dir", default=None, help="where to save viz PNGs (default: <frames_dir>/viz)")
    ap.add_argument("--verify_only", action="store_true")
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = yaml.safe_load(open(a.config))
    fov_x, fov_y = np.deg2rad(cfg["fov_x_deg"]), np.deg2rad(cfg["fov_y_deg"])
    meta = json.load(open(a.urls_json))
    projection = a.projection or cfg.get("projection", "calibration")
    viz_dir = a.viz_dir or os.path.join(a.frames_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    rows = []
    for seq in a.seqs:
        # VRS is fetched for calibration AND IMU when calibration projection is requested
        want_vrs = (projection == "calibration")
        seq_dir, mp4, vrs_path = io.download_sequence(seq, meta, a.raw_dir, want_calibration=want_vrs)

        g_ts, g_yaw, g_pit = io.load_gaze_raw(seq_dir)
        imu_ts, imu_arr = io.load_imu_from_vrs(vrs_path)   # IMU straight from the VRS stream
        print(f"[{seq}] IMU: {'MISSING' if imu_ts is None else f'{len(imu_ts)} samples from VRS'}")
        paths, f_ts, native = io.subsample_frames(
            mp4, os.path.join(a.frames_dir, seq), target_fps=cfg["target_fps"])

        rate = gg.measure_gaze_rate(g_ts)
        print(f"[{seq}] gaze ~{rate['implied_hz']:.1f} Hz ({rate['n_samples']} samples); "
              f"video native fps={native:.1f}")

        if projection == "calibration":
            assert vrs_path and os.path.exists(vrs_path), f"[{seq}] need VRS for calibration"
            project = gg.make_gaze_projector("calibration", vrs_path,
                                             depth_m=cfg.get("gaze_depth_m", 1.0),
                                             rotate_cw90=cfg.get("rotate_cw90", True))
        else:
            project = gg.make_gaze_projector("pinhole", None, fov_x, fov_y,
                                             cfg["yaw_sign"], cfg["pitch_sign"])

        cls_list, patch_list, grid_list, gxy = [], [], [], []
        oof_fallback = 0
        for p, t in zip(paths, f_ts):
            frame = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            j = int(np.clip(np.searchsorted(g_ts, t), 0, len(g_ts) - 1))
            out = project(g_yaw[j], g_pit[j])
            if projection == "calibration":
                xy = out[:2]; oof_fallback += int(out[2])
            else:
                xy = out
            cls, patches, grid = enc.frame_features(frame, a.device, cfg["dino_input_patch"])
            cls_list.append(cls); patch_list.append(patches); grid_list.append(grid)
            gxy.append((xy, g_yaw[j], g_pit[j], frame))
        if projection == "calibration":
            print(f"[{seq}] out-of-FOV gaze fallback: {oof_fallback}/{len(paths)} "
                  f"({100*oof_fallback/max(1,len(paths)):.1f}%)")

        # ---- ALWAYS save a few visualizations (not just under --verify_only) ----
        n_viz = min(a.n_viz, max(0, len(paths) - 1))
        for k in range(1, n_viz + 1):
            (xy0, _, _, fa), (xy1, _, _, fb) = gxy[k-1], gxy[k]
            fs = enc.cosine(cls_list[k-1], cls_list[k])
            pt0 = enc.patch_token_at_gaze(patch_list[k-1], grid_list[k-1], xy0)
            pt1 = enc.patch_token_at_gaze(patch_list[k],   grid_list[k],   xy1)
            enc.show_pair(fa, fb, xy0, xy1, fs, enc.cosine(pt0, pt1),
                          save_path=os.path.join(viz_dir, f"{seq}_pair_{k:03d}.png"),
                          grid=grid_list[k])
        if a.verify_only:
            continue   # skip CSV when only verifying

        for i in range(1, len(paths)):
            (xy0, y0, p0, _), (xy1, y1, p1, _) = gxy[i-1], gxy[i]
            frame_sim = enc.cosine(cls_list[i-1], cls_list[i])
            pt0 = enc.patch_token_at_gaze(patch_list[i-1], grid_list[i-1], xy0)
            pt1 = enc.patch_token_at_gaze(patch_list[i],   grid_list[i],   xy1)
            patch_sim = enc.cosine(pt0, pt1)

            rat = gg.window_gaze_rates(g_ts, g_yaw, g_pit, f_ts[i-1], f_ts[i])
            win = gg.window_gaze_samples(g_ts, g_yaw, g_pit, f_ts[i-1], f_ts[i], project=project)
            imu_win = io.window_imu_samples(imu_ts, imu_arr, f_ts[i-1], f_ts[i])
            vec = gg.yawpitch_to_unit_vec(y1, p1)

            rows.append(dict(
                idx=len(rows), sequence=seq,
                path_frame_1=paths[i-1], path_frame_2=paths[i],
                n_gaze_in_gap=win["n"],
                n_oof_in_gap=win["n_oof"],   # gaze samples outside the lens -> (0.5,0.5) filler
                yaw_pitch_window=";".join(f"{y:.5f},{p:.5f}" for y, p in zip(win["yaw"], win["pitch"])),
                vec3d_window=";".join(f"{v[0]:.5f},{v[1]:.5f},{v[2]:.5f}" for v in win["vec3d"]),
                xy_window=";".join(f"{x:.4f},{y:.4f}" for x, y in zip(win["x"], win["y"])),
                n_velocities=rat["n_velocities"],
                # (n-1) x 3 signal: omega_yaw, omega_pitch, omega_mag  (rad/s), one
                # comma-separated triple per step, steps separated by ';'
                gaze_rates_window=";".join(
                    f"{r[0]:.5f},{r[1]:.5f},{r[2]:.5f}" for r in rat["rates"]),
                n_imu_in_gap=imu_win["n"],
                imu_window=";".join(",".join(f"{v:.5f}" for v in r) for r in imu_win["rows"]),
                vec_3d=f"{vec[0]:.5f},{vec[1]:.5f},{vec[2]:.5f}",
                yaw=round(float(y1), 5), pitch=round(float(p1), 5),
                x=round(xy1[0], 4), y=round(xy1[1], 4),
                gaze_patch_token_sim=round(patch_sim, 4),
                frame_similarity=round(frame_sim, 4),
            ))

        del cls_list, patch_list, grid_list, gxy
        if a.device == "cuda":
            torch.cuda.empty_cache()

    if not a.verify_only:
        os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)
        pd.DataFrame(rows).to_csv(a.out_csv, index=False)
        print(f"wrote {len(rows)} rows -> {a.out_csv}")


if __name__ == "__main__":
    main()
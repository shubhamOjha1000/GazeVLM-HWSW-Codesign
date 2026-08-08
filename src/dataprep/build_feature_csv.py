"""Build the Loss-1 / Loss-2 feature dataset from the AEA download-links JSON.

Samples N videos, runs the temporal pipeline on each, saves one DINOv2 feature file per
frame, and writes a single CSV whose rows `src.loss1.dataset.Loss1Dataset` consumes
directly:

    idx, sequence, feat_frame_1, feat_frame_2,
    gaze_patch_token_sim, frame_similarity, n_velocities, gaze_rates_window,
    n_gaze_in_gap, n_oof_in_gap, gaze_xy_window, gaze_vec3d_rates_window,
    n_imu_in_gap, imu_window, imu_accel_mag_mean, imu_gyro_mag_mean

The gaze window is described two ways -- WHERE the eye pointed and HOW FAST it moved:

    gaze_xy_window           "x,y;..."      ~10 x 2, projected image coords in [0, 1]
    gaze_vec3d_rates_window  "x,y,z;..."     ~9 x 3, d/dt of the 3D unit gaze vector, 1/s
    n_gaze_in_gap            how many samples the window actually holds
    n_oof_in_gap             how many of them fell OUTSIDE the lens (see below)

`gaze_vec3d_rates_window` is the 3D counterpart of `gaze_rates_window` and has the same
n-1 length, so the two line up step for step. It is the better of the two for speed:
because the gaze vector lies on the unit sphere, the NORM of its three channels is the
exact angular speed, whereas `gaze_rates_window`'s omega_mag treats yaw and pitch as
orthogonal and overstates the yaw term by 1/cos(pitch) -- ~33% at the -41 deg pitch this
dataset reaches. Its three channels are not independent: dv/dt is perpendicular to v, so
they carry two degrees of freedom, not three.

`n_oof_in_gap` matters: out-of-FOV samples are written as (0.5, 0.5), which is
indistinguishable from a genuine centre gaze. Rows with a non-zero count have invented
coordinates in `gaze_xy_window` and should be filtered or masked, not trusted. Both rate
columns are unaffected -- they are pure trigonometry on yaw/pitch and need no camera
model, so they are always real.

The IMU columns come from the VRS, which is already downloaded for the calibration, so
they cost extraction time but no extra bandwidth:

    n_imu_in_gap        RAW IMU samples in the window (~800-1000 at Aria's rate)
    imu_window          "ax,ay,az,gx,gy,gz;..."  --imu_hz bins x 6, m/s^2 and rad/s
    imu_accel_mag_mean  mean ||accel|| over the RAW samples
    imu_gyro_mag_mean   mean ||gyro||  over the RAW samples

`imu_window` is BINNED, not raw: at ~1 kHz a 1 s window holds ~1000x6 values, which is
~50 KB per CSV row and over a GB across all 143 sequences. `--imu_hz 10` (the default)
averages into 10 bins so the column lines up one-to-one with `gaze_xy_window`. Pass
`--imu_hz 0` for every raw sample, and check the resulting file size before scaling up.

The two magnitude columns are computed on the UNBINNED samples on purpose: averaging
vectors within a bin cancels equal-and-opposite motion, so a mean of the binned rows
would understate how much the head actually moved.

Unpack any of the packed columns with `src.loss1.dataset.parse_rates`, which splits on
';' then ',' and so handles the 2-, 3- and 6-wide forms alike.

Unlike `build_similarity_csv.py`, which stores paths to JPEG frames, this stores paths to
**encoded features** -- one .npz per frame holding the CLS token, the patch grid, the
projected gaze point and the timestamp. Consecutive rows share files: frame t is written
once and referenced twice, as feat_frame_2 of one row and feat_frame_1 of the next.

The consequence is that DINOv2, the video and the VRS are needed exactly once. Training
reads only the .npz files, so everything under --raw_dir and --frames_dir is disposable
(--cleanup_raw does it during the run).

Example
-------
    python -m src.dataprep.build_feature_csv \
        --urls_json  aea_download_urls.json \
        --out_csv    data/feature_dataset.csv \
        --raw_dir    data/raw \
        --frames_dir data/frames_1fps \
        --feat_dir   data/features \
        --n_videos 5 --cleanup_raw

Growing an existing dataset
---------------------------
Point --exclude_csv at what you already have and the new sample cannot overlap it:

    python -m src.dataprep.build_feature_csv \
        --urls_json aea_download_urls.json \
        --out_csv    data/batch2.csv \
        --exclude_csv data/batch1.csv \
        --n_videos 40 --resume --cleanup_raw

Exclusion happens BEFORE sampling, so --n_videos 40 yields 40 new videos rather than 40
minus however many collided. The CSV is checkpointed after every video and --resume skips
sequences already in it, so an interrupted multi-hour build restarts where it stopped
instead of starting over.

By default every video is used in full. AEA sequences average ~193 s (range 67-456 s), so
at 1 FPS that is ~190 rows each. Pass --max_seconds only to trim a debug run: capping
discards footage the download has already paid for.
"""

import argparse
import json
import os
import random
import shutil
import time

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from src.common import aria_io as io
from src.common import encoders as enc
from src.common import gaze_geometry as gg


def parse_args():
    ap = argparse.ArgumentParser("Feature-CSV builder for Loss-1 / Loss-2 training")
    ap.add_argument("--urls_json", required=True, help="AEA download-links JSON")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--feat_dir", required=True, help="where the per-frame .npz files go")
    ap.add_argument("--config", default="configs/temporal_analysis.yaml")

    ap.add_argument("--n_videos", type=int, default=5, help="how many videos to sample")
    ap.add_argument("--seqs", nargs="+", default=None,
                    help="pin explicit sequence names instead of sampling")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--exclude", nargs="+", default=None,
                    help="sequence names to keep OUT of the sample")
    ap.add_argument("--exclude_csv", nargs="+", default=None,
                    help="CSVs whose `sequence` values to exclude. Point this at the "
                         "dataset you already built and the new sample cannot overlap "
                         "it -- safer than re-listing names by hand.")
    ap.add_argument("--resume", action="store_true",
                    help="if --out_csv already exists, keep its rows and skip the "
                         "sequences in it. The CSV is checkpointed after every video, "
                         "so an interrupted multi-hour build restarts where it stopped.")

    ap.add_argument("--target_fps", type=float, default=1.0)
    ap.add_argument("--max_seconds", type=float, default=0.0,
                    help="cap how many seconds of each video to use; 0 (the default) "
                         "means the whole video. Capping discards footage already paid "
                         "for in the download, so only set it to trim run time.")
    ap.add_argument("--imu_hz", type=float, default=10.0,
                    help="bins per second for imu_window; 10 (the default) matches the "
                         "~10 gaze samples per window. 0 keeps every raw sample, which "
                         "is ~50 KB per CSV row -- check the file size before scaling.")
    ap.add_argument("--no_imu", action="store_true",
                    help="skip IMU extraction entirely (saves ~30-60 s per video)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cleanup_raw", action="store_true",
                    help="delete each video's raw files once its features are written, "
                         "so peak disk stays at ~1 video instead of n_videos")
    return ap.parse_args()


def write_csv(rows, path):
    """Write the table, renumbering `idx` to 0..N-1.

    idx is assigned at write time rather than trusted from the row dicts because a
    --resume run inherits rows from an earlier file: if that file was ever filtered or
    concatenated, its idx values need not be contiguous, and continuing from len(rows)
    would then duplicate or skip indices. Renumbering makes idx exactly the row number,
    which is all anything downstream uses it for.
    """
    df = pd.DataFrame(rows)
    if len(df):
        df["idx"] = range(len(df))
    df.to_csv(path, index=False)
    return df


def pick_sequences(meta, n_videos, seed, explicit=None, exclude=None):
    """Sample without replacement from the videos NOT already excluded.

    Excluding before sampling (rather than sampling then filtering) is what makes the
    result deterministic in the size of the request: ask for 40 and you get 40 new ones,
    not 40 minus however many collided.
    """
    seqs = list(meta["sequences"])
    exclude = set(exclude or ())
    if explicit:
        chosen = explicit
        missing = [s for s in chosen if s not in meta["sequences"]]
        if missing:
            raise KeyError(f"sequences not in the JSON: {missing}")
        clash = [s for s in chosen if s in exclude]
        if clash:
            print(f"!! {len(clash)} pinned sequences are also excluded, dropping: {clash}")
            chosen = [s for s in chosen if s not in exclude]
    else:
        pool = [s for s in seqs if s not in exclude]
        if exclude:
            print(f"excluding {len(exclude)} already-built videos -> "
                  f"{len(pool)} of {len(seqs)} available")
        if n_videos > len(pool):
            raise ValueError(
                f"asked for {n_videos} videos but only {len(pool)} remain after "
                f"excluding {len(exclude)}. Lower --n_videos.")
        random.seed(seed)
        chosen = random.sample(pool, n_videos)
        overlap = set(chosen) & exclude          # cannot happen; assert it anyway
        assert not overlap, f"sampler returned excluded sequences: {overlap}"

    total = 0
    print(f"selected {len(chosen)} of {len(seqs)} videos"
          f"{'' if explicit else f' (seed {seed})'}:\n")
    for i, s in enumerate(chosen, 1):
        e = meta["sequences"][s]
        for k in ("video_main_rgb", "mps_eye_gaze", "main_vrs"):
            if k not in e:
                raise KeyError(f"{s}: missing required entry '{k}'")
        mb = sum(e[k]["file_size_bytes"] for k in
                 ("video_main_rgb", "mps_eye_gaze", "main_vrs")) / 1e6
        total += mb
        print(f"   {i}. {s:30s} {mb:9,.0f} MB")
    print(f"\n   total download: {total/1e3:.1f} GB")
    return chosen


def encode_and_save(seq, feat_dir, paths, f_ts, g_ts, g_yaw, g_pit, project,
                    device, dino_size):
    """One DINOv2 pass per frame; write a self-contained .npz beside it.

    Returns the in-memory tokens (so the pair loop needs no reload), the file paths, the
    per-frame gaze point, the grid size, and the out-of-FOV count.
    """
    out_dir = os.path.join(feat_dir, seq)
    os.makedirs(out_dir, exist_ok=True)

    cls_l, pat_l, gxy, feat_paths, oof = [], [], [], [], 0
    grid = None
    for k, (p, t) in enumerate(zip(paths, f_ts)):
        frame = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        j = int(np.clip(np.searchsorted(g_ts, t), 0, len(g_ts) - 1))
        out = project(g_yaw[j], g_pit[j])
        x, y = float(out[0]), float(out[1])
        o = bool(out[2]) if len(out) > 2 else False
        oof += int(o)

        cls, patches, grid = enc.frame_features(frame, device, dino_size)

        fp = os.path.join(out_dir, f"feat_{k:05d}.npz")
        np.savez_compressed(
            fp,
            cls=cls.cpu().numpy().astype(np.float32),
            patches=patches.cpu().numpy().astype(np.float32),
            grid=np.int32(grid),
            gaze_xy=np.array([x, y], dtype=np.float32),
            t_us=np.float64(t),
        )
        feat_paths.append(fp)
        cls_l.append(cls); pat_l.append(patches); gxy.append((x, y, o))

    return cls_l, pat_l, gxy, feat_paths, grid, oof


def main():
    a = parse_args()
    cfg = yaml.safe_load(open(a.config)) if os.path.exists(a.config) else {}
    dino_size = int(cfg.get("dino_input_patch", 98))
    depth_m = float(cfg.get("gaze_depth_m", 1.0))
    rotate_cw90 = bool(cfg.get("rotate_cw90", True))
    max_seconds = a.max_seconds if a.max_seconds and a.max_seconds > 0 else None

    meta = json.load(open(a.urls_json))

    # ---- what must NOT be sampled -------------------------------------------------
    exclude = set(a.exclude or ())
    for p in (a.exclude_csv or ()):
        if not os.path.exists(p):
            raise FileNotFoundError(f"--exclude_csv: no such file {p}")
        got = set(pd.read_csv(p, usecols=["sequence"])["sequence"].unique())
        print(f"excluding {len(got)} sequences already in {p}")
        exclude |= got

    # ---- resume an interrupted run ------------------------------------------------
    rows = []
    if a.resume and os.path.exists(a.out_csv):
        prev = pd.read_csv(a.out_csv)
        done = set(prev["sequence"].unique())
        rows = prev.to_dict("records")
        exclude |= done
        print(f"RESUMING {a.out_csv}: {len(prev)} rows from {len(done)} videos kept, "
              f"those videos will be skipped")

    chosen = pick_sequences(meta, a.n_videos, a.seed, a.seqs, exclude)

    for d in (a.raw_dir, a.frames_dir, a.feat_dir):
        os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)

    per_seq = []
    t_all = time.time()

    for si, seq in enumerate(chosen, 1):
        print(f"\n{'='*78}\n[{si}/{len(chosen)}]  {seq}\n{'='*78}")
        t_seq = time.time()

        # ---- Stage 0: acquisition ------------------------------------------------
        seq_dir, mp4, vrs_path = io.download_sequence(seq, meta, a.raw_dir,
                                                      want_calibration=True)

        # ---- Stage 1: 1 FPS frames + the raw gaze stream -------------------------
        cap = cv2.VideoCapture(mp4)
        native = cap.get(cv2.CAP_PROP_FPS) or 20.0
        cap.release()
        max_frames = int(max_seconds * native) if max_seconds else None
        paths, f_ts, native = io.subsample_frames(
            mp4, os.path.join(a.frames_dir, seq), a.target_fps, max_frames)
        g_ts, g_yaw, g_pit = io.load_gaze_raw(seq_dir)
        print(f"   frames {len(paths)}   gaze {len(g_ts):,} @ "
              f"{1/np.median(np.diff(g_ts)/1e6):.1f} Hz over {g_ts[-1]/1e6:.0f} s")

        # ---- IMU, from the VRS already on disk for the calibration ---------------
        imu_ts, imu_arr = (None, None)
        if not a.no_imu:
            t_imu = time.time()
            imu_ts, imu_arr = io.load_imu_from_vrs(vrs_path)
            if imu_ts is None:
                print("   IMU MISSING -- imu columns will be empty for this sequence")
            else:
                r = io.measure_imu_rate(imu_ts)
                print(f"   imu    {r['n_samples']:,} @ {r['implied_hz']:.0f} Hz over "
                      f"{r['span_s']:.0f} s   ({time.time()-t_imu:.0f}s to extract)")
                # the three streams are each rebased to their own first sample, so a
                # span mismatch means the frame<->IMU join is off, not just noisy
                drift = abs(r["span_s"] - g_ts[-1] / 1e6)
                if drift > 2.0:
                    print(f"   !! imu span differs from gaze by {drift:.1f}s -- the "
                          f"windows may not line up")

        # ---- the projector is per sequence: calibration is device-specific -------
        project = gg.make_gaze_projector("calibration", vrs_path,
                                         depth_m=depth_m, rotate_cw90=rotate_cw90)

        # ---- Stage 2: encode once, save features --------------------------------
        cls_l, pat_l, gxy, feat_paths, grid, oof = encode_and_save(
            seq, a.feat_dir, paths, f_ts, g_ts, g_yaw, g_pit, project,
            a.device, dino_size)

        # ---- Stages 3 + 4: similarities and the 3-channel rate signal -----------
        n_before = len(rows)
        for i in range(1, len(paths)):
            frame_sim = enc.cosine(cls_l[i-1], cls_l[i])
            pt0 = enc.patch_token_at_gaze(pat_l[i-1], grid, gxy[i-1][:2])
            pt1 = enc.patch_token_at_gaze(pat_l[i],   grid, gxy[i][:2])
            patch_sim = enc.cosine(pt0, pt1)

            rat = gg.window_gaze_rates(g_ts, g_yaw, g_pit, f_ts[i-1], f_ts[i])
            # the raw samples the rates were differenced FROM: ~10 per 1 s window.
            # Reuses the same `project` object as the per-frame gaze, so the CPF->camera
            # transform and the native->upright rotation apply here too -- there is only
            # one projection implementation.
            win = gg.window_gaze_samples(g_ts, g_yaw, g_pit, f_ts[i-1], f_ts[i],
                                         project=project)
            vrat = gg.window_gaze_vec_rates(g_ts, g_yaw, g_pit, f_ts[i-1], f_ts[i])

            # ---- IMU for the same window ------------------------------------
            n_bins = int(round(a.imu_hz * (f_ts[i] - f_ts[i-1]) / 1e6))
            if a.imu_hz > 0 and n_bins > 0:
                iw = io.window_imu_binned(imu_ts, imu_arr, f_ts[i-1], f_ts[i], n_bins)
                imu_rows, n_imu = iw["rows"], iw["n_raw"]
            else:                                   # --imu_hz 0 -> every raw sample
                iw = io.window_imu_samples(imu_ts, imu_arr, f_ts[i-1], f_ts[i])
                imu_rows, n_imu = iw["rows"], iw["n"]

            # magnitudes on the UNBINNED samples: averaging vectors within a bin
            # cancels opposing motion and would understate the true head movement
            a_mag = g_mag = float("nan")
            if imu_ts is not None and n_imu:
                m = (imu_ts >= f_ts[i-1]) & (imu_ts < f_ts[i])
                raw = imu_arr[m]
                a_mag = float(np.linalg.norm(raw[:, :3], axis=1).mean())
                g_mag = float(np.linalg.norm(raw[:, 3:], axis=1).mean())

            rows.append(dict(
                idx=len(rows),                       # GLOBAL, across all videos
                sequence=seq,
                feat_frame_1=feat_paths[i-1],
                feat_frame_2=feat_paths[i],
                gaze_patch_token_sim=round(patch_sim, 4),
                frame_similarity=round(frame_sim, 4),
                n_velocities=rat["n_velocities"],
                gaze_rates_window=";".join(
                    f"{r[0]:.5f},{r[1]:.5f},{r[2]:.5f}" for r in rat["rates"]),
                # --- raw per-sample gaze: n entries, vs n-1 above --------------------
                n_gaze_in_gap=win["n"],
                n_oof_in_gap=win["n_oof"],           # (0.5,0.5) fillers -> do not trust
                gaze_xy_window=";".join(
                    f"{x:.4f},{y:.4f}" for x, y in zip(win["x"], win["y"])),
                gaze_vec3d_rates_window=";".join(
                    f"{r[0]:.5f},{r[1]:.5f},{r[2]:.5f}" for r in vrat["rates"]),
                # --- IMU: accel xyz (m/s^2) then gyro xyz (rad/s) --------------------
                n_imu_in_gap=n_imu,                  # RAW count, not the number of bins
                imu_window=";".join(
                    ",".join(f"{v:.5f}" for v in r) for r in imu_rows),
                imu_accel_mag_mean=round(a_mag, 5),
                imu_gyro_mag_mean=round(g_mag, 5),
            ))

        n_rows = len(rows) - n_before
        feat_mb = sum(os.path.getsize(f) for f in feat_paths) / 1e6
        new = rows[n_before:]
        n_samp = sum(r["n_gaze_in_gap"] for r in new)
        per_seq.append(dict(sequence=seq, frames=len(paths), rows=n_rows,
                            oof_frames=oof,
                            gaze_samples=n_samp,
                            # per-sample out-of-FOV rate; a high value means many
                            # (0.5, 0.5) fillers sit in gaze_xy_window
                            oof_samples=sum(r["n_oof_in_gap"] for r in new),
                            samples_per_row=round(n_samp / max(1, n_rows), 1),
                            imu_per_row=round(
                                sum(r["n_imu_in_gap"] for r in new) / max(1, n_rows), 1),
                            feat_MB=round(feat_mb, 1),
                            secs=round(time.time() - t_seq, 1)))
        # CHECKPOINT after every video. A 40-video build runs for hours and Colab
        # disconnects; without this a crash at video 38 would discard all 38.
        write_csv(rows, a.out_csv)

        done_n, left_n = si, len(chosen) - si
        eta = (time.time() - t_all) / max(1, done_n) * left_n / 60
        print(f"   -> {n_rows} rows   (running total {len(rows)})   "
              f"features {feat_mb:.1f} MB   {time.time()-t_seq:.0f}s")
        print(f"   checkpointed {len(rows)} rows -> {a.out_csv}"
              f"   [{done_n}/{len(chosen)} done, ~{eta:.0f} min left]")

        if a.cleanup_raw:
            shutil.rmtree(seq_dir, ignore_errors=True)
            shutil.rmtree(os.path.join(a.frames_dir, seq), ignore_errors=True)
            print(f"   cleaned up raw + frames for {seq}")

        del cls_l, pat_l
        if a.device == "cuda":
            torch.cuda.empty_cache()

    df = write_csv(rows, a.out_csv)

    print(f"\n{'='*78}\nDONE in {time.time()-t_all:.0f}s\n{'='*78}\n")
    print(pd.DataFrame(per_seq).to_string(index=False))
    print(f"\nTOTAL ROWS = {len(df)}   (idx 0 .. {len(df)-1})")
    print(f"   videos  : {df['sequence'].nunique()}")
    print(f"   columns : {df.shape[1]}")
    print(f"   CSV     : {a.out_csv}  ({os.path.getsize(a.out_csv)/1e3:.0f} KB)")
    print(f"   idx contiguous: {list(df['idx']) == list(range(len(df)))}")
    print(f"\nfeed it to Loss 1 with:\n"
          f"   python -m src.loss1.train --csv {a.out_csv} --out_dir runs/loss1")


if __name__ == "__main__":
    main()

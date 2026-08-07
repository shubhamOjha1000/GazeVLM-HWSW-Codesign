"""Build the Loss-1 / Loss-2 feature dataset from the AEA download-links JSON.

Samples N videos, runs the temporal pipeline on each, saves one DINOv2 feature file per
frame, and writes a single CSV whose rows `src.loss1.dataset.Loss1Dataset` consumes
directly:

    idx, sequence, feat_frame_1, feat_frame_2,
    gaze_patch_token_sim, frame_similarity, n_velocities, gaze_rates_window

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
        --n_videos 5 --max_seconds 90 --cleanup_raw
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

    ap.add_argument("--target_fps", type=float, default=1.0)
    ap.add_argument("--max_seconds", type=float, default=90.0,
                    help="cap per video; 0 or negative means the whole video")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cleanup_raw", action="store_true",
                    help="delete each video's raw files once its features are written, "
                         "so peak disk stays at ~1 video instead of n_videos")
    return ap.parse_args()


def pick_sequences(meta, n_videos, seed, explicit=None):
    """Sample without replacement, then assert every required entry is present."""
    seqs = list(meta["sequences"])
    if explicit:
        chosen = explicit
        missing = [s for s in chosen if s not in meta["sequences"]]
        if missing:
            raise KeyError(f"sequences not in the JSON: {missing}")
    else:
        random.seed(seed)
        chosen = random.sample(seqs, min(n_videos, len(seqs)))

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
    chosen = pick_sequences(meta, a.n_videos, a.seed, a.seqs)

    for d in (a.raw_dir, a.frames_dir, a.feat_dir):
        os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)

    rows, per_seq = [], []
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
            ))

        n_rows = len(rows) - n_before
        feat_mb = sum(os.path.getsize(f) for f in feat_paths) / 1e6
        per_seq.append(dict(sequence=seq, frames=len(paths), rows=n_rows,
                            oof_frames=oof, feat_MB=round(feat_mb, 1),
                            secs=round(time.time() - t_seq, 1)))
        print(f"   -> {n_rows} rows   (running total {len(rows)})   "
              f"features {feat_mb:.1f} MB   {time.time()-t_seq:.0f}s")

        if a.cleanup_raw:
            shutil.rmtree(seq_dir, ignore_errors=True)
            shutil.rmtree(os.path.join(a.frames_dir, seq), ignore_errors=True)
            print(f"   cleaned up raw + frames for {seq}")

        del cls_l, pat_l
        if a.device == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(a.out_csv, index=False)

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

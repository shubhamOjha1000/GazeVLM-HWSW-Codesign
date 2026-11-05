"""Check 2: projected gaze drawn on frames for a person to look at, plus the yaw check. CPU only.

    python -u -m accuracy_tables.hpc.newvideos.draw_gaze --lists pilot,control

Writes one sheet per video to newvideos/check_gaze/<video>.jpg: 12 frames spread through the video. The red circle is the point
the fovea is placed on (vlmeval.spatial.gaze_fixation, exactly as the VLM runs get it), and the yellow dots are the second of
gaze samples before the frame. The sheets show video frames, so keep them on the cluster or your own machine: don't publish them.

For a video with left and right yaw it prints how far Aria's combined yaw is from their plain mean, in degrees.
"""
import argparse
import os

import cv2
import numpy as np

from accuracy_tables.common import GAZE_WINDOW
from accuracy_tables.hpc.newvideos import paths
from accuracy_tables.hpc.newvideos.score_gate import merge_rows
from vlmeval import spatial as sp

TILE, TILES_PER_ROW, N_FRAMES = 448, 4, 12


def pick_frames(n, k=N_FRAMES):
    """k frames spread evenly through 1 .. n - 1 (frame 0 never has gaze)."""
    return sorted({int(round(v)) for v in np.linspace(1, n - 1, k)})


def yaw_report(video):
    """A line about one video's yaw."""
    with np.load(paths.gaze_npz(video)) as z:
        yaw, mean, fmt = z["yaw"], z["yaw_mean"], str(z["gaze_format"])
    if not np.isfinite(mean).any():
        return f"   {video}: {fmt}"
    d = np.degrees(np.abs(yaw - mean))
    d = d[np.isfinite(d)]
    missing = int(np.isnan(yaw).sum())
    return (f"   {video}: {fmt}; combined minus mean in degrees: median {np.median(d):.3f}, p99 {np.percentile(d, 99):.3f}, "
            f"max {d.max():.3f}; {missing} of {len(yaw)} rows without a combined yaw")


def draw_tile(path, track, t, label):
    """One frame with its gaze, as a TILE x TILE image."""
    img = cv2.imread(path)
    img = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA)
    text = label
    if track is not None:
        ts, xy = track
        for u, v in xy[(ts > t - GAZE_WINDOW) & (ts <= t)]:
            cv2.circle(img, (int(u * TILE), int(v * TILE)), 3, (0, 255, 255), -1)
    g = sp.gaze_fixation(track, t, window=GAZE_WINDOW) if track is not None else None
    if g is None:
        text += " NO GAZE"
    else:
        cv2.circle(img, (int(g[0] * TILE), int(g[1] * TILE)), 14, (0, 0, 255), 3)
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def draw_video(video, track, out_dir):
    meta = paths.frame_meta(video)
    d = paths.frames_dir(video)
    tiles = [draw_tile(os.path.join(d, f"frame_{k:05d}.jpg"), track, float(meta["ts"][k]), f"frame {k}")
             for k in pick_frames(meta["n"])]
    while len(tiles) % TILES_PER_ROW:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + TILES_PER_ROW]) for i in range(0, len(tiles), TILES_PER_ROW)])
    path = os.path.join(out_dir, f"{video}.jpg")
    cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lists", required=True, help="comma list of lists written by plan.py")
    a = ap.parse_args()
    videos = [v for name in a.lists.split(",") for v in paths.read_list(name.strip())]
    merge_rows("draw", videos)
    metas = {v: paths.frame_meta(v) for v in videos}

    def frame_time(v, k):
        return metas[v]["ts"][k] if k < metas[v]["n"] else k

    tracks, dropped = sp.gaze_tracks(paths.folds_dir("draw"), videos, frame_time)
    print(f"{dropped} windows dropped for out-of-view or missing gaze")
    out_dir = os.path.join(paths.root(), "check_gaze")
    os.makedirs(out_dir, exist_ok=True)
    print("yaw:")
    for v in videos:
        print(yaw_report(v))
    print("sheets to look at:")
    for v in videos:
        print("  ", draw_video(v, tracks.get(v), out_dir), flush=True)


if __name__ == "__main__":
    main()

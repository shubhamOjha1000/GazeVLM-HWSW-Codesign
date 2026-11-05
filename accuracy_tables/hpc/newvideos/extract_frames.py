"""The 1 fps frames of the new videos, CPU only, with the same function and settings as the 12 known videos.

    python -u -m accuracy_tables.hpc.newvideos.extract_frames --video <name>

A video whose meta.json exists is skipped: meta.json is written last, so its presence means the frames are complete.
"""
import argparse
import os

from accuracy_tables.hpc.newvideos import paths
from vlmeval.frames import extract_1fps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    paths.add_video_args(ap)
    a = ap.parse_args()
    for v in paths.chosen_videos(a):
        if os.path.exists(os.path.join(paths.frames_dir(v), "meta.json")):
            print(f"   {v}: frames already extracted", flush=True)
            continue
        mp4 = os.path.join(paths.video_dir(v), f"{v}.mp4")
        meta = extract_1fps(mp4, os.path.join(paths.new_frames_root(), v), paths.FRAME_SIZE)
        print(f"   {v}: {meta['n']} frames at step {meta['step']} of {meta['native_fps']:g} fps", flush=True)


if __name__ == "__main__":
    main()

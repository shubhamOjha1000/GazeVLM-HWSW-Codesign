"""Where the new videos' files live. Standard library only, so the `aria` environment can import it too.

Everything is written under $GV_ROOT/newvideos/. $EGO_ROOT, $GV_ROOT/data/ and $GV_ROOT/frames_1344/ are only read.
"""
import json
import os

FRAME_SIZE = 1344        # long side of the 1 fps frames: vlmeval.qwen_fovea.IMG, which a test checks


def root():
    return os.path.join(os.environ["GV_ROOT"], "newvideos")


def video_dir(video):
    return os.path.join(os.environ["EGO_ROOT"], "videos", video)


def questions_json():
    return os.path.join(os.environ["EGO_ROOT"], "Egoeverything_VQA.json")


def known_folds_dir():
    return os.path.join(os.environ["GV_ROOT"], "data", "folds")


def known_preds_csv():
    return os.path.join(os.environ["GV_ROOT"], "data", "vlm", "egoeverything", "oof_preds_full.csv")


def ckpt_dir():
    return os.path.join(os.environ["GV_ROOT"], "data", "runs", "eyehead")


def new_frames_root():
    return os.path.join(root(), "frames")


def frames_dir(video):
    """The video's 1 fps frames: the existing folder for the 12 known videos, otherwise the new one."""
    known = os.path.join(os.environ["GV_ROOT"], "frames_1344", video)
    return known if os.path.exists(os.path.join(known, "meta.json")) else os.path.join(new_frames_root(), video)


def frame_meta(video):
    with open(os.path.join(frames_dir(video), "meta.json")) as f:
        return json.load(f)


def gaze_npz(video):
    return os.path.join(root(), "gaze", f"{video}.npz")


def rows_csv(video):
    return os.path.join(root(), "rows", f"{video}.csv")


def rows_stats(video):
    return os.path.join(root(), "rows", f"{video}.json")


def folds_dir(group):
    return os.path.join(root(), group, "folds")


def preds_csv(group):
    return os.path.join(root(), group, "preds.csv")


def list_path(name):
    return os.path.join(root(), "lists", f"{name}.txt")


def read_list(name):
    with open(list_path(name)) as f:
        return [line.strip() for line in f if line.strip()]


def write_list(name, videos):
    os.makedirs(os.path.dirname(list_path(name)), exist_ok=True)
    with open(list_path(name), "w") as f:
        f.write("".join(v + "\n" for v in videos))


def add_video_args(ap):
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", help="one video name")
    g.add_argument("--list", help="a list written by plan.py: aea12, aea13, nymeria, pilot, control or rest")


def chosen_videos(a):
    return [a.video] if a.video else read_list(a.list)

"""Prepares what the accuracy runs read on NYU Torch and checks it, CPU only:

    python -u -m accuracy_tables.hpc.prepare_data --config accuracy_tables/hpc/config.torch.yaml

1. Checks that the files copied from Drive and the questions file are there.
2. Extracts the 1 fps frames of every gate video with questions from $EGO_ROOT/videos/<video>/<video>.mp4, with the rule
   and size of the Colab runs (vlmeval.frames.extract_1fps). Frames already extracted are kept, and a video with a Colab
   tar in paths.frames_cache_dir is left for load_data to unpack.
3. Runs load_data on all and on test, and stops if the numbers differ from the Colab runs': the frames, gaze and gate
   data are then arranged wrongly, and accuracy would not be comparable.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from accuracy_tables import common  # noqa: E402

EXPECTED = {"all": ["834 questions on "],
            "test": ["204 questions on 4 videos (4 test videos)",
                     "gate frames per video: mean 41.8, gaze missing on 4 of 167",
                     "uniform frames per video: mean 86.8"]}


def check_inputs(p):
    need = [os.path.join(p["folds_dir"], f) for f in ("train_val_fold1.csv", "test.csv")] + [p["oof_cache"], p["questions_json"]]
    missing = [f for f in need if not os.path.exists(f)]
    if missing:
        sys.exit("missing, see the runbook: " + ", ".join(missing))


def extract_frames(cfg):
    from vlmeval import qwen_fovea as qf
    from vlmeval.frames import extract_1fps
    cfg["run"]["videos"] = "all"
    videos = sorted(set(common._questions_and_clips(cfg)[0].video))
    for v in videos:
        out_dir = os.path.join(cfg["paths"]["frames_local_dir"], v)
        if os.path.exists(os.path.join(out_dir, "meta.json")):         # written last, so its presence means complete
            print(f"   {v}: already extracted", flush=True)
            continue
        if os.path.exists(os.path.join(cfg["paths"]["frames_cache_dir"], f"{v}_px{qf.IMG}.tar")):
            print(f"   {v}: from its tar", flush=True)
            continue
        mp4 = os.path.join(os.environ["EGO_ROOT"], "videos", v, f"{v}.mp4")
        if not os.path.exists(mp4):
            sys.exit(f"no video at {mp4}")
        print(f"   {v}: {extract_1fps(mp4, out_dir, qf.IMG)['n']} frames", flush=True)


def check_data(cfg):
    for videos, lines in EXPECTED.items():
        cfg["run"]["videos"] = videos
        text = common.describe(common.load_data(cfg))
        print(f"--- {videos}\n{text}")
        wrong = [line for line in lines if not any(t.startswith(line) for t in text.splitlines())]
        if wrong:
            sys.exit(f"DATA CHECK FAILED on {videos}, expected: {wrong}")
    print("data check OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    cfg = common.load_config(a.config)
    check_inputs(cfg["paths"])
    extract_frames(cfg)
    check_data(cfg)


if __name__ == "__main__":
    main()

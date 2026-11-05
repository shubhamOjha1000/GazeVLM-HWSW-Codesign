"""Writes the video lists every later step reads, and stops if the groups aren't the expected size.

    python -u -m accuracy_tables.hpc.newvideos.plan

aea12    AEA videos with questions that the gate already has data for
aea13    AEA videos with questions that the gate has never seen: built and checked, not scored on a VLM
nymeria  Nymeria videos with questions
pilot    5 Nymeria videos, spread evenly through the sorted names, so the choice depends on no result
control  1 of the aea12 videos, drawn beside the pilot videos as a reference
rest     aea13 + nymeria without the pilot videos: what the array job builds
"""
import json
import os
import sys

from accuracy_tables.hpc.newvideos import paths

EXPECTED = dict(aea12=12, aea13=13, nymeria=325)
CONTROL = "loc4_script3_seq2_rec2"


def split_groups(question_videos, gate_videos):
    """Names starting with `loc` are AEA, the others Nymeria (context section 2)."""
    videos = sorted(set(question_videos))
    aea = [v for v in videos if v.startswith("loc")]
    return dict(aea12=[v for v in aea if v in gate_videos], aea13=[v for v in aea if v not in gate_videos],
                nymeria=[v for v in videos if not v.startswith("loc")])


def pilot_videos(nymeria, n=5):
    """The middle video of each of n equal parts of the sorted list."""
    return [nymeria[(2 * i + 1) * len(nymeria) // (2 * n)] for i in range(n)]


def main():
    import pandas as pd
    with open(paths.questions_json()) as f:
        question_videos = [r["video_name"] for r in json.load(f)]
    d = paths.known_folds_dir()
    gate_videos = set()
    for name in ("train_val_fold1.csv", "test.csv"):
        gate_videos |= set(pd.read_csv(os.path.join(d, name), usecols=["sequence"])["sequence"])
    groups = split_groups(question_videos, gate_videos)
    sizes = {k: len(v) for k, v in groups.items()}
    print("groups:", sizes)
    if sizes != EXPECTED:
        sys.exit(f"expected {EXPECTED}: stop and tell the user")
    if CONTROL not in groups["aea12"]:
        sys.exit(f"{CONTROL} is not in aea12: stop and tell the user")
    pilot = pilot_videos(groups["nymeria"])
    rest = [v for v in groups["aea13"] + groups["nymeria"] if v not in pilot]
    for name, videos in dict(groups, pilot=pilot, control=[CONTROL], rest=rest).items():
        paths.write_list(name, videos)
    print("pilot:", pilot)
    print(f"rest: {len(rest)} videos, so the array is --array=0-{len(rest) - 1}%20")


if __name__ == "__main__":
    main()

"""Joins the three video groups into one data folder, picks the pilot and checks it all. CPU only, base environment.

    python -u -m accuracy_tables.hpc.all350.prepare

Writes under newvideos/all350/, and only reads everything else:

    folds/test.csv               the 12 known videos' rows from data/folds/, then the rows of aea13 and nymeria
    folds/train_val_fold1.csv    empty: load_labels and gaze_tracks read both files
    preds.csv                    the 12 videos' lines of oof_preds_full.csv, then aea13/preds.csv and nymeria/preds.csv, unchanged
    frames/<video>               symlinks to frames_1344/ and newvideos/frames/

The 12 known videos keep their original rows because their predictions are matched by the exact frame names in those rows, and
results/hpc_all/ and results/hpc_budget/ were scored on them. Only the columns the code reads are kept.

Checks, each of which stops the job: the question and video counts; each video's labels, frames and gaze match what its own group
gives; and every record to be reused has the frames and tokens the joined data gives today. Ends with REUSED RECORDS MATCH.
"""
import collections
import json
import os
import sys

import numpy as np
import pandas as pd

from accuracy_tables import common
from accuracy_tables.hpc.all350 import seed_results
from accuracy_tables.hpc.newvideos import paths, plan
from accuracy_tables.models import ADAPTERS

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.all350.yaml")
COLUMNS = ["sequence", "feat_frame_1", "feat_frame_2", "quad", "gaze_xy_window", "n_oof_in_gap"]     # what load_labels and gaze_tracks read
GROUPS = dict(aea12=(834, 12), aea13=(1478, 13), nymeria=(3673, 325))      # questions, videos
TOTAL = (5985, 350)
NYMERIA_SPLIT = (21, 304)                                                  # videos with many questions, with 1 question


# ---------------------------------------------------------------------------- paths
def root():
    return os.path.join(paths.root(), "all350")


def folds_dir():
    return os.path.join(root(), "folds")


def preds_csv():
    return os.path.join(root(), "preds.csv")


def frames_root():
    return os.path.join(root(), "frames")


# ---------------------------------------------------------------------------- joining
def known_rows(videos):
    """The known videos' rows as they are in data/folds/, in order."""
    d = paths.known_folds_dir()
    rows = pd.concat([pd.read_csv(os.path.join(d, f), usecols=COLUMNS) for f in ("train_val_fold1.csv", "test.csv")],
                     ignore_index=True)
    return rows[rows["sequence"].isin(set(videos))]


def known_preds(videos):
    p = pd.read_csv(paths.known_preds_csv())
    return p[p["sequence"].isin(set(videos))]


def join_problems(rows, preds, videos):
    """What is wrong with the joined rows and predictions -> list of messages."""
    out = []
    have = set(rows["sequence"])
    if have != set(videos):
        out.append(f"rows cover {len(have)} videos, expected {len(set(videos))}: missing {sorted(set(videos) - have)[:5]}, "
                   f"extra {sorted(have - set(videos))[:5]}")
    for name, names in (("rows", rows["feat_frame_2"]), ("predictions", preds["feat_frame_2"])):
        if names.duplicated().any():
            out.append(f"{int(names.duplicated().sum())} duplicate frame names in the {name}")
    a, b = set(rows["feat_frame_2"]), set(preds["feat_frame_2"])
    if a != b:
        out.append(f"{len(a - b)} rows have no prediction and {len(b - a)} predictions have no row")
    return out


def write_csv(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def join_data(lists):
    """Writes the fold files and preds.csv from {group: videos}. Stops if they don't fit together."""
    rows = pd.concat([known_rows(lists["aea12"]),
                      pd.read_csv(os.path.join(paths.folds_dir("aea13"), "test.csv"), usecols=COLUMNS),
                      pd.read_csv(os.path.join(paths.folds_dir("nymeria"), "test.csv"), usecols=COLUMNS)],
                     ignore_index=True)[COLUMNS]
    preds = pd.concat([known_preds(lists["aea12"]), pd.read_csv(paths.preds_csv("aea13")), pd.read_csv(paths.preds_csv("nymeria"))],
                      ignore_index=True)
    found = join_problems(rows, preds, [v for videos in lists.values() for v in videos])
    if found:
        sys.exit("the groups don't join: " + "; ".join(found) + ". Stop and tell the user.")
    os.makedirs(folds_dir(), exist_ok=True)
    write_csv(rows, os.path.join(folds_dir(), "test.csv"))
    write_csv(rows.head(0), os.path.join(folds_dir(), "train_val_fold1.csv"))
    write_csv(preds, preds_csv())
    print(f"joined {len(rows)} frame pairs on {rows['sequence'].nunique()} videos", flush=True)


def link_frames(videos):
    os.makedirs(frames_root(), exist_ok=True)
    for v in videos:
        target = paths.frames_dir(v)
        if not os.path.exists(os.path.join(target, "meta.json")):
            sys.exit(f"{v}: no frames at {target}. Stop and tell the user.")
        link = os.path.join(frames_root(), v)
        if os.path.islink(link):
            os.remove(link)
        os.symlink(target, link)


# ---------------------------------------------------------------------------- the pilot
def question_counts():
    with open(paths.questions_json()) as f:
        return collections.Counter(q["video_name"] for q in json.load(f))


def pick_pilot(nymeria, aea13, counts):
    """2 Nymeria videos with many questions, 4 with one, 1 of the new AEA videos, each spread through the sorted names, so the
    choice depends on no result. The first is the check job's Nymeria video."""
    many = sorted(v for v in nymeria if counts[v] > 1)
    one = sorted(v for v in nymeria if counts[v] == 1)
    if (len(many), len(one)) != NYMERIA_SPLIT:
        sys.exit(f"Nymeria has {len(many)} videos with many questions and {len(one)} with one, expected {NYMERIA_SPLIT}: "
                 "stop and tell the user")
    return plan.pilot_videos(many, 2) + plan.pilot_videos(one, 4) + plan.pilot_videos(sorted(aea13), 1)


# ---------------------------------------------------------------------------- checking
def group_config(group):
    """The config reading one group from where it was built, as the runs so far did."""
    cfg = common.load_config(CONFIG)
    if group == "aea12":
        cfg["paths"].update(folds_dir=paths.known_folds_dir(), oof_cache=paths.known_preds_csv(),
                            frames_local_dir=os.path.join(os.environ["GV_ROOT"], "frames_1344"))
    else:
        cfg["paths"].update(folds_dir=paths.folds_dir(group), oof_cache=paths.preds_csv(group),
                            frames_local_dir=paths.new_frames_root())
    return cfg


def check_counts(name, data, expected):
    got = (len(data["qs"]), len(data["videos"]))
    print(f"{name}: {got[0]} questions on {got[1]} videos", flush=True)
    if got != expected:
        sys.exit(f"{name}: expected {expected[0]} questions on {expected[1]} videos. Stop and tell the user.")


def _frame_fields(frames):
    """What a run takes from a frame, without its path, which goes through a symlink in the joined data."""
    return [(fr["key"], fr["t"], fr["gaze"], fr["gaze_trace"]) for fr in frames]


def smolvlm_budget_frames(cfg, data):
    """{fraction: {video: [frame]}} under the budget rule, on the one model whose token count is fixed already."""
    stock = ADAPTERS["smolvlm"].stock_tokens
    return {f: common.budget_frames(cfg, data, "smolvlm", stock, float(f), use_gate=True)[0] for f in cfg["budget"]["fractions"]}


def compare_group(group_data, group_budget, joined, joined_budget):
    """Differences between one group's data, read from where it was built, and the joined data -> list of messages."""
    out = []
    for v in group_data["videos"]:
        if v not in joined["labels"]:
            out.append(f"{v}: not in the joined data")
            continue
        if not np.array_equal(group_data["labels"][v], joined["labels"][v]):
            out.append(f"{v}: gate labels differ")
        if group_data["metas"][v]["n"] != joined["metas"][v]["n"]:
            out.append(f"{v}: frame count differs")
        for name in ("uniform", "gate"):
            if _frame_fields(group_data[name][v]) != _frame_fields(joined[name][v]):
                out.append(f"{v}: {name} frames or gaze differ")
        for f in group_budget:
            if _frame_fields(group_budget[f][v]) != _frame_fields(joined_budget[f][v]):
                out.append(f"{v}: budget {f} frames or gaze differ")
    return out


def expected_frames(cfg, data, budget, model, row):
    """({condition: {video: frames}}, tokens per frame) that a record of this row should have on the joined data."""
    names = seed_results.condition_names(model, row)
    stock = ADAPTERS[model].stock_tokens
    if row == "blind":
        return {names[0]: {v: 0 for v in data["videos"]}}, 0
    if row == "uniform":
        cap = cfg["run"]["max_frames"].get(model)
        return {names[0]: {v: len(common.cap_frames(data["uniform"][v], cap)) for v in data["videos"]}}, stock
    per = {name: {v: len(frames) for v, frames in budget[f].items()} for name, f in zip(names, cfg["budget"]["fractions"])}
    return per, cfg["budget"]["tokens_per_frame"][model]


def reused_differences(cfg, data, budget):
    """Reused records whose frames or tokens aren't what the joined data gives -> list of messages. Reads no answers."""
    out = []
    for model in seed_results.MODELS:
        for row, path, _ in seed_results.sources(model):
            want, per_frame = expected_frames(cfg, data, budget, model, row)
            bad = 0
            records = seed_results.read_records(path)
            for r in records:
                n = want.get(r["condition"], {}).get(r["video"])
                bad += n is None or r.get("n_frames") != n or r.get("n_image_tokens") != n * per_frame       # an error record has no tokens
            print(f"   {path}: {len(records)} records, {bad} differ", flush=True)
            if bad:
                out.append(f"{path}: {bad} of {len(records)} records")
    return out


def stop_on(found, stop_line, good_line):
    for line in found[:20]:
        print("  ", line, flush=True)
    if found:
        sys.exit(f"{stop_line}: {len(found)} differences. Stop and tell the user.")
    print(good_line, flush=True)


def main():
    lists = {g: paths.read_list(g) for g in GROUPS}
    videos = sorted(v for vs in lists.values() for v in vs)
    join_data(lists)
    link_frames(videos)
    counts = question_counts()
    pilot = pick_pilot(lists["nymeria"], lists["aea13"], counts)
    paths.write_list("all350_pilot", pilot)
    print(f"pilot, in {paths.list_path('all350_pilot')}:", flush=True)
    for v in pilot:
        print(f"   {v}: {counts[v]} questions", flush=True)
    print(f"   {sum(counts[v] for v in pilot)} questions in all", flush=True)

    cfg = common.load_config(CONFIG)
    joined = common.load_data(cfg)
    print(common.describe(joined), flush=True)
    check_counts("all350", joined, TOTAL)
    joined_budget = smolvlm_budget_frames(cfg, joined)
    found = []
    for group, expected in GROUPS.items():
        group_cfg = group_config(group)
        data = common.load_data(group_cfg)
        check_counts(group, data, expected)
        found += compare_group(data, smolvlm_budget_frames(group_cfg, data), joined, joined_budget)
    stop_on(found, "COMBINED DIFFERS", "COMBINED MATCHES")
    stop_on(reused_differences(cfg, joined, joined_budget), "REUSED RECORDS DIFFER", "REUSED RECORDS MATCH")


if __name__ == "__main__":
    main()

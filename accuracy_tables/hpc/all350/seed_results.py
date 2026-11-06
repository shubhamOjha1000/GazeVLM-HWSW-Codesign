"""Copies the records that can be reused into a model's new results folder, so the run scores only what's missing.

    python -u -m accuracy_tables.hpc.all350.seed_results --model phi --out <results folder>

    SmolVLM2              uniform and blind from hpc_all (12 AEA videos) and nymeria_smolvlm (325 Nymeria videos);
                          budget from hpc_budget (12) and nymeria_smolvlm (325)
    Qwen2.5, Qwen3, Phi   uniform and blind from hpc_all (12)

The source folders are only read, and no-gate records aren't copied. A row whose file already exists in <out> is left alone, so a
rerun after a timeout keeps what was scored since. Anything unsafe stops the job before that row is written: an error record, a
duplicate, a record of another video, a condition with another name, or the wrong number of records. Prints no accuracy.
"""
import argparse
import collections
import json
import os
import sys

from accuracy_tables import common, run
from accuracy_tables.hpc.newvideos import paths
from accuracy_tables.models import ADAPTERS

MODELS = ("qwen2.5", "qwen3", "phi", "smolvlm")
PROTECTED = ("hpc_all", "hpc_budget", "nymeria_smolvlm")


def results_root():
    return os.path.join(os.environ["GV_ROOT"], "results")


def sources(model):
    """[(row, source file, the video list it must cover)] for one model."""
    r = results_root()
    out = [(row, os.path.join(r, "hpc_all", model, f"{row}.jsonl"), "aea12") for row in ("uniform", "blind")]
    if model == "smolvlm":
        out.append(("budget", os.path.join(r, "hpc_budget", model, "budget.jsonl"), "aea12"))
        out += [(row, os.path.join(r, "nymeria_smolvlm", model, f"{row}.jsonl"), "nymeria") for row in ("uniform", "blind", "budget")]
    return out


def condition_names(model, row):
    """The condition names a run writes today, from the protocol defaults. A reused record under any other name would not be
    skipped by the run, so it would be scored again."""
    d = common.DEFAULTS
    if row == "uniform":
        cap = d["run"]["max_frames"].get(model)
        return [run.uniform_name(min(d["run"]["K"], cap) if cap else d["run"]["K"])]
    if row == "blind":
        return [run.BLIND]
    ad = ADAPTERS[model]()
    label = ad.setting_label(ad.fovea_settings(d["ours"])[0])
    return [run.budget_name(f, d["run"]["policy"], label) for f in d["budget"]["fractions"]]


def read_records(path):
    if not os.path.exists(path):
        sys.exit(f"no results to reuse at {path}: stop and tell the user")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def question_videos():
    """The video of each question, by qid (its position in the questions file)."""
    with open(paths.questions_json()) as f:
        return [q["video_name"] for q in json.load(f)]


def record_problems(records, names, videos, video_of):
    """What makes these records unsafe to reuse -> list of messages. names: the conditions, videos: the videos they must cover."""
    videos = set(videos)
    want = sum(video_of[q] in videos for q in range(len(video_of)))
    seen, per_condition, found = set(), collections.Counter(), collections.Counter()
    for r in records:
        key = (r["condition"], r["qid"])
        found["duplicate"] += key in seen
        seen.add(key)
        found["error record"] += r.get("error") is not None
        found["other condition"] += r["condition"] not in names
        found["other video"] += r["video"] not in videos
        found["qid of another video"] += not 0 <= r["qid"] < len(video_of) or video_of[r["qid"]] != r["video"]
        per_condition[r["condition"]] += 1
    out = [f"{n} x {what}" for what, n in found.items() if n]
    out += [f"{name}: {per_condition[name]} records, expected {want}" for name in names if per_condition[name] != want]
    return out


def refuse_protected(out):
    real = os.path.realpath(out)
    for name in PROTECTED:
        p = os.path.realpath(os.path.join(results_root(), name))
        if real == p or real.startswith(p + os.sep):
            sys.exit(f"{out} is inside results/{name}, which is only read")


def seed_model(model, out):
    refuse_protected(out)
    target_dir = os.path.join(out, model)
    os.makedirs(target_dir, exist_ok=True)
    video_of = question_videos()
    rows = {}
    for row, path, group in sources(model):
        rows.setdefault(row, []).append((path, group))
    for row, parts in rows.items():
        target = os.path.join(target_dir, f"{row}.jsonl")
        if os.path.exists(target):
            print(f"   {model} {row}: already there, left alone", flush=True)
            continue
        kept = []
        for path, group in parts:
            records = read_records(path)
            found = record_problems(records, condition_names(model, row), paths.read_list(group), video_of)
            if found:
                sys.exit(f"{path} can't be reused: {'; '.join(found)}. Stop and tell the user.")
            print(f"   {model} {row}: {len(records)} records from {path}", flush=True)
            kept += records
        tmp = target + ".tmp"
        with open(tmp, "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in kept))
        os.replace(tmp, target)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--out", required=True, help="the new results folder; records go to <out>/<model>/")
    a = ap.parse_args()
    seed_model(a.model, a.out)


if __name__ == "__main__":
    main()

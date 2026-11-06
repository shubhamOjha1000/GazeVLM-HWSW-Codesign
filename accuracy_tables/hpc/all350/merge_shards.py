"""Joins a run's shards into the results an unsharded run would have written. CPU only. Reads no accuracy.

    python -u -m accuracy_tables.hpc.all350.merge_shards --root <results folder> --model qwen3 --targets <targets.csv> --config <yaml> \
        --rows <comma list> --calibration-gpu <gpu_types.txt>

Writes <root>/<model>/<row>.jsonl, gpu_types.txt and merge_info.json, so summarize.py and report.py read it like any other results folder.
Each line is the shard's own line, unchanged, in the order an unsharded run writes it. That order matters: a video's confidence interval
depends on the order its results come in.

Everything is checked before anything is written. It stops, leaving <root>/<model>/ as it was, on: a missing question or condition, a
duplicate, a half-written or unreadable line, an error record, a record of a video that isn't that shard's, a manifest that doesn't
split the videos cleanly, or any disagreement between jobs in GPU name, commit, code, targets file, settings or library versions.
"""
import argparse
import collections
import csv
import json
import os
import shutil
import sys

from accuracy_tables.hpc.all350 import seed_results, shards
from accuracy_tables.hpc.check_gpu import recorded_names

BY_CONDITION = ("budget", "nogate")        # scored one condition at a time; the other rows, every condition of a question together


def read_targets(path):
    """{row: its conditions in the file's order, each once}."""
    out = {}
    with open(path, newline="") as f:
        for line in csv.DictReader(f):
            names = out.setdefault(line["row"], [])
            if line["condition"] not in names:
                names.append(line["condition"])
    return out


def job_problems(manifest, root, model, targets, config, calibration_gpu):
    """Do all the jobs agree with each other, and with the files as they are now? -> (messages, {shard: its job lines}, the GPU name)."""
    out, infos = [], {}
    for s in range(manifest["n_shards"]):
        try:
            infos[s] = shards.read_info(shards.info_path(root, s, model))
        except ValueError:
            infos[s] = []
            out.append(f"shard {s}: a half-written line in shard_info.jsonl")
        if not infos[s]:
            out.append(f"shard {s}: no job has started it")
    lines = [line for s in sorted(infos) for line in infos[s]]
    if lines:
        out += shards.disagreements(lines[0], infos)
        for field, name, path in (("targets_sha256", "targets", targets), ("config_sha256", "settings", config),
                                  ("manifest_sha256", "manifest", shards.manifest_path(root))):
            if lines[0].get(field) != shards.file_sha256(path):
                out.append(f"the {name} file isn't the one the shards ran with: {path}")
    names = set()
    for s in sorted(infos):
        found = recorded_names(os.path.join(shards.shard_folder(root, s, model), "gpu_types.txt"))
        if len(found) != 1:
            out.append(f"shard {s}: gpu_types.txt holds {sorted(found)}, not one GPU name")
        names |= found
    if lines:
        names.add(lines[0].get("gpu"))
    if calibration_gpu:
        found = recorded_names(calibration_gpu)
        if len(found) != 1:
            out.append(f"calibration: {calibration_gpu} holds {sorted(found)}, not one GPU name")
        names |= found
    if len(names) > 1:
        out.append(f"more than one GPU name across the jobs: {sorted(str(n) for n in names)}")
    return out, infos, (sorted(str(n) for n in names)[0] if names else None)


def read_shard_row(path, shard, row, videos, conditions, video_of):
    """One shard's file for one row -> ({(condition, qid): (video, the line as written)}, messages)."""
    if not os.path.exists(path):
        return {}, [f"shard {shard} {row}: no results file"]
    with open(path) as f:
        lines = f.readlines()
    kept, wrong = {}, collections.Counter()
    for line in lines:
        if not line.strip():
            continue
        if not line.endswith("\n"):
            wrong["half-written last line"] += 1
            continue
        try:
            r = json.loads(line)
            key, video = (r["condition"], r["qid"]), r["video"]
        except (ValueError, KeyError, TypeError):
            wrong["unreadable line"] += 1
            continue
        if r.get("error") is not None:
            wrong[f"error record ({r['error']})"] += 1
        elif key[0] not in conditions:
            wrong["condition not in the targets"] += 1
        elif video not in videos:
            wrong["record of a video that isn't this shard's"] += 1
        elif not 0 <= key[1] < len(video_of) or video_of[key[1]] != video:
            wrong["qid of another video"] += 1
        elif key in kept:
            wrong["duplicate"] += 1
        else:
            kept[key] = (video, line)
    return kept, [f"shard {shard} {row}: {n} x {what}" for what, n in wrong.items()]


def line_order(row, rank):
    """The order an unsharded run writes this row: a condition's questions are always by video, then question."""
    if row in BY_CONDITION:
        return lambda item: (rank[item[0][0]], item[1][0], item[0][1])
    return lambda item: (item[1][0], item[0][1], rank[item[0][0]])


def plan_merge(root, model, targets, config, rows, calibration_gpu=None):
    """Every check, and the merged lines if they pass. Writes nothing. -> ({row: [lines]}, info, messages)."""
    found = []
    manifest = shards.read_manifest(root)
    video_of = seed_results.question_videos()
    found += shards.manifest_problems(manifest, collections.Counter(video_of))
    wanted = read_targets(targets)
    if set(wanted) != set(rows):
        found.append(f"the rows to merge {sorted(rows)} aren't the rows of the targets {sorted(wanted)}")
    jobs, infos, gpu = job_problems(manifest, root, model, targets, config, calibration_gpu)
    found += jobs
    qids = collections.defaultdict(list)
    for q, v in enumerate(video_of):
        qids[v].append(q)
    merged = {}
    for row in rows:
        conditions = wanted.get(row, [])
        rank = {c: i for i, c in enumerate(conditions)}
        joined = {}
        for s in manifest["shards"]:
            videos = set(s["videos"])
            path = os.path.join(shards.shard_folder(root, s["shard"], model), f"{row}.jsonl")
            kept, messages = read_shard_row(path, s["shard"], row, videos, rank, video_of)
            found += messages
            for key in kept:
                if key in joined:
                    found.append(f"{row}: {key} is in more than one shard")
            joined.update(kept)
            need = {(c, q) for c in conditions for v in videos for q in qids[v]}
            gone = need - set(kept)
            if gone and not any("no results file" in m for m in messages):
                found.append(f"shard {s['shard']} {row}: {len(gone)} questions missing, such as {sorted(gone)[0]}")
        merged[row] = [item[1][1] for item in sorted(joined.items(), key=line_order(row, rank))]
    info = dict(model=model, gpu=gpu, n_questions=manifest["n_questions"], n_videos=manifest["n_videos"], scope=manifest["scope"],
                commit=next((lines[0].get("commit") for lines in infos.values() if lines), None),
                shards=[dict(shard=s["shard"], questions=s["questions"], videos=len(s["videos"]),
                             nodes=sorted({str(line.get("node")) for line in infos.get(s["shard"], [])}),
                             jobs=sorted({str(line.get("job")) for line in infos.get(s["shard"], [])})) for s in manifest["shards"]],
                records={row: len(lines) for row, lines in merged.items()})
    return merged, info, found


def write_final(root, model, merged, info):
    """Builds the folder beside the final one and swaps it in, so a crash can't leave a partial result under the final name."""
    final, tmp, old = os.path.join(root, model), os.path.join(root, f"{model}.merging"), os.path.join(root, f"{model}.old")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for row, lines in merged.items():
        with open(os.path.join(tmp, f"{row}.jsonl"), "w") as f:
            f.writelines(lines)
    with open(os.path.join(tmp, "gpu_types.txt"), "w") as f:
        f.write(info["gpu"] + "\n")
    with open(os.path.join(tmp, "merge_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(final):
        os.rename(final, old)
    os.rename(tmp, final)
    shutil.rmtree(old, ignore_errors=True)


def merge(root, model, targets, config, rows, calibration_gpu=None):
    """Checks everything, then writes. A failed check stops the job with nothing written. -> info."""
    shards.refuse_protected(root)
    merged, info, found = plan_merge(root, model, targets, config, rows, calibration_gpu)
    for line in found[:40]:
        print("  ", line, flush=True)
    if found:
        sys.exit(f"MERGE STOPPED: {len(found)} problems, and nothing was written. Rerun the unfinished shard, or tell the user.")
    write_final(root, model, merged, info)
    for s in info["shards"]:
        print(f"   shard {s['shard']}: {s['questions']} questions on {s['videos']} videos, node {', '.join(s['nodes'])}, job {', '.join(s['jobs'])}", flush=True)
    print(f"   {info['gpu']}, commit {info['commit']}", flush=True)
    print(f"MERGED: {sum(info['records'].values())} records of {len(rows)} rows into {os.path.join(root, model)}", flush=True)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--calibration-gpu", help="the calibrate job's gpu_types.txt")
    a = ap.parse_args()
    merge(a.root, a.model, a.targets, a.config, a.rows.split(","), a.calibration_gpu)


if __name__ == "__main__":
    main()

"""Splitting one run across separate one-GPU jobs: which videos each job scores, and what each job ran under.

    python -u -m accuracy_tables.hpc.all350.shards write --root <results folder> --shards 3 (--all | --list all350_pilot | --smallest)
    python -u -m accuracy_tables.hpc.all350.shards start --root <results folder> --shard 0 --model qwen3 --config <yaml> --targets <csv> \
        --rows <comma list> --workers 8

write assigns whole videos to shards, balanced by question count, and writes <root>/manifest/: manifest.json, shard_<n>.txt and
all_videos.txt. The assignment depends only on the questions file. Writing it again changes nothing, and it stops if the result would differ.

start runs at the top of a shard's job, before any scoring. It compares this job with every earlier job of every shard (commit, code,
targets, settings, GPU, library versions), stops on any difference, and otherwise adds a line to <root>/shards/<n>/<model>/shard_info.jsonl,
which merge_shards.py checks again. Each shard writes only to its own folder. Reads no answers.
"""
import argparse
import hashlib
import json
import os
import sys
import time

from accuracy_tables.hpc.all350 import prepare, seed_results
from accuracy_tables.hpc.newvideos import paths

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CODE_FOLDERS = ("accuracy_tables", "vlmeval", "eyehead")
CODE_FILES = (".py", ".yaml", ".sh", ".SBATCH")
AGREE = ("model", "rows", "workers", "commit", "code_sha256", "targets_sha256", "config_sha256", "manifest_sha256", "gpu", "versions")
PROTECTED = ("hpc_all", "hpc_budget", "nymeria_smolvlm", "hpc350_check", "hpc350_pilot", "smolvlm_350", "smolvlm_350_pilot", "smolvlm_350_check")


# ---------------------------------------------------------------------------- where things are
def manifest_path(root):
    return os.path.join(root, "manifest", "manifest.json")


def shard_folder(root, shard, model):
    return os.path.join(root, "shards", str(shard), model)


def info_path(root, shard, model):
    return os.path.join(shard_folder(root, shard, model), "shard_info.jsonl")


def refuse_protected(root):
    """Stops if this folder is inside a results folder the user has ruled read-only."""
    real = os.path.realpath(root)
    for name in PROTECTED:
        p = os.path.realpath(os.path.join(seed_results.results_root(), name))
        if real == p or real.startswith(p + os.sep):
            sys.exit(f"{root} is inside results/{name}, which is only read")


# ---------------------------------------------------------------------------- the assignment
def assign(counts, n_shards):
    """Whole videos to shards, balanced by question count: the videos with the most questions first, each to the shard with the
    fewest so far (then the fewest videos, then the lowest number). Depends only on the counts. -> [sorted videos] per shard."""
    members, loads = [[] for _ in range(n_shards)], [0] * n_shards
    for video in sorted(counts, key=lambda v: (-counts[v], v)):
        shard = min(range(n_shards), key=lambda s: (loads[s], len(members[s]), s))
        members[shard].append(video)
        loads[shard] += counts[video]
    return [sorted(m) for m in members]


def manifest_of(counts, n_shards, scope):
    """counts: {video: questions}, of the videos this run covers."""
    return dict(scope=scope, n_shards=n_shards, n_videos=len(counts), n_questions=sum(counts.values()),
                shards=[dict(shard=i, questions=sum(counts[v] for v in videos), videos=videos)
                        for i, videos in enumerate(assign(counts, n_shards))])


def smallest_videos(counts):
    """A small set for the check job, from the questions file alone: the AEA video with the fewest questions, the Nymeria video with the
    fewest above one, and the first 3 Nymeria videos with one."""
    by_size = lambda videos: sorted(videos, key=lambda v: (counts[v], v))
    aea = by_size(v for v in counts if v.startswith("loc"))
    nymeria = [v for v in counts if not v.startswith("loc")]
    return aea[:1] + by_size(v for v in nymeria if counts[v] > 1)[:1] + sorted(v for v in nymeria if counts[v] == 1)[:3]


def manifest_problems(manifest, counts):
    """What's wrong with a manifest, given every video's question count -> list of messages."""
    out, seen = [], set()
    for s in manifest["shards"]:
        if not s["videos"]:
            out.append(f"shard {s['shard']}: no videos")
        for v in s["videos"]:
            if v in seen:
                out.append(f"{v}: in more than one shard")
            seen.add(v)
            if not counts.get(v):
                out.append(f"{v}: has no questions")
        if s["questions"] != sum(counts.get(v, 0) for v in s["videos"]):
            out.append(f"shard {s['shard']}: {s['questions']} questions in the manifest, {sum(counts.get(v, 0) for v in s['videos'])} in the questions file")
    if manifest["scope"] == "all":
        out += [f"{v}: in no shard" for v in sorted(set(counts) - seen)]
    return out


def read_manifest(root):
    path = manifest_path(root)
    if not os.path.exists(path):
        sys.exit(f"no {path}: the calibrate job writes it")
    with open(path) as f:
        return json.load(f)


def write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def write_manifest(root, manifest):
    """Writes it once. Writing it again changes nothing, and a different assignment stops the job."""
    refuse_protected(root)
    path = manifest_path(root)
    if os.path.exists(path):
        if read_manifest(root) != manifest:
            sys.exit(f"{path} holds a different assignment: stop and tell the user")
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for s in manifest["shards"]:
        write_text(os.path.join(root, "manifest", f"shard_{s['shard']}.txt"), "".join(v + "\n" for v in s["videos"]))
    write_text(os.path.join(root, "manifest", "all_videos.txt"), "".join(v + "\n" for s in manifest["shards"] for v in s["videos"]))
    write_text(path, json.dumps(manifest, indent=1) + "\n")
    return True


def write_cmd(a):
    counts = prepare.question_counts()
    if a.all:
        chosen, scope = dict(counts), "all"
        if (sum(chosen.values()), len(chosen)) != prepare.TOTAL:
            sys.exit(f"{sum(chosen.values())} questions on {len(chosen)} videos, expected {prepare.TOTAL}: stop and tell the user")
    else:
        videos, scope = (paths.read_list(a.list), a.list) if a.list else (smallest_videos(counts), "smallest")
        chosen = {v: counts[v] for v in videos}
    manifest = manifest_of(chosen, a.shards, scope)
    found = manifest_problems(manifest, counts)
    if found:
        sys.exit("the assignment is wrong: " + "; ".join(found[:5]) + ". Stop and tell the user.")
    wrote = write_manifest(a.root, manifest)
    print(f"{'written' if wrote else 'already there, unchanged'}: {manifest_path(a.root)}", flush=True)
    for s in manifest["shards"]:
        print(f"   shard {s['shard']}: {s['questions']} questions on {len(s['videos'])} videos", flush=True)


# ---------------------------------------------------------------------------- what a job ran under
def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git_commit(repo=REPO):
    """The checked-out commit, read from .git so that the container needs no git."""
    git = os.path.join(repo, ".git")
    with open(os.path.join(git, "HEAD")) as f:
        head = f.read().strip()
    if not head.startswith("ref: "):
        return head
    ref = head[len("ref: "):]
    if os.path.exists(os.path.join(git, ref)):
        with open(os.path.join(git, ref)) as f:
            return f.read().strip()
    with open(os.path.join(git, "packed-refs")) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    sys.exit(f"can't find commit of {ref} in {git}: stop and tell the user")


def code_sha256(repo=REPO):
    """One hash of the code the run depends on, committed or not, so a local edit between two jobs shows up."""
    h = hashlib.sha256()
    for folder in CODE_FOLDERS:
        for base, _, files in sorted(os.walk(os.path.join(repo, folder))):
            for name in sorted(files):
                if name.endswith(CODE_FILES):
                    path = os.path.join(base, name)
                    h.update(os.path.relpath(path, repo).encode())
                    h.update(file_sha256(path).encode())
    return h.hexdigest()


def fingerprint(root, model, config, targets, rows, workers, gpu, versions, repo=REPO):
    """Everything all of a run's jobs must agree on."""
    return dict(model=model, rows=list(rows), workers=int(workers), commit=git_commit(repo), code_sha256=code_sha256(repo),
                targets_sha256=file_sha256(targets), config_sha256=file_sha256(config), manifest_sha256=file_sha256(manifest_path(root)),
                gpu=gpu, versions=versions)


def read_info(path):
    """A shard's job lines (none if it hasn't started). Run repair_results first: a half-written last line stops here."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def disagreements(mine, infos):
    """Where other jobs differ from this one -> list of messages. infos: {shard: its job lines}."""
    return [f"shard {shard}: {field} {line.get(field)} against {mine.get(field)}"
            for shard, lines in sorted(infos.items()) for line in lines for field in AGREE if line.get(field) != mine.get(field)]


def start(root, shard, model, config, targets, rows, workers, gpu, versions, repo=REPO):
    """Checks this job against every earlier job of every shard, then records it."""
    refuse_protected(root)
    manifest = read_manifest(root)
    if not 0 <= shard < manifest["n_shards"]:
        sys.exit(f"shard {shard}: this manifest has shards 0 to {manifest['n_shards'] - 1}")
    mine = fingerprint(root, model, config, targets, rows, workers, gpu, versions, repo)
    found = disagreements(mine, {s: read_info(info_path(root, s, model)) for s in range(manifest["n_shards"])})
    if found:
        sys.exit("this job doesn't match the shards before it, and the merge would refuse them: " + "; ".join(found[:5]) + ". Stop and tell the user.")
    path = info_path(root, shard, model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = dict(mine, shard=shard, node=os.environ.get("SLURMD_NODENAME"), job=os.environ.get("SLURM_JOB_ID"), started=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(path, "a") as f:
        f.write(json.dumps(line) + "\n")
    print(f"shard {shard}: commit {mine['commit']}, {gpu}, targets {mine['targets_sha256'][:12]}", flush=True)


def start_cmd(a):
    import torch
    import transformers
    gpu = torch.cuda.get_device_properties(0).name
    versions = dict(python=sys.version.split()[0], torch=torch.__version__, transformers=transformers.__version__)
    start(a.root, a.shard, a.model, a.config, a.targets, a.rows.split(","), a.workers, gpu, versions)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write", help="write the assignment, once")
    w.add_argument("--root", required=True)
    w.add_argument("--shards", type=int, required=True)
    which = w.add_mutually_exclusive_group(required=True)
    which.add_argument("--all", action="store_true", help="every video with questions")
    which.add_argument("--list", help="the videos of a list written by the earlier steps")
    which.add_argument("--smallest", action="store_true", help="the small set for the check job")
    w.set_defaults(run=write_cmd)
    s = sub.add_parser("start", help="check and record this job, before it scores anything")
    s.add_argument("--root", required=True)
    s.add_argument("--shard", type=int, required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--config", required=True)
    s.add_argument("--targets", required=True)
    s.add_argument("--rows", required=True)
    s.add_argument("--workers", type=int, required=True)
    s.set_defaults(run=start_cmd)
    a = ap.parse_args()
    a.run(a)


if __name__ == "__main__":
    main()

"""Section 15's FLOPs-matched run split across 3 one-GPU jobs per model: the split, the sealed plan, each shard's check, and the merge.

    python -u -m accuracy_tables.hpc.all350.flops_shards split --source <section 12 or 13 results> --root <flops results>
    python -u -m accuracy_tables.hpc.all350.flops_shards seal  --root <flops results> --model qwen2.5
    python -u -m accuracy_tables.hpc.all350.flops_shards start --root <flops results> --shard 0 --model qwen2.5 --config <yaml> \
        --rows <comma list> --workers 8
    python -u -m accuracy_tables.hpc.all350.flops_shards merge --root <flops results> --model qwen2.5 --config <yaml> --rows <comma list>

split   copies the assignment sections 12 and 13 ran on (<source>/manifest/manifest.json, only read) to <root>/manifest/, after checking
        it against the questions file, and prints it. It never recomputes the assignment.
seal    records the hashes of <root>/calibrate/<model>/flops_plan.json and flops_targets.csv in flops_plan.sha256, once calibrate is done.
start   runs at the top of a shard's job: the plan must exist and be the sealed one, and the config the plan's. Then shards.py's own check
        and record (commit, code, targets, settings, GPU, library versions against every earlier job of every shard).
merge   merge_shards.py's checks and output, with every row in the order an unsharded flops_run.py writes it (one condition at a time),
        plus: every shard ran the sealed plan, and every record is on the ledger with one GPU name. It prints any oom or nan record. A
        problem stops it with nothing written (MERGE STOPPED, INCOMPLETE).
Reads no answers.
"""
import argparse
import collections
import json
import os
import sys

from accuracy_tables.hpc.all350 import merge_shards, prepare, seed_results, shards

N_SHARDS = 3
LEDGER = "gpu_ledger.jsonl"                 # flops_run.LEDGER, named here too so this module needs no torch
READ_ONLY = shards.PROTECTED + ("qwen3_350", "qwen3_350_pilot", "qwen3_350_check", "qwen25_350", "smolvlm_flops", "smolvlm_350")


def refuse_read_only(root):
    """Stops if root is inside a results folder this section only reads."""
    real = os.path.realpath(root)
    for name in READ_ONLY:
        p = os.path.realpath(os.path.join(seed_results.results_root(), name))
        if real == p or real.startswith(p + os.sep):
            sys.exit(f"{root} is inside results/{name}, which is only read")


def plan_files(root, model):
    cal = os.path.join(root, "calibrate", model)
    return dict(plan=os.path.join(cal, "flops_plan.json"), targets=os.path.join(cal, "flops_targets.csv"),
                seal=os.path.join(cal, "flops_plan.sha256"), gpu=os.path.join(cal, "gpu_types.txt"))


# ---------------------------------------------------------------------------- split
def split(source, root):
    refuse_read_only(root)
    manifest = shards.read_manifest(source)
    counts = prepare.question_counts()
    found = shards.manifest_problems(manifest, counts)
    if manifest.get("scope") != "all" or manifest.get("n_shards") != N_SHARDS:
        found.append(f"scope {manifest.get('scope')} with {manifest.get('n_shards')} shards, not all with {N_SHARDS}")
    if (sum(counts.values()), len(counts)) != prepare.TOTAL:
        found.append(f"{sum(counts.values())} questions on {len(counts)} videos, expected {prepare.TOTAL}")
    if found:
        sys.exit(f"the split in {shards.manifest_path(source)} is wrong: " + "; ".join(found[:5]) + ". Stop and tell the user.")
    wrote = shards.write_manifest(root, manifest)
    print(f"the split sections 12 and 13 ran on, from {shards.manifest_path(source)}: "
          f"{'copied' if wrote else 'already here, unchanged'} to {shards.manifest_path(root)}", flush=True)
    for s in manifest["shards"]:
        print(f"   shard {s['shard']}: {s['questions']} questions on {len(s['videos'])} videos", flush=True)
    return manifest


# ---------------------------------------------------------------------------- the sealed plan
def seal(root, model):
    refuse_read_only(root)
    f = plan_files(root, model)
    for k in ("plan", "targets"):
        if not os.path.exists(f[k]):
            sys.exit(f"no {f[k]}: the calibrate job writes it")
    if os.path.exists(f["seal"]):
        sys.exit(f"{f['seal']} exists: the plan was sealed already. Stop and tell the user.")
    line = dict(plan_sha256=shards.file_sha256(f["plan"]), targets_sha256=shards.file_sha256(f["targets"]))
    shards.write_text(f["seal"], json.dumps(line) + "\n")
    print(f"sealed: plan {line['plan_sha256'][:12]}, targets {line['targets_sha256'][:12]} in {f['seal']}", flush=True)


def sealed_plan(root, model, config=None):
    """The plan, if it is the sealed one (and was made with config) -> (plan, its files); stops otherwise."""
    f = plan_files(root, model)
    for k in ("plan", "targets", "seal"):
        if not os.path.exists(f[k]):
            sys.exit(f"no {f[k]}: run the calibrate job first")
    with open(f["seal"]) as fh:
        want = json.load(fh)
    if shards.file_sha256(f["plan"]) != want["plan_sha256"] or shards.file_sha256(f["targets"]) != want["targets_sha256"]:
        sys.exit(f"the plan or the targets in {os.path.dirname(f['plan'])} aren't the ones calibrate sealed. Stop and tell the user.")
    with open(f["plan"]) as fh:
        plan = json.load(fh)
    if plan["model"] != model:
        sys.exit(f"the plan is for {plan['model']}, not {model}. Stop and tell the user.")
    if config is not None and shards.file_sha256(config) != plan["config_sha256"]:
        sys.exit(f"{config} changed since the plan was made. Stop and tell the user.")
    return plan, f


def start(root, shard, model, config, rows, workers, gpu, versions, repo=shards.REPO):
    refuse_read_only(root)
    _, f = sealed_plan(root, model, config)
    shards.start(root, shard, model, config, f["targets"], rows, workers, gpu, versions, repo)


# ---------------------------------------------------------------------------- the merge
def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def error_counts(root, model, rows, manifest):
    """{(shard, row): {error: records}} of the shards' records with an error (oom, nan)."""
    out = {}
    for s in range(manifest["n_shards"]):
        for row in rows:
            path = os.path.join(shards.shard_folder(root, s, model), f"{row}.jsonl")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for line in f:
                    try:
                        err = json.loads(line).get("error")
                    except ValueError:
                        continue
                    if err is not None:
                        out.setdefault((s, row), collections.Counter())[err] += 1
    return out


def folder_gpus(folder, rows, name):
    """One results folder's records attributed to the GPU on its ledger -> (Counter {GPU name: records}, messages)."""
    gpus, found = collections.Counter(), []
    path = os.path.join(folder, LEDGER)
    entries = read_jsonl(path) if os.path.exists(path) else []
    for row in rows:
        n = count_lines(os.path.join(folder, f"{row}.jsonl"))
        mine = sorted((e for e in entries if e["row"] == row), key=lambda e: e["first_line"])
        if n and (not mine or mine[0]["first_line"] != 0):
            found.append(f"{name} {row}: records before the first ledger line")
            continue
        for i, e in enumerate(mine):
            end = mine[i + 1]["first_line"] if i + 1 < len(mine) else n
            if min(end, n) > e["first_line"]:
                gpus[e["gpu"]] += min(end, n) - e["first_line"]
    return gpus, found


def ledger_gpus(root, model, rows, manifest):
    """Each shard's records attributed to the GPU on the ledger -> ({GPU name: records}, messages)."""
    gpus, found = collections.Counter(), []
    for s in range(manifest["n_shards"]):
        got, msgs = folder_gpus(shards.shard_folder(root, s, model), rows, f"shard {s}")
        gpus.update(got)
        found += msgs
    if len(gpus) > 1:
        found.append(f"records from more than one GPU on the ledger: {dict(gpus)}")
    return dict(gpus), found


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def run_info_problems(root, model, manifest, plan_sha):
    found = []
    for s in range(manifest["n_shards"]):
        path = os.path.join(shards.shard_folder(root, s, model), "run_info_flops.json")
        if not os.path.exists(path):
            found.append(f"shard {s}: no run_info_flops.json")
            continue
        with open(path) as f:
            got = json.load(f).get("plan_sha256")
        if got != plan_sha:
            found.append(f"shard {s}: ran plan {str(got)[:12]}, not the sealed {plan_sha[:12]}")
    return found


def by_condition(lines, rank):
    """A row's lines in the order flops_run.py writes it whole: condition by condition, then by video and question."""
    def key(line):
        r = json.loads(line)
        return rank[r["condition"]], r["video"], r["qid"]
    return sorted(lines, key=key)


def merge(root, model, config, rows):
    refuse_read_only(root)
    plan, f = sealed_plan(root, model)
    manifest = shards.read_manifest(root)
    errors = error_counts(root, model, rows, manifest)
    for (s, row), n in sorted(errors.items()):
        print(f"   records with an error: shard {s} {row}: {dict(n)}", flush=True)
    print(f"oom / nan records: {sum(sum(n.values()) for n in errors.values())}", flush=True)
    merged, info, found = merge_shards.plan_merge(root, model, f["targets"], config, rows, f["gpu"])
    wanted = merge_shards.read_targets(f["targets"])
    if not found:
        merged = {row: by_condition(lines, {c: i for i, c in enumerate(wanted[row])}) for row, lines in merged.items()}
    gpus, ledger = ledger_gpus(root, model, rows, manifest)
    found += ledger + run_info_problems(root, model, manifest, shards.file_sha256(f["plan"]))
    info.update(gpu_per_record=gpus, plan_sha256=shards.file_sha256(f["plan"]))
    for line in found[:40]:
        print("  ", line, flush=True)
    if found:
        sys.exit(f"MERGE STOPPED, INCOMPLETE: {len(found)} problems, and nothing was written. Rerun the unfinished shard, or tell the user.")
    merge_shards.write_final(root, model, merged, info)
    for s in info["shards"]:
        print(f"   shard {s['shard']}: {s['questions']} questions on {s['videos']} videos, node {', '.join(s['nodes'])}, "
              f"job {', '.join(s['jobs'])}", flush=True)
    print(f"   {info['gpu']}, commit {info['commit']}; GPU per record: {gpus}", flush=True)
    print(f"MERGED: {sum(info['records'].values())} records of {len(rows)} rows into {os.path.join(root, model)}", flush=True)
    return info


# ---------------------------------------------------------------------------- commands
def start_cmd(a):
    import torch
    import transformers
    gpu = torch.cuda.get_device_properties(0).name
    versions = dict(python=sys.version.split()[0], torch=torch.__version__, transformers=transformers.__version__)
    start(a.root, a.shard, a.model, a.config, a.rows.split(","), a.workers, gpu, versions)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("split")
    s.add_argument("--source", required=True)
    s.add_argument("--root", required=True)
    s.set_defaults(run=lambda a: split(a.source, a.root))
    s = sub.add_parser("seal")
    s.add_argument("--root", required=True)
    s.add_argument("--model", required=True)
    s.set_defaults(run=lambda a: seal(a.root, a.model))
    s = sub.add_parser("start")
    for k in ("--root", "--model", "--config", "--rows"):
        s.add_argument(k, required=True)
    s.add_argument("--shard", type=int, required=True)
    s.add_argument("--workers", type=int, required=True)
    s.set_defaults(run=start_cmd)
    s = sub.add_parser("merge")
    for k in ("--root", "--model", "--config", "--rows"):
        s.add_argument(k, required=True)
    s.set_defaults(run=lambda a: merge(a.root, a.model, a.config, a.rows.split(",")))
    a = ap.parse_args(argv)
    a.run(a)


if __name__ == "__main__":
    main()

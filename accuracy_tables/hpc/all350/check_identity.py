"""At full budget, four baselines keep every token, so they must give Uniform K's answers. Prints no accuracy.

    python -u -m accuracy_tables.hpc.all350.check_identity --write <file> --config <config>     a config with those rows at full budget
    python -u -m accuracy_tables.hpc.all350.check_identity --compare <folder>                   a model's results scored with it

Run's parity check covers a model's own path, not a baseline's. This checks that the baselines' encoding, packing and prompt reach the same
answers when nothing is dropped: a 9 x 9 pool of SmolVLM's 9 x 9 grid, DTD dropping nothing, DyCoke pruning nothing, StreamingTOM keeping all
tokens. FastVID, PruneVid and both GazeVLMs can't reproduce stock input, so they only get parity and the thread check.
"""
import argparse
import json
import os
import sys

import yaml

from accuracy_tables import common

ROWS = ("avgpool", "dtd", "dycoke", "streamingtom")


def write_config(config, path):
    """A copy of the config with the 4 rows at full budget and every other setting as it was."""
    with open(config) as f:
        raw = yaml.safe_load(f)
    raw["avgpool"] = dict(budget=[1.0])
    raw["dtd"] = dict(settings=[dict(threshold=0.0, absolute=False)])
    raw["dycoke"] = dict(settings=[dict(window=common.DEFAULTS["dycoke"]["settings"][0]["window"], prune=0.0)])
    raw["streamingtom"] = dict(ratio=[1.0])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)


def read_records(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compare(uniform, records):
    """Each record against Uniform K on the same question -> dict(records, missing, answers, tokens, frames, max_prob_diff)."""
    ref = {r["qid"]: r for r in uniform}
    out = dict(records=len(records), missing=0, answers=0, tokens=0, frames=0, max_prob_diff=0.0)
    for r in records:
        u = ref.get(r["qid"])
        if u is None:
            out["missing"] += 1
            continue
        out["answers"] += r.get("pred") != u.get("pred")
        out["tokens"] += r.get("n_image_tokens") != u.get("n_image_tokens")
        out["frames"] += r.get("n_frames") != u.get("n_frames")
        if r.get("p") and u.get("p"):
            out["max_prob_diff"] = max(out["max_prob_diff"], max(abs(a - b) for a, b in zip(r["p"], u["p"])))
    return out


def problems(r):
    return r["missing"] + r["answers"] + r["tokens"] + r["frames"] + (r["records"] == 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", help="write a config with the 4 rows at full budget to this file")
    ap.add_argument("--config", help="the config to copy, with --write")
    ap.add_argument("--compare", help="a model's results folder, scored with that config")
    a = ap.parse_args()
    if a.write:
        write_config(a.config, a.write)
        print(f"written {a.write}", flush=True)
        return
    uniform = read_records(os.path.join(a.compare, "uniform.jsonl"))
    bad = 0
    for row in ROWS:
        r = compare(uniform, read_records(os.path.join(a.compare, f"{row}.jsonl")))
        print(f"{row}: {r}", flush=True)
        bad += problems(r)
    if bad:
        sys.exit("NOT THE SAME AS UNIFORM K: stop and tell the user")
    print("SAME AS UNIFORM K", flush=True)


if __name__ == "__main__":
    main()

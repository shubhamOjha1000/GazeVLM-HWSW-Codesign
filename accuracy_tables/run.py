"""Score the accuracy-table rows Uniform K, Blind and Ours for one or more VLMs.

    python -m accuracy_tables.run --config accuracy_tables/config.yaml --model qwen2.5
    python -m accuracy_tables.run --config accuracy_tables/config.yaml --model all --rows uniform,blind

Run from the repository root (it imports vlmeval/ and eyehead/). Results are appended one
question at a time to <out_dir>/<model>/<row>.jsonl, so an interrupted run resumes where it
stopped; summarize.py turns them into the table.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accuracy_tables import common  # noqa: E402
from accuracy_tables.models import ADAPTERS, build_messages  # noqa: E402

ROWS = ("uniform", "blind", "ours")
BLIND = "blind (question and options only)"


def uniform_name(k):
    return f"uniform K={k} (stock input)"


def ours_name(policy, label):
    return f"ours {policy} gate + fovea {label}"


def _done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                done.add((r["condition"], r["qid"]))
    return done


@torch.no_grad()
def run_blind(ad, items, out_path, log_every=50):
    """Question and options only, through each model's own text path; resumable."""
    done = _done(out_path)
    todo = [it for it in items if (BLIND, it["qid"]) not in done]
    print(f"{len(done)} already scored, {len(todo)} to go -> {out_path}")
    t_start = time.time()
    with open(out_path, "a") as fh:
        for n, it in enumerate(todo, 1):
            t0 = time.time()
            ids = ad.blind_ids(build_messages(it["question"], it["options"], [], 0))
            p = torch.softmax(ad.blind_logits(ids)[ad.letter_ids].float(), -1).cpu().numpy()
            err = None if np.isfinite(p).all() else "nan"
            pred = "ABCDE"[int(np.nanargmax(p))] if np.isfinite(p).any() else "?"
            fh.write(json.dumps(dict(
                condition=BLIND, qid=int(it["qid"]), video=it["video"], answer=it["answer"], pred=pred,
                correct=int(pred == it["answer"]), p=[round(float(v), 4) for v in p], n_frames=0,
                n_image_tokens=0, n_tokens=int(ids.shape[1]), sec=round(time.time() - t0, 3), error=err)) + "\n")
            fh.flush()
            if n % log_every == 0:
                print(f"   {n}/{len(todo)}  {(time.time() - t_start) / n:.2f} s each", flush=True)
    print(f"done: {len(todo)} new in {(time.time() - t_start) / 60:.1f} min")


def run_model(key, cfg, data, rows, info_extra=None):
    run, p = cfg["run"], cfg["paths"]
    ad = ADAPTERS[key](cfg["models"].get(key), cfg).load()
    out_dir = os.path.join(p["out_dir"], key)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {key}: {ad.model_id}, {torch.cuda.memory_allocated() / 1e9:.1f} GB on GPU")
    cap = run["max_frames"].get(key)
    info = dict(model=key, model_id=ad.model_id, rows=list(rows), config=cfg, max_frames=cap,
                stock_tokens_per_frame=ad.stock_tokens, **(info_extra or {}))
    try:
        if run["parity"]:
            q0 = data["qs"].sort_values(["video", "qid"]).iloc[0]
            m0 = data["metas"][q0.video]
            k0 = int(np.argmin(np.abs(np.asarray(m0["ts"]) - q0.t0))) if np.isfinite(q0.t0) else 0
            from vlmeval.frames import frame_path
            ok, rep = ad.parity(dict(path=frame_path(m0, k0)), q0.question, q0.options)
            print("parity, our pipeline vs the model's own forward on one frame:")
            for k, v in rep.items():
                print(f"   {k:<44} {v}")
            info["parity"] = dict(ok=bool(ok), report=rep)
            if not ok:
                raise RuntimeError(f"{key}: parity failed; nothing scored (see the report above)")
            print("parity OK")

        if "uniform" in rows:
            k = min(run["K"], cap) if cap else run["K"]
            frames = {v: common.cap_frames(f, cap) for v, f in data["uniform"].items()}
            items = common.make_items(data["qs"], frames)
            print(f"\n--- Uniform K: {k} frames per video at stock input, {len(items)} questions")
            ad.score(items, {uniform_name(k): ad.stock_condition()}, os.path.join(out_dir, "uniform.jsonl"))

        if "blind" in rows:
            print(f"\n--- Blind: question and options only")
            run_blind(ad, common.make_items(data["qs"], {}), os.path.join(out_dir, "blind.jsonl"))

        if "ours" in rows:
            frames = {v: common.cap_frames(f, cap) for v, f in data["gate"].items()}
            items = common.make_items(data["qs"], frames)
            conds = {ours_name(run["policy"], ad.setting_label(s)): ad.fovea_condition(s)
                     for s in ad.fovea_settings(cfg["ours"])}
            print(f"\n--- Ours: {run['policy']} gate frames (mean "
                  f"{np.mean([len(it['frames']) for it in items]):.1f} per question) x {len(conds)} fovea settings")
            ad.score(items, conds, os.path.join(out_dir, "ours.jsonl"))
    finally:
        try:
            import transformers
            info["versions"] = dict(transformers=transformers.__version__, torch=torch.__version__)
        except Exception:
            pass
        with open(os.path.join(out_dir, "run_info.json"), "w") as f:
            json.dump(info, f, indent=1, default=str)
        ad.free()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML config (copy config.example.yaml)")
    ap.add_argument("--model", required=True, help=f"one of {', '.join(ADAPTERS)}, a comma list, or all")
    ap.add_argument("--rows", default=",".join(ROWS), help=f"comma list of {', '.join(ROWS)} (default: all)")
    ap.add_argument("--videos", help="override run.videos: test or all")
    ap.add_argument("--no-parity", action="store_true", help="skip the parity check (not recommended)")
    a = ap.parse_args(argv)

    cfg = common.load_config(a.config)
    if a.videos:
        cfg["run"]["videos"] = a.videos
    if a.no_parity:
        cfg["run"]["parity"] = False
    keys = list(ADAPTERS) if a.model == "all" else [k.strip() for k in a.model.split(",")]
    rows = [r.strip() for r in a.rows.split(",")]
    bad = [k for k in keys if k not in ADAPTERS] + [r for r in rows if r not in ROWS]
    if bad:
        ap.error(f"unknown model(s) / row(s): {bad}")

    data = common.load_data(cfg)
    print(common.describe(data))
    failed = []
    for key in keys:
        try:
            run_model(key, cfg, data, rows)
        except Exception as ex:                       # keep going with the next model
            failed.append(key)
            print(f"!! {key} stopped: {type(ex).__name__}: {ex}")
            if len(keys) == 1:
                raise
    if failed:
        sys.exit(f"models that did not finish: {failed}")


if __name__ == "__main__":
    main()

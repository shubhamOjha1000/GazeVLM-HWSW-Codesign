"""Score the accuracy-table rows (Uniform K, Blind, Ours, GazeVLM, StreamingTOM, DyCoke, FastVID, PruneVid) for one or more VLMs.

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

ROWS = ("uniform", "blind", "ours", "gazevlm", "streamingtom", "dycoke", "fastvid", "prunevid")
BLIND = "blind (question and options only)"


def uniform_name(k):
    return f"uniform K={k} (stock input)"


def ours_name(policy, label):
    return f"ours {policy} gate + fovea {label}"


def gazevlm_name(rho, g, k):
    """Every parameter in the name, so settings never mix in one result file."""
    return (f"GazeVLM rho={float(rho):g} (K={k}, sigma={g['sigma']:g}, R_min={g['r_min']:g}, "
            f"thumb {g['thumb']}x{g['thumb']}, {g['gaze']} gaze)")


def streamingtom_name(ratio, s, k):
    """Every parameter in the name, so settings never mix in one result file."""
    return (f"StreamingTOM CTR G={float(ratio):g}N (K={k}, tau={s['tau']:g}, k={s['knn']}, beta={s['beta']:g}"
            f"{', released DPC' if s['released_dpc'] else ''})")


def dycoke_name(window, prune, k):
    return f"DyCoke TTM prune={float(prune):g} window={int(window)} (K={k})"


def fastvid_name(r, s, k):
    """Every parameter in the name, so settings never mix in one result file."""
    return (f"FastVID r={float(r):g} (K={k}, c={s['c']}, tau={s['tau']:g}, ignore={s['ignore']:g}, d={s['d']:g}, "
            f"p={s['p']}, beta={s['beta']:g})")


def prunevid_name(c, s, k):
    """Every parameter in the name, so settings never mix in one result file."""
    return (f"PruneVid tau={float(c['tau']):g} beta={float(c['beta']):g} (K={k}, gamma={float(s['gamma']):g}, "
            f"k={int(s['knn'])}, sets > {int(s['min_tokens'])} merged)")


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

        k = min(run["K"], cap) if cap else run["K"]
        uniform_items = common.make_items(data["qs"], {v: common.cap_frames(f, cap) for v, f in data["uniform"].items()})
        if "uniform" in rows:
            print(f"\n--- Uniform K: {k} frames per video at stock input, {len(uniform_items)} questions")
            ad.score(uniform_items, {uniform_name(k): ad.stock_condition()}, os.path.join(out_dir, "uniform.jsonl"))

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

        if "gazevlm" in rows:
            from vlmeval import gazevlm as gz
            g = cfg["gazevlm"]
            conds = {gazevlm_name(r, g, k): dict(rho=float(r), sigma=float(g["sigma"]), r_min=float(g["r_min"]),
                                                 thumb=int(g["thumb"]), gaze=g["gaze"]) for r in g["rho"]}
            print(f"\n--- GazeVLM: the {k} Uniform K frames per video, ROI + {g['thumb']}x{g['thumb']} thumbnail "
                  f"tokens, rho in {list(g['rho'])}")
            gz.run_questions(ad.gazevlm_backend(), uniform_items, conds, os.path.join(out_dir, "gazevlm.jsonl"),
                             build_messages)

        if "streamingtom" in rows:
            from vlmeval import streamingtom as stm
            s = cfg["streamingtom"]
            conds = {streamingtom_name(r, s, k): dict(ratio=float(r), tau=float(s["tau"]), knn=int(s["knn"]),
                                                      beta=float(s["beta"]), released_dpc=bool(s["released_dpc"]))
                     for r in s["ratio"]}
            print(f"\n--- StreamingTOM (CTR): the {k} Uniform K frames per video in time order, G / N in {list(s['ratio'])}")
            stm.run_questions(ad.streamingtom_backend(), uniform_items, conds,
                              os.path.join(out_dir, "streamingtom.jsonl"), build_messages)

        if "dycoke" in rows:
            from vlmeval import dycoke as dy
            conds = {dycoke_name(s["window"], s["prune"], k): dict(window=int(s["window"]), prune=float(s["prune"]))
                     for s in cfg["dycoke"]["settings"]}
            print(f"\n--- DyCoke (TTM): the {k} Uniform K frames per video in windows; settings "
                  f"{[(c['window'], c['prune']) for c in conds.values()]}")
            dy.run_questions(ad.dycoke_backend(), uniform_items, conds, os.path.join(out_dir, "dycoke.jsonl"),
                             build_messages)

        if "fastvid" in rows:
            from vlmeval import fastvid as fv
            s = cfg["fastvid"]
            conds = {fastvid_name(r, s, k): dict(r=float(r), c=int(s["c"]), tau=float(s["tau"]),
                                                 ignore=float(s["ignore"]), d=float(s["d"]), p=int(s["p"]),
                                                 beta=float(s["beta"])) for r in s["r"]}
            print(f"\n--- FastVID: the {k} Uniform K frames per video (offline), r in {list(s['r'])}")
            fv.run_questions(ad.fastvid_backend(), uniform_items, conds, os.path.join(out_dir, "fastvid.jsonl"),
                             build_messages)

        if "prunevid" in rows:
            from vlmeval import prunevid as pv
            s = cfg["prunevid"]
            conds = {prunevid_name(c, s, k): dict(tau=float(c["tau"]), beta=float(c["beta"]), gamma=float(s["gamma"]),
                                                  knn=int(s["knn"]), min_tokens=int(s["min_tokens"]))
                     for c in s["settings"]}
            print(f"\n--- PruneVid (token merging): the {k} Uniform K frames per video (offline); settings "
                  f"{[(c['tau'], c['beta']) for c in conds.values()]} (tau, beta)")
            pv.run_questions(ad.prunevid_backend(), uniform_items, conds, os.path.join(out_dir, "prunevid.jsonl"),
                             build_messages)
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

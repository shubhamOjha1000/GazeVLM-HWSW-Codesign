"""Score the accuracy-table rows (Uniform K, Blind, Ours, GazeVLM, StreamingTOM, DyCoke, FastVID, PruneVid, DTD,
Uniform K + average pooling, the two-view GazeVLM) for one or more VLMs.

    python -m accuracy_tables.run --config accuracy_tables/config.yaml --model qwen2.5
    python -m accuracy_tables.run --config accuracy_tables/config.yaml --model all --rows uniform,blind

Smoke test (does every method run on every model?): one video, one question, a separate output folder,
and a failing method does not stop the others; a pass / fail table is printed at the end:

    python -m accuracy_tables.run --config accuracy_tables/config.yaml --model all \
        --videos loc2_script1_seq2_rec1 --max-questions 1 --out-dir /content/smoke --keep-going

Run from the repository root (it imports vlmeval/ and eyehead/). Results are appended one
question at a time to <out_dir>/<model>/<row>.jsonl, so an interrupted run resumes where it
stopped; summarize.py turns them into the table.
"""

import argparse
import contextlib
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accuracy_tables import common  # noqa: E402
from accuracy_tables.models import ADAPTERS, build_messages  # noqa: E402

ROWS = ("uniform", "blind", "ours", "gazevlm", "streamingtom", "dycoke", "fastvid", "prunevid", "dtd", "avgpool", "gazevlm2")
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


def dtd_name(s, k):
    if s["absolute"]:
        return f"DTD feature tau={float(s['threshold']):g} (absolute, K={k})"
    return f"DTD feature drop={float(s['threshold']):g} (relative, K={k})"


def avgpool_name(side, k):
    return f"uniform K={k} + average pooling {side}x{side}"


def gazevlm2_name(tokens, g, k):
    """Every parameter in the name, so settings never mix in one result file."""
    return (f"GazeVLM two-view B={int(tokens)} per frame (K={k}, alpha={float(g['alpha']):g}, "
            f"beta={float(g['beta']):g}, fixation gaze)")


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


def _file_errors(path):
    """(records, records with an error) in a result file."""
    n = bad = 0
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                n += 1
                bad += json.loads(line).get("error") is not None
    return n, bad


@contextlib.contextmanager
def _row(name, status, keep_going, path=None):
    """Runs one row; with keep_going a failure is recorded in status and the next row runs."""
    t0 = time.time()
    try:
        yield
        n, bad = _file_errors(path) if path else (0, 0)
        status[name] = f"ok ({n} records{f', {bad} with errors (oom / nan)' if bad else ''}, {time.time() - t0:.0f} s)"
    except Exception as ex:
        if not keep_going:
            raise
        traceback.print_exc()
        status[name] = f"FAILED: {type(ex).__name__}: {str(ex)[:300]}"
        torch.cuda.empty_cache()


def check_parity(ad, data, key):
    """The model's parity report on the first question's keyframe (it also fixes how prompts are tokenized for this
    transformers version, so run it before scoring) -> report; raises if it fails."""
    q0 = data["qs"].sort_values(["video", "qid"]).iloc[0]
    m0 = data["metas"][q0.video]
    k0 = int(np.argmin(np.abs(np.asarray(m0["ts"]) - q0.t0))) if np.isfinite(q0.t0) else 0
    from vlmeval.frames import frame_path
    ok, rep = ad.parity(dict(path=frame_path(m0, k0)), q0.question, q0.options)
    print("parity, our pipeline vs the model's own forward on one frame:")
    for k, v in rep.items():
        print(f"   {k:<44} {v}")
    if not ok:
        raise RuntimeError(f"{key}: parity failed; nothing scored (see the report above)")
    print("parity OK")
    return rep


def row_jobs(ad, cfg, data, rows):
    """One job per requested row: dict(row, msg, items, conds, score), score(items, conds, out_path) scoring any
    subset of the row's conditions on any subset of its items (run_model passes all of them; system_perf/ one
    condition and one question at a time)."""
    run, key = cfg["run"], ad.key
    cap = run["max_frames"].get(key)
    k = min(run["K"], cap) if cap else run["K"]
    uniform_items = common.make_items(data["qs"], {v: common.cap_frames(f, cap) for v, f in data["uniform"].items()})
    jobs = []
    add = lambda row, msg, items, conds, score: jobs.append(dict(row=row, msg=msg, items=items, conds=conds, score=score))
    if "uniform" in rows:
        add("uniform", f"--- Uniform K: {k} frames per video at stock input, {len(uniform_items)} questions",
            uniform_items, {uniform_name(k): ad.stock_condition()}, ad.score)
    if "blind" in rows:
        add("blind", "--- Blind: question and options only", common.make_items(data["qs"], {}), {BLIND: None},
            lambda items, conds, path: run_blind(ad, items, path))
    if "ours" in rows:
        frames = {v: common.cap_frames(f, cap) for v, f in data["gate"].items()}
        items = common.make_items(data["qs"], frames)
        conds = {ours_name(run["policy"], ad.setting_label(s)): ad.fovea_condition(s) for s in ad.fovea_settings(cfg["ours"])}
        add("ours", f"--- Ours: {run['policy']} gate frames (mean {np.mean([len(it['frames']) for it in items]):.1f} per "
            f"question) x {len(conds)} fovea settings", items, conds, ad.score)
    if "gazevlm" in rows:
        from vlmeval import gazevlm as gz
        g = cfg["gazevlm"]
        conds = {gazevlm_name(r, g, k): dict(rho=float(r), sigma=float(g["sigma"]), r_min=float(g["r_min"]),
                                             thumb=int(g["thumb"]), gaze=g["gaze"]) for r in g["rho"]}
        add("gazevlm", f"--- GazeVLM: the {k} Uniform K frames per video, ROI + {g['thumb']}x{g['thumb']} thumbnail tokens, "
            f"rho in {list(g['rho'])}", uniform_items, conds,
            lambda items, conds, path: gz.run_questions(ad.gazevlm_backend(), items, conds, path, build_messages))
    if "streamingtom" in rows:
        from vlmeval import streamingtom as stm
        s = cfg["streamingtom"]
        conds = {streamingtom_name(r, s, k): dict(ratio=float(r), tau=float(s["tau"]), knn=int(s["knn"]),
                                                  beta=float(s["beta"]), released_dpc=bool(s["released_dpc"]))
                 for r in s["ratio"]}
        add("streamingtom", f"--- StreamingTOM (CTR): the {k} Uniform K frames per video in time order, G / N in "
            f"{list(s['ratio'])}", uniform_items, conds,
            lambda items, conds, path: stm.run_questions(ad.streamingtom_backend(), items, conds, path, build_messages))
    if "dycoke" in rows:
        from vlmeval import dycoke as dy
        conds = {dycoke_name(s["window"], s["prune"], k): dict(window=int(s["window"]), prune=float(s["prune"]))
                 for s in cfg["dycoke"]["settings"]}
        add("dycoke", f"--- DyCoke (TTM): the {k} Uniform K frames per video in windows; settings "
            f"{[(c['window'], c['prune']) for c in conds.values()]}", uniform_items, conds,
            lambda items, conds, path: dy.run_questions(ad.dycoke_backend(), items, conds, path, build_messages))
    if "fastvid" in rows:
        from vlmeval import fastvid as fv
        s = cfg["fastvid"]
        conds = {fastvid_name(r, s, k): dict(r=float(r), c=int(s["c"]), tau=float(s["tau"]), ignore=float(s["ignore"]),
                                             d=float(s["d"]), p=int(s["p"]), beta=float(s["beta"])) for r in s["r"]}
        add("fastvid", f"--- FastVID: the {k} Uniform K frames per video (offline), r in {list(s['r'])}", uniform_items, conds,
            lambda items, conds, path: fv.run_questions(ad.fastvid_backend(), items, conds, path, build_messages))
    if "prunevid" in rows:
        from vlmeval import prunevid as pv
        s = cfg["prunevid"]
        conds = {prunevid_name(c, s, k): dict(tau=float(c["tau"]), beta=float(c["beta"]), gamma=float(s["gamma"]),
                                              knn=int(s["knn"]), min_tokens=int(s["min_tokens"])) for c in s["settings"]}
        add("prunevid", f"--- PruneVid (token merging): the {k} Uniform K frames per video (offline); settings "
            f"{[(c['tau'], c['beta']) for c in conds.values()]} (tau, beta)", uniform_items, conds,
            lambda items, conds, path: pv.run_questions(ad.prunevid_backend(), items, conds, path, build_messages))
    if "dtd" in rows:
        from vlmeval import dtd
        conds = {dtd_name(s, k): dict(threshold=float(s["threshold"]), absolute=bool(s["absolute"]))
                 for s in cfg["dtd"]["settings"]}
        add("dtd", f"--- DTD (TimeChat-Online): the {k} Uniform K frames per video in time order; settings "
            f"{[(c['threshold'], 'absolute' if c['absolute'] else 'relative') for c in conds.values()]}", uniform_items, conds,
            lambda items, conds, path: dtd.run_questions(ad.dtd_backend(), items, conds, path, build_messages))
    if "avgpool" in rows:
        from vlmeval import avgpool as ap
        be = ad.avgpool_backend()
        sides = sorted({ap.pool_side(float(b), ad.stock_tokens, be.side, getattr(be, "row_breaks", False))
                        for b in cfg["avgpool"]["budget"]})
        conds = {avgpool_name(s, k): dict(side=s) for s in sides}
        add("avgpool", f"--- Uniform K + average pooling: the {k} Uniform K frames, each pooled to "
            f"{[f'{s}x{s}' for s in sides]} tokens (budgets {list(cfg['avgpool']['budget'])} of {ad.stock_tokens} tokens "
            f"per frame)", uniform_items, conds,
            lambda items, conds, path: ap.run_questions(be, items, conds, path, build_messages))
    if "gazevlm2" in rows:
        from vlmeval import gazevlm2 as gv2
        g = cfg["gazevlm2"]
        tokens = sorted({max(2, int(round(float(b) * ad.stock_tokens))) for b in g["budget"]})
        conds = {gazevlm2_name(t, g, k): dict(tokens=t, alpha=float(g["alpha"]), beta=float(g["beta"])) for t in tokens}
        add("gazevlm2", f"--- GazeVLM (two-view): the {k} Uniform K frames, each a global and a gaze-centred view; "
            f"{tokens} tokens per frame (budgets {list(g['budget'])} of {ad.stock_tokens})", uniform_items, conds,
            lambda items, conds, path: gv2.run_questions(ad.gazevlm2_backend(), items, conds, path, build_messages))
    return jobs


def run_model(key, cfg, data, rows, info_extra=None, keep_going=False):
    """Scores the rows for one model -> {row: 'ok (...)' or 'FAILED: ...'}."""
    run, p = cfg["run"], cfg["paths"]
    ad = ADAPTERS[key](cfg["models"].get(key), cfg).load()
    out_dir = os.path.join(p["out_dir"], key)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {key}: {ad.model_id}, {torch.cuda.memory_allocated() / 1e9:.1f} GB on GPU")
    cap = run["max_frames"].get(key)
    info = dict(model=key, model_id=ad.model_id, rows=list(rows), config=cfg, max_frames=cap,
                stock_tokens_per_frame=ad.stock_tokens, **(info_extra or {}))
    status = {}
    out = lambda name: os.path.join(out_dir, f"{name}.jsonl")
    try:
        if run["parity"]:
            info["parity"] = dict(ok=True, report=check_parity(ad, data, key))
        for job in row_jobs(ad, cfg, data, rows):
            with _row(job["row"], status, keep_going, out(job["row"])):
                print("\n" + job["msg"])
                job["score"](job["items"], job["conds"], out(job["row"]))
    finally:
        try:
            import transformers
            info["versions"] = dict(transformers=transformers.__version__, torch=torch.__version__)
        except Exception:
            pass
        info["status"] = status
        with open(os.path.join(out_dir, "run_info.json"), "w") as f:
            json.dump(info, f, indent=1, default=str)
        ad.free()
    return status


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML config (copy config.example.yaml)")
    ap.add_argument("--model", required=True, help=f"one of {', '.join(ADAPTERS)}, a comma list, or all")
    ap.add_argument("--rows", default=",".join(ROWS), help=f"comma list of {', '.join(ROWS)} (default: all)")
    ap.add_argument("--videos", help="override run.videos: test, all, or a comma list of video names")
    ap.add_argument("--max-questions", type=int, help="score only the first N questions (by video, then qid)")
    ap.add_argument("--out-dir", help="override paths.out_dir (e.g. a scratch folder for a smoke test)")
    ap.add_argument("--keep-going", action="store_true",
                    help="a failing row or model does not stop the rest; a pass / fail table is printed at the end")
    ap.add_argument("--no-parity", action="store_true", help="skip the parity check (not recommended)")
    a = ap.parse_args(argv)

    cfg = common.load_config(a.config)
    if a.videos:
        cfg["run"]["videos"] = a.videos if a.videos in ("test", "all") else [v.strip() for v in a.videos.split(",")]
    if a.out_dir:
        cfg["paths"]["out_dir"] = os.path.abspath(os.path.expanduser(a.out_dir))
    if a.no_parity:
        cfg["run"]["parity"] = False
    keys = list(ADAPTERS) if a.model == "all" else [k.strip() for k in a.model.split(",")]
    rows = [r.strip() for r in a.rows.split(",")]
    bad = [k for k in keys if k not in ADAPTERS] + [r for r in rows if r not in ROWS]
    if bad:
        ap.error(f"unknown model(s) / row(s): {bad}")

    data = common.load_data(cfg)
    if a.max_questions:
        data["qs"] = data["qs"].sort_values(["video", "qid"]).head(a.max_questions)
    print(common.describe(data))
    failed, table = [], {}
    for key in keys:
        try:
            table[key] = run_model(key, cfg, data, rows, keep_going=a.keep_going)
        except Exception as ex:                       # keep going with the next model
            failed.append(key)
            table[key] = {"model": f"FAILED before scoring: {type(ex).__name__}: {str(ex)[:300]}"}
            print(f"!! {key} stopped: {type(ex).__name__}: {ex}")
            if a.keep_going:
                traceback.print_exc()
            elif len(keys) == 1:
                raise
            torch.cuda.empty_cache()
    if a.keep_going:
        print("\n=== pass / fail per model and row")
        for key, status in table.items():
            for row, s in status.items():
                print(f"   {key:<9} {row:<13} {s}")
        n_fail = sum(s.startswith("FAILED") for st in table.values() for s in st.values())
        print(f"=== {n_fail} failed of {sum(len(st) for st in table.values())}")
        if n_fail:
            sys.exit(1)
    if failed:
        sys.exit(f"models that did not finish: {failed}")


if __name__ == "__main__":
    main()

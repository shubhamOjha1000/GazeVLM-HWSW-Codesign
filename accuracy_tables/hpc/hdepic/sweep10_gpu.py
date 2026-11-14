"""The 10% FLOPs sweep on one model's GPU (sweep10.py holds the protocol). Diagnostic dev work only: writes $GV_ROOT/data/hdepic/sweep10/
<model>/ and nothing else; no results table, no test question.

    python -u -m accuracy_tables.hpc.hdepic.sweep10_gpu checks --model <m> --workers <W>     S1
    python -u -m accuracy_tables.hpc.hdepic.sweep10_gpu plan --model <m> --workers <W>       S2
    python -u -m accuracy_tables.hpc.hdepic.sweep10_gpu run --model <m> --workers <W>        S3

checks  the snapshot; parity; stock tokens per frame and every representation's tokens, encoded; identity: ours_res_1.00 and
        nogate_res_1.00 on Uniform's frames, and the twoview code with the whole frame alone, must give Uniform's answers (SAME ANSWERS,
        max_prob_diff 0.0), and the twoview's whole-frame view must be the stock input pixel for pixel; threads: the sweep's own scorer
        with loading threads against none; the FLOPs units (the LLM's prefill quadratic, checked against its config; one frame's encoder at
        every representation; the gate), then every representation scored on 2 questions under the counter, which must match them
        exactly; on the 2.2B, 50 of v2 Stage 1's Blind records rescored. Writes units.json. Ends with CHECKS DONE.
plan    each question's blind prompt length; the prompt fits for one and two images a frame; the targets (10% of each question's Uniform-64
        FLOPs, summed over the 940); K' for every condition (flops.choose_k: closest, the smaller on a tie); the baselines' knobs by the
        rho rule on their own K' frames (calibrate.py's counts); every condition scored on 2 questions under the counter (vision and
        prefill as planned, image tokens as planned); a timing run of 8 questions per condition and the projected hours. Writes plan.json
        and plan.csv. Ends with PLAN WRITTEN.
run     every planned condition on the 940 dev questions into answers/<condition>.jsonl, resumable; a line per condition in
        answers/gpu_ledger.jsonl. Ends with RUN DONE.
"""
import argparse
import contextlib
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

from accuracy_tables import common, flops, run
from accuracy_tables import calibrate as cal
from accuracy_tables import flops_calibrate as fc
from accuracy_tables.hpc.hdepic import sweep10 as sw
from accuracy_tables.hpc.hdepic import sweep10_views as views
from accuracy_tables.models import ADAPTERS, build_messages

IDENTITY_QUESTIONS, THREAD_QUESTIONS, VERIFY_QUESTIONS, TIMING_QUESTIONS, BLIND_RECHECK = 20, 10, 2, 8, 50
# Each knob over its whole range. calibrate.py's grids (sections 14 to 16, unchanged) stop at GazeVLM's rho 0.95, FastVID's r 0.505 and
# PruneVid's beta 0.60, which keep under the shares the sweep needs: GazeVLM's rho is a gaze-mass fraction, and the box holding 95% of
# a Gaussian of sigma 0.1 of the side is about half the side, a third of the tokens. Its share climbs steeply towards rho = 1 (the whole
# frame), hence the fine steps of 1 - rho down to 1e-10 and rho = 1 itself.
GRID_RHO = sorted(set(cal.GRID_RHO) | {round(1 - 10 ** (-x / 20), 12) for x in range(27, 201)} | {1.0})
GRID_R = [round(0.005 * i, 3) for i in range(2, 201)]            # 0.01 .. 1.00
GRID_BETA = [round(0.01 * i, 2) for i in range(1, 101)]          # 0.01 .. 1.00
RHO_SHORT_PTS = 0.0                                              # a method whose whole range keeps less than rho: flagged rho_short
PRUNE_KNOBS = ("dycoke", "dtd")                                  # knobs that say what to drop: the share falls as they rise
CONNECTED_SPAN = 0.25                                            # a knob whose whole range moves the share less than this isn't connected
DIP_TOKENS = dict(gazevlm=2.0)                                   # GazeVLM grows a small ROI to r_min about its centre, so its share can
                                                                 # dip a little at tiny rho; nothing else may fall as its keep knob rises
PREFLIGHT_CLIPS = 24                                             # clips the knob ranges are checked on before the slow counting
PROMPT_QUESTIONS, PROMPT_FRAMES = 24, (1, 4, 16, 64)


# ---------------------------------------------------------------------------- setup
def setup(model, workers):
    if model not in sw.MODELS:
        sys.exit(f"{model}: the sweep runs {sorted(sw.MODELS)}")
    cfg = common.load_config(sw.CONFIG)
    cfg["run"]["prefetch_workers"] = int(workers)
    cfg["paths"]["out_dir"] = sw.model_dir(model, "unused")
    data = common.load_data(cfg)
    if len(data["qs"]) != sw.N_DEV:
        sys.exit(f"{len(data['qs'])} questions loaded, not the {sw.N_DEV} dev questions. Stop and tell the user.")
    print(common.describe(data), flush=True)
    ad = ADAPTERS[model](cfg["models"].get(model), cfg).load()
    if ad.stock_tokens != sw.MODELS[model]["stock_tokens"]:
        sys.exit(f"{model}: {ad.stock_tokens} stock tokens, the sweep expects {sw.MODELS[model]['stock_tokens']}")
    be = views.backend(ad)
    if be.stock_side != sw.MODELS[model]["stock_side"]:
        sys.exit(f"{model}: a {be.stock_side} px stock frame, the sweep expects {sw.MODELS[model]['stock_side']}")
    return cfg, data, ad, be


def snapshot_from_path(path, model_id):
    """The commit hash in a cached file's path, <hub>/models--<org>--<name>/snapshots/<hash>/<file>, or None if the path isn't that."""
    if not isinstance(path, str):
        return None
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 4 and parts[-3] == "snapshots" and parts[-4] == "models--" + model_id.replace("/", "--"):
        return parts[-2]
    return None


def snapshot(ad):
    """(the snapshot the model's files were loaded from, its config.json path, the config's own _commit_hash). The path is resolved the way
    from_pretrained resolves it from the local cache: at the adapter's pinned revision (the 256M's) or refs/main."""
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(ad.model_id, "config.json", revision=ad.download_kwargs().get("revision"))
    return snapshot_from_path(path, ad.model_id), path, getattr(ad.model.config, "_commit_hash", None)


def check_snapshot(model, ad):
    h, path, attr = snapshot(ad)
    want = sw.MODELS[model]["snapshot"]
    print(f"SNAPSHOT {model}: {ad.model_id} at {h}, from {path}" + (f" (expected {want}...)" if want else ""), flush=True)
    if h is None:
        sys.exit(f"SNAPSHOT UNREADABLE: no snapshots/<hash>/config.json for {ad.model_id} in the cache. Stop and tell the user.")
    if attr and attr != h:
        sys.exit(f"SNAPSHOT DIFFERS: {model}'s config says {attr}, its files are {h}. Stop and tell the user.")
    if want and not h.startswith(want):
        sys.exit(f"SNAPSHOT DIFFERS: {model} loaded {h}, the sweep expects {want}. Stop and tell the user.")
    return h


def versions():
    import transformers
    return dict(torch=torch.__version__, transformers=transformers.__version__)


def fresh(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


class Counting:
    """flops_calibrate's counting context: the attention implementation in impl, torch's attention formulas taking grouped-query
    attention, and a watch for attention the counter can't see."""

    def __init__(self, ad):
        self.ad, self.impl, self.missed = ad, None, set()

    @contextlib.contextmanager
    def __call__(self):
        with flops.attention_implementation(self.ad.model, self.impl), flops.gqa_attention(), flops.AttentionWatch() as watch:
            yield
        self.missed |= watch.missed


def items_on(data, frames_of, n=None):
    items = common.make_items(data["qs"], frames_of)
    return items[:n] if n else items


def view_condition(ln):
    """(the condition's name in the records, its spec for views.run_views) for ours' and the no-gate control's lines."""
    who = f"ours gate L={ln['K']}" if ln["kind"] == "ours" else f"no gate L={ln['K']} (slot middles)"
    if ln["method"].endswith("_res"):
        return f"{who} + whole frame {ln['side']} px ({ln['tokens_per_frame']} tokens)", ("res", int(ln["side"]))
    where = "centre" if ln["method"].endswith("centre") else "gaze"
    return (f"{who} + two views: whole frame {ln['side']} px + {where} tile {ln['tile_src']} of {sw.SOURCE} px at {ln['side']} px "
            f"({ln['tokens_per_frame']} tokens)"), ("twoview", dict(gaze=where, tile=True))


def build_job(ad, cfg, data, be, inputs, ln, workers):
    """run.py's own job for the references, uniform_flops and the baselines (flops_calibrate.planned_job at the line's K' and knob);
    the sweep's scorer for ours and the no-gate control, on the budget rule's or the slot middles' frames at L = K'."""
    kind = ln["kind"]
    if kind in ("uniform", "blind"):
        job = run.row_jobs(ad, cfg, data, [kind])[0]
    elif kind in ("uniform_flops", "baseline"):
        job = fc.planned_job(ad, cfg, inputs(ln["K"]), dict(row=ln["row"], K=ln["K"], knob=ln["knob"], budget_pct=100 * sw.BUDGET))
    else:
        name, spec = view_condition(ln)
        items = items_on(data, sw.gate_frames(data, ln["K"], use_gate=kind == "ours"))
        job = dict(row=ln["name"], msg=f"--- {ln['name']}: {name}", items=items, conds={name: spec},
                   score=lambda items, conds, path: views.run_views(be, items, conds, path, workers))
    job["name"] = ln["name"]
    return job


# ---------------------------------------------------------------------------- S1
def tokens_table(model, be, fr):
    """Each representation encoded on one frame: its tokens against the plan's. -> problems."""
    bad = []
    g = sw.MODELS[model]
    for r in sw.representations(model):
        if r["name"].startswith(("avgpool", "gazevlm2")):
            continue
        cond = ("res", r["side"]) if r["name"].startswith("res") else ("twoview", dict(gaze="gaze", tile=True))
        got = [len(be.encode(v)[0]) for v in be.load(fr, cond)[0]]
        print(f"TOKENS {model} {r['name']}: {r['side']} px a view, {got} tokens encoded, {sum(got)} a frame (planned {r['tokens']}, rho "
              f"{r['rho_real']:.3f})", flush=True)
        if sum(got) != r["tokens"]:
            bad.append(f"{r['name']}: {sum(got)} tokens, planned {r['tokens']}")
    note = (" (each frame is its own image: Qwen's 2-frame temporal patch holds the frame twice, so a frame is 256 tokens, not 128)"
            if model.startswith("qwen") else "")
    print(f"STOCK TOKENS {model}: {g['stock_tokens']} a frame at {g['stock_side']} px{note}", flush=True)
    return bad


def pixels_of(be, view):
    return view[0] if isinstance(view, tuple) else view


def stock_reference(be, fr):
    """The model's stock input for a frame, made by its fovea module's own path (not this code's)."""
    if isinstance(be, views.Smol):
        from PIL import Image
        with Image.open(fr["path"]) as im:
            return be.m.stock_input(be.ip, im.convert("RGB"))["pv"]
    qf = be.qf
    return qf.resize01(qf.load_frame01(fr["path"], qf.IMG), be.stock_side)


def identity(model, ad, cfg, data, be, out, workers):
    """ours_res_1.00, nogate_res_1.00 and the twoview code's whole frame alone, on Uniform's frames, against Uniform -> problems."""
    out = fresh(os.path.join(out, "identity"))
    job = run.row_jobs(ad, cfg, data, ["uniform"])[0]
    items = job["items"][:IDENTITY_QUESTIONS]
    ref_path = os.path.join(out, "uniform.jsonl")
    job["score"](items, job["conds"], ref_path)
    ref = sw.read_jsonl(ref_path)
    S, bad = be.stock_side, []
    conds = {"ours_res_1.00 on Uniform's frames": ("res", S), "nogate_res_1.00 on Uniform's frames": ("res", S),
             "the twoview code, whole frame only, on Uniform's frames": ("twoview", dict(gaze="gaze", tile=False))}
    for i, (name, spec) in enumerate(conds.items()):
        path = os.path.join(out, f"identity_{i}.jsonl")
        views.run_views(be, items, {name: spec}, path, workers)
        n, differ, prob = sw.same_by_qid(sw.read_jsonl(path), ref)
        print(f"   {name}: {n} records, {differ} differ from Uniform's, max_prob_diff {prob}", flush=True)
        if differ or prob > 0 or n != len(items):
            bad.append(f"{name}: {differ} of {n} differ, max_prob_diff {prob}")
    worst, frames, tiles = 0.0, 0, 0
    for it in items[:5]:
        for fr in it["frames"]:
            vs, _ = be.load(fr, ("twoview", dict(gaze="gaze", tile=True)))
            worst = max(worst, float((pixels_of(be, vs[0]).float() - stock_reference(be, fr).float()).abs().max()))
            frames += 1
            tiles += tuple(pixels_of(be, vs[1]).shape) == tuple(pixels_of(be, vs[0]).shape)
    print(f"   the twoview's whole-frame view against the stock input on {frames} frames: max abs difference {worst}; the tile the stock "
          f"view's shape on {tiles} of {frames}", flush=True)
    if worst > 0 or tiles != frames:
        bad.append(f"the twoview's whole view differs from the stock input by {worst}, or a tile isn't the stock shape")
    return bad


def threads(model, data, be, out, workers):
    """The sweep's scorer on 10 questions (the budget rule's frames at L = 4) with loading threads against none -> problems."""
    out = fresh(os.path.join(out, "threads"))
    items = items_on(data, sw.gate_frames(data, 4), THREAD_QUESTIONS)
    conds = {"res 0.50": ("res", sw.res_grid(model, 0.5)[1]), "twoview gaze": ("twoview", dict(gaze="gaze", tile=True))}
    bad = []
    for i, (name, spec) in enumerate(conds.items()):
        paths = [os.path.join(out, f"{i}_{w}.jsonl") for w in (0, workers)]
        for path, w in zip(paths, (0, workers)):
            views.run_views(be, items, {name: spec}, path, w)
        n, differ, prob = sw.same_by_qid(sw.read_jsonl(paths[1]), sw.read_jsonl(paths[0]))
        print(f"   threads {name}: {n} records with {workers} loading threads against none, {differ} differ, max_prob_diff {prob}", flush=True)
        if differ or prob > 0 or n != len(items):
            bad.append(f"threads {name}: {differ} of {n} differ")
    return bad


def measure_units(model, ad, cfg, data, be):
    """The FLOPs units: the gate's decision, the LLM's prefill quadratic (checked against the config), and one frame's encoder at every
    representation (res sides, the twoview's whole view and tile) -> (units, counting)."""
    measure = flops.MEASURE[model]
    gate = flops.measure_gate_ckpt(os.path.join(cfg["paths"]["gate_ckpt_dir"], "fold1.pt"))
    fr = data["uniform"][data["videos"][0]][0]
    sides = sorted({sw.res_grid(model, r)[1] for r in sw.RHOS})
    counting = Counting(ad)

    def count_all():
        with counting(), torch.no_grad():
            pts = [(n, measure["prefill"](ad, n)) for n in fc.PREFILL_LENGTHS]
            vis = {}
            for side in sides:
                v = be.load(fr, ("res", side))[0][0]
                vis[f"res {side}"] = flops.count_flops(lambda v=v: be.encode(v))[0]
            two, _ = be.load(fr, ("twoview", dict(gaze="gaze", tile=True)))
            vis["whole"] = flops.count_flops(lambda: be.encode(two[0]))[0]
            vis["tile"] = flops.count_flops(lambda: be.encode(two[1]))[0]
        return pts, vis

    pts, vis = count_all()
    if counting.missed:
        print(f"the counter has no FLOP formula for {sorted(counting.missed)} under the model's own attention: every count is redone "
              "with eager attention", flush=True)
        counting.impl = "eager"
        counting.missed.clear()
        pts, vis = count_all()
    if counting.missed:
        sys.exit(f"ATTENTION NOT COUNTED: {sorted(counting.missed)}, even with eager attention. Stop and tell the user.")
    c = flops.fit_prefill(pts)
    print(f"LLM forward: {c[0]:.4g} + {c[1]:.4g} n + {c[2]:.4g} n^2 FLOPs for n tokens; at "
          + ", ".join(f"{n}: {f / 1e9:,.1f} G" for n, f in pts), flush=True)
    shape = measure["llm_shape"](ad)
    chk = flops.llm_shape_check(c, shape)
    tol = measure.get("llm_shape_tolerance", flops.LLM_SHAPE_TOLERANCE)
    print(f"n^2 term {c[2]:,.0f} against 4 x {shape['layers']} layers x {shape['heads']} heads x {shape['head_dim']} = {chk['want_c']:,} "
          f"from the model's config: {100 * chk['c_off']:+.4f}% (tolerance {100 * tol:g}%)", flush=True)
    if abs(chk["c_off"]) > tol:
        sys.exit(f"c DIFFERS FROM CONFIG by {100 * chk['c_off']:+.2f}%. Stop and tell the user.")
    S = be.stock_side
    if vis["whole"] != vis[f"res {S}"]:
        sys.exit(f"FLOPS DIFFER: the twoview's whole view costs {vis['whole']:,}, a stock frame {vis[f'res {S}']:,}. Stop and tell the user.")
    units = dict(prefill=c, gate_per_decision=float(gate), v_stock=float(vis[f"res {S}"]), v_tile=float(vis["tile"]),
                 v_twoview=float(vis["whole"] + vis["tile"]), v_res={str(s): float(vis[f"res {s}"]) for s in sides},
                 stock_tokens=int(ad.stock_tokens), stock_side=int(S), attention=counting.impl or "the model's own")
    for s in sides:
        print(f"VISION {model}: a {s} px frame {units['v_res'][str(s)] / 1e9:,.2f} GFLOPs "
              f"({100 * units['v_res'][str(s)] / units['v_stock']:.1f}% of a stock frame)", flush=True)
    print(f"VISION {model}: the twoview {units['v_twoview'] / 1e9:,.2f} GFLOPs a frame (the tile {units['v_tile'] / 1e9:,.2f}); the gate "
          f"{gate:,} FLOPs a decision per fold model, x {flops.GATE_MODELS}", flush=True)
    return units, counting


def check_counted(got, v_frame, units, tpf, baseline):
    """A question counted while scored against the units -> (vision ratio, prefill ratio, ok)."""
    r_ve = got["ve"] / (got["frames"] * v_frame) if got["frames"] else (1.0 if got["ve"] == 0 else float("inf"))
    r_pf = got["pf"] / float(flops.prefill(units["prefill"], got["n_tokens"]))
    hi = 1 + (flops.VISION_TOLERANCE if baseline else flops.EXACT_TOLERANCE)
    tokens_ok = baseline or tpf is None or got.get("n_image_tokens") == got["frames"] * tpf
    return r_ve, r_pf, bool(1 - flops.EXACT_TOLERANCE <= r_ve <= hi and abs(r_pf - 1) <= flops.EXACT_TOLERANCE and tokens_ok)


def flops_exact(model, ad, cfg, data, be, units, counting, out, workers):
    """Uniform's stock path and every representation of this code, each on 2 questions at 4 Uniform frames, counted while scored
    -> problems."""
    from system_perf import probe
    out = fresh(os.path.join(out, "flops"))
    decoder, tmp, inputs = probe.find_decoder(ad.model), os.path.join(out, "count.jsonl"), fc.Inputs(data)
    stock = units["stock_tokens"]
    fams = [("uniform's stock path", fc.planned_job(ad, cfg, inputs(4), dict(row="uniform_flops", K=4, knob=None, budget_pct=10)),
             units["v_stock"], stock)]
    items = items_on(data, inputs(4)["uniform"])
    for rho in sw.RHOS:
        _, side, t = sw.res_grid(model, rho)
        fams.append((f"res {side} px", dict(row=f"res {side}", items=items, conds={f"res {side}": ("res", side)},
                                            score=lambda it, c, p: views.run_views(be, it, c, p, workers)), units["v_res"][str(side)], t))
    fams.append(("twoview", dict(row="twoview", items=items, conds={"twoview": ("twoview", dict(gaze="gaze", tile=True))},
                                 score=lambda it, c, p: views.run_views(be, it, c, p, workers)), units["v_twoview"], 2 * stock))
    bad, n = [], 0
    qids = [int(it["qid"]) for it in items[:VERIFY_QUESTIONS]]
    with counting():
        for name, job, v, tpf in fams:
            for qid in qids:
                got = fc.counted(job, qid, decoder, tmp)
                r_ve, r_pf, ok = check_counted(got, v, units, tpf, False)
                n += 1
                print(f"   {'ok ' if ok else 'BAD'} {name:<22} q{qid}: vision {r_ve:.6f}, prefill {r_pf:.6f} x the units "
                      f"({got['frames']} frames, {got['n_image_tokens']} image tokens, {got['n_tokens']} tokens)", flush=True)
                if not ok:
                    bad.append(f"{name} q{qid}: vision {r_ve:.6f}, prefill {r_pf:.6f}, image tokens {got['n_image_tokens']}")
    if counting.missed:
        bad.append(f"attention not counted: {sorted(counting.missed)}")
    if not bad:
        print(f"FLOPS EXACT: {n} counts on {len(fams)} representations match the units", flush=True)
    return bad


def blind_recheck(ad, data, out):
    """50 of v2 Stage 1's Blind records rescored here (the 2.2B, a B200, as Stage 1) -> problems."""
    path = os.path.join(fresh(os.path.join(out, "blind")), "blind.jsonl")
    items = common.make_items(data["qs"], {})[:BLIND_RECHECK]
    run.run_blind(ad, items, path)
    n, differ, prob = sw.same_by_qid(sw.read_jsonl(path), sw.read_jsonl(sw.v2_path("stage1", "blind.jsonl")), ("pred", "p", "n_tokens"))
    print(f"BLIND RECHECK: {n} dev questions rescored against v2 Stage 1's blind records: {differ} differ, max_prob_diff {prob}", flush=True)
    return [] if not differ and n == len(items) else [f"blind recheck: {differ} of {n} differ"]


def checks_cmd(a):
    model = a.model
    cfg, data, ad, be = setup(model, a.workers)
    out = sw.model_dir(model, "checks")
    os.makedirs(out, exist_ok=True)
    snap = check_snapshot(model, ad)
    rep = run.check_parity(ad, data, model)
    bad = tokens_table(model, be, data["uniform"][data["videos"][0]][len(data["uniform"][data["videos"][0]]) // 2])
    bad += identity(model, ad, cfg, data, be, out, a.workers)
    bad += threads(model, data, be, out, a.workers)
    units, counting = measure_units(model, ad, cfg, data, be)
    bad += flops_exact(model, ad, cfg, data, be, units, counting, out, a.workers)
    if model == "smolvlm":
        bad += blind_recheck(ad, data, out)
    for b in bad:
        print("   " + b, flush=True)
    if bad:
        sys.exit(f"CHECKS FAILED on {model}: {len(bad)} problems. Stop and tell the user.")
    print(f"SAME ANSWERS: ours_res_1.00, nogate_res_1.00 and the twoview's whole frame give Uniform's answers on {IDENTITY_QUESTIONS} "
          "questions at max_prob_diff 0.0; the threads change no answer", flush=True)
    sw.write_json(sw.model_dir(model, "units.json"), dict(model=model, model_id=ad.model_id, snapshot=snap, gpu=torch.cuda.get_device_name(),
                                                          config_sha256=fc.sha256(sw.CONFIG), units=units, parity=rep, versions=versions()))
    ad.free()
    print(f"CHECKS DONE: {model}. STOP: show the user and wait for the OK before S2.", flush=True)


# ---------------------------------------------------------------------------- S2
def knobs_key(cfg, side):
    """calibrate.counts_key's settings with the sweep's grids, so a file of calibrate's counts is refused rather than reused."""
    import hashlib
    parts = dict(grids="sweep10 whole ranges", rho=GRID_RHO, r=GRID_R, beta=GRID_BETA, K=cfg["run"]["K"], max_frames=cfg["run"]["max_frames"],
                 side=side, gazevlm={k: cfg["gazevlm"][k] for k in ("sigma", "r_min", "thumb", "gaze")},
                 fastvid={k: cfg["fastvid"][k] for k in ("c", "tau", "ignore", "d", "p", "beta")},
                 prunevid=dict(tau=cfg["prunevid"]["settings"][0]["tau"], gamma=cfg["prunevid"]["gamma"], knn=cfg["prunevid"]["knn"],
                               min_tokens=cfg["prunevid"]["min_tokens"]))
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def knob_counts(path, frames, backend, cfg, side, check=True):
    """calibrate.resumable_counts over the sweep's grids: each clip's kept tokens at every GazeVLM rho, FastVID r and PruneVid beta, by
    calibrate's own token arithmetic, checked against the methods' whole plans on the first clips (check). Saved per clip, so a rerun
    continues; path None keeps them in memory (the preflight). -> (kept_gz, kept_fv, kept_pv), each {knob: {clip: tokens}}."""
    key = knobs_key(cfg, side)
    done = cal.read_counts(path, key) if path else {}
    if path:
        print(f"   {len(done)} of {len(frames)} clips already counted in {path}", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    g, workers = cfg["gazevlm"], int(cfg["run"]["prefetch_workers"])
    checked = 0 if check else cal.COUNTS_CROSS_CHECK
    with torch.no_grad(), (open(path, "a") if path else contextlib.nullcontext()) as fh:
        for v in frames:
            if v in done:
                continue
            prefixes = [cal.heat_prefix(fr, g) for fr in frames[v]]
            gaze = [cal.gazevlm_tokens(prefixes, r, g, side) for r in GRID_RHO]
            del prefixes
            if workers:
                backend.prefetch(frames[v], workers)
            states = []
            for fr in frames[v]:
                st = backend.frame_state(fr)
                states.append(dict(tokens=st["tokens"], sal=st["sal"]))
            fast, prune = cal.fastvid_counts(states, GRID_R, cfg["fastvid"]), cal.prunevid_counts(states, GRID_BETA, cfg["prunevid"])
            if checked < cal.COUNTS_CROSS_CHECK:
                checked += 1
                if (fast != [cal.fastvid_kept(states, r, cfg["fastvid"]) for r in GRID_R]
                        or prune != [cal.prunevid_kept(states, b, cfg["prunevid"]) for b in GRID_BETA]):
                    sys.exit(f"{v}: the token arithmetic differs from the methods' own plans: stop and tell the user")
                print(f"   {v}: the token arithmetic matches the methods' own plans at every r and beta", flush=True)
            rec = dict(video=v, key=key, gazevlm=gaze, fastvid=fast, prunevid=prune)
            if fh is not None:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            done[v] = rec
            del states
            torch.cuda.empty_cache()
    of = lambda name, j: {v: done[v][name][j] for v in frames}
    return ({r: of("gazevlm", j) for j, r in enumerate(GRID_RHO)}, {r: of("fastvid", j) for j, r in enumerate(GRID_R)},
            {b: of("prunevid", j) for j, b in enumerate(GRID_BETA)})


def knob_choices(cands, rho, k):
    """{row: [(share, knob)]} -> ({row: (share, knob)} by the rho rule, {row: (most share, its knob)} for every row whose whole range keeps
    less than rho). Stops, naming every such row, if a row can't keep the 0.50 floor at all."""
    out, short, bad = {}, {}, []
    for row, c in cands.items():
        try:
            out[row] = sw.pick(c, rho, 1.0, name=row)
        except ValueError as ex:
            bad.append(str(ex))
            continue
        top = max(c, key=lambda x: x[0])
        if top[0] < rho - RHO_SHORT_PTS:
            short[row] = top
    if bad:
        sys.exit(f"RHO UNREACHABLE at rho {rho:.2f} on Uniform K'={k}: " + "; ".join(bad) + ". Stop and tell the user.")
    return out, short


def share_candidates(n, stock, N, window, kept_gz, kept_fv, kept_pv, nq=None):
    """{row: [(token share, knob)]} of the six knob-continuous baselines, calibrate.py's candidates over the sweep's grids. n: {clip: input
    frames}; N: tokens a frame (the GazeVLM grid's side squared, the stock tokens on all four models); kept_*: knob_counts'."""
    nq = nq or {v: 1 for v in n}
    full = {v: n[v] * stock for v in n}
    w = lambda kept: cal.weighted(kept, full, nq)
    first = {v: -(-n[v] // window) for v in n}
    return dict(
        streamingtom=[(w({v: n[v] * G for v in n}), G / N) for G in range(1, N + 1)],
        dycoke=[(w({v: first[v] * N + (n[v] - first[v]) * kk for v in n}), 1 - (kk + 0.5) / N) for kk in range(1, N + 1)],
        dtd=[(w({v: N + (n[v] - 1) * kk for v in n}), (N - kk + 0.5) / N) for kk in range(1, N + 1)],
        gazevlm=[(w(kept_gz[r]), r) for r in GRID_RHO],
        fastvid=[(w(kept_fv[r]), r) for r in GRID_R],
        prunevid=[(w(kept_pv[b]), b) for b in GRID_BETA])


def grid_candidates(model):
    """{row: [(token share, knob)]} of average pooling's s x s grids and the two-view GazeVLM's even B, both exact from the token counts."""
    stock, grid = sw.MODELS[model]["stock_tokens"], sw.MODELS[model]["grid"]
    return dict(avgpool=[(s * s / stock, s * s / stock) for s in range(1, grid + 1)],
                gazevlm2=[(b / stock, b / stock) for b in range(2, 2 * stock + 1, 2)])


def knob_problems(cands, stock):
    """A knob that doesn't move its row's share (not connected), or a share that falls as the knob keeps more (wrong direction) ->
    named problems. Keeping more: a larger knob, except DyCoke's and DTD's, which say what to drop."""
    out = []
    for row, c in cands.items():
        shares = [x[0] for x in sorted(c, key=lambda x: x[1])]
        if row in PRUNE_KNOBS:
            shares = shares[::-1]
        if max(shares) - min(shares) < CONNECTED_SPAN:
            out.append(f"KNOB NOT CONNECTED: {row} keeps {100 * min(shares):.2f}% to {100 * max(shares):.2f}% of the stock tokens over its "
                       f"whole knob range")
        drop = max((a - b for a, b in zip(shares, shares[1:])), default=0.0)
        if drop > DIP_TOKENS.get(row, 0.0) / stock + 1e-12:
            out.append(f"KNOB NOT MONOTONE: {row}'s share falls by {100 * drop:.2f} points as its knob keeps more")
    return out


def knob_preflight(model, ad, cfg, inputs, ks):
    """Before the slow counting: every baseline's knob range on PREFLIGHT_CLIPS clips at each rho's K' (ks: {rho: K'}). Each row's share
    must move with its knob, keep more as it keeps more, and keep at least 0.50 somewhere in its range, or the plan stops, naming the rows.
    A row whose whole range stays under rho is reported (RHO SHORT, sample)."""
    side = ad.gazevlm_backend().side
    window = int(cfg["dycoke"]["settings"][0]["window"])
    for rho, k in ks.items():
        d = inputs(k)
        clips = sorted(d["uniform"])
        sample = clips[:: max(1, len(clips) // PREFLIGHT_CLIPS)][:PREFLIGHT_CLIPS]
        cfg_k = fc.with_knob(cfg, "uniform", None, k)
        cap = cfg_k["run"]["max_frames"].get(ad.key)
        frames = {v: common.cap_frames(d["uniform"][v], cap) for v in sample}
        kept = knob_counts(None, frames, ad.fastvid_backend(), cfg_k, side, check=False)
        cands = dict(share_candidates({v: len(f) for v, f in frames.items()}, ad.stock_tokens, side * side, window, *kept),
                     **grid_candidates(model))
        for row, c in cands.items():
            lo, hi = min(c, key=lambda x: x[1]), max(c, key=lambda x: x[1])
            print(f"KNOB RANGE {model} rho {rho:.2f} K'={k} ({len(sample)} clips): {row} knob {lo[1]:g} to {hi[1]:g} keeps "
                  f"{100 * lo[0]:.2f}% to {100 * hi[0]:.2f}% ({len(c)} settings)", flush=True)
        bad = knob_problems(cands, ad.stock_tokens)
        if bad:
            sys.exit("KNOB RANGE FAILED: " + "; ".join(bad) + ". Stop and tell the user.")
        _, short = knob_choices(cands, rho, k)
        for row, (most, at) in short.items():
            print(f"RHO SHORT (sample): {row} at rho {rho:.2f} keeps at most {100 * most:.2f}% over its whole range (at {at})", flush=True)
    print(f"KNOB RANGES OK: every baseline's share moves with its knob, the right way, and keeps 0.50 somewhere, on {PREFLIGHT_CLIPS} "
          "clips at each rho's K'", flush=True)


def baseline_knobs(ad, cfg, data_k, k, rho, counts_path):
    """Each knob-continuous single-view baseline's knob on its Uniform K' frames by the rho rule (closest to rho of the stock tokens among
    settings keeping at least 0.50), each knob over its whole range -> ({row: (token share, knob)}, {row: its most, if under rho})."""
    cfg_k = fc.with_knob(cfg, "uniform", None, k)
    cap = cfg_k["run"]["max_frames"].get(ad.key)
    nq = data_k["qs"].video.value_counts().to_dict()
    frames = {v: common.cap_frames(data_k["uniform"][v], cap) for v in nq}
    side = ad.gazevlm_backend().side
    window = int(cfg_k["dycoke"]["settings"][0]["window"])
    kept = knob_counts(counts_path, frames, ad.fastvid_backend(), cfg_k, side)
    cands = share_candidates({v: len(frames[v]) for v in frames}, ad.stock_tokens, side * side, window, *kept, nq=nq)
    bad = knob_problems(cands, ad.stock_tokens)
    if bad:
        sys.exit(f"KNOB RANGE FAILED on all {len(frames)} clips at K'={k}: " + "; ".join(bad) + ". Stop and tell the user.")
    return knob_choices(cands, rho, k)


def prompt_fits(model, ad, be, data, blind):
    measure = flops.MEASURE[model]
    qs = data["qs"].sort_values(["video", "qid"]).reset_index(drop=True)
    one, two, differ = [], [], 0
    for r in qs.iloc[np.linspace(0, len(qs) - 1, PROMPT_QUESTIONS).astype(int)].drop_duplicates("qid").itertuples():
        n_v = data["metas"][r.video]["n"]
        for f in PROMPT_FRAMES:
            frs = [data["frame"](r.video, int(i)) for i in common.uniform_indices(n_v, f)]
            n1 = views.prompt_len(be, build_messages, r.question, r.options, frs, 1)
            differ += n1 != measure["prompt_len"](ad, r.question, list(r.options), frs, [0] * len(frs))
            one.append((len(frs), n1 - blind[int(r.qid)]))
            two.append((len(frs), views.prompt_len(be, build_messages, r.question, r.options, frs, 2) - blind[int(r.qid)]))
    if differ:
        sys.exit(f"PROMPT DIFFERS: on {differ} prompts this code's one-image prompt isn't the rows' own. Stop and tell the user.")
    p1, p2 = flops.fit_prompt(one), flops.fit_prompt(two)
    for name, p in (("one image a frame", p1), ("two images a frame", p2)):
        print(f"a video prompt, {name} = its blind prompt + {p['c0']:.1f} + {p['o']:.2f} per frame + its image tokens (largest miss "
              f"{p['max_residual']:.1f} tokens over {len(one)} prompts)", flush=True)
    return p1, p2


def make_lines(model, ad, cfg, data, units, p1, p2, blind, log=print):
    """Every planned condition: K' (or L) at 10%, the knobs, the planned FLOPs share."""
    g = sw.MODELS[model]
    stock, S = g["stock_tokens"], g["stock_side"]
    qs = data["qs"].sort_values(["video", "qid"]).reset_index(drop=True)
    vids = list(qs.video)
    K = np.array([int(data["metas"][v]["n"]) for v in vids], dtype=np.int64)
    b = np.array([blind[int(q)] for q in qs.qid], dtype=np.float64)
    dec = K - 1
    total = float(sw.question_flops(units, K, stock * K, units["v_stock"], b, p1).sum())

    def share(frames, tpf, v, prompt, decisions=0):
        frames = np.asarray(frames, dtype=np.float64)
        return float(sw.question_flops(units, frames, tpf * frames, v, b, prompt, decisions).sum()) / total

    ours_n = {}
    for L in range(1, sw.K_MAX + 1):
        pos = sw.kept_positions(data, L, True)
        ours_n[L] = np.array([len(pos[v]) for v in vids])
    lines = []

    def add(name, method, kind, rho, tpf, views_, side, k, s, v, prompt=1, row=None, knob=None, rho_real=None, frames=None, **extra):
        pct = 100 * s
        line = dict(name=name, method=method, kind=kind, row=row, rho=rho, rho_real=rho_real if rho_real is not None else tpf / stock,
                    tokens_per_frame=tpf, views=views_, side=side, K=int(k), knob=knob, v_frame=float(v), prompt=prompt,
                    target_pct=100 * sw.BUDGET if kind not in ("uniform", "blind") else None, planned_pct=pct,
                    miss_pts=pct - 100 * sw.BUDGET, nearest_only=sw.nearest_only(pct) if kind not in ("uniform", "blind") else False,
                    frames_per_q=float(np.mean(frames)) if frames is not None else None, rho_short=False)
        line.update(extra)
        lines.append(line)

    add("uniform", "uniform", "uniform", 1.0, stock, 1, S, sw.K_MAX, 1.0, units["v_stock"], frames=K, row="uniform")
    add("blind", "blind", "blind", None, 0, 0, None, 0, float(flops.prefill(units["prefill"], b).sum()) / total, 0.0, rho_real=0.0,
        frames=np.zeros_like(K))
    k, s = sw.best_k(lambda k: share(np.minimum(k, K), stock, units["v_stock"], p1))
    add("uniform_flops", "uniform_flops", "uniform_flops", 1.0, stock, 1, S, k, s, units["v_stock"], row="uniform_flops",
        frames=np.minimum(k, K))
    reps = [(f"res_{r:.2f}", r) + sw.res_grid(model, r) for r in sw.RHOS]
    for name, rho, m, side, t in reps:
        v = units["v_res"][str(side)]
        L, s = sw.best_k(lambda L: share(ours_n[L], t, v, p1, dec))
        add(f"ours_{name}", "ours_res", "ours", rho, t, 1, side, L, s, v, frames=ours_n[L], grid=f"{m}x{m}")
        L, s = sw.best_k(lambda L: share(np.minimum(L, K), t, v, p1))
        add(f"nogate_{name}", "nogate_res", "nogate", rho, t, 1, side, L, s, v, frames=np.minimum(L, K), grid=f"{m}x{m}")
    tv, v = sw.twoview(model), units["v_twoview"]
    L, s = sw.best_k(lambda L: share(ours_n[L], tv["tokens"], v, p2, dec))
    for name in ("ours_twoview", "ours_twoview_centre"):
        add(name, name, "ours", None, tv["tokens"], 2, S, L, s, v, prompt=2, frames=ours_n[L], tile_src=tv["tile_src"])
    L, s = sw.best_k(lambda L: share(np.minimum(L, K), tv["tokens"], v, p2))
    add("nogate_twoview", "nogate_twoview", "nogate", None, tv["tokens"], 2, S, L, s, v, prompt=2, frames=np.minimum(L, K),
        tile_src=tv["tile_src"])
    inputs = fc.Inputs(data)
    ks = {rho: sw.best_k(lambda k: share(np.minimum(k, K), rho * stock, units["v_stock"], p1))[0] for rho in sw.RHOS[:2]}
    knob_preflight(model, ad, cfg, inputs, ks)
    for rho in sw.RHOS[:2]:
        k1 = ks[rho]
        log(f"   counting the baselines' tokens on Uniform K'={k1} frames (rho {rho:.2f}) ...")
        knobs, short = baseline_knobs(ad, cfg, inputs(k1), k1, rho, sw.model_dir(model, f"knob_counts_K{k1}.jsonl"))
        for row, (most, at) in short.items():
            log(f"RHO SHORT: {row} at rho {rho:.2f} keeps at most {100 * most:.2f}% of the stock tokens over its knob's whole range (at "
                f"{at}); by the rho rule it runs at its setting nearest rho above the floor. The user decides before S3.")
        for row in sw.BASELINE_ROWS:
            if row == "avgpool":
                s_ = sw.avgpool_side(model, rho)
                tpf, views_, knob, k = s_ * s_, 1, s_ * s_ / stock, None
            elif row == "gazevlm2":
                tpf = sw.gazevlm2_tokens(model, rho)
                views_, knob, k = 2, tpf / stock, None
            else:
                sh, knob = knobs[row]
                tpf, views_, k = sh * stock, 1, k1
            vf = views_ * units["v_stock"]
            if k is None:
                k, _ = sw.best_k(lambda k: share(np.minimum(k, K), tpf, vf, p1))
            s = share(np.minimum(k, K), tpf, vf, p1)
            add(f"{row}_{rho:.2f}", row, "baseline", rho, float(tpf), views_, S, k, s, vf, row=row, knob=float(knob), frames=np.minimum(k, K),
                rho_short=row in short)
    return lines, total, {int(q): int(k_) for q, k_ in zip(qs.qid, K)}


def plan_cmd(a):
    model = a.model
    path = sw.model_dir(model, "plan.json")
    if os.path.exists(path):
        sys.exit(f"{path} exists, and the run reads it. Planning again needs the user's OK: stop and tell them.")
    doc = sw.read_json(sw.model_dir(model, "units.json"))
    cfg, data, ad, be = setup(model, a.workers)
    snap = check_snapshot(model, ad)
    if (doc["model_id"], doc["snapshot"], doc["config_sha256"], doc["gpu"]) != (ad.model_id, snap, fc.sha256(sw.CONFIG),
                                                                               torch.cuda.get_device_name()):
        sys.exit("the model, snapshot, config or GPU differs from the checks job's units.json. Stop and tell the user.")
    units = doc["units"]
    run.check_parity(ad, data, model)
    blind = {int(r.qid): int(ad.blind_ids(build_messages(r.question, list(r.options), [], 0)).shape[1]) for r in data["qs"].itertuples()}
    p1, p2 = prompt_fits(model, ad, be, data, blind)
    lines, total, k_of = make_lines(model, ad, cfg, data, units, p1, p2, blind)
    if sorted(ln["name"] for ln in lines) != sorted(c[0] for c in sw.conditions()):
        sys.exit("the planned conditions aren't the protocol's 28. Stop and tell the user.")
    inputs = fc.Inputs(data)
    jobs = {}
    for ln in lines:
        jobs[ln["name"]] = build_job(ad, cfg, data, be, inputs, ln, a.workers)
        conds = list(jobs[ln["name"]]["conds"])
        if len(conds) != 1:
            sys.exit(f"{ln['name']}: {len(conds)} conditions. Stop and tell the user.")
        ln["condition"] = conds[0]

    print(f"\n--- every condition on {VERIFY_QUESTIONS} questions, counted while scored", flush=True)
    from system_perf import probe
    chk = fresh(sw.model_dir(model, "plan_checks"))
    decoder, tmp, counting, bad = probe.find_decoder(ad.model), os.path.join(chk, "count.jsonl"), Counting(ad), []
    counting.impl = None if units.get("attention") == "the model's own" else units.get("attention")
    qids = [int(q) for q in data["qs"].sort_values(["video", "qid"]).qid[:VERIFY_QUESTIONS]]
    with counting():
        for ln in lines:
            if ln["kind"] == "blind":
                continue
            baseline = ln["kind"] == "baseline"
            tpf = None if baseline else int(ln["tokens_per_frame"])
            v = units["v_stock"] * ln["views"] if baseline else ln["v_frame"]
            ln["verified"] = []
            for qid in qids:
                got = fc.counted(jobs[ln["name"]], qid, decoder, tmp)
                r_ve, r_pf, ok = check_counted(got, v, units, tpf, baseline)
                ln["verified"].append(dict(qid=qid, vision_ratio=r_ve, prefill_ratio=r_pf, ok=ok))
                print(f"   {'ok ' if ok else 'BAD'} {ln['name']:<20} q{qid}: vision {r_ve:.5f}, prefill {r_pf:.6f} x planned "
                      f"({got['frames']} frames, {got['n_image_tokens']} image tokens, {got['n_tokens']} tokens)", flush=True)
                if not ok:
                    bad.append(f"{ln['name']} q{qid}")
    if counting.missed:
        bad.append(f"attention not counted: {sorted(counting.missed)}")
    if bad:
        sys.exit(f"FLOPS DIFFER on {len(bad)} counts ({bad[:5]}): the plan doesn't describe these rows. Stop and tell the user.")

    print(f"\n--- timing: {TIMING_QUESTIONS} questions per condition, as the run scores them", flush=True)
    tdir = fresh(sw.model_dir(model, "plan_timing"))
    for ln in lines:
        job = jobs[ln["name"]]
        items = job["items"][:TIMING_QUESTIONS]
        t0 = time.time()
        job["score"](items, job["conds"], os.path.join(tdir, f"{ln['name']}.jsonl"))
        ln["sec_per_q"] = (time.time() - t0) / max(1, len(items))
    hours = sum(ln["sec_per_q"] for ln in lines) * sw.N_DEV / 3600
    plan = dict(model=model, model_id=ad.model_id, snapshot=snap, gpu=torch.cuda.get_device_name(), config=sw.CONFIG,
                config_sha256=fc.sha256(sw.CONFIG), units=units, prompt_one=p1, prompt_two=p2, uniform_total_flops=total,
                K_candidates={str(q): k for q, k in k_of.items()}, verify_qids=qids, lines=lines, projected_gpu_hours=hours,
                budget=sw.BUDGET, rhos=list(sw.RHOS), floor=sw.FLOOR, versions=versions())
    import pandas as pd
    t = pd.DataFrame([{k: v for k, v in ln.items() if k != "verified"} for ln in lines])
    t.to_csv(sw.model_dir(model, "plan.csv"), index=False)
    pd.set_option("display.width", 250)
    print(f"\n=== {model}: target 10% of its Uniform-64 FLOPs ({total / sw.N_DEV / 1e12:,.2f} TFLOPs a question on average); nearest_only: "
          f"more than {flops.NEAREST_ONLY_PTS} points off", flush=True)
    print(t[["name", "rho", "rho_real", "rho_short", "tokens_per_frame", "views", "side", "K", "knob", "target_pct", "planned_pct", "miss_pts",
             "nearest_only", "frames_per_q", "sec_per_q"]].round(4).to_string(index=False), flush=True)
    print(f"knob ranges searched: GazeVLM rho {GRID_RHO[0]:g} to {GRID_RHO[-1]:g} ({len(GRID_RHO)} values), FastVID r {GRID_R[0]:g} to "
          f"{GRID_R[-1]:g}, PruneVid beta {GRID_BETA[0]:g} to {GRID_BETA[-1]:g}; RHO SHORT rows: "
          f"{[ln['name'] for ln in lines if ln['rho_short']] or 'none'}", flush=True)
    for method, settings in sw.method_settings().items():
        print(f"   {method}: rho levels {sorted(settings)}", flush=True)
    print(f"PROJECTED RUN: {hours:.2f} GPU-hours on one {sw.MODELS[model]['gpu']} for {len(lines)} conditions x {sw.N_DEV} questions "
          f"(timing on {TIMING_QUESTIONS} questions each)", flush=True)
    sw.write_json(path, plan)
    ad.free()
    print(f"PLAN WRITTEN: {path}. STOP: show the user and wait for the OK before S3.", flush=True)


# ---------------------------------------------------------------------------- S3
def run_cmd(a):
    from accuracy_tables import flops_run
    model = a.model
    plan = sw.read_json(sw.model_dir(model, "plan.json"))
    if plan["model"] != model or plan["config_sha256"] != fc.sha256(sw.CONFIG):
        sys.exit(f"the plan is for {plan['model']}, or {sw.CONFIG} changed since it was made. Stop and tell the user.")
    cfg, data, ad, be = setup(model, a.workers)
    if (ad.model_id, check_snapshot(model, ad)) != (plan["model_id"], plan["snapshot"]):
        sys.exit("the model or its snapshot differs from the plan's. Stop and tell the user.")
    run.check_parity(ad, data, model)
    out = sw.model_dir(model, "answers")
    os.makedirs(out, exist_ok=True)
    inputs, jobs = fc.Inputs(data), []
    for ln in plan["lines"]:
        job = build_job(ad, cfg, data, be, inputs, ln, a.workers)
        if set(job["conds"]) != {ln["condition"]}:
            sys.exit(f"{ln['name']}: the job's condition {sorted(job['conds'])} isn't the plan's {ln['condition']!r}. Stop and tell the user.")
        jobs.append(job)
    gpu, status = flops_run.gpu_name(), {}
    print(f"{len(jobs)} conditions x {sw.N_DEV} questions on {gpu}; projected {plan['projected_gpu_hours']:.2f} GPU-hours", flush=True)
    try:
        for job in jobs:
            path = os.path.join(out, f"{job['name']}.jsonl")
            flops_run.note_gpu(out, job, path, gpu)
            with run._row(job["name"], status, True, path):
                print("\n" + job["msg"], flush=True)
                job["score"](job["items"], job["conds"], path)
    finally:
        ad.free()
    print("\n=== pass / fail per condition", flush=True)
    for name, s in status.items():
        print(f"   {name:<22} {s}", flush=True)
    n_fail = sum(s.startswith("FAILED") for s in status.values())
    if n_fail:
        sys.exit(f"RUN FAILED: {n_fail} of {len(status)} conditions. Stop and tell the user.")
    print(f"RUN DONE: {model}, {len(status)} conditions x {sw.N_DEV} dev questions -> {out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("checks", "plan", "run"))
    ap.add_argument("--model", required=True, choices=sorted(sw.MODELS))
    ap.add_argument("--workers", type=int, default=8, help="frame loading threads (the baselines' and the sweep's scorer)")
    a = ap.parse_args(argv)
    dict(checks=checks_cmd, plan=plan_cmd, run=run_cmd)[a.command](a)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""Match every row on total FLOPs per question instead of tokens (runbook section 14 on SmolVLM, section 15 on Qwen2.5-VL and Qwen3-VL).

    python -m accuracy_tables.flops_calibrate --config accuracy_tables/hpc/config.all350.yaml --model smolvlm \
        --out-dir <results folder> --prefetch-workers 8

The model is an argument: any model with measuring functions in flops.MEASURE (smolvlm, qwen2.5, qwen3). On Qwen a foveated frame's
encoder cost follows its layout, so step 1 also counts the encoder on the stock layout and on foveated layouts at 7 gaze points and fits
it to the layout's patches and attention sizes (flops.fit_vision, checked on every point), and ours' and the no-gate control's vision
and image tokens are summed over each video's own frames. All counting runs under flops.AttentionWatch: if an attention op runs that
the counter has no formula for, every count is redone with eager attention, and ATTENTION NOT COUNTED is printed if one is still missed.

Ours sets the targets and changes nothing. The budget rule at its 3 token budgets (5, 10 and 20%, the smallest fovea) costs some share of
Uniform K's FLOPs, and those 3 shares are the FLOPs budgets. accuracy_tables/flops.py says what a question's FLOPs are. The other rows:
  uniform_flops   Uniform K' at stock input, K' the frame count whose FLOPs come closest to the target
  each baseline   its own method on Uniform K' frames, keeping ours' tokens per frame (17 of 81 on SmolVLM) through its budget
                  parameter, which calibrate.py's own matching sets. K' is the frame count whose FLOPs come closest to the target. The
                  two-view GazeVLM encodes 2 images a frame, so it gets its own K'. Nothing else about a baseline changes.
  nogate          the no-gate control at ours' token budgets, as it is, for reference
Only token counts and FLOPs are used, never an answer.

Steps, all printed:
  1. parity; one gate decision's FLOPs; the LLM's FLOPs at 5 prompt lengths (a quadratic, checked, and its n^2 term against the
     config); one stock frame's and one foveated frame's FLOPs, counted while scoring one question each into a temporary file
  2. each question's blind prompt length, and the intro and per-frame tokens of a video prompt, fitted on 24 questions
  3. ours' FLOPs share at each budget (the targets), K' per row, and each baseline's budget parameter on its K' frames
  4. every planned condition scored on 2 questions under the counter: its vision and prefill FLOPs must match flops.py's (exactly for
     Uniform K and ours, within 2% above for the baselines, whose own work flops.py counts as zero), or the job stops
Writes <out>/<model>/flops_plan.json and flops_targets.csv, and ends with PLAN WRITTEN.
"""

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accuracy_tables import calibrate as cal  # noqa: E402
from accuracy_tables import common, flops, run  # noqa: E402
from accuracy_tables.models import ADAPTERS, build_messages  # noqa: E402

REF_ROWS = ("uniform", "blind", "budget", "nogate")
BASELINES = run.BASELINE_ROWS
SINGLE_VIEW = tuple(r for r in BASELINES if r not in flops.TWO_VIEWS)
PREFILL_LENGTHS = (256, 2048, 6144, 1024, 4096)        # the first 3 fix the quadratic, all 5 must agree
PROMPT_FRAMES = (1, 4, 16, 64)
PROMPT_QUESTIONS = 24
VERIFY_QUESTIONS = 2


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ---------------------------------------------------------------------------- the rows' inputs at another frame count
def with_uniform(data, k):
    """data with k evenly spaced frames per video as its Uniform K frames: load_data's rule at another K."""
    frames = {v: [data["frame"](v, int(f)) for f in common.uniform_indices(data["metas"][v]["n"], int(k))] for v in data["videos"]}
    return dict(data, uniform=frames)


def with_knob(cfg, row, knob, k):
    """cfg with K = k and the row's budget parameter set to knob alone, as calibrate.matched_config sets it. Nothing else changes."""
    c = copy.deepcopy(cfg)
    c["run"]["K"] = int(k)
    if row in ("uniform", "uniform_flops"):
        return c
    knob = float(knob)
    window = int(cfg["dycoke"]["settings"][0]["window"])
    tau = float(cfg["prunevid"]["settings"][0]["tau"])
    if row == "streamingtom":
        c["streamingtom"]["ratio"] = [knob]
    elif row == "dycoke":
        c["dycoke"]["settings"] = [dict(window=window, prune=knob)]
    elif row == "dtd":
        c["dtd"]["settings"] = [dict(threshold=knob, absolute=False)]
    elif row == "avgpool":
        c["avgpool"]["budget"] = [knob]
    elif row == "gazevlm2":
        c["gazevlm2"]["budget"] = [knob]
    elif row == "gazevlm":
        c["gazevlm"]["rho"] = [knob]
    elif row == "prunevid":
        c["prunevid"]["settings"] = [dict(tau=tau, beta=knob)]
    elif row == "fastvid":
        c["fastvid"]["r"] = [knob]
    else:
        raise ValueError(f"no budget parameter for {row}")
    return c


def planned_job(ad, cfg, data_k, line):
    """run.py's own job for one planned line, on data_k (with_uniform at the line's K) -> the job, its row renamed to the result file
    it writes (uniform_flops for Uniform K', so it stays apart from Uniform K's file)."""
    base = "uniform" if line["row"] == "uniform_flops" else line["row"]
    jobs = run.row_jobs(ad, with_knob(cfg, base, line["knob"], line["K"]), data_k, [base])
    if len(jobs) != 1 or len(jobs[0]["conds"]) != 1:
        raise RuntimeError(f"{line['row']}: expected one job with one condition, got {[(j['row'], list(j['conds'])) for j in jobs]}")
    job = jobs[0]
    job["row"] = line["row"]
    job["name"] = f"{line['row']} {line['budget_pct']:g}%"
    return job


class Inputs:
    """with_uniform's data, made once per frame count."""

    def __init__(self, data):
        self.data, self.cache = data, {}

    def __call__(self, k):
        if k not in self.cache:
            self.cache[k] = with_uniform(self.data, k)
        return self.cache[k]


# ---------------------------------------------------------------------------- the baselines' budget parameters
def knobs_at(ad, cfg_k, data_k, share, counts_path, log=print):
    """Each single-view baseline's budget parameter keeping `share` of the stock tokens of its Uniform K' frames, chosen as
    calibrate.calibrate chooses it -> {row: (predicted share, knob)}. cfg_k: with_knob(cfg, 'uniform', None, K')."""
    cap = cfg_k["run"]["max_frames"].get(ad.key)
    nq = data_k["qs"].video.value_counts().to_dict()
    frames = {v: common.cap_frames(data_k["uniform"][v], cap) for v in nq}
    n = {v: len(frames[v]) for v in frames}
    stock = ad.stock_tokens
    full = {v: n[v] * stock for v in n}
    side = ad.gazevlm_backend().side
    N = side * side
    ap_be = ad.avgpool_backend()
    window = int(cfg_k["dycoke"]["settings"][0]["window"])
    kept_gz, kept_fv, kept_pv = cal.resumable_counts(counts_path, frames, ad.fastvid_backend(), cfg_k, side, log)
    return dict(
        streamingtom=cal.match_streamingtom(share, n, nq, full, N),
        dycoke=cal.match_dycoke(share, n, nq, full, N, window),
        dtd=cal.match_dtd(share, n, nq, full, N),
        avgpool=cal.match_avgpool(share, stock, ap_be.side, getattr(ap_be, "row_breaks", False)),
        gazevlm=cal.closest([(cal.weighted(kept_gz[r], full, nq), r) for r in cal.GRID_RHO], share),
        fastvid=cal.closest([(cal.weighted(kept_fv[r], full, nq), r) for r in cal.GRID_R], share),
        prunevid=cal.closest([(cal.weighted(kept_pv[b], full, nq), b) for b in cal.GRID_BETA], share))


# ---------------------------------------------------------------------------- counting real questions
def verify_qids(qs, n):
    """n questions from different videos for the FLOPs check: the first question of each video in name order, Nymeria and AEA
    (names starting with loc) taking turns, so both kinds of video are covered."""
    firsts = qs.sort_values(["video", "qid"]).groupby("video", sort=True).qid.first()
    pools = [[int(q) for v, q in firsts.items() if not v.startswith("loc")], [int(q) for v, q in firsts.items() if v.startswith("loc")]]
    out = []
    while len(out) < n and any(pools):
        for pool in pools:
            if pool and len(out) < n:
                out.append(pool.pop(0))
    return out


def counted(job, qid, decoder, tmp):
    """One question of a job scored through the run's own code under the counter (system_perf's count_flops), into a temporary
    file -> dict(ve, pf, frames, n_tokens, n_image_tokens, video): vision-side FLOPs, prefill FLOPs, images given to the row, the
    prompt's tokens, its image tokens and the question's video."""
    from system_perf import measure as spm
    item = next(it for it in job["items"] if int(it["qid"]) == qid)
    cond = next(iter(job["conds"]))
    ve, pf = spm.count_flops(decoder, job, cond, item, tmp)
    with open(tmp) as f:
        rec = [json.loads(line) for line in f if line.strip()][-1]
    if rec.get("error"):
        raise RuntimeError(f"{job['row']} {cond}, question {qid}: {rec['error']}")
    return dict(ve=float(ve), pf=float(pf), frames=len(item["frames"]), n_tokens=int(rec["n_tokens"]),
                n_image_tokens=rec.get("n_image_tokens"), video=item["video"])


def check(row, got, units, want_ve=None):
    """got (from counted) against flops.py -> (vision ratio, prefill ratio, ok). A baseline's vision may be up to VISION_TOLERANCE
    above (its own work, counted as zero); everything else must match within EXACT_TOLERANCE. want_ve: the question's vision FLOPs
    from its frames' layouts (Qwen's foveated rows), instead of frames x one frame's FLOPs. A frame count that the encoder paired
    into temporal patches would cost other than frames x one frame, and fail here."""
    unit = units["v_fovea"] if row in flops.FOVEA_ROWS else units["v_stock"]
    views = 2 if row in flops.TWO_VIEWS else 1
    if want_ve is None:
        want_ve = views * got["frames"] * unit
    r_ve = got["ve"] / want_ve if want_ve else (1.0 if got["ve"] == 0 else math.inf)
    r_pf = got["pf"] / float(flops.prefill(units["prefill"], got["n_tokens"]))
    hi = 1 + (flops.VISION_TOLERANCE if row in BASELINES else flops.EXACT_TOLERANCE)
    ok = 1 - flops.EXACT_TOLERANCE <= r_ve <= hi and abs(r_pf - 1) <= flops.EXACT_TOLERANCE
    return r_ve, r_pf, ok


# ---------------------------------------------------------------------------- the plan
def frames_per_video(job):
    """{video: frames each of its questions is given} from a run.py job's own items."""
    return {it["video"]: len(it["frames"]) for it in job["items"]}


def shares(q, units, prompt, ref, tokens_per_frame, decisions, fovea=None):
    """FLOPs share of ours' and the no-gate control's runs: ({name: share of the Uniform K FLOPs}, Uniform K's FLOPs total).
    ref: {run.py job name ('uniform', 'budget 5%', 'nogate 5%', ...): frames_per_video(job)}; q: per question, video, blind.
    fovea: {name: {video: (vision FLOPs, image tokens, frames)}} from each frame's layout (Qwen), in place of frames x one foveated frame."""
    fu = q["video"].map(ref["uniform"]).to_numpy()
    stock = units["stock_tokens"]
    total = float(flops.question_flops(units, fu, "v_stock", flops.prompt_tokens(q["blind"], fu, stock * fu, prompt)).sum())
    out = {}
    for name, per_video in ref.items():
        if name == "uniform":
            continue
        f = q["video"].map(per_video).to_numpy()
        dec = q["video"].map(decisions).to_numpy() if name.startswith("budget") else 0
        if fovea and name in fovea:
            vision = q["video"].map({v: x[0] for v, x in fovea[name].items()}).to_numpy()
            tokens = q["video"].map({v: x[1] for v, x in fovea[name].items()}).to_numpy()
            n = flops.prompt_tokens(q["blind"], f, tokens, prompt)
            out[name] = float(flops.question_flops_vision(units, vision, n, decisions=dec).sum()) / total
            continue
        n = flops.prompt_tokens(q["blind"], f, tokens_per_frame * f, prompt)
        out[name] = float(flops.question_flops(units, f, "v_fovea", n, decisions=dec).sum()) / total
    return out, total


def fovea_per_video(job, measure, coef):
    """{video: (vision FLOPs, image tokens, frames)} of one foveated job's frames, summed over each frame's own layout (flops.fit_vision's
    coefficients). A video's questions all get the same frames."""
    cond = next(iter(job["conds"]))
    out = {}
    for it in job["items"]:
        if it["video"] in out:
            continue
        vision = tokens = 0
        for fr in it["frames"]:
            cells, side = measure["layout"](job["conds"], cond, fr)
            vision += float(np.dot(measure["features"](cells, side), coef))
            tokens += len(cells)
        out[it["video"]] = (vision, int(tokens), len(it["frames"]))
    return out


def plan_frame_counts(q, units, prompt, total, targets, k_max, families):
    """{(family, fraction): (K', predicted share)}; families: {family: (tokens per frame, views)}."""
    qq = dict(blind=q["blind"].to_numpy(), video_frames=q["video_frames"].to_numpy())
    out = {}
    for fam, (tpf, views) in families.items():
        for b, target in targets.items():
            out[fam, b] = flops.choose_k(lambda k: flops.uniform_share(qq, units, prompt, total, k, tpf, views), target, k_max)
    return out


def line(row, b, target, predicted, K, knob, condition, token_share=None):
    return dict(row=row, budget_pct=round(100 * float(b), 6), target_pct=None if target is None else 100 * float(target),
                predicted_pct=100 * float(predicted), K=int(K), knob=None if knob is None else float(knob), condition=condition,
                token_share_pct=None if token_share is None else 100 * float(token_share))


def make_plan(ad, cfg, data, out_model_dir, verify_n=VERIFY_QUESTIONS, log=print):
    from system_perf import probe
    key, stock = ad.key, ad.stock_tokens
    measure = flops.MEASURE[key]
    cap = cfg["run"]["max_frames"].get(key)
    k_ref = min(cfg["run"]["K"], cap) if cap else cfg["run"]["K"]
    tpf = int(cfg["budget"]["tokens_per_frame"][key])
    share = tpf / stock
    fractions = [float(b) for b in cfg["budget"]["fractions"]]
    run.check_parity(ad, data, key)                     # also fixes how prompts are tokenized for this transformers version
    decoder = probe.find_decoder(ad.model)
    tmp = os.path.join(tempfile.mkdtemp(prefix="flops_"), "rec.jsonl")
    qs = data["qs"].sort_values(["video", "qid"]).reset_index(drop=True)
    qids = verify_qids(qs, verify_n)

    log("\n--- step 1: FLOPs units")
    gate = flops.measure_gate_ckpt(os.path.join(cfg["paths"]["gate_ckpt_dir"], "fold1.pt"))
    log(f"gate: {gate:,} FLOPs per decision per fold model, counted as {flops.GATE_MODELS} fold models")
    ref_jobs = {job.get("name", job["row"]): job for job in run.row_jobs(ad, cfg, data, ["uniform", "budget", "nogate"])}
    top = f"budget {100 * max(fractions):g}%"
    per_layout = "features" in measure                 # a foveated frame's encoder cost follows its layout (Qwen)
    impl, missed = [None], set()

    @contextlib.contextmanager
    def counting():
        """Every count runs in here: at the attention implementation in impl, watching for attention the counter can't see."""
        with flops.attention_implementation(ad.model, impl[0]), flops.AttentionWatch() as watch:
            yield
        missed.update(watch.missed)

    def count_units():
        with counting():
            pts = [(n, measure["prefill"](ad, n)) for n in PREFILL_LENGTHS]
            u_ = counted(ref_jobs["uniform"], qids[0], decoder, tmp)
            o_ = counted(ref_jobs[top], qids[0], decoder, tmp)
            lay = []
            if per_layout:
                conds = dict(fovea=ad.fovea_condition(ad.fovea_settings(cfg["ours"])[0]))
                shapes = [measure["stock"]()] + [measure["layout"](conds, "fovea", dict(gaze=g)) for g in flops.VISION_FIT_GAZES]
                lay = [(measure["features"](cells, side), float(measure["vision"](ad, cells, side)), len(cells)) for cells, side in shapes]
        return pts, u_, o_, lay

    points, u, o, layouts = count_units()
    if missed:
        log(f"the counter has no FLOP formula for {sorted(missed)} under the model's own attention, so every count is redone with "
            "eager attention, which runs attention as matrix products it counts")
        impl[0] = "eager"
        missed.clear()
        points, u, o, layouts = count_units()
    if missed:
        log(f"ATTENTION NOT COUNTED: {sorted(missed)} ran without a FLOP formula, even with eager attention. Tell the user.")
    c = flops.fit_prefill(points)
    log(f"LLM forward: {c[0]:.4g} + {c[1]:.4g} n + {c[2]:.4g} n^2 FLOPs for n tokens; at "
        + ", ".join(f"{n}: {f / 1e9:,.1f} G" for n, f in points))
    if c[2] <= 0:
        log("ATTENTION NOT COUNTED: the counter saw no n^2 term, so this torch's attention kernel has no FLOP formula. "
            "Every row is still counted the same way. Tell the user.")
    llm_check = None
    if "llm_shape" in measure:
        shape = measure["llm_shape"](ad)
        llm_check = dict(shape, **flops.llm_shape_check(c, shape))
        log(f"n^2 term {c[2]:,.0f} against 4 x {shape['layers']} layers x {shape['heads']} heads x {shape['head_dim']} = "
            f"{llm_check['want_c']:,} from the model's config: {100 * llm_check['c_off']:+.3f}%")
        log(f"n term {c[1]:.4g} against 2 x {shape['non_embedding_params']:,} non-embedding parameters = {llm_check['want_b']:.4g}: "
            f"{100 * llm_check['b_off']:+.2f}%, for information only")
        if abs(llm_check["c_off"]) > flops.LLM_SHAPE_TOLERANCE:
            sys.exit(f"c DIFFERS FROM CONFIG by {100 * llm_check['c_off']:+.2f}%: the counter doesn't see the LLM's attention as its "
                     "config describes it. Stop and tell the user.")
    units = dict(v_stock=u["ve"] / u["frames"], v_fovea=o["ve"] / o["frames"], prefill=c, gate_per_decision=float(gate),
                 stock_tokens=stock, tokens_per_frame=tpf)
    log(f"one stock frame {units['v_stock'] / 1e9:,.2f} GFLOPs ({u['frames']} frames), one foveated frame "
        f"{units['v_fovea'] / 1e9:,.2f} GFLOPs ({o['frames']} frames)")
    coef = None
    if per_layout:
        coef = flops.fit_vision([(f, v) for f, v, _ in layouts])
        units["vision_coef"] = coef
        fov = layouts[1:]
        log(f"the encoder per frame = {' + '.join(f'{x:.6g} x{i}' for i, x in enumerate(coef))} FLOPs over its layout's terms "
            f"x0, x1, ... (flops.py's features), exact on all {len(layouts)} layouts counted; at {len(fov)} gaze points a foveated "
            f"frame costs {min(v for _, v, _ in fov) / 1e9:,.2f} to {max(v for _, v, _ in fov) / 1e9:,.2f} GFLOPs with "
            f"{min(n for *_, n in fov)} to {max(n for *_, n in fov)} tokens")
        r = units["v_stock"] / layouts[0][1]
        if abs(r - 1) > flops.EXACT_TOLERANCE:
            sys.exit(f"FLOPS DIFFER: a scored stock frame costs {r:.6f} x the stock layout counted alone. Stop and tell the user.")
    for name, got in (("uniform", u), (top, o)):
        r = got["pf"] / float(flops.prefill(c, got["n_tokens"]))
        if abs(r - 1) > flops.EXACT_TOLERANCE:
            sys.exit(f"{name}: the scored question's prefill is {r:.6f} x the fitted one. Stop and tell the user.")

    log("\n--- step 2: prompt lengths")
    blind = {int(r.qid): int(ad.blind_ids(build_messages(r.question, list(r.options), [], 0)).shape[1]) for r in qs.itertuples()}
    samples = []
    for r in qs.iloc[np.linspace(0, len(qs) - 1, min(PROMPT_QUESTIONS, len(qs))).astype(int)].drop_duplicates("qid").itertuples():
        n_v = data["metas"][r.video]["n"]
        for f in PROMPT_FRAMES:
            frs = [data["frame"](r.video, int(i)) for i in common.uniform_indices(n_v, f)]
            if frs:
                n0 = measure["prompt_len"](ad, r.question, list(r.options), frs, [0] * len(frs))
                samples.append((len(frs), n0 - blind[int(r.qid)]))
    prompt = flops.fit_prompt(samples)
    log(f"a video prompt = its blind prompt + {prompt['c0']:.1f} + {prompt['o']:.2f} per frame + its image tokens "
        f"(largest miss {prompt['max_residual']:.1f} tokens over {len(samples)} prompts)")

    log("\n--- step 3: the targets, K' per row, the baselines' budget parameters")
    q = pd.DataFrame(dict(qid=qs.qid.astype(int), video=qs.video, blind=[blind[int(x)] for x in qs.qid],
                          video_frames=[int(data["metas"][v]["n"]) for v in qs.video]))
    decisions = {v: len(data["labels"][v]) for v in data["videos"]}
    ref = {name: frames_per_video(job) for name, job in ref_jobs.items()}
    fovea = {}
    if per_layout:
        for name, job in ref_jobs.items():
            if job["row"] in flops.FOVEA_ROWS:
                fovea[name] = fovea_per_video(job, measure, coef)
                n_frames = sum(ref[name][v] for v in fovea[name])
                log(f"{name}: {sum(x[1] for x in fovea[name].values()) / max(1, n_frames):.2f} image tokens and "
                    f"{sum(x[0] for x in fovea[name].values()) / max(1, n_frames) / 1e9:,.2f} GFLOPs per foveated frame over "
                    f"{n_frames} frames, from each frame's own layout")
    got, total = shares(q, units, prompt, ref, tpf, decisions, fovea)
    targets = {b: got[f"budget {100 * b:g}%"] for b in fractions}
    for b in fractions:
        log(f"budget {100 * b:g}% (tokens): ours costs {100 * targets[b]:.3f}% of Uniform K's FLOPs, the no-gate control "
            f"{100 * got[f'nogate {100 * b:g}%']:.3f}%")
    _, b_two = cal.match_gazevlm2(share, stock, ad.gazevlm_backend().side ** 2)
    tpf_two = max(2, int(round(b_two * stock)))
    families = dict(uniform_flops=(stock, 1), single=(tpf, 1), two_view=(tpf_two, 2))
    ks = plan_frame_counts(q, units, prompt, total, targets, k_max=cap or cfg["run"]["K"], families=families)
    inputs = Inputs(data)
    knobs = {}
    for b in fractions:
        k = ks["single", b][0]
        if k not in knobs:
            log(f"   counting the baselines' tokens on Uniform K'={k} frames ...")
            knobs[k] = knobs_at(ad, with_knob(cfg, "uniform", None, k), inputs(k), share,
                                os.path.join(out_model_dir, f"flops_counts_K{k}.jsonl"), log)

    names = {name: next(iter(job["conds"])) for name, job in ref_jobs.items()}
    blind_share = float(flops.prefill(c, q["blind"].to_numpy()).sum()) / total
    lines = [line("uniform", 1.0, 1.0, 1.0, k_ref, None, names["uniform"], 1.0),
             line("blind", 0.0, 0.0, blind_share, 0, None, run.BLIND, 0.0)]
    qq = dict(blind=q["blind"].to_numpy(), video_frames=q["video_frames"].to_numpy())
    for b in fractions:
        pct = f"{100 * b:g}%"
        lines.append(line("budget", b, targets[b], targets[b], 0, None, names[f"budget {pct}"], share))
        lines.append(line("nogate", b, None, got[f"nogate {pct}"], 0, None, names[f"nogate {pct}"], share))
        k, s = ks["uniform_flops", b]
        lines.append(line("uniform_flops", b, targets[b], s, k, None, None, 1.0))
        k = ks["single", b][0]
        for row in SINGLE_VIEW:
            pred, knob = knobs[k][row]
            s = flops.uniform_share(qq, units, prompt, total, k, pred * stock, 1)
            lines.append(line(row, b, targets[b], s, k, knob, None, pred))
        k = ks["two_view", b][0]
        s = flops.uniform_share(qq, units, prompt, total, k, tpf_two, 2)
        lines.append(line("gazevlm2", b, targets[b], s, k, b_two, None, tpf_two / stock))
    for ln in lines:
        if ln["condition"] is None:
            ln["condition"] = next(iter(planned_job(ad, cfg, inputs(ln["K"]), ln)["conds"]))

    log(f"\n--- step 4: every planned condition on questions {qids}, under the counter")
    seen, bad = {}, []
    with counting():
        for ln in lines:
            if ln["row"] == "blind":
                continue
            if (ln["row"], ln["condition"]) in seen:                  # two budgets sharing a condition: checked once
                ln["verified"] = seen[ln["row"], ln["condition"]]
                continue
            name = None
            if ln["row"] in REF_ROWS:
                name = "uniform" if ln["row"] == "uniform" else f"{ln['row']} {ln['budget_pct']:g}%"
                job = ref_jobs[name]
            else:
                job = planned_job(ad, cfg, inputs(ln["K"]), ln)
            ln["verified"] = seen[ln["row"], ln["condition"]] = []
            for qid in qids:
                got_q = counted(job, qid, decoder, tmp)
                want = fovea[name][got_q["video"]] if name in fovea else None
                r_ve, r_pf, ok = check(ln["row"], got_q, units, None if want is None else want[0])
                tokens_ok = want is None or got_q.get("n_image_tokens") == want[1]
                ln["verified"].append(dict(qid=qid, vision_ratio=r_ve, prefill_ratio=r_pf, ok=bool(ok and tokens_ok)))
                log(f"   {'ok ' if ok and tokens_ok else 'BAD'} {ln['row']:<13} {ln['budget_pct']:>5g}%  q{qid}: vision {r_ve:.5f}, "
                    f"prefill {r_pf:.6f} x flops.py ({got_q['frames']} images, {got_q['n_tokens']} tokens)")
                if not ok:
                    bad.append(f"{ln['row']} {ln['condition']} q{qid}: vision {r_ve:.5f}, prefill {r_pf:.6f}")
                if not tokens_ok:
                    bad.append(f"{ln['row']} {ln['condition']} q{qid}: {got_q.get('n_image_tokens')} image tokens scored, {want[1]} "
                               "from the frames' layouts")
    if missed:
        log(f"ATTENTION NOT COUNTED: {sorted(missed)} ran without a FLOP formula while scoring. Tell the user.")
    if bad:
        for b_ in bad:
            log("   " + b_)
        sys.exit(f"FLOPS DIFFER on {len(bad)} counts: flops.py doesn't describe these rows. Stop and tell the user.")

    videos = {v: dict(frames=int(data["metas"][v]["n"]), decisions=int(decisions[v])) for v in data["videos"]}
    plan = dict(model=key, model_id=ad.model_id, share=share, tokens_per_frame=tpf, stock_tokens=stock, K_reference=k_ref,
                fractions=fractions, units=units, prompt=prompt, targets={f"{100 * b:g}": targets[b] for b in fractions},
                videos=videos, verify_qids=qids, lines=lines,
                attention=dict(implementation=impl[0] or "the model's own", missed=sorted(missed)), llm_check=llm_check)
    if per_layout:      # ours' and the no-gate control's vision FLOPs and image tokens per question, by condition and video
        plan["fovea"] = {names[name]: {v: list(x) for v, x in per_video.items()} for name, per_video in fovea.items()}
        plan["vision_layouts"] = [dict(features=list(f), flops=v, tokens=n) for f, v, n in layouts]
    return plan


def targets_table(plan):
    """One line per row and budget, for the report and for the user."""
    t = pd.DataFrame([{k: v for k, v in ln.items() if k != "verified"} for ln in plan["lines"]])
    t.insert(0, "model", plan["model"])
    t["miss_pts"] = t.predicted_pct - t.target_pct
    t["nearest_only"] = t.miss_pts.abs() > flops.NEAREST_ONLY_PTS
    return t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="smolvlm", help=f"one of {', '.join(flops.MEASURE)}")
    ap.add_argument("--videos", help="override run.videos: all, or a comma list of video names")
    ap.add_argument("--out-dir", required=True, help="the results folder: writes <out-dir>/<model>/flops_plan.json and flops_targets.csv")
    ap.add_argument("--prefetch-workers", type=int, help="override run.prefetch_workers: threads loading frames ahead while encoding")
    ap.add_argument("--verify-questions", type=int, default=VERIFY_QUESTIONS, help="questions per condition counted in step 4")
    a = ap.parse_args(argv)
    if a.model not in flops.MEASURE:
        sys.exit(f"{a.model}: FLOPs are only measured for {sorted(flops.MEASURE)}: its LLM call, prompt and encoder need their own "
                 "functions in accuracy_tables/flops.py")
    cfg = common.load_config(a.config)
    if a.prefetch_workers is not None:
        cfg["run"]["prefetch_workers"] = a.prefetch_workers
    if a.videos:
        cfg["run"]["videos"] = a.videos if a.videos in ("test", "all") else [v.strip() for v in a.videos.split(",")]
    out_dir = os.path.abspath(os.path.expanduser(a.out_dir))
    cfg["paths"]["out_dir"] = out_dir
    out_model = os.path.join(out_dir, a.model)
    os.makedirs(out_model, exist_ok=True)

    data = common.load_data(cfg)
    print(common.describe(data), flush=True)
    ad = ADAPTERS[a.model](cfg["models"].get(a.model), cfg).load()
    try:
        plan = make_plan(ad, cfg, data, out_model, a.verify_questions)
    finally:
        ad.free()
    import transformers
    plan.update(config=os.path.abspath(a.config), config_sha256=sha256(a.config), videos_arg=cfg["run"]["videos"],
                versions=dict(torch=torch.__version__, transformers=transformers.__version__),
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")
    path = os.path.join(out_model, "flops_plan.json")
    with open(path + ".tmp", "w") as f:
        json.dump(plan, f, indent=1)
    os.replace(path + ".tmp", path)
    table = targets_table(plan)
    table.to_csv(os.path.join(out_model, "flops_targets.csv"), index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 70)
    print("\nFLOPs share of Uniform K per row and budget, from token counts and FLOPs only. nearest_only: more than "
          f"{flops.NEAREST_ONLY_PTS} points off. nogate keeps ours' token budget and has no FLOPs target:")
    print(table[["row", "budget_pct", "target_pct", "predicted_pct", "miss_pts", "nearest_only", "K", "knob", "token_share_pct"]]
          .round(3).to_string(index=False))
    print(f"{table.condition.nunique()} conditions for {len(table)} lines")
    print(f"PLAN WRITTEN: {path}")


if __name__ == "__main__":
    main()

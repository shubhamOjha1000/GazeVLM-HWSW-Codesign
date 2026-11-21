"""Ablation on EgoEverything's AEA videos: what the temporal gate and the spatial fovea each add, and the two combined, in accuracy and
TFLOPs per question. One A100 per model (qwen2.5, qwen3, smolvlm, smolvlm256), every row scored fresh.

    python -u -m accuracy_tables.hpc.ablation_aea plan    --model M --config C --root R --workers 16
    python -u -m accuracy_tables.hpc.ablation_aea prepare --model M --config C --root R --procs P
    python -u -m accuracy_tables.hpc.ablation_aea run     --model M --config C --root R --worker i --of P --workers W
    python -u -m accuracy_tables.hpc.ablation_aea status  --model M --root R
    python -u -m accuracy_tables.hpc.ablation_aea report  --root R

The videos: the gate videos whose names start with `loc` (AEA), 25 videos and 2,312 questions; Nymeria is left out. The rows, a 2 x 2 of
frames {evenly spaced, the gate's} x input {stock, foveated}, all at the 10% token budget's frame limit, plus Uniform K:

    uniform          Uniform K: K evenly spaced frames at stock input (run.py's row)                  the 100% reference
    uniform_frames   the frame limit's evenly spaced frames (the no-gate control's) at stock input     neither
    temporal         the gate's frames (the budget rule's) at stock input                              temporal only
    nogate           the no-gate control: the evenly spaced frames, foveated (run.py's row)            spatial only
    budget           the budget rule: the gate's frames, foveated (run.py's row, EgoWAVE)               combined

The frame limit is the budget rule's own (vlmeval/budget_rule.py, at the model's foveated tokens per frame), so the 4 rows of the 2 x 2
send the same number of frames per video, and differ only in which frames and how each is encoded. Every setting is common.py's and the
model's config's; nothing is tuned here.

A question's FLOPs (accuracy_tables/flops.py): the gate (5 fold models x one decision x the video's decisions) on the rows that use the
gate, plus the vision encoder (frames x one stock frame, or the foveated frames' own layouts), plus the LLM's one forward over the prompt.
`plan` measures these units on the real model, checks every row on 2 questions under the counter, and writes <root>/<model>/plan.json.
`report` counts each answer's FLOPs from its own record and those units. No winner is declared.

Writes only <root>/<model>/ and <root>/summary_ablation.csv. Never commit what it writes.
"""

import argparse
import contextlib
import glob
import json
import os
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from accuracy_tables import common, flops, run  # noqa: E402
from accuracy_tables import flops_calibrate as fc  # noqa: E402

FRACTION = 0.1
EXPECT_VIDEOS, EXPECT_QUESTIONS = 25, 2312
MODELS = ("qwen2.5", "qwen3", "smolvlm", "smolvlm256")
ROWS = ("uniform", "uniform_frames", "temporal", "nogate", "budget")
LABELS = {"uniform": "Uniform K (all frames, stock)", "uniform_frames": "Uniform, same frames (stock)",
          "temporal": "Temporal only (gate frames, stock)", "nogate": "Spatial only (uniform frames, fovea)",
          "budget": "Temporal + spatial (EgoWAVE)"}
FOVEA_ROWS = ("nogate", "budget")
GATE_ROWS = ("temporal", "budget")
VERIFY = 2


# ---------------------------------------------------------------------------- shared
def model_dir(a):
    return os.path.join(os.path.abspath(a.root), a.model)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(path + ".tmp", path)


def aea_videos(cfg):
    """The AEA gate videos that have questions, sorted."""
    c = dict(cfg, run=dict(cfg["run"], videos="all"))
    _, clips, _ = common._questions_and_clips(c)
    return sorted({x["sequence"] for x in clips if x["sequence"].startswith("loc")})


def ablation_cfg(config, videos=None, workers=None):
    """The model's config with the budget at 10% only and run.videos the AEA videos (or the given ones)."""
    cfg = common.load_config(config)
    cfg["budget"]["fractions"] = [FRACTION]
    cfg["run"]["videos"] = list(videos) if videos is not None else aea_videos(cfg)
    if workers is not None:
        cfg["run"]["prefetch_workers"] = int(workers)
    return cfg


def condition_name(row, policy):
    pct = f"{100 * FRACTION:g}%"
    if row == "temporal":
        return f"temporal only: {policy} gate frames at the {pct} limit, stock input"
    return f"uniform frames at the {pct} limit (slot middles), stock input"


def build_jobs(ad, cfg, data):
    """{row: run.py-style job} for the 5 rows. uniform, nogate and budget are run.py's own jobs; uniform_frames and temporal take the
    no-gate control's and the budget rule's frames (common.budget_frames, the same call run.py makes) at the model's stock input."""
    jobs = {}
    for job in run.row_jobs(ad, cfg, data, ["uniform", "budget", "nogate"]):
        jobs[job["row"]] = job
    for row, use_gate in (("uniform_frames", False), ("temporal", True)):
        frames, why = common.budget_frames(cfg, data, ad.key, ad.stock_tokens, FRACTION, use_gate=use_gate)
        items = common.make_items(data["qs"], frames)
        name = condition_name(row, cfg["run"]["policy"])
        jobs[row] = dict(row=row, name=row, items=items, conds={name: ad.stock_condition()}, score=ad.score,
                         msg=f"--- {name}: mean {np.mean([len(it['frames']) for it in items]):.1f} frames per question; frames kept, "
                             f"by reason: {why}")
    return {r: jobs[r] for r in ROWS}


def frames_of(job):
    return {it["video"]: len(it["frames"]) for it in job["items"]}


def load_adapter(key, cfg):
    from accuracy_tables.models import ADAPTERS
    return ADAPTERS[key](cfg["models"].get(key), cfg).load()


def check_aea(data):
    nv, nq = len(data["videos"]), len(data["qs"])
    print(f"AEA: {nq} questions on {nv} videos", flush=True)
    if (nv, nq) != (EXPECT_VIDEOS, EXPECT_QUESTIONS):
        sys.exit(f"AEA has {nq} questions on {nv} videos here, not {EXPECT_QUESTIONS} on {EXPECT_VIDEOS}. Stop and tell the user.")


# ---------------------------------------------------------------------------- plan: the FLOPs units, measured
def plan_cmd(a):
    from system_perf import probe
    out = model_dir(a)
    path = os.path.join(out, "plan.json")
    if os.path.exists(path):
        print(f"{path} exists: kept as it is (the run reads it)", flush=True)
        return
    os.makedirs(out, exist_ok=True)
    cfg = ablation_cfg(a.config, workers=a.workers)
    data = common.load_data(cfg)
    check_aea(data)
    ad = load_adapter(a.model, cfg)
    key = ad.key
    measure = flops.MEASURE[key]
    run.check_parity(ad, data, key)
    print("parity OK", flush=True)
    jobs = build_jobs(ad, cfg, data)
    decoder = probe.find_decoder(ad.model)
    tmp = os.path.join(tempfile.mkdtemp(prefix="ablation_"), "rec.jsonl")
    qs = data["qs"].sort_values(["video", "qid"]).reset_index(drop=True)
    qids = fc.verify_qids(qs, VERIFY)
    per_layout = "features" in measure
    impl, missed = [None], set()

    @contextlib.contextmanager
    def counting():
        with flops.attention_implementation(ad.model, impl[0]), flops.gqa_attention(), flops.AttentionWatch() as watch:
            yield
        missed.update(watch.missed)

    def count_units():
        with counting():
            pts = [(n, measure["prefill"](ad, n)) for n in fc.PREFILL_LENGTHS]
            u_ = fc.counted(jobs["uniform"], qids[0], decoder, tmp)
            o_ = fc.counted(jobs["budget"], qids[0], decoder, tmp)
            lay = []
            if per_layout:
                lay = [dict(kind=kind, label=label, features=list(measure["features"](cells, side)),
                            flops=float(measure["vision"](ad, cells, side)), tokens=len(cells))
                       for kind, label, cells, side in fc.vision_fit_layouts(ad, cfg, measure)]
        return pts, u_, o_, lay

    print("\n--- step 1: FLOPs units", flush=True)
    gate = flops.measure_gate_ckpt(os.path.join(cfg["paths"]["gate_ckpt_dir"], "fold1.pt"))
    print(f"gate: {gate:,} FLOPs per decision per fold model, counted as {flops.GATE_MODELS} fold models", flush=True)
    points, u, o, layouts = count_units()
    if missed:
        print(f"no FLOP formula for {sorted(missed)} under the model's own attention: every count redone with eager attention", flush=True)
        impl[0] = "eager"
        missed.clear()
        points, u, o, layouts = count_units()
    if missed:
        print(f"ATTENTION NOT COUNTED: {sorted(missed)} ran without a FLOP formula, even with eager attention. Tell the user.", flush=True)
    c = flops.fit_prefill(points)
    print(f"LLM forward: {c[0]:.4g} + {c[1]:.4g} n + {c[2]:.4g} n^2 FLOPs for n tokens", flush=True)
    llm_check = None
    if "llm_shape" in measure:
        shape = measure["llm_shape"](ad)
        llm_check = dict(shape, **flops.llm_shape_check(c, shape))
        print(f"n^2 term against the model's config: {100 * llm_check['c_off']:+.3f}%", flush=True)
        if abs(llm_check["c_off"]) > measure.get("llm_shape_tolerance", flops.LLM_SHAPE_TOLERANCE):
            sys.exit(f"c DIFFERS FROM CONFIG by {100 * llm_check['c_off']:+.2f}%. Stop and tell the user.")
    units = dict(v_stock=u["ve"] / u["frames"], v_fovea=o["ve"] / o["frames"], prefill=c, gate_per_decision=float(gate),
                 stock_tokens=ad.stock_tokens)
    print(f"one stock frame {units['v_stock'] / 1e9:,.2f} GFLOPs, one foveated frame {units['v_fovea'] / 1e9:,.2f} GFLOPs "
          f"(mean of question {qids[0]}'s frames)", flush=True)
    coef = None
    if per_layout:
        coef = flops.fit_vision([(x["features"], x["flops"]) for x in layouts])
        units["vision_coef"] = coef
        print(f"the encoder per frame fitted on {len(layouts)} layouts, exact on all", flush=True)
        r = units["v_stock"] / layouts[0]["flops"]
        if abs(r - 1) > flops.EXACT_TOLERANCE:
            sys.exit(f"FLOPS DIFFER: a scored stock frame costs {r:.6f} x the stock layout counted alone. Stop and tell the user.")
    for name, got in (("uniform", u), ("budget", o)):
        r = got["pf"] / float(flops.prefill(c, got["n_tokens"]))
        if abs(r - 1) > flops.EXACT_TOLERANCE:
            sys.exit(f"{name}: the scored question's prefill is {r:.6f} x the fitted one. Stop and tell the user.")

    fovea = {}
    if per_layout:
        for row in FOVEA_ROWS:
            fovea[row] = fc.fovea_per_video(jobs[row], measure, coef)

    print(f"\n--- step 2: every row on questions {qids}, under the counter", flush=True)
    bad, verified = [], {}
    with counting():
        for row in ROWS:
            verified[row] = []
            for qid in qids:
                got = fc.counted(jobs[row], qid, decoder, tmp)
                want = fovea[row][got["video"]] if row in fovea else None
                r_ve, r_pf, ok = fc.check(row, got, units, None if want is None else want[0])
                tokens_ok = want is None or got.get("n_image_tokens") == want[1]
                verified[row].append(dict(qid=qid, vision_ratio=r_ve, prefill_ratio=r_pf, ok=bool(ok and tokens_ok)))
                print(f"   {'ok ' if ok and tokens_ok else 'BAD'} {row:<15} q{qid}: vision {r_ve:.5f}, prefill {r_pf:.6f} x flops.py "
                      f"({got['frames']} images, {got['n_tokens']} tokens)", flush=True)
                if not (ok and tokens_ok):
                    bad.append(f"{row} q{qid}")
    if missed:
        print(f"ATTENTION NOT COUNTED: {sorted(missed)} ran without a FLOP formula while scoring. Tell the user.", flush=True)
    if bad:
        sys.exit(f"FLOPS DIFFER on {bad}: flops.py doesn't describe these rows. Stop and tell the user.")

    decisions = {v: len(data["labels"][v]) for v in data["videos"]}
    qn = data["qs"].groupby("video").size().to_dict()
    rows = {r: dict(label=LABELS[r], condition=next(iter(jobs[r]["conds"])), frames=frames_of(jobs[r])) for r in ROWS}
    plan = dict(model=key, model_id=ad.model_id, config=os.path.abspath(a.config), config_sha256=fc.sha256(a.config),
                fraction=FRACTION, n_questions=int(len(data["qs"])),
                videos={v: dict(frames=int(data["metas"][v]["n"]), decisions=int(decisions[v]), questions=int(qn.get(v, 0)))
                        for v in data["videos"]},
                units=units, llm_check=llm_check, rows=rows, verified=verified,
                attention=dict(implementation=impl[0] or "the model's own", missed=sorted(missed)))
    if per_layout:
        plan["fovea"] = {r: {v: list(x) for v, x in per_video.items()} for r, per_video in fovea.items()}
    print("\nframes per question:", flush=True)
    for r in ROWS:
        f = np.array([rows[r]["frames"][v] for v in data["qs"].video])
        print(f"   {r:<15} {f.mean():6.1f}   {rows[r]['condition']}", flush=True)
    write_json(path, plan)
    print(f"PLAN WRITTEN: {path}", flush=True)


# ---------------------------------------------------------------------------- prepare: repair, drop out-of-memory records, split
def worker_dirs(out):
    return sorted(glob.glob(os.path.join(out, "answers", "w*")))


def prepare_cmd(a):
    from accuracy_tables.hpc import repair_results
    out = model_dir(a)
    plan = read_json(os.path.join(out, "plan.json"))
    dropped = {}
    for d in worker_dirs(out):
        for p in repair_results.repair(d):
            print("dropped a half-written last line:", p, flush=True)
        for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            with open(path) as f:
                lines = [x for x in f if x.strip()]
            keep = [x for x in lines if json.loads(x).get("error") != "oom"]
            if len(keep) != len(lines):
                with open(path + ".tmp", "w") as f:
                    f.writelines(keep)
                os.replace(path + ".tmp", path)
                dropped[os.path.relpath(path, out)] = len(lines) - len(keep)
    print(f"OOM RECORDS DROPPED: {dropped or 'none'}; every answer kept", flush=True)
    path = os.path.join(out, "answers", "assignment.json")
    if os.path.exists(path):
        asg = read_json(path)
        if asg["procs"] != a.procs:
            sys.exit(f"answers/assignment.json is for {asg['procs']} processes, not {a.procs}: resubmit with PROCS={asg['procs']}")
        print(f"assignment kept: {asg['procs']} processes", flush=True)
        return
    # whole videos onto processes, the heaviest first onto the least loaded; weight = questions x frames over the 5 rows
    weight = {v: x["questions"] * sum(plan["rows"][r]["frames"].get(v, 0) for r in ROWS) for v, x in plan["videos"].items()}
    groups, loads = [[] for _ in range(a.procs)], [0] * a.procs
    for v in sorted(weight, key=lambda v: (-weight[v], v)):
        i = min(range(a.procs), key=lambda j: (loads[j], j))
        groups[i].append(v)
        loads[i] += weight[v]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_json(path, dict(procs=a.procs, videos=[sorted(g) for g in groups], weights=loads))
    for i, g in enumerate(groups):
        print(f"   w{i}: {len(g)} videos, {sum(plan['videos'][v]['questions'] for v in g)} questions", flush=True)


# ---------------------------------------------------------------------------- run: one process's videos, all 5 rows
def run_cmd(a):
    import torch
    out = model_dir(a)
    plan = read_json(os.path.join(out, "plan.json"))
    if fc.sha256(a.config) != plan["config_sha256"]:
        sys.exit(f"{a.config} changed since the plan was made. Stop and tell the user.")
    asg = read_json(os.path.join(out, "answers", "assignment.json"))
    if asg["procs"] != a.of:
        sys.exit(f"the assignment is for {asg['procs']} processes, not {a.of}")
    videos = asg["videos"][a.worker]
    wdir = os.path.join(out, "answers", f"w{a.worker}")
    os.makedirs(wdir, exist_ok=True)
    cfg = ablation_cfg(a.config, videos=videos, workers=a.workers)
    data = common.load_data(cfg)
    ad = load_adapter(a.model, cfg)
    status = {}
    try:
        run.check_parity(ad, data, ad.key)
        print(f"w{a.worker}: parity OK, {len(data['qs'])} questions on {len(videos)} videos", flush=True)
        jobs = build_jobs(ad, cfg, data)
        for r, job in jobs.items():
            if next(iter(job["conds"])) != plan["rows"][r]["condition"]:
                sys.exit(f"{r}: the condition here differs from the plan's. Stop and tell the user.")
            want = {v: plan["rows"][r]["frames"][v] for v in videos}
            if frames_of(job) != want:
                sys.exit(f"{r}: frames per video differ from the plan's. Stop and tell the user.")
        open(os.path.join(out, "answers", f".scoring_w{a.worker}"), "w").close()
        for r, job in jobs.items():
            path = os.path.join(wdir, f"{r}.jsonl")
            with run._row(r, status, False, path):
                print("\n" + job["msg"], flush=True)
                job["score"](job["items"], job["conds"], path)
    finally:
        ad.free()
        with open(os.path.join(wdir, "gpu.txt"), "a") as f:
            f.write((torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu") + "\n")
    for name, s in status.items():
        print(f"   w{a.worker} {name:<15} {s}", flush=True)


# ---------------------------------------------------------------------------- status and report
def load_records(out):
    """Every record of the 5 rows over all processes -> DataFrame with a row column."""
    from vlmeval import egoev
    parts = []
    for d in worker_dirs(out):
        for r in ROWS:
            df = egoev.load_results(os.path.join(d, f"{r}.jsonl"))
            if len(df):
                parts.append(df.assign(row=r))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["row", "qid", "video"])


def status_cmd(a):
    out = model_dir(a)
    plan = read_json(os.path.join(out, "plan.json"))
    res = load_records(out)
    done = True
    for r in ROWS:
        g = res[res.row == r] if len(res) else res
        err = int(g["error"].notna().sum()) if "error" in g else 0
        n = len(g) - err
        done &= n == plan["n_questions"] and err == 0
        print(f"   {r:<15} {n:5d} of {plan['n_questions']} answered, {err} error records", flush=True)
    print("RUN DONE" if done else "RUN INCOMPLETE: resubmit the same job (it continues), or tell the user if it repeats", flush=True)


def add_flops(res, plan):
    u, videos = plan["units"], plan["videos"]
    res = res.copy()
    n_frames = res["n_frames"].fillna(0).to_numpy(dtype=np.float64)
    fovea_row = res.row.isin(FOVEA_ROWS).to_numpy()
    vision = n_frames * np.where(fovea_row, u["v_fovea"], u["v_stock"])
    if plan.get("fovea"):
        planned = [plan["fovea"][r][v][0] if r in FOVEA_ROWS else 0.0 for r, v in zip(res.row, res.video)]
        vision = np.where(fovea_row, np.asarray(planned, dtype=np.float64), vision)
    dec = res.video.map({v: x["decisions"] for v, x in videos.items()}).to_numpy(dtype=np.float64)
    gate = np.where(res.row.isin(GATE_ROWS).to_numpy(), flops.GATE_MODELS * u["gate_per_decision"] * dec, 0.0)
    res["gate"], res["vision"] = gate, vision
    res["flops"] = gate + vision + flops.prefill(u["prefill"], res["n_tokens"].to_numpy())
    return res


def model_table(out, model):
    from vlmeval import egoev
    plan = read_json(os.path.join(out, "plan.json"))
    res = load_records(out)
    problems = []
    if "error" in res:
        bad = res[res["error"].notna()]
        problems += [f"{model} {r}: {n} error records" for r, n in bad.groupby("row").size().items()]
        res = res[res["error"].isna()]
    res = add_flops(res, plan)
    if plan.get("fovea"):
        off = [r for r, v, n in zip(res.row, res.video, res.n_image_tokens) if r in FOVEA_ROWS and plan["fovea"][r][v][1] != n]
        if off:
            problems.append(f"{model}: {len(off)} foveated records whose image tokens aren't the plan's")
    ref = res[res.row == "uniform"].set_index("qid")
    lines = []
    for r in ROWS:
        g = res[res.row == r].copy()
        if not len(g):
            problems.append(f"{model} {r}: no records")
            continue
        g["correct"] = g["correct"].astype(int)
        acc = egoev.accuracy_table(g).iloc[0]
        common_q = g.set_index("qid").index.intersection(ref.index)
        gq = g.set_index("qid").loc[common_q]
        share = 100 * gq["flops"].sum() / ref.loc[common_q, "flops"].sum() if len(common_q) else np.nan
        tok = 100 * gq["n_image_tokens"].sum() / ref.loc[common_q, "n_image_tokens"].sum() if len(common_q) else np.nan
        if int(acc.n) != plan["n_questions"]:
            problems.append(f"{model} {r}: n = {int(acc.n)}, expected {plan['n_questions']}")
        lines.append(dict(model=model, row=r, label=LABELS[r], condition=plan["rows"][r]["condition"], n=int(acc.n),
                          accuracy=acc.accuracy, ci_low=acc.ci_low, ci_high=acc.ci_high, tflops_per_q=g["flops"].mean() / 1e12,
                          flops_pct_of_uniform=share, token_pct_of_uniform=tok, gate_gflops_per_q=g["gate"].mean() / 1e9,
                          vision_tflops_per_q=g["vision"].mean() / 1e12, frames_per_q=g["n_frames"].mean(),
                          image_tokens_per_q=g["n_image_tokens"].mean(), prompt_tokens_per_q=g["n_tokens"].mean(),
                          videos=g.video.nunique()))
    gpus = set()
    for d in worker_dirs(out):
        p = os.path.join(d, "gpu.txt")
        if os.path.exists(p):
            gpus |= {x.strip() for x in open(p) if x.strip()}
    if len(gpus) > 1 or any("A100" not in x for x in gpus):
        problems.append(f"{model}: GPUs recorded {sorted(gpus)}, expected one A100 type")
    return pd.DataFrame(lines), problems, sorted(gpus)


def report_cmd(a):
    root = os.path.abspath(a.root)
    tables, problems = [], []
    for m in MODELS:
        out = os.path.join(root, m)
        if not os.path.exists(os.path.join(out, "plan.json")):
            print(f"{m}: no plan.json, left out", flush=True)
            continue
        t, p, gpus = model_table(out, m)
        print(f"\n=== {m} (GPU {gpus}), AEA, rows at the {100 * FRACTION:g}% token budget's frame limit; no winner is declared", flush=True)
        cols = ["label", "n", "accuracy", "ci_low", "ci_high", "tflops_per_q", "flops_pct_of_uniform", "frames_per_q", "image_tokens_per_q"]
        print(t[cols].round(2).to_string(index=False), flush=True)
        tables.append(t)
        problems += p
    if not tables:
        sys.exit("no model has a plan")
    allt = pd.concat(tables, ignore_index=True)
    allt.to_csv(os.path.join(root, "summary_ablation.csv"), index=False)
    acc = allt.pivot_table(index="row", columns="model", values="accuracy").reindex(ROWS)
    tf = allt.pivot_table(index="row", columns="model", values="tflops_per_q").reindex(ROWS)
    acc["avg"] = acc.mean(axis=1)
    tf["avg"] = tf.mean(axis=1)
    paper = pd.concat({"accuracy (%)": acc.round(2), "TFLOPs / question": tf.round(3)}, axis=1)
    paper.index = [LABELS[r] for r in paper.index]
    print("\n=== the ablation table: accuracy and TFLOPs per question, per model", flush=True)
    print(paper.to_string(), flush=True)
    print(f"\nwrote {os.path.join(root, 'summary_ablation.csv')}", flush=True)
    if problems:
        for p in problems:
            print("   " + p, flush=True)
        sys.exit("INCOMPLETE: tell the user")
    print("EVERY n IS COMPLETE", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("plan", "prepare", "run", "status", "report"))
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--config", help="config.all350.yaml (config.smolvlm256.yaml for smolvlm256)")
    ap.add_argument("--root", required=True, help="results root: <root>/<model>/")
    ap.add_argument("--workers", type=int, default=8, help="frame loading threads")
    ap.add_argument("--procs", type=int, default=1, help="prepare: processes sharing the GPU")
    ap.add_argument("--worker", type=int, help="run: this process's share")
    ap.add_argument("--of", type=int, help="run: the processes the assignment is for")
    a = ap.parse_args(argv)
    if a.command != "report" and not a.model:
        ap.error("--model is needed")
    if a.command in ("plan", "run") and not a.config:
        ap.error("--config is needed")
    dict(plan=plan_cmd, prepare=prepare_cmd, run=run_cmd, status=status_cmd, report=report_cmd)[a.command](a)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)

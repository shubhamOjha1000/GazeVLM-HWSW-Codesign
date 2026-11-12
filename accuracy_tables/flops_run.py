"""Score the FLOPs-matched table that flops_calibrate.py planned (runbook section 14).

    python -m accuracy_tables.flops_run --config accuracy_tables/hpc/config.all350.yaml --model smolvlm \
        --plan <results folder>/smolvlm/flops_plan.json --out-dir <results folder> --prefetch-workers 8 --keep-going

Every condition goes through run.py's own jobs, with nothing changed but its frame count and its budget parameter:
  uniform, blind, budget, nogate   as run.py scores them (Uniform K, Blind, the budget rule and the no-gate control at 5, 10 and 20%)
  uniform_flops and the baselines  run.py's job for that row on Uniform K' frames, at the plan's K' and budget parameter
Results are appended one question at a time to <out-dir>/<model>/<row>.jsonl (Uniform K' to uniform_flops.jsonl), so a rerun
continues where it stopped. The job stops before scoring if the config isn't the one the plan was made with, or if a condition's
name here differs from the plan's.
"""

import argparse
import json
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accuracy_tables import common, run  # noqa: E402
from accuracy_tables import flops_calibrate as fc  # noqa: E402
from accuracy_tables.models import ADAPTERS  # noqa: E402

REF_ROWS = fc.REF_ROWS


def reference_jobs(ad, cfg, data, plan):
    """run.py's jobs for Uniform K, Blind, the budget rule and the no-gate control; their conditions must be the plan's."""
    jobs = run.row_jobs(ad, cfg, data, list(REF_ROWS))
    have = {(j["row"], c) for j in jobs for c in j["conds"]}
    want = {(ln["row"], ln["condition"]) for ln in plan["lines"] if ln["row"] in REF_ROWS}
    if have != want:
        sys.exit(f"the reference rows' conditions differ from the plan's: here only {sorted(have - want)}, "
                 f"in the plan only {sorted(want - have)}. Stop and tell the user.")
    return jobs


def plan_jobs(ad, cfg, data, plan):
    """One job per planned condition of Uniform K' and the baselines (a condition two budgets share is scored once)."""
    inputs, jobs, seen = fc.Inputs(data), [], set()
    for ln in plan["lines"]:
        if ln["row"] in REF_ROWS or (ln["row"], ln["condition"]) in seen:
            continue
        seen.add((ln["row"], ln["condition"]))
        job = fc.planned_job(ad, cfg, inputs(ln["K"]), ln)
        if set(job["conds"]) != {ln["condition"]}:
            sys.exit(f"{ln['row']} at {ln['budget_pct']:g}%: run.py names it {sorted(job['conds'])}, the plan {ln['condition']!r}. "
                     "Stop and tell the user.")
        jobs.append(job)
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the config the plan was made with")
    ap.add_argument("--model", default="smolvlm")
    ap.add_argument("--plan", required=True, help="flops_plan.json from flops_calibrate.py")
    ap.add_argument("--videos", help="override run.videos: all, or a comma list of video names")
    ap.add_argument("--max-questions", type=int, help="score only the first N questions (by video, then qid)")
    ap.add_argument("--out-dir", required=True, help="results: <out-dir>/<model>/<row>.jsonl")
    ap.add_argument("--prefetch-workers", type=int, help="override run.prefetch_workers")
    ap.add_argument("--keep-going", action="store_true", help="a failing row doesn't stop the rest; a pass / fail table ends the log")
    a = ap.parse_args(argv)

    with open(a.plan) as f:
        plan = json.load(f)
    if plan["model"] != a.model:
        sys.exit(f"the plan is for {plan['model']}, not {a.model}")
    if fc.sha256(a.config) != plan["config_sha256"]:
        sys.exit(f"{a.config} changed since the plan was made. Stop and tell the user.")
    cfg = common.load_config(a.config)
    if a.prefetch_workers is not None:
        cfg["run"]["prefetch_workers"] = a.prefetch_workers
    if a.videos:
        cfg["run"]["videos"] = a.videos if a.videos in ("test", "all") else [v.strip() for v in a.videos.split(",")]
    cfg["paths"]["out_dir"] = os.path.abspath(os.path.expanduser(a.out_dir))
    if run.baselines_without_threads([a.model], run.BASELINE_ROWS, cfg["run"]["prefetch_workers"]):
        ap.error(f"frame loading threads aren't built for the baseline rows of {a.model}: leave out --prefetch-workers")

    data = common.load_data(cfg)
    if a.max_questions:
        data["qs"] = data["qs"].sort_values(["video", "qid"]).head(a.max_questions)
    print(common.describe(data), flush=True)
    out_dir = os.path.join(cfg["paths"]["out_dir"], a.model)
    os.makedirs(out_dir, exist_ok=True)
    ad = ADAPTERS[a.model](cfg["models"].get(a.model), cfg).load()
    status = {}
    info = dict(model=a.model, model_id=ad.model_id, plan=os.path.abspath(a.plan), plan_sha256=fc.sha256(a.plan),
                config=cfg, stock_tokens_per_frame=ad.stock_tokens)
    try:
        info["parity"] = dict(ok=True, report=run.check_parity(ad, data, a.model))
        jobs = reference_jobs(ad, cfg, data, plan) + plan_jobs(ad, cfg, data, plan)
        print(f"{len(jobs)} jobs, {sum(len(j['conds']) for j in jobs)} conditions", flush=True)
        for job in jobs:
            path = os.path.join(out_dir, f"{job['row']}.jsonl")
            with run._row(job.get("name", job["row"]), status, a.keep_going, path):
                print("\n" + job["msg"], flush=True)
                job["score"](job["items"], job["conds"], path)
    finally:
        try:
            import transformers
            info["versions"] = dict(transformers=transformers.__version__, torch=torch.__version__)
        except Exception:
            pass
        info["status"] = status
        with open(os.path.join(out_dir, "run_info_flops.json"), "w") as f:
            json.dump(info, f, indent=1, default=str)
        ad.free()
    print("\n=== pass / fail per row")
    for name, s in status.items():
        print(f"   {name:<20} {s}")
    n_fail = sum(s.startswith("FAILED") for s in status.values())
    print(f"=== {n_fail} failed of {len(status)}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)

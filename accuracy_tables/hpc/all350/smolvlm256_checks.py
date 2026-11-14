"""SmolVLM2-256M's checks around the FLOPs-matched run (runbook section 16). Prints no accuracy, except vs2p2b, which reads two finished
summaries.

    python -u -m accuracy_tables.hpc.all350.smolvlm256_checks pick-videos (--questions 60 | --every 25)
    python -u -m accuracy_tables.hpc.all350.smolvlm256_checks tokens --config <yaml> --reference <results/smolvlm_flops/smolvlm>
    python -u -m accuracy_tables.hpc.all350.smolvlm256_checks same --new <folder> --ref <folder> --rows <list> [--all]
    python -u -m accuracy_tables.hpc.all350.smolvlm256_checks identity --folder <folder>
    python -u -m accuracy_tables.hpc.all350.smolvlm256_checks estimate --timing <folder>
    python -u -m accuracy_tables.hpc.all350.smolvlm256_checks records --folder <folder> --plan <flops_plan.json> --rows <list>
    python -u -m accuracy_tables.hpc.all350.smolvlm256_checks vs2p2b --ours <csv> --other <csv> --out <csv>

pick-videos  a deterministic set of videos: AEA and Nymeria taking turns in name order until --questions are covered, or every
             --every-th video in name order (the calibrate job's timing sample)
tokens       the 256M's stock tokens per frame from its pinned config, its smallest fovea's tokens on every frame of all 350 videos
             (stops unless one number), the frame limits at each budget (the same proposer as section 10, measure_tokens.limits_table),
             then SAME FRAMES AS 2.2B: where a video's 256M limit equals its 2.2B limit, the budget rule must keep the same frames, and
             the 2.2B's frames recomputed here must have the frame counts of every budget record in --reference. Ends with PROPOSED.
same         every record of --new against --ref on the same (condition, question): answer, probabilities, frames and tokens. Ends with
             SAME ANSWERS or stops. --all: --ref may not have records --new lacks (threads on against off).
identity     check_identity's comparison, also requiring max_prob_diff 0.0: SAME AS UNIFORM K
estimate     the full run's time, from the calibrate job's timing run on a sample of videos
records      one job's folder: oom / nan records, the GPU per record from its ledger, and every planned answer there
vs2p2b       the side table: per group, row and budget, accuracy, interval and FLOPs share on both models. No test, no winner.
"""
import argparse
import collections
import json
import os
import re
import sys

from accuracy_tables.hpc.all350 import prepare

MODEL, REFERENCE_MODEL = "smolvlm256", "smolvlm"
KEYS = ("pred", "p", "n_frames", "n_image_tokens", "n_tokens", "error")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------- pick-videos
def pick_by_questions(counts, n):
    """AEA (names starting with loc) and Nymeria taking turns, each in name order, until n questions are covered."""
    pools = [sorted(v for v in counts if not v.startswith("loc")), sorted(v for v in counts if v.startswith("loc"))]
    out, got = [], 0
    while got < n and any(pools):
        for pool in pools:
            if pool and got < n:
                v = pool.pop(0)
                out.append(v)
                got += counts[v]
    return out


def pick_every(counts, k):
    return sorted(counts)[::int(k)]


# ---------------------------------------------------------------------------- tokens and the 2.2B's frames
def video_limits(cfg, data, key, stock, fraction):
    """{video: the budget rule's frame limit}, as common.budget_frames computes it."""
    from accuracy_tables import common
    from vlmeval import budget_rule as br
    per_frame = cfg["budget"]["tokens_per_frame"][key]
    cap = cfg["run"]["max_frames"].get(key)
    out = {}
    for c in data["clips"]:
        v = c["sequence"]
        n = int((c["frame_no"] < data["metas"][v]["n"]).sum())
        full = len(common.cap_frames(data["uniform"][v], cap)) * stock
        out[v] = br.frame_limit(float(fraction), full, per_frame, n, cap)
    return out


def same_frames(cfg, data, reference):
    """(pairs where the limits are equal, of them with other frames, pairs where the check doesn't apply, 2.2B budget records whose
    frame count isn't the one recomputed here, records read)."""
    from accuracy_tables import common, run
    from accuracy_tables.models import ADAPTERS
    label = ADAPTERS[REFERENCE_MODEL]().setting_label(ADAPTERS[REFERENCE_MODEL]().fovea_settings(cfg["ours"])[0])
    applies = differ = na = 0
    ref_frames = {}
    for f in cfg["budget"]["fractions"]:
        frames = {m: common.budget_frames(cfg, data, m, ADAPTERS[m].stock_tokens, float(f), use_gate=True)[0] for m in (REFERENCE_MODEL, MODEL)}
        limits = {m: video_limits(cfg, data, m, ADAPTERS[m].stock_tokens, f) for m in (REFERENCE_MODEL, MODEL)}
        for v in data["videos"]:
            if limits[REFERENCE_MODEL][v] == limits[MODEL][v]:
                applies += 1
                differ += [fr["key"] for fr in frames[REFERENCE_MODEL][v]] != [fr["key"] for fr in frames[MODEL][v]]
            else:
                na += 1
        ref_frames[run.budget_name(f, cfg["run"]["policy"], label)] = {v: len(x) for v, x in frames[REFERENCE_MODEL].items()}
    records = read_jsonl(os.path.join(reference, "budget.jsonl"))
    wrong = sum(ref_frames.get(r["condition"], {}).get(r["video"]) != r.get("n_frames") for r in records)
    return applies, differ, na, wrong, len(records)


def tokens_cmd(a):
    from transformers import AutoConfig, AutoProcessor

    from accuracy_tables import common
    from accuracy_tables.hpc.all350 import measure_tokens
    from accuracy_tables.models import ADAPTERS
    from vlmeval import smolvlm256_fovea as s2
    cfg = common.load_config(a.config)
    mc = AutoConfig.from_pretrained(s2.MODEL_ID, revision=s2.REVISION)
    proc = AutoProcessor.from_pretrained(s2.MODEL_ID, revision=s2.REVISION)
    found = s2.geometry_problems(mc, proc.image_processor, proc)
    if found:
        sys.exit("SmolVLM2-256M's config isn't the module's geometry: " + "; ".join(found) + ". Stop and tell the user.")
    v = mc.vision_config
    stock = (int(v.image_size) // int(v.patch_size) // int(mc.scale_factor)) ** 2
    if stock != ADAPTERS[MODEL].stock_tokens:
        sys.exit(f"the config gives {stock} stock tokens per frame, the adapter {ADAPTERS[MODEL].stock_tokens}. Stop and tell the user.")
    print(f"SmolVLM2-256M at {s2.REVISION}: {v.image_size} px, {v.patch_size} px patches, pixel shuffle {mc.scale_factor}: "
          f"{stock} stock tokens per frame (SmolVLM2-2.2B: 81); text context {mc.text_config.max_position_embeddings} positions", flush=True)

    data = common.load_data(cfg)
    k = ADAPTERS[MODEL]().fovea_settings(cfg["ours"])[0]["k"]
    counts, frames, no_gaze = collections.Counter(), 0, 0
    for i, vid in enumerate(data["videos"]):
        for f in range(data["metas"][vid]["n"]):
            fr = data["frame"](vid, f)
            g = fr.get("gaze")
            no_gaze += g is None
            counts[len(s2.foveated_cells(*(g if g is not None else (0.5, 0.5)), k))] += 1
            frames += 1
        if i % 50 == 0:
            print(f"   {i} of {len(data['videos'])} videos", flush=True)
    if len(counts) != 1:
        sys.exit(f"FOVEATED TOKENS DIFFER between frames: {dict(counts)}. Stop and tell the user.")
    tpf = next(iter(counts))
    print(f"foveated tokens per frame at k = {k}: {tpf} on every one of {frames} frames ({no_gaze} with no gaze, fovea at the centre)",
          flush=True)
    if tpf != s2.tokens_at(k) or tpf != int(cfg["budget"]["tokens_per_frame"][MODEL]):
        sys.exit(f"{tpf} measured, {s2.tokens_at(k)} by the layout, {cfg['budget']['tokens_per_frame'][MODEL]} in {a.config}. "
                 "Stop and tell the user.")
    table = measure_tokens.limits_table({REFERENCE_MODEL: float(cfg["budget"]["tokens_per_frame"][REFERENCE_MODEL]), MODEL: float(tpf)}, cfg)
    print("\nlimits for a video with at least K frames:", flush=True)
    print(table.round(2).to_string(index=False), flush=True)

    applies, differ, na, wrong, n_records = same_frames(cfg, data, a.reference)
    if not n_records or wrong:
        sys.exit(f"the 2.2B's budget frames recomputed here don't give the frame counts of {wrong} of {n_records} records in "
                 f"{a.reference}. Stop and tell the user.")
    print(f"\nthe 2.2B's budget frames recomputed here give the frame count of all {n_records} budget records in {a.reference}", flush=True)
    if not applies:
        print(f"SAME FRAMES AS 2.2B: the check doesn't apply, no video has the same limit on both models at any budget ({na} pairs)",
              flush=True)
    elif differ:
        sys.exit(f"FRAMES DIFFER FROM 2.2B on {differ} of {applies} video-budget pairs with the same limit. Stop and tell the user.")
    else:
        print(f"SAME FRAMES AS 2.2B: the same kept frames on all {applies} video-budget pairs where both models have the same limit; "
              f"the check doesn't apply to the other {na}, whose limits differ", flush=True)
    mine = table[table.model == MODEL]
    limits = "/".join(str(int(x)) for x in mine.limit)
    budgets = "/".join(f"{x:g}" for x in mine.budget_pct)
    print(f"\nPROPOSED, needs the user's OK: smolvlm256 tokens per frame: stock {stock}, foveated {tpf}; frame limits {limits} at "
          f"{budgets}% (SmolVLM2-2.2B: 81, 17, {'/'.join(str(int(x)) for x in table[table.model == REFERENCE_MODEL].limit)}). "
          "STOP: nothing is calibrated until the user approves these.", flush=True)


# ---------------------------------------------------------------------------- same answers
def differences(new, ref, need_all=False):
    """(records compared, records that differ, max probability difference, records only ref has)."""
    ref_of = {(r["condition"], r["qid"]): r for r in ref}
    keys = {(r["condition"], r["qid"]) for r in new}
    differ, prob = 0, 0.0
    for r in new:
        b = ref_of.get((r["condition"], r["qid"]))
        if b is None or r.get("error") is not None:
            differ += 1
            continue
        if r.get("p") and b.get("p"):
            prob = max(prob, max(abs(x - y) for x, y in zip(r["p"], b["p"])))
        differ += any(r.get(k) != b.get(k) for k in KEYS)
    ref_only = sum(k not in keys for k in ref_of) if need_all else 0
    return len(new), differ, prob, ref_only


def same_cmd(a):
    total = bad = 0
    worst = 0.0
    for row in a.rows.split(","):
        n, differ, prob, ref_only = differences(read_jsonl(os.path.join(a.new, f"{row}.jsonl")),
                                                read_jsonl(os.path.join(a.ref, f"{row}.jsonl")), a.all)
        print(f"   {row}: {n} records, {differ} differ, max_prob_diff {prob}" + (f", {ref_only} only in --ref" if a.all else ""), flush=True)
        total += n
        bad += differ + ref_only + (n == 0)
        worst = max(worst, prob)
    if bad or worst > 0:
        sys.exit(f"ANSWERS DIFFER: {bad} of {total} records, max_prob_diff {worst}. Stop and tell the user.")
    print(f"SAME ANSWERS: 0 of {total} records differ, max_prob_diff 0.0", flush=True)


def identity_cmd(a):
    from accuracy_tables.hpc.all350 import check_identity as ci
    uniform = ci.read_records(os.path.join(a.folder, "uniform.jsonl"))
    bad = 0
    for row in ci.ROWS:
        r = ci.compare(uniform, ci.read_records(os.path.join(a.folder, f"{row}.jsonl")))
        print(f"   {row}: {r}", flush=True)
        bad += ci.problems(r) + (r["max_prob_diff"] > 0)
    if bad:
        sys.exit("NOT THE SAME AS UNIFORM K: stop and tell the user")
    print("SAME AS UNIFORM K at max_prob_diff 0.0", flush=True)


# ---------------------------------------------------------------------------- the run's time
STATUS = re.compile(r"ok \((\d+) records.*?, (\d+) s\)")


def estimate(timing, totals=None):
    """-> dict(seconds, questions, videos, by_questions_h, by_videos_h) from a timing run's run_info_flops.json and records."""
    with open(os.path.join(timing, "run_info_flops.json")) as f:
        status = json.load(f)["status"]
    seconds = 0
    for s in status.values():
        m = STATUS.search(s)
        if not m:
            raise SystemExit(f"a timing row didn't finish: {s}. Stop and tell the user.")
        seconds += int(m.group(2))
    uniform = read_jsonl(os.path.join(timing, "uniform.jsonl"))
    q, v = len({r["qid"] for r in uniform}), len({r["video"] for r in uniform})
    total_q, total_v = totals or prepare.TOTAL                # (questions, videos) of the full run; EgoEverything's by default
    return dict(seconds=seconds, questions=q, videos=v, by_questions_h=seconds * total_q / q / 3600, by_videos_h=seconds * total_v / v / 3600)


def estimate_cmd(a):
    e = estimate(a.timing, (a.questions, a.videos) if a.questions else None)
    lo, hi = sorted((e["by_questions_h"], e["by_videos_h"]))
    print(f"timing run: {e['seconds']} s of scoring for {e['questions']} questions on {e['videos']} videos", flush=True)
    print(f"PROJECTED full run: about {lo:.1f} to {hi:.1f} hours of scoring (scaled by questions and by videos), plus about 10 minutes "
          "loading the data", flush=True)


# ---------------------------------------------------------------------------- one job's records
def records_cmd(a):
    from accuracy_tables.hpc.all350 import flops_shards
    with open(a.plan) as f:
        plan = json.load(f)
    rows = a.rows.split(",")
    total_q = a.questions or prepare.TOTAL[0]
    want = {}
    for ln in plan["lines"]:
        want.setdefault(ln["row"], set()).add(ln["condition"])
    errors, have = collections.Counter(), {}
    for row in rows:
        recs = read_jsonl(os.path.join(a.folder, f"{row}.jsonl"))
        have[row] = len(recs)
        for r in recs:
            if r.get("error") is not None:
                errors[(row, r["error"])] += 1
    for (row, err), n in sorted(errors.items()):
        print(f"   records with an error: {row}: {n} x {err}", flush=True)
    print(f"oom / nan records: {sum(errors.values())}", flush=True)
    gpus, found = flops_shards.folder_gpus(a.folder, rows, "run")
    print(f"GPU per record: {dict(gpus)}", flush=True)
    found += [f"records from more than one GPU: {dict(gpus)}"] if len(gpus) > 1 else []
    expected = sum(len(want.get(row, ())) for row in rows) * total_q
    n = sum(have.values())
    found += [f"{row}: {have[row]} records, {len(want.get(row, ())) * total_q} planned" for row in rows
              if have[row] != len(want.get(row, ())) * total_q]
    if errors:
        found.append(f"{sum(errors.values())} records with an error")
    for line in found[:40]:
        print("  ", line, flush=True)
    print(f"RECORDS: {n} of {expected} planned answers in {a.folder} (one job, so nothing to merge)", flush=True)
    if found:
        sys.exit(f"INCOMPLETE: {len(found)} problems. Rerun the full job, or tell the user.")


# ---------------------------------------------------------------------------- the side table
SIDE = ("accuracy", "ci_low", "ci_high", "flops_pct")


def vs_table(ours, other):
    """Per group, row and budget: accuracy, interval and FLOPs share of each model's own run, side by side."""
    keys = ["group", "row", "budget_pct"]
    a = ours[keys + list(SIDE)].rename(columns={c: f"{c}_256m" for c in SIDE})
    b = other[keys + list(SIDE)].rename(columns={c: f"{c}_2p2b" for c in SIDE})
    t = b.merge(a, on=keys, how="outer")
    t["note"] = "two models, each matched on its own FLOPs; no paired test, no winner"
    return t


def vs2p2b_cmd(a):
    import pandas as pd
    t = vs_table(pd.read_csv(a.ours), pd.read_csv(a.other))
    t.to_csv(a.out, index=False)
    pd.set_option("display.width", 220)
    print(t.drop(columns="note").round(3).to_string(index=False), flush=True)
    print(f"written {a.out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("pick-videos")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--questions", type=int)
    g.add_argument("--every", type=int)
    s.set_defaults(run=lambda a: print(",".join(pick_by_questions(prepare.question_counts(), a.questions) if a.questions
                                                else pick_every(prepare.question_counts(), a.every))))
    s = sub.add_parser("tokens")
    s.add_argument("--config", required=True)
    s.add_argument("--reference", required=True, help="the 2.2B's FLOPs-matched results folder, only read")
    s.set_defaults(run=tokens_cmd)
    s = sub.add_parser("same")
    for k in ("--new", "--ref", "--rows"):
        s.add_argument(k, required=True)
    s.add_argument("--all", action="store_true", help="--ref may not have records --new lacks")
    s.set_defaults(run=same_cmd)
    s = sub.add_parser("identity")
    s.add_argument("--folder", required=True)
    s.set_defaults(run=identity_cmd)
    s = sub.add_parser("estimate")
    s.add_argument("--timing", required=True)
    s.add_argument("--questions", type=int, help="the full run's questions (default: EgoEverything's 5,985), with --videos")
    s.add_argument("--videos", type=int, help="the full run's videos or clips (default: 350)")
    s.set_defaults(run=estimate_cmd)
    s = sub.add_parser("records")
    for k in ("--folder", "--plan", "--rows"):
        s.add_argument(k, required=True)
    s.add_argument("--questions", type=int, help="questions per planned condition (default: EgoEverything's 5,985)")
    s.set_defaults(run=records_cmd)
    s = sub.add_parser("vs2p2b")
    for k in ("--ours", "--other", "--out"):
        s.add_argument(k, required=True)
    s.set_defaults(run=vs2p2b_cmd)
    a = ap.parse_args(argv)
    a.run(a)


if __name__ == "__main__":
    main()

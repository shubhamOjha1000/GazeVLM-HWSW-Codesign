"""High tokens per frame at 10% FLOPs, on 4 models and the 940 dev questions (2026-11-14). Diagnostic dev work only: nothing here feeds a
results table, no test-set question is used, and v1, sections 17 and 18 and v2 Stage 1 stay as they are. Everything is written to
$GV_ROOT/data/hdepic/sweep10/; every results folder is only read.

    python -u -m accuracy_tables.hpc.hdepic.sweep10 blind-source      S0  did v2 Stage 1's Blind run on the 940 dev questions?
    python -u -m accuracy_tables.hpc.hdepic.sweep10 plan              S0  every dev question's section 18 candidates (K = min(64, distinct))
    python -u -m accuracy_tables.hpc.hdepic.sweep10 gate              S0  the gate's rows, its 5 models' scores, its statistics, the VOR check
    python -u -m accuracy_tables.hpc.hdepic.sweep10 frames --index i  S0  one planned video's frames (an array task)
    python -u -m accuracy_tables.hpc.hdepic.sweep10 frames-check      S0  every planned frame there; meta.json per question
    python -u -m accuracy_tables.hpc.hdepic.sweep10 data              S0  the manifest, then the data check through common.load_data
    python -u -m accuracy_tables.hpc.hdepic.sweep10 sizes                 every representation's px and tokens, per model
    python -u -m accuracy_tables.hpc.hdepic.sweep10 summary           S4  the tables, the best rho per method, the gate and gaze effects

The protocol, the same for every model and method:
  questions   exactly the 940 of data/hdepic/v2/dev_ids.json (fine_grained action_recognition outside section 18's sample, seed 1)
  candidates  section 18's rule for all 4 models: K = min(64, distinct MP4 frames in the span), its stamps and prompt unchanged (Qwen ran
              K = 96 on EgoEverything; here it is 64, the same candidate frames for every model)
  budget      10% of the model's own Uniform-64 FLOPs per question (gate + vision encoder + LLM prefill, section 14's units); each
              condition takes the frame count K' (at least 1) closest to it; nearest_only past 0.5 points
  rho         each representation keeps a share rho of the model's stock tokens per frame, rho in 0.50, 0.75, 1.00: of the settings a
              method can take, the one whose tokens per frame come closest to rho x stock among those keeping at least 0.50 x stock
              (one rule for every method; a grid that can't reach rho exactly lands at its nearest size above the floor)
  best rho    per method, its highest dev accuracy; a tie goes to the measured FLOPs share nearest 10%, then to the larger rho. Fixed
              before any accuracy is seen, the same for every method.
Conditions (sweep10_gpu.py scores them): uniform (K = 64, stock), blind, uniform_flops (stock, K'), the 8 baselines at rho 0.50 and 0.75
(each its own knob; at 1.00 they are uniform_flops), ours_res_rho (the gate's frames, the whole frame resized before encoding),
ours_twoview (the whole frame at stock size plus a tile half the source's side around the gaze, 2x stock detail), ours_twoview_centre
(the tile at the centre), and the no-gate control (slot middles) at every representation.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

from accuracy_tables.hpc.hdepic import data, groups, s18

BUDGET = 0.10
RHOS = (0.50, 0.75, 1.00)
FLOOR = 0.50
SOURCE = 1344                          # the stored frames' side
TILE_SHARE = 0.5                       # the twoview tile: half the source's side, so the stock side shows it at 2x the stock detail
N_DEV = 940
K_MAX = s18.K
CONFIG = "accuracy_tables/hpc/config.sweep10.yaml"
BASELINE_ROWS = ("gazevlm", "streamingtom", "dycoke", "fastvid", "prunevid", "dtd", "avgpool", "gazevlm2")      # run.BASELINE_ROWS
TWO_VIEW_BASELINES = ("gazevlm2",)
# tokens are base px cells (patch x merge); stock_side is the model's own input for a whole frame; grid tokens per side at stock
MODELS = {
    "smolvlm": dict(name="SmolVLM2-2.2B", gpu="B200", base=42, grid=9, stock_side=384, stock_tokens=81, snapshot="482adb5"),
    "smolvlm256": dict(name="SmolVLM2-256M", gpu="B200", base=64, grid=8, stock_side=512, stock_tokens=64, snapshot="9cda5c1"),
    "qwen2.5": dict(name="Qwen2.5-VL-3B", gpu="A100", base=28, grid=16, stock_side=448, stock_tokens=256, snapshot="66285546"),  # section 15's
    "qwen3": dict(name="Qwen3-VL-4B", gpu="A100", base=32, grid=16, stock_side=512, stock_tokens=256, snapshot=None),
}
BOOTSTRAP, BOOTSTRAP_SEED = 2000, 0


# ---------------------------------------------------------------------------- where things live
def root(*parts):
    d = os.path.join(os.environ["GV_ROOT"], "data", "hdepic", "sweep10", *parts)
    data.refuse_inside_raw(d)
    return d


def pseudo(i):
    return f"sw10_{int(i):05d}"


def plan_json():
    return root("plan.json")


def rows_csv(name):
    return root("rows", f"{name}.csv")


def folds_dir():
    return root("folds")


def preds_csv():
    return root("preds.csv")


def frames_dir(name):
    return root("frames", name)


def questions_json():
    return root("questions.json")


def model_dir(model, *parts):
    return root(model, *parts)


def v2_path(*parts):
    return os.path.join(os.environ["GV_ROOT"], "data", "hdepic", "v2", *parts)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)


# ---------------------------------------------------------------------------- the rho rule and the representations
def pick(cands, rho, scale, floor=FLOOR, name="this method"):
    """cands [(amount, setting)] -> the one whose amount comes closest to rho x scale among those at or above floor x scale, the larger
    amount on a tie. Amounts are tokens per frame (scale = the stock tokens) or token shares (scale = 1). name: the row, for the error."""
    ok = [c for c in cands if c[0] >= floor * scale - 1e-9]
    if not ok:
        top = max(cands, key=lambda c: c[0]) if cands else (float("nan"), None)
        raise ValueError(f"{name}: no setting keeps {floor} x {scale} (rho {rho:g}); the most any of its {len(cands)} settings keeps is "
                         f"{top[0]:.4g}, at {top[1]}")
    return min(ok, key=lambda c: (abs(c[0] - rho * scale), -c[0]))


def res_grid(model, rho):
    """(tokens per side m, side px, tokens) of the whole frame resized before encoding at rho: the m x m grid by the rho rule, its side m
    x base px (the model's own stock input at the full grid)."""
    g = MODELS[model]
    m = pick([(k * k, k) for k in range(1, g["grid"] + 1)], rho, g["stock_tokens"])[1]
    return m, (g["stock_side"] if m == g["grid"] else m * g["base"]), m * m


def avgpool_side(model, rho):
    """Average pooling's s x s grid at rho, by the same rule (the stock grid pooled; the same token counts as res_grid)."""
    return res_grid(model, rho)[0]


def gazevlm2_tokens(model, rho):
    """The two-view GazeVLM's B tokens per frame at rho (B even: B // 2 a view), by the rho rule."""
    stock = MODELS[model]["stock_tokens"]
    return pick([(b, b) for b in range(2, 2 * stock + 1, 2)], rho, stock)[1]


def twoview(model):
    """The two views of a frame: the whole frame at the stock side and a tile of TILE_SHARE of the source's side around the gaze, at the
    stock side too: dict(side, tile_src, tokens per frame, detail against stock)."""
    g = MODELS[model]
    src = int(round(SOURCE * TILE_SHARE))
    return dict(side=g["stock_side"], tile_src=src, tokens=2 * g["stock_tokens"], detail=SOURCE / src)


def representations(model):
    """Every representation of the sweep on this model: [dict(name, rho, side, tokens, rho_real, views)]."""
    g = MODELS[model]
    out = []
    for rho in RHOS:
        m, side, t = res_grid(model, rho)
        out.append(dict(name=f"res_{rho:.2f}", rho=rho, side=side, grid=f"{m}x{m}", tokens=t, rho_real=t / g["stock_tokens"], views=1))
    tv = twoview(model)
    out.append(dict(name="twoview", rho=None, side=tv["side"], grid=f"2 x {g['grid']}x{g['grid']}", tokens=tv["tokens"],
                    rho_real=tv["tokens"] / g["stock_tokens"], views=2, tile_src=tv["tile_src"]))
    for rho in RHOS[:2]:
        s = avgpool_side(model, rho)
        out.append(dict(name=f"avgpool_{rho:.2f}", rho=rho, side=g["stock_side"], grid=f"{s}x{s}", tokens=s * s,
                        rho_real=s * s / g["stock_tokens"], views=1))
        b = gazevlm2_tokens(model, rho)
        out.append(dict(name=f"gazevlm2_{rho:.2f}", rho=rho, side=g["stock_side"], grid=f"B={b}", tokens=b, rho_real=b / g["stock_tokens"],
                        views=2))
    return out


def sizes_lines(model):
    g = MODELS[model]
    lines = [f"SIZES {model} ({g['name']}, {g['gpu']}): stock {g['stock_side']} px = {g['stock_tokens']} tokens a frame "
             f"({g['grid']} x {g['grid']} tokens of {g['base']} px)"]
    for r in representations(model):
        extra = f", tile {r['tile_src']} of {SOURCE} source px at {r['side']} px" if r.get("tile_src") else ""
        lines.append(f"SIZES {model} {r['name']}: {r['side']} px, {r['grid']}, {r['tokens']} tokens a frame, rho {r['rho_real']:.3f}, "
                     f"{r['views']} image(s) encoded a frame{extra}")
    return lines


# ---------------------------------------------------------------------------- the conditions
def conditions():
    """[(name, method, rho, kind)], every model the same, in run order. kind: uniform, blind, uniform_flops, baseline, ours, nogate."""
    out = [("uniform", "uniform", 1.0, "uniform"), ("blind", "blind", None, "blind"), ("uniform_flops", "uniform_flops", 1.0, "uniform_flops")]
    out += [(f"ours_res_{r:.2f}", "ours_res", r, "ours") for r in RHOS]
    out += [("ours_twoview", "ours_twoview", None, "ours"), ("ours_twoview_centre", "ours_twoview_centre", None, "ours")]
    out += [(f"nogate_res_{r:.2f}", "nogate_res", r, "nogate") for r in RHOS] + [("nogate_twoview", "nogate_twoview", None, "nogate")]
    out += [(f"{row}_{r:.2f}", row, r, "baseline") for row in BASELINE_ROWS for r in RHOS[:2]]
    return out


def method_settings():
    """{method: {rho: condition}} for the best-rho choice. A baseline at 1.00 is uniform_flops (it keeps every token); the two views,
    about 2 x stock tokens, are their methods' one setting."""
    out = {"uniform_flops": {1.0: "uniform_flops"}}
    for row in BASELINE_ROWS:
        out[row] = {0.5: f"{row}_0.50", 0.75: f"{row}_0.75", 1.0: "uniform_flops"}
    out["ours_res"] = {r: f"ours_res_{r:.2f}" for r in RHOS}
    out["ours_twoview"] = {2.0: "ours_twoview"}
    out["ours_twoview_centre"] = {2.0: "ours_twoview_centre"}
    out["nogate_res"] = {r: f"nogate_res_{r:.2f}" for r in RHOS}
    out["nogate_twoview"] = {2.0: "nogate_twoview"}
    return out


def kept_positions(data, limit, use_gate=True):
    """{clip: frame positions} at a frame limit: the budget rule's slots and choice on the gate's labels (vlmeval.budget_rule.keep_frames,
    each segment its own frame 0) or the no-gate control's slot middles, then any forced frame, exactly as common.budget_frames picks
    them at a token budget. Here the limit is the K' the FLOPs budget allows (at most the clip's frames)."""
    from accuracy_tables import common
    from vlmeval import budget_rule as br
    out = {}
    for c in data["clips"]:
        v, fn = c["sequence"], c["frame_no"]
        n = int((fn < data["metas"][v]["n"]).sum())
        L = max(1, min(int(limit), n))
        if use_gate:
            kept, _ = br.keep_frames(data["labels"][v][:n - 1], L, starts=data.get("segment_starts", {}).get(v, ()))
        else:
            kept = br.slot_middles(n, L)
        out[v] = [int(fn[t]) for t in common.include_forced(kept, data.get("forced", {}).get(v, ()))]
    return out


def gate_frames(data, limit, use_gate=True):
    """kept_positions as frame dicts (data['frame'])."""
    return {v: [data["frame"](v, f) for f in pos] for v, pos in kept_positions(data, limit, use_gate).items()}


KEYS = ("pred", "p", "n_frames", "n_image_tokens", "n_tokens", "error")


def same_by_qid(new, ref, keys=KEYS):
    """(records, records that differ from ref's on the same question in any of keys, max probability difference); a question ref
    lacks counts as different."""
    ref_of = {int(r["qid"]): r for r in ref}
    differ, prob = 0, 0.0
    for r in new:
        b = ref_of.get(int(r["qid"]))
        if b is None or r.get("error") is not None:
            differ += 1
            continue
        if r.get("p") and b.get("p"):
            prob = max(prob, max(abs(x - y) for x, y in zip(r["p"], b["p"])))
        differ += any(r.get(k) != b.get(k) for k in keys)
    return len(new), differ, prob


GATE_PAIRS = [(f"ours_res_{r:.2f}", f"nogate_res_{r:.2f}") for r in RHOS] + [("ours_twoview", "nogate_twoview")]
GAZE_PAIRS = [("ours_twoview", "ours_twoview_centre")]


# ---------------------------------------------------------------------------- FLOPs arithmetic (accuracy_tables/flops.py's units)
def question_flops(units, frames, image_tokens, v_frame, blind, prompt, decisions=0):
    """FLOPs per question (arrays): the gate (5 fold models x its decisions) + frames x one frame's vision FLOPs + the LLM's prefill on
    the prompt the frames and their image tokens make."""
    from accuracy_tables import flops
    frames = np.asarray(frames, dtype=np.float64)
    n = flops.prompt_tokens(blind, frames, image_tokens, prompt)
    gate = flops.GATE_MODELS * units["gate_per_decision"] * np.asarray(decisions, dtype=np.float64)
    return gate + frames * float(v_frame) + flops.prefill(units["prefill"], n)


def best_k(share_of, k_max=K_MAX, target=BUDGET):
    """flops.choose_k: the frame count in 1..k_max closest to the target share, the smaller on a tie -> (K, share)."""
    from accuracy_tables import flops
    return flops.choose_k(share_of, target, k_max)


def nearest_only(pct, target_pct=100 * BUDGET):
    from accuracy_tables import flops
    return bool(abs(pct - target_pct) > flops.NEAREST_ONLY_PTS)


# ---------------------------------------------------------------------------- S0: did v2's Blind run on the dev questions?
def dev_answers(plan, folder=None):
    """{dev qid: answer letter} from HD-EPIC's own question files (only read)."""
    folder = folder or data.questions_dir()
    files, _ = groups.question_files(folder)
    raw, out = {}, {}
    for q in plan["questions"]:
        key = (q["group"], q["type"])
        if key not in raw:
            with open(os.path.join(folder, files[q["group"]][q["type"]])) as f:
                raw[key] = json.load(f)
        r = raw[key][q["hdepic_qid"]] if isinstance(raw[key], dict) else raw[key][int(q["hdepic_qid"])]
        out[int(q["qid"])] = "ABCDE"[int(r["correct_idx"])]
    return out


def blind_source(v2_blind, dev_plan, answers, s18_blind, s18_questions, sample_ids, blind_name):
    """Facts about v2 Stage 1's blind records -> (facts, problems). The records must be one per dev question (its qid and clip), with that
    question's answer key, no dev question may be in section 18's sample or run, and their probability vectors must not be section 18's."""
    dev = {int(q["qid"]): q for q in dev_plan["questions"]}
    problems = []
    qids = [int(r["qid"]) for r in v2_blind]
    if len(v2_blind) != len(dev) or set(qids) != set(dev) or len(set(qids)) != len(qids):
        problems.append(f"{len(v2_blind)} records for {len(set(qids))} qids; the dev plan has {len(dev)}")
    if {r["condition"] for r in v2_blind} != {blind_name}:
        problems.append(f"conditions {sorted({r['condition'] for r in v2_blind})}")
    clips = sum(r["video"] != dev.get(int(r["qid"]), {}).get("pseudo") for r in v2_blind)
    keys = sum(r["answer"] != answers.get(int(r["qid"])) for r in v2_blind)
    if clips:
        problems.append(f"{clips} records name another clip than their dev question's")
    if keys:
        problems.append(f"{keys} records hold another answer key than their dev question's")
    ar = {int(q["qid"]): q for q in s18_questions if q["type"] == "action_recognition"}
    s18_ar = [r for r in s18_blind if int(r["qid"]) in ar]
    dev_ids = {q["hdepic_qid"] for q in dev.values()}
    in_sample = len(dev_ids & set(sample_ids))
    in_run = len(dev_ids & {q["hdepic_qid"] for q in s18_questions})
    if in_sample or in_run:
        problems.append(f"{in_sample} dev questions in section 18's sample, {in_run} in its run")
    s18_p = collections.Counter(tuple(r["p"]) for r in s18_ar if r.get("p"))
    same_p = sum(tuple(r.get("p") or ()) in s18_p for r in v2_blind)
    if v2_blind and same_p == len(v2_blind):
        problems.append("every probability vector is one of section 18's")
    facts = dict(records=len(v2_blind), correct=sum(int(r["correct"]) for r in v2_blind), s18_records=len(s18_ar),
                 s18_correct=sum(int(r["correct"]) for r in s18_ar), in_sample=in_sample, in_run=in_run, same_p=same_p,
                 s18_distinct_p=len(s18_p), clips_differ=clips, keys_differ=keys)
    return facts, problems


def blind_source_cmd(a):
    from accuracy_tables.hpc.hdepic import v2
    BLIND = "blind (question and options only)"                      # run.BLIND
    dev_plan = read_json(v2_path("dev_plan.json"))
    v2_blind = read_jsonl(v2_path("stage1", "blind.jsonl"))
    s18_blind = read_jsonl(os.path.join(s18.results_dir(), "smolvlm", "blind.jsonl"))
    sample_ids = {x for ids in read_json(groups.sample_json())["qids"].values() for x in ids}
    facts, problems = blind_source(v2_blind, dev_plan, dev_answers(dev_plan), s18_blind, read_json(s18.questions_json()), sample_ids,
                                   BLIND)
    print(f"v2 Stage 1's blind.jsonl ({v2.root('stage1', 'blind.jsonl')}): {facts['records']} records, one per dev question by qid and "
          f"clip ({facts['clips_differ']} name another clip), answer keys from HD-EPIC's own files ({facts['keys_differ']} differ)", flush=True)
    print(f"dev questions in section 18's 2,000-question sample: {facts['in_sample']}; in section 18's run: {facts['in_run']}", flush=True)
    print(f"probability vectors equal to one of section 18's {facts['s18_records']} action_recognition blind records "
          f"({facts['s18_distinct_p']} distinct): {facts['same_p']} of {facts['records']}", flush=True)
    print(f"correct: v2 Stage 1 {facts['correct']} of {facts['records']}; section 18's action_recognition {facts['s18_correct']} of "
          f"{facts['s18_records']}", flush=True)
    if problems:
        sys.exit("BLIND SOURCE: NOT SHOWN to be the dev questions: " + "; ".join(problems) + ". Stop and tell the user.")
    same = "the same count" if (facts["correct"], facts["records"]) == (facts["s18_correct"], facts["s18_records"]) else "a different count"
    print(f"BLIND SOURCE: v2 Stage 1's Blind ran on the 940 dev questions (v2.stage1_cmd calls run.run_blind on the dev items; its records "
          f"are the dev questions' own by qid, clip and answer key, none of those questions is in section 18's sample or run, and "
          f"{facts['same_p']} of {facts['records']} probability vectors equal any of section 18's). Section 18's action_recognition "
          f"figure is {same} of correct answers on other questions, not a copy. S1 also rescores 50 of them on the 2.2B.", flush=True)


# ---------------------------------------------------------------------------- S0: the plan
def dev_ids():
    body = read_json(v2_path("dev_ids.json"))
    ids = list(body["qids"])
    if len(ids) != N_DEV or len(set(ids)) != N_DEV or body.get("seed") != 1 or body.get("type") != "action_recognition":
        sys.exit(f"DEV IDS DIFFER: {len(ids)} ids ({len(set(ids))} distinct), seed {body.get('seed')}, type {body.get('type')}. Stop and "
                 "tell the user.")
    return ids


def time_cache(videos):
    import pandas as pd
    from accuracy_tables.hpc.hdepic import plan as plan17
    cache, info = {}, {}
    for i, v in enumerate(sorted(videos)):
        mp4_ns, vrs_ns = data.read_time_csv(os.path.join(data.video_dir(v), f"{v}_mp4_to_vrs_time_ns.csv"))
        cache[v] = dict(mp4_ns=mp4_ns, vrs_ns=vrs_ns)
        info[v] = plan17.video_info(v)
        cache[v]["gaze_us"] = pd.read_csv(info[v]["gaze_csv"], usecols=["tracking_timestamp_us"])["tracking_timestamp_us"].to_numpy(np.float64)
        info[v]["gaze_hz"] = data.rate_hz(cache[v]["gaze_us"])
        if i % 25 == 0:
            print(f"   {i} of {len(videos)} videos read", flush=True)
    return cache, info


def v2_problems(entries, dev_plan):
    """Each dev question's 9 v2 Stage 1 frames must be Uniform K' = 9 of its candidates here (the same questions, the same candidates)."""
    from accuracy_tables import common
    found = []
    v2q = {int(q["qid"]): q for q in dev_plan["questions"]}
    for e in entries:
        q = v2q.get(int(e["qid"]))
        sel = common.uniform_indices(e["K"], 9)
        if q is None or q["hdepic_qid"] != e["hdepic_qid"] or q["mp4_frames"] != [e["mp4_frames"][j] for j in sel]:
            found.append(f"dev qid {e['qid']} ({e['hdepic_qid']})")
    return found


def plan_cmd(a):
    from accuracy_tables.hpc.hdepic import plan18, v2
    ids = dev_ids()
    loaded, _ = groups.load_groups()
    fg = [q for qs in loaded[groups.SAMPLE_GROUP].values() for q in qs]
    sample, quota = groups.stratified_sample(fg)
    groups.write_or_check_sample(sample, quota, len(fg))
    in_sample = set(ids) & {x for xs in sample.values() for x in xs}
    pool = {q["qid"]: q for q in v2.dev_pool(loaded, sample)}
    outside = [x for x in ids if x not in pool]
    if in_sample or outside:
        sys.exit(f"DEV IDS DIFFER: {len(in_sample)} in section 18's sample, {len(outside)} outside v2's dev pool. Stop and tell the user.")
    cache, info = time_cache({s["video"] for x in ids for s in pool[x]["segments"]})
    entries, bad = [], []
    for i, x in enumerate(ids):
        e, why = plan18.plan_question(pool[x], i, cache)
        if e is None:
            bad.append(f"{x}: {why}")
            continue
        e["pseudo"] = pseudo(i)
        entries.append(e)
    if bad:
        sys.exit(f"PLAN STOPPED: {len(bad)} dev questions have no candidates, for example {bad[:3]}. Stop and tell the user.")
    found = v2_problems(entries, read_json(v2.root("dev_plan.json")))
    if found:
        sys.exit(f"CANDIDATES DIFFER FROM V2: {len(found)} dev questions, for example {found[:3]}. Stop and tell the user.")
    videos = sorted({v for e in entries for v in e["videos"]})
    plan = dict(rule="the 940 dev questions (v2 dev_ids.json), section 18's candidates (D1, D2, D5)", videos=videos,
                video_info={v: info[v] for v in videos}, questions=entries)
    write_json(plan_json(), plan)
    k = np.array([e["K"] for e in entries])
    print(f"{len(entries)} questions on {len(videos)} videos; K min {k.min()}, median {np.median(k):g}, max {k.max()}, under 64 on "
          f"{int((k < K_MAX).sum())}; several segments {sum(e['n_segments'] > 1 for e in entries)}; query-frame exception "
          f"{sum(e['exception'] for e in entries)}; candidates outside their video's gaze {sum(e['frames_outside_gaze'] for e in entries)} "
          f"of {int(k.sum())}", flush=True)
    print(f"each question's 9 v2 Stage 1 frames are Uniform 9 of its candidates here: {len(entries)} of {len(entries)}", flush=True)
    print(f"SWEEP PLAN WRITTEN: {plan_json()}", flush=True)


# ---------------------------------------------------------------------------- S0: the gate (section 18's step 2 code, on the dev plan)
def gate_stats(plan, folds, preds):
    """Section 18's gate statistics (gate18 stats) on this plan -> the table; ours against AEA and Nymeria."""
    import pandas as pd
    from accuracy_tables.hpc.newvideos import check_gate, paths
    from eyehead.infer import load_ckpt
    span = {}
    for q in plan["questions"]:
        bounds = [0] + list(q["starts"]) + [q["K"]]
        span[q["pseudo"]] = sum(q["mp4_s"][s1 - 1] - q["mp4_s"][s0] for s0, s1 in zip(bounds, bounds[1:]))
    ck = load_ckpt(os.path.join(paths.ckpt_dir(), "fold1.pt"), "cpu")
    cols = ["sequence", "feat_frame_2", "n_gaze_in_gap", "n_oof_in_gap", "n_imu_in_gap", check_gate.EYE_COL, check_gate.IMU_COL]
    df = pd.read_csv(os.path.join(folds, "test.csv"), usecols=cols).merge(
        pd.read_csv(preds, usecols=["feat_frame_2", "pred"]), on="feat_frame_2", how="left", validate="one_to_one")
    if df["pred"].isna().any():
        sys.exit(f"{int(df['pred'].isna().sum())} rows have no prediction. Rerun the gate step.")
    df["frame"] = df["feat_frame_2"].map(check_gate.pol.frame_no)
    df = df.sort_values(["sequence", "frame"]).reset_index(drop=True)

    def summary(part):
        per = pd.DataFrame([check_gate.video_summary(v, x["pred"].to_numpy().astype(int), x["n_gaze_in_gap"].to_numpy(),
                                                     x["n_oof_in_gap"].to_numpy(), x["n_imu_in_gap"].to_numpy())
                            for v, x in part.groupby("sequence", sort=False)])
        return per, dict(check_gate.group_summary(per, part["pred"].to_numpy().astype(int)), **check_gate.input_stats(part, ck))

    per, s = summary(df)
    seconds = float(sum(span[v] for v in per.video))
    s["marks_per_min"] = 60 * per["marks"].sum() / seconds if seconds > 0 else float("nan")
    s["seconds_p50"] = float(np.median([span[v] for v in per.video]))
    table = {"sweep10 dev": s}
    for ref in ("aea12", "nymeria"):
        try:
            table[ref] = summary(check_gate.load_group(ref))[1]
        except Exception as ex:
            print(f"   {ref}: not compared ({type(ex).__name__}: {str(ex)[:120]})", flush=True)
    return pd.DataFrame(table)


def gate_cmd(a):
    import pandas as pd
    from accuracy_tables.hpc.hdepic import gate, gate18
    plan = read_json(plan_json())
    streams = gate18.Streams()
    for i, q in enumerate(sorted(plan["questions"], key=lambda q: (q["videos"][0], q["qid"]))):
        out = rows_csv(q["pseudo"])
        if os.path.exists(out):
            continue
        rows = pd.DataFrame(gate18.question_rows(q, streams))
        if len(rows) != q["K"] - 1:
            sys.exit(f"{q['pseudo']}: {len(rows)} rows for {q['K']} candidates. Stop and tell the user.")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        rows.to_csv(out + ".tmp", index=False)
        os.replace(out + ".tmp", out)
        if i % 200 == 0:
            print(f"   rows: {i} of {len(plan['questions'])} questions", flush=True)
    print(f"ROWS WRITTEN: {len(plan['questions'])} questions", flush=True)
    out = gate18.score_rows(plan["questions"], folds_dir(), preds_csv(), lambda q: pd.read_csv(rows_csv(q["pseudo"])))
    print(f"SCORED: {len(out)} rows of {out.sequence.nunique()} questions -> {preds_csv()}", flush=True)
    t = gate_stats(plan, folds_dir(), preds_csv())
    pd.set_option("display.width", 240)
    print(t.round(3).to_string(), flush=True)
    t.to_csv(root("gate_stats.csv"))
    vor = t.loc["vor_median_cos", "sweep10 dev"]
    if not np.isfinite(vor) or vor > gate.VOR_CLEARLY_NEGATIVE:
        sys.exit(f"VOR CHECK FAILED: vor_median_cos {vor:.3f}, not clearly negative (AEA and Nymeria about -0.88). Stop and tell the user.")
    print(f"VOR CHECK OK: vor_median_cos {vor:.3f}", flush=True)


# ---------------------------------------------------------------------------- S0: the frames
def frame_index():
    """{(video, MP4 frame): (path, decoder s)} of every frame sections 17 and 18 and v2 Stage 1 already extracted."""
    from accuracy_tables.hpc.hdepic import v2
    out = v2.frame_index()
    path = v2_path("dev_plan.json")
    if not os.path.exists(path):
        return out
    for q in read_json(path)["questions"]:
        meta = os.path.join(v2.dev_frames_dir(q["pseudo"]), "meta.json")
        if os.path.exists(meta):
            m = read_json(meta)
            for j, (vi, fr, dec) in enumerate(zip(m["video_of"], m["mp4_frames"], m["decoder_s"])):
                out.setdefault((m["videos"][vi], int(fr)), (os.path.join(m["dir"], f"frame_{j:05d}.jpg"), float(dec)))
    return out


def frames_check(plan):
    """-> (problems, frame files, distinct files, worst decoder offset); writes meta.json (load18's) for every complete question."""
    from accuracy_tables.hpc.hdepic.frames import DECODER_TOLERANCE_S, SIZE
    problems, files, inodes, worst = [], 0, set(), 0.0
    for q in plan["questions"]:
        d = frames_dir(q["pseudo"])
        dec = {}
        for v in q["videos"]:
            p = os.path.join(d, f"part_{v}.json")
            if os.path.exists(p):
                dec.update({int(k): float(x) for k, x in read_json(p).items()})
        paths_ = [os.path.join(d, f"frame_{j:05d}.jpg") for j in range(q["K"])]
        have = [p for p in paths_ if os.path.exists(p)]
        if len(have) != q["K"] or sorted(dec) != list(range(q["K"])):
            problems.append(f"{q['pseudo']}: {len(have)} of {q['K']} frames, decoder times for {len(dec)}")
            continue
        decoder = [dec[j] for j in range(q["K"])]
        off = float(np.abs(np.asarray(decoder) - np.asarray(q["mp4_s"])).max())
        worst = max(worst, off)
        if off > DECODER_TOLERANCE_S:
            problems.append(f"{q['pseudo']}: the decoder's frame times are up to {off:.3f} s from the time CSV's")
            continue
        write_json(os.path.join(d, "meta.json"), dict(n=q["K"], dir=d, ts=q["mp4_s"], device_us=q["device_us"], mp4_frames=q["mp4_frames"],
                                                      video_of=q["video_of"], videos=q["videos"], video_n=q["video_n"], starts=q["starts"],
                                                      forced=q["forced"], decoder_s=decoder, size=SIZE))
        files += len(have)
        inodes |= {(os.stat(p).st_dev, os.stat(p).st_ino) for p in have}
    return problems, files, len(inodes), worst


def frames_cmd(a):
    from accuracy_tables.hpc.hdepic import v2
    plan = read_json(plan_json())
    i = a.index if a.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= i < len(plan["videos"]):
        sys.exit(f"index {i}: the plan has {len(plan['videos'])} videos (0 to {len(plan['videos']) - 1})")
    v2.extract_video(plan, plan["videos"][i], frame_index(), folder=frames_dir)


def frames_check_cmd(a):
    plan = read_json(plan_json())
    problems, files, unique, worst = frames_check(plan)
    print(f"the decoder's frame times are at most {worst:.4f} s from the time CSV's", flush=True)
    for p in problems[:40]:
        print("  ", p, flush=True)
    if problems:
        sys.exit(f"DEV FRAMES DIFFER: {len(problems)} problems. Rerun the frames array for those videos, or tell the user.")
    print(f"DEV FRAMES MATCH: {files} frame files for {len(plan['questions'])} questions, {unique} distinct on disk", flush=True)


# ---------------------------------------------------------------------------- S0: the manifest and the data check
def data_cmd(a):
    from accuracy_tables import common
    from accuracy_tables.hpc.hdepic import manifest18
    plan = read_json(plan_json())
    qs = manifest18.entries(plan)
    write_json(questions_json(), qs)
    cfg = common.load_config(a.config)
    d = common.load_data(cfg)
    print(common.describe(d), flush=True)
    found, missing = manifest18.data_problems(d, plan, cfg)
    ids = dev_ids()
    got = d["qs"].sort_values("qid")
    if list(got.hdepic_qid) != ids or list(got.qid) != list(range(N_DEV)):
        found.append("the loaded questions aren't the dev ids in their order")
    if len(set(d["videos"])) != len(d["qs"]):
        found.append(f"{len(set(d['videos']))} clips for {len(d['qs'])} questions")
    for p in found[:40]:
        print("  ", p, flush=True)
    if found:
        sys.exit(f"DATA CHECK FAILED: {len(found)} problems. Stop and tell the user.")
    k = np.array([d["metas"][v]["n"] for v in d["videos"]])
    print(f"frames without gaze (their fovea or tile at the centre, section 9's rule): {sum(missing.values())} of {int(k.sum())}", flush=True)
    print(f"data check OK: {len(d['qs'])} questions, one clip each, the dev ids in order; every clip's {int(k.sum())} candidates, gate "
          "labels (K - 1), Uniform K = all candidates, segment starts and stamps; every frame file there", flush=True)


# ---------------------------------------------------------------------------- S4: the summary
def record_flops(rec, line, units, k_q):
    """One answer's FLOPs from its own record and the plan's units: the gate on ours' rows (its K - 1 decisions), frames encoded x one
    frame's vision FLOPs (a baseline is charged all its input frames, x 2 for the two-view GazeVLM), and the prefill on its prompt."""
    from accuracy_tables import flops
    kind = line["kind"]
    if kind == "blind":
        vision = 0.0
    elif kind == "baseline":
        vision = line["views"] * min(int(line["K"]), int(k_q)) * units["v_stock"]
    else:
        vision = int(rec["n_frames"]) * float(line["v_frame"])
    gate = flops.GATE_MODELS * units["gate_per_decision"] * (int(k_q) - 1) if kind == "ours" else 0.0
    return gate + vision + float(flops.prefill(units["prefill"], rec["n_tokens"]))


def paired(a, b, n_boot=BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """a, b: {qid: correct 0/1}. On the questions both have: (a - b in points, 95% interval from a paired bootstrap over questions, n)."""
    q = sorted(set(a) & set(b))
    if not q:
        return float("nan"), float("nan"), float("nan"), 0
    d = np.array([a[x] - b[x] for x in q], dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    return 100 * d.mean(), 100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5), len(q)


def best_of(rows):
    """rows: [dict(rho, accuracy, flops_pct)] of one method -> the best: the highest accuracy, then the FLOPs share nearest 10%, then the
    larger rho."""
    return max(rows, key=lambda r: (round(r["accuracy"], 9), -abs(r["flops_pct"] - 100 * BUDGET), r["rho"]))


def model_table(model, plan, answers):
    """One row per planned condition. answers: {condition name: [records]}."""
    from accuracy_tables.hpc.hdepic.diag18 import wilson
    units = plan["units"]
    k_of = {int(q): int(k) for q, k in plan["K_candidates"].items()}
    lines = {ln["name"]: ln for ln in plan["lines"]}
    good = {name: {int(r["qid"]): r for r in recs if r.get("error") is None} for name, recs in answers.items()}
    uni = good.get("uniform", {})
    uni_flops = {q: record_flops(r, lines["uniform"], units, k_of[q]) for q, r in uni.items()}
    uni_correct = {q: int(r["correct"]) for q, r in uni.items()}
    rows = []
    for name, ln in lines.items():
        recs = good.get(name, {})
        q = sorted(set(recs) & set(uni_flops))
        f = sum(record_flops(recs[x], ln, units, k_of[x]) for x in q)
        pct = 100 * f / sum(uni_flops[x] for x in q) if q else float("nan")
        acc, lo, hi = wilson(sum(int(r["correct"]) for r in recs.values()), len(recs))
        d, dlo, dhi, _ = paired({x: int(r["correct"]) for x, r in recs.items()}, uni_correct)
        rows.append(dict(model=model, condition=name, method=ln["method"], rho=ln["rho"], rho_real=ln["rho_real"], K=ln["K"],
                         tokens_per_frame=ln["tokens_per_frame"], views=ln["views"],
                         image_tokens_per_q=np.mean([r.get("n_image_tokens", 0) for r in recs.values()]) if recs else float("nan"),
                         frames_per_q=np.mean([r.get("n_frames", 0) for r in recs.values()]) if recs else float("nan"),
                         planned_pct=ln["planned_pct"], flops_pct=pct, nearest_only=nearest_only(pct) if ln["kind"] not in ("uniform", "blind") else False,
                         n=len(recs), errors=len(answers.get(name, [])) - len(recs), accuracy=acc, ci_low=lo, ci_high=hi,
                         d_uniform=d, d_low=dlo, d_high=dhi))
    return rows


def best_rows(model, table):
    by = {r["condition"]: r for r in table}
    out = []
    for method, settings in method_settings().items():
        cands = [dict(by[c], rho=rho) for rho, c in settings.items() if c in by and by[c]["n"]]
        if cands:
            b = best_of(cands)
            out.append(dict(model=model, method=method, best_rho=b["rho"], condition=b["condition"], K=b["K"], accuracy=b["accuracy"],
                            ci_low=b["ci_low"], ci_high=b["ci_high"], d_uniform=b["d_uniform"], d_low=b["d_low"], d_high=b["d_high"],
                            flops_pct=b["flops_pct"], settings=len(cands)))
    return out


def effect_rows(model, answers):
    good = {name: {int(r["qid"]): int(r["correct"]) for r in recs if r.get("error") is None} for name, recs in answers.items()}
    out = []
    for effect, pairs in (("gate (ours - nogate, same representation)", GATE_PAIRS), ("gaze (gaze tile - centre tile)", GAZE_PAIRS)):
        for a_, b_ in pairs:
            if a_ in good and b_ in good:
                d, lo, hi, n = paired(good[a_], good[b_])
                out.append(dict(model=model, effect=effect, a=a_, b=b_, n=n, diff=d, low=lo, high=hi))
    return out


def summary_cmd(a):
    import pandas as pd
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    tables, best, effects = [], [], []
    for model in MODELS:
        path = model_dir(model, "plan.json")
        if not os.path.exists(path):
            print(f"{model}: no plan, skipped", flush=True)
            continue
        plan = read_json(path)
        answers = {ln["name"]: read_jsonl(model_dir(model, "answers", f"{ln['name']}.jsonl")) for ln in plan["lines"]}
        missing = [n for n, r in answers.items() if len({int(x["qid"]) for x in r}) != N_DEV]
        if missing:
            print(f"{model}: INCOMPLETE: {missing}", flush=True)
        t = model_table(model, plan, answers)
        tables += t
        best += best_rows(model, t)
        effects += effect_rows(model, answers)
        cols = ["condition", "rho", "rho_real", "K", "tokens_per_frame", "image_tokens_per_q", "planned_pct", "flops_pct", "nearest_only",
                "n", "accuracy", "ci_low", "ci_high", "d_uniform", "d_low", "d_high"]
        print(f"\n=== {model} ({MODELS[model]['name']}): 10% of its Uniform-64 FLOPs; target 10.000%; dev only", flush=True)
        print(pd.DataFrame(t)[cols].round(3).to_string(index=False), flush=True)
        print(f"--- {model}: each method's best rho (highest dev accuracy; ties: FLOPs nearest 10%, then the larger rho)", flush=True)
        print(pd.DataFrame([b for b in best if b["model"] == model]).drop(columns="model").round(3).to_string(index=False), flush=True)
    out = root("summary")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(tables).to_csv(os.path.join(out, "conditions.csv"), index=False)
    pd.DataFrame(best).to_csv(os.path.join(out, "best_rho.csv"), index=False)
    pd.DataFrame(effects).to_csv(os.path.join(out, "effects.csv"), index=False)
    if best:
        b = pd.DataFrame(best)
        b["cell"] = [f"rho {r:g}: {x:.2f} ({d:+.2f})" for r, x, d in zip(b.best_rho, b.accuracy, b.d_uniform)]
        cross = b.pivot(index="method", columns="model", values="cell")
        print("\n=== across models: each method's best setting, accuracy and its difference from that model's Uniform-64", flush=True)
        print(cross.to_string(), flush=True)
        cross.to_csv(os.path.join(out, "cross_model.csv"))
    if effects:
        print("\n=== the gate and the gaze, paired over questions (95% intervals from a paired bootstrap)", flush=True)
        print(pd.DataFrame(effects).round(3).to_string(index=False), flush=True)
    print(f"SUMMARY DONE (dev only, no winner): {out}. STOP.", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("blind-source", "plan", "gate", "frames-check", "sizes", "summary"):
        sub.add_parser(name)
    s = sub.add_parser("frames")
    s.add_argument("--index", type=int, help="the planned video's position (default: $SLURM_ARRAY_TASK_ID)")
    s = sub.add_parser("data")
    s.add_argument("--config", default=CONFIG)
    a = ap.parse_args(argv)
    if a.command == "sizes":
        for m in MODELS:
            for line in sizes_lines(m):
                print(line, flush=True)
        return
    dict(**{"blind-source": blind_source_cmd, "plan": plan_cmd, "gate": gate_cmd, "frames": frames_cmd, "frames-check": frames_check_cmd,
            "data": data_cmd, "summary": summary_cmd})[a.command](a)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)

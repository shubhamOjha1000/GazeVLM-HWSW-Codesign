"""Runbook section 18, diagnostic C1 to C4: why the foveated rows (budget, nogate) sit at Blind on HD-EPIC. Diagnostic only: no method
setting changes, nothing here feeds a results table, and every output is labelled diagnostic. Every results folder is only read; writes go
to $GV_ROOT/data/hdepic/diagnose18/ only.

    python -u -m accuracy_tables.hpc.hdepic.diag18 c1      (CPU) frame provenance and prompts, through the GPU run's own job builders
    python -u -m accuracy_tables.hpc.hdepic.diag18 c2      (CPU) answer agreement from the saved answers, HD-EPIC against EgoEverything
    python -u -m accuracy_tables.hpc.hdepic.diag18 c3      (B200) the foveated builder at full resolution against the stock path
    python -u -m accuracy_tables.hpc.hdepic.diag18 c4      (B200) the ablation ladder on action_recognition's 940 questions, K' = 9 frames

c1  For 20 action_recognition questions (one a real video) and 5 of section 17's "where looking" questions, each row's frames as the GPU run
    builds them: flops_run.reference_jobs (Uniform K, budget, nogate) and flops_run.plan_jobs (uniform_flops and GazeVLM at 10%), the same
    functions, no copies. Per frame: file, MP4 frame, time, whether it's inside the question's span and among the plan's candidates, and
    whether the JPEG is that MP4 frame (decoded again, mean absolute difference). For budget and nogate: the gaze and the fovea's box in
    source pixels. Then each row's prompt with the image tokens collapsed, ours' against Uniform's on the same frames.
c2  For action_recognition, all of fine_grained, 3d_perception, section 17's gaze_estimation and section 14's EgoEverything: how often budget,
    nogate, Blind, Uniform and GazeVLM pick the same letter, each row's letters, and where Uniform and Blind differ, which side budget takes.
c3  Section 14's SmolVLM2-2.2B on 50 action_recognition and 50 EgoEverything questions, Uniform K frames, two conditions: the stock frame and
    the foveated builder with every cell at full resolution (k = 9: its layout is the stock 81 cells). In bf16 the two can't be equal to
    the last bit (the foveated path embeds 14 px patches one by one; the stock path convolves the image; the sums round differently), so the
    identity is also checked in fp32 on the vision side, as the parity check does. Then, on the run's own 17-token layout, whether the
    fovea's cells are the stock pixels with the stock position ids, and how far each cell's token sits from the stock token of that area.
c4  On action_recognition's 940 questions, the same 9 frames per question (Uniform K' = 9), no gate, no budget: (a) stock, 81 tokens;
    (b) average pooling 4 x 4 after the encoder, 16; (c) ours' periphery only (all 9 cells coarse), 9; (d) ours' foveated frame, 17;
    (e1) the frame shrunk to the periphery's resolution (126 px) on the stock path, 9; (e2) shrunk to 168 px, 16; (g) ours' foveated frame
    with the fovea at the centre, 17. Then (f): the gate's frames (budget 10%) encoded stock, which needs a rerun if C1 finds a frame bug.
    Blind comes from section 18's saved answers. Each with n, accuracy and its Wilson interval.
"""
import argparse
import collections
import difflib
import json
import os
import re
import sys

import numpy as np

from accuracy_tables.hpc.hdepic import data, s18

MODEL = "smolvlm"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONFIG18 = os.path.join(REPO, "accuracy_tables", "hpc", "config.hdepic18.yaml")
CONFIG17 = os.path.join(REPO, "accuracy_tables", "hpc", "config.hdepic.yaml")
CONFIG14 = os.path.join(REPO, "accuracy_tables", "hpc", "config.all350.yaml")
SAME_FILE_MAD = 4.0                    # mean absolute difference (0-255) between a JPEG and its MP4 frame decoded again, JPEG noise included
DIAG = "diagnostic"
WORKERS = 16                           # threads loading frames ahead on the B200 jobs (the threads check: the same answers either way)


def out_dir(*parts):
    d = os.path.join(os.environ["GV_ROOT"], "data", "hdepic", "diagnose18", *parts)
    data.refuse_inside_raw(d)
    os.makedirs(d, exist_ok=True)
    return d


def results(name):
    return os.path.join(os.environ["GV_ROOT"], "results", name)


def wilson(k, n, z=1.96):
    """(accuracy %, low %, high %) with Wilson's interval."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * p, 100 * (mid - half), 100 * (mid + half)


def collapse(text, image="<image>"):
    """A prompt with every run of image tokens as <image xN>."""
    return re.sub(f"(?:{re.escape(image)})+", lambda m: f"<image x{len(m.group(0)) // len(image)}>", text)


def strip_counts(text):
    return re.sub(r"<image x\d+>", "<image>", text)


# ---------------------------------------------------------------------------- C1
def real_video(meta, f):
    return meta["videos"][meta["video_of"][f]] if "videos" in meta else meta["video"]


def spans18(q, folder):
    """{real video: [(start, end)]} of a section 18 question's segments as D1 bounds them (a single instant widened by 1 s)."""
    from accuracy_tables.hpc.hdepic import groups
    raw = groups.raw_question(folder, q["group"], q["type"], q["hdepic_qid"])
    row = groups.question_row(q["group"], q["type"], q["hdepic_qid"], raw)
    out = collections.defaultdict(list)
    for s in row["segments"]:
        mp4, _ = data.read_time_csv(os.path.join(data.video_dir(s["video"]), f"{s['video']}_mp4_to_vrs_time_ns.csv"))
        out[s["video"]].append(groups.segment_bounds(s, mp4, s18.AROUND_S))
    return out


def jobs_for(cfg, d, plan, pct=10.0, extra_rows=("uniform_flops", "gazevlm")):
    """{row label: {clip: [frame]}} from the GPU run's own builders: flops_run.reference_jobs and flops_run.plan_jobs."""
    from accuracy_tables import flops_run
    from accuracy_tables.models import ADAPTERS
    ad = ADAPTERS[MODEL](cfg["models"].get(MODEL), cfg)             # not loaded: building jobs needs no weights
    out = {}
    for job in flops_run.reference_jobs(ad, cfg, d, plan):
        name = job.get("name", job["row"])
        if job["row"] in ("uniform", "budget", "nogate") and (job["row"] == "uniform" or name.endswith(f" {pct:g}%")):
            out["uniform" if job["row"] == "uniform" else name] = {it["video"]: it["frames"] for it in job["items"]}
    lines = [ln for ln in plan["lines"] if ln["row"] in extra_rows and float(ln["budget_pct"]) == pct]
    for job in flops_run.plan_jobs(ad, cfg, d, dict(plan, lines=lines)):
        out[job["name"]] = {it["video"]: it["frames"] for it in job["items"]}
    return out


def decoded_mad(video, mp4_frame, jpeg_path, cv2):
    """(mean abs difference of the JPEG from the MP4 frame decoded again, and from the frame 30 later; decoder time s)."""
    cap = cv2.VideoCapture(os.path.join(data.video_dir(video), f"{video}.mp4"))
    out = []
    for f in (mp4_frame, mp4_frame + 30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, fr = cap.read()
        out.append((fr, cap.get(cv2.CAP_PROP_POS_MSEC) / 1e3) if ok else (None, None))
    cap.release()
    img = cv2.imread(jpeg_path)
    mads = []
    for fr, _ in out:
        if fr is None or img is None:
            mads.append(float("nan"))
            continue
        small = cv2.resize(fr, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        mads.append(float(np.abs(small.astype(np.float32) - img.astype(np.float32)).mean()))
    return mads[0], mads[1], out[0][1]


def fovea_box(g, side_px):
    """The k = 1 fovea's box (x0, y0, x1, y1) in a side_px frame, for gaze g in [0, 1] (the centre when g is None)."""
    from vlmeval import smolvlm_fovea as sm
    g = g if g is not None else (0.5, 0.5)
    fine = np.asarray([c for c in np.asarray(sm.foveated_cells(*g, 1)).tolist() if c[2] == sm.BASE])
    y0, x0 = fine[:, 0].min(), fine[:, 1].min()
    y1, x1 = (fine[:, 0] + fine[:, 2]).max(), (fine[:, 1] + fine[:, 2]).max()
    f = side_px / sm.IMG
    return tuple(int(round(v * f)) for v in (x0, y0, x1, y1))


def prompt_text(proc, item, frames, tokens_per_frame):
    from accuracy_tables.models import build_messages
    from vlmeval import smolvlm_fovea as sm
    msgs = build_messages(item["question"], list(item["options"]), [(fr["path"], fr["t"]) for fr in frames], sm.IMG * sm.IMG)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return collapse(sm.expand_images(proc, text, [tokens_per_frame] * len(frames)), sm.image_markers(proc)[2])


def c1_dataset(name, cfg_path, plan_path, picks, span_of, cand_of, proc, cv2, rows_csv, log_lines=2):
    """The C1 checks on one dataset -> counters."""
    import pandas as pd
    from accuracy_tables import common
    cfg = common.load_config(cfg_path)
    d = common.load_data(cfg)
    with open(plan_path) as f:
        plan = json.load(f)
    rows = jobs_for(cfg, d, plan)
    qs = d["qs"].set_index("video")
    c = collections.Counter()
    for v in picks(d):
        item = qs.loc[v]
        it = dict(question=item.question, options=list(item.options))
        spans, cands = span_of(v), cand_of(v)
        print(f"\n--- {DIAG} C1 {name} {v} ({item.get('hdepic_qid', '')}): K = {d['metas'][v]['n']}; rows: "
              + ", ".join(f"{r} {len(f[v])} frames" for r, f in rows.items()), flush=True)
        for r, per in rows.items():
            frames = per[v]
            m = d["metas"][v]
            for i, fr in enumerate(frames):
                pos = fr["key"][1]
                rv, mf, t = real_video(m, pos), int(m["mp4_frames"][pos]), float(m["ts"][pos])
                inside = any(a - 1e-6 <= t <= b + 1e-6 for a, b in spans.get(rv, []))
                cand = (rv, mf) in cands
                c["frames"] += 1
                c["in_span"] += inside
                c["in_plan"] += cand
                if r.split()[0] in ("budget", "nogate"):
                    c["foveated"] += 1
                    c["foveated_in_plan"] += cand
                rec = dict(dataset=name, clip=v, row=r, i=i, position=pos, file=os.path.basename(fr["path"]), video=rv, mp4_frame=mf,
                           mp4_s=t, in_span=inside, in_plan=cand, gaze=None if fr["gaze"] is None else tuple(round(x, 4) for x in fr["gaze"]))
                if r.split()[0] in ("budget", "nogate"):
                    rec["fovea_box_1344"] = fovea_box(fr["gaze"], 1344)
                    rec["fovea_box_mp4"] = fovea_box(fr["gaze"], 1408)
                if i in (0, len(frames) // 2, len(frames) - 1) and r in ("uniform",) or (i < 2 and r.split()[0] == "budget"):
                    mad, mad_far, dec = decoded_mad(rv, mf, fr["path"], cv2)
                    rec.update(mad=mad, mad_1s_later=mad_far, decoder_s=dec)
                    c["decoded"] += 1
                    c["file_matches"] += bool(np.isfinite(mad) and mad < SAME_FILE_MAD and dec is not None and abs(dec - t) < 0.05)
                rows_csv.append(rec)
                if i < log_lines or not inside or not cand:
                    print(f"   {r:<22} #{i:<3} pos {pos:<3} {rec['file']} {rv} frame {mf} at {t:.3f} s; in span {inside}; a candidate {cand}"
                          + (f"; gaze {rec['gaze']}, fovea box (1344 px) {rec.get('fovea_box_1344')}, (MP4 px) {rec.get('fovea_box_mp4')}"
                             if "fovea_box_1344" in rec else "")
                          + (f"; JPEG vs MP4 frame MAD {rec['mad']:.2f} (1 s later {rec['mad_1s_later']:.2f}), decoder {rec['decoder_s']}"
                             if "mad" in rec else ""), flush=True)
        ref = prompt_text(proc, it, rows["uniform"][v], 81)
        for r, per in rows.items():
            if r.split()[0] not in ("budget", "nogate"):
                continue
            ours = prompt_text(proc, it, per[v], 17)
            same = strip_counts(ours) == strip_counts(prompt_text(proc, it, per[v], 81))
            c["prompts"] += 1
            c["prompts_match"] += same
            if c["prompts"] <= 2 * log_lines:
                print(f"   prompt, {r} against Uniform K (image tokens collapsed):", flush=True)
                for line in difflib.unified_diff(ref.splitlines(), ours.splitlines(), "uniform", r, lineterm="", n=1):
                    print("      " + line[:220], flush=True)
            if not same:
                print(f"   PROMPT DIFFERS outside the images: {r} on {v}", flush=True)
    return c


def c1_cmd(a):
    import cv2
    import pandas as pd
    from transformers import AutoProcessor
    from vlmeval import smolvlm_fovea as sm
    proc = AutoProcessor.from_pretrained(sm.MODEL_CANDIDATES[0])
    rows_csv, total = [], collections.Counter()
    folder = data.questions_dir()
    with open(s18.plan_json()) as f:
        p18 = {q["pseudo"]: q for q in json.load(f)["questions"]}
    with open(s18.questions_json()) as f:
        q18 = {q["video"]: q for q in json.load(f)}

    def picks18(d):
        seen, out = set(), []
        for v in sorted(d["videos"]):
            q = q18[v]
            if q["type"] == "action_recognition" and q["real_videos"][0] not in seen:
                out.append(v)
                seen.add(q["real_videos"][0])
            if len(out) == a.questions:
                break
        return out

    total.update(c1_dataset("section 18 action_recognition", CONFIG18, os.path.join(results("hdepic_all_smolvlm_flops"), MODEL, "flops_plan.json"),
                            picks18, lambda v: spans18(q18[v], folder),
                            lambda v: {(p18[v]["videos"][k], f) for k, f in zip(p18[v]["video_of"], p18[v]["mp4_frames"])}, proc, cv2, rows_csv))
    p17 = {q["pseudo"]: q for q in data.read_plan()["questions"]}

    def picks17(d):
        ge = sorted(v for v in d["videos"] if p17[v]["type"] == "gaze_estimation")
        return ge[:: max(1, len(ge) // a.gaze_questions)][: a.gaze_questions]

    total.update(c1_dataset("section 17 gaze_estimation", CONFIG17, os.path.join(results("hdepic_gaze_smolvlm_flops"), MODEL, "flops_plan.json"),
                            picks17, lambda v: {p17[v]["video"]: [(p17[v]["start_s"], p17[v]["end_s"])]},
                            lambda v: {(p17[v]["video"], f) for f in p17[v]["mp4_frames"]}, proc, cv2, rows_csv))
    path = os.path.join(out_dir("c1"), "frames.csv")
    pd.DataFrame(rows_csv).to_csv(path, index=False)
    print(f"\n{DIAG} C1: written {path} (ids, times and boxes, no text)", flush=True)
    print(f"FRAMES IN SPAN: {total['in_span']} of {total['frames']}", flush=True)
    print(f"FOVEATED FRAMES = PLAN'S CANDIDATES: {total['foveated_in_plan']} of {total['foveated']}", flush=True)
    print(f"FILES MATCH THEIR MP4 FRAMES: {total['file_matches']} of {total['decoded']} (JPEG vs the MP4 frame decoded again, "
          f"MAD under {SAME_FILE_MAD:g} and the decoder within 0.05 s)", flush=True)
    print(f"PROMPTS MATCH OUTSIDE THE IMAGES: {total['prompts_match']} of {total['prompts']}", flush=True)
    print("C1 DONE (diagnostic)", flush=True)


# ---------------------------------------------------------------------------- C2
def load_labelled(folder, plan_path=None):
    """A results folder's records with each record's row label (uniform, blind, '<row> <b>%')."""
    import pandas as pd
    from accuracy_tables.hpc.all350 import flops_report as rep
    from accuracy_tables.hpc.hdepic import diagnose as dg
    with open(plan_path or os.path.join(folder, MODEL, "flops_plan.json")) as f:
        plan = json.load(f)
    res, _ = rep.load(folder, MODEL)
    if not len(res) and os.path.isdir(os.path.join(folder, "shard0")):
        res = pd.concat([rep.load(os.path.join(folder, f"shard{s}"), MODEL)[0] for s in range(3)], ignore_index=True)
    lines = pd.DataFrame(plan["lines"])[["row", "condition", "budget_pct"]].drop_duplicates(["row", "condition"])
    res = res.merge(lines, on=["row", "condition"], how="left")
    res["label"] = [dg.label(r, b) for r, b in zip(res.row, res.budget_pct)]
    return res


AGREE = ("uniform", "blind", "budget 10%", "nogate 10%", "gazevlm 10%")


def agreement(res):
    """-> (pairwise same-letter %, letters % per label, where Uniform != Blind: % budget 10% = Blind and = Uniform, n)."""
    import pandas as pd
    p = res.pivot_table(index="qid", columns="label", values="pred", aggfunc="first")
    labs = [x for x in AGREE if x in p.columns]
    p = p[labs].dropna()
    pair = pd.DataFrame({x: {y: 100 * float((p[x] == p[y]).mean()) for y in labs} for x in labs})
    letters = pd.DataFrame({x: (100 * p[x].value_counts(normalize=True)).reindex(list("ABCDE")).fillna(0) for x in labs}).T
    side = {}
    if {"uniform", "blind", "budget 10%"} <= set(labs):
        split = p[p.uniform != p.blind]
        side = dict(n=len(split), with_blind=100 * float((split["budget 10%"] == split.blind).mean()) if len(split) else float("nan"),
                    with_uniform=100 * float((split["budget 10%"] == split.uniform).mean()) if len(split) else float("nan"))
    return pair, letters, side, len(p)


def c2_cmd(a):
    import pandas as pd
    pd.set_option("display.width", 220)
    with open(s18.questions_json()) as f:
        q18 = pd.DataFrame(json.load(f))
    with open(data.questions_json()) as f:
        q17 = pd.DataFrame(json.load(f))
    r18 = load_labelled(results("hdepic_all_smolvlm_flops"))
    r18["type"], r18["group"] = r18.video.map(dict(zip(q18.video, q18.type))), r18.video.map(dict(zip(q18.video, q18.group)))
    r17 = load_labelled(results("hdepic_gaze_smolvlm_flops"))
    r17["type"] = r17.video.map(dict(zip(q17.video, q17.type)))
    r14 = load_labelled(results("smolvlm_flops"))
    sets = [("section 18 action_recognition", r18[r18.type == "action_recognition"]), ("section 18 fine_grained", r18[r18.group == "fine_grained"]),
            ("section 18 3d_perception", r18[r18.group == "3d_perception"]), ("section 17 gaze_estimation", r17[r17.type == "gaze_estimation"]),
            ("section 14 EgoEverything", r14)]
    summary = {}
    for name, res in sets:
        pair, letters, side, n = agreement(res)
        print(f"\n--- {DIAG} C2 {name}: {n} questions with every row; the same letter (%)", flush=True)
        print(pair.round(1).to_string(), flush=True)
        print("letters chosen (%)", flush=True)
        print(letters.round(1).to_string(), flush=True)
        print(f"AGREEMENT {name}: budget 10% = Blind on {pair.loc['budget 10%', 'blind']:.1f}%, = Uniform on "
              f"{pair.loc['budget 10%', 'uniform']:.1f}%; where Uniform and Blind differ ({side.get('n', 0)}), budget sides with Blind on "
              f"{side.get('with_blind', float('nan')):.1f}%, with Uniform on {side.get('with_uniform', float('nan')):.1f}%", flush=True)
        summary[name] = dict(pairs=pair.round(3).to_dict(), letters=letters.round(3).to_dict(), side=side, n=n)
    path = os.path.join(out_dir("c2"), "agreement.json")
    with open(path, "w") as f:
        json.dump(summary, f, default=float)
    print(f"\n{DIAG} C2: written {path}\nC2 DONE (diagnostic)", flush=True)


# ---------------------------------------------------------------------------- C3
def vision_fp32(model, recs, mean, std):
    """[(cos min, max abs diff)] of the foveated builder on the stock 81-cell layout against the stock encoder, in fp32 with exact
    attention, as smolvlm_fovea.parity_report checks one frame."""
    import torch
    import torch.nn.functional as F
    from contextlib import nullcontext
    from vlmeval import smolvlm_fovea as sm
    vis, connector = sm.parts(model)
    dtype = next(vis.parameters()).dtype
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        exact = sdpa_kernel(SDPBackend.MATH)
    except Exception:
        exact = nullcontext()
    cells, out = sm.uniform_cells(), []
    try:
        vis.float()
        connector.float()
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        with exact, torch.no_grad():
            for rec in recs:
                x = sm.encode_cells(model, rec["x01"], cells, mean, std).float()
                y = sm.encode_pixels(model, rec["pv"]).float()
                out.append((F.cosine_similarity(x, y, dim=-1).min().item(), (x - y).abs().max().item()))
    finally:
        vis.to(dtype)
        connector.to(dtype)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    return out


def fovea_inputs(rec, cells, mean, std):
    """(largest |patch - the stock pixel_values' patch at the same place| over the fovea's 42 px cells, position ids all the stock grid's)."""
    from vlmeval import smolvlm_fovea as sm
    p = sm.cells_to_patches(rec["x01"], cells, mean, std)
    h, w = sm.patch_positions(cells)
    worst, ids = 0.0, True
    for i, (y, x, s) in enumerate(np.asarray(cells).tolist()):
        if s != sm.BASE:
            continue
        for a in range(3):
            for b in range(3):
                k = 9 * i + 3 * a + b
                ref = rec["pv"][:, y + a * sm.PATCH:y + (a + 1) * sm.PATCH, x + b * sm.PATCH:x + (b + 1) * sm.PATCH]
                worst = max(worst, float((p[k] - ref.float()).abs().max()))
                ids &= bool(h[k] == y // sm.PATCH + a and w[k] == x // sm.PATCH + b)
    return worst, ids


def compare_records(path_a, path_b):
    """(records, answers that differ, max |p diff|, the first differing pair) of two record files on the same questions."""
    def read(p):
        return {(r["qid"]): r for r in (json.loads(x) for x in open(p) if x.strip()) if r.get("error") is None}
    ra, rb = read(path_a), read(path_b)
    common_q = sorted(set(ra) & set(rb))
    diff = [q for q in common_q if ra[q]["pred"] != rb[q]["pred"]]
    mx = max((float(np.max(np.abs(np.asarray(ra[q]["p"]) - np.asarray(rb[q]["p"])))) for q in common_q), default=float("nan"))
    first = (ra[diff[0]], rb[diff[0]]) if diff else None
    return len(common_q), len(diff), mx, first


def load_model(cfg, d):
    from accuracy_tables import run
    from accuracy_tables.models import ADAPTERS
    ad = ADAPTERS[MODEL](cfg["models"].get(MODEL), cfg).load()
    run.check_parity(ad, d, MODEL)                                        # also fixes how prompts are tokenized, as every run does
    return ad


def c3_dataset(name, cfg, d, ad, items, folder):
    from PIL import Image
    from vlmeval import smolvlm_fovea as sm
    conds = {"stock": ("uniform", sm.IMG), "fovea_every_cell_full": ("fovea", dict(k=9, gaze="gaze"))}
    paths = {c: os.path.join(folder, f"{c}.jsonl") for c in conds}
    for c, cond in conds.items():
        ad.score(items, {c: cond}, paths[c])
    n, nd, mx, first = compare_records(paths["stock"], paths["fovea_every_cell_full"])
    if nd == 0 and mx == 0.0:
        print(f"FOVEATED IDENTITY ({name}): SAME ANSWERS: 0 of {n} records differ, max_prob_diff 0.0", flush=True)
    else:
        print(f"FOVEATED IDENTITY ({name}): {nd} of {n} records differ, max_prob_diff {mx:.4f} (bf16: the foveated path embeds 14 px "
              "patches one by one, the stock path convolves the image, and the sums round differently; the fp32 line below is the identity)",
              flush=True)
    if first:
        a, b = first
        print(f"   first divergence, question {a['qid']} ({a['video']}): stock {a['pred']} {a['p']}, foveated {b['pred']} {b['p']}; image "
              f"tokens {a['n_image_tokens']} and {b['n_image_tokens']}, prompt tokens {a['n_tokens']} and {b['n_tokens']}, frames "
              f"{a['n_frames']} and {b['n_frames']}", flush=True)
    ip = ad.proc.image_processor
    mean, std = ip.image_mean, ip.image_std
    same_layout = np.array_equal(np.asarray(sm.foveated_cells(0.3, 0.7, 9)), np.asarray(sm.uniform_cells()))
    recs, worst, ids, drift = [], 0.0, True, collections.defaultdict(list)
    for it in items:
        fr = it["frames"][0]
        rec = sm.stock_input(ip, Image.open(fr["path"]).convert("RGB"))
        recs.append(rec)
        cells = sm.foveated_cells(*(fr["gaze"] if fr["gaze"] is not None else (0.5, 0.5)), 1)
        w, ok = fovea_inputs(rec, cells, mean, std)
        worst, ids = max(worst, w), ids and ok
        t = sm.fovea_feature_drift(ad.model, rec, cells, mean, std)
        for px, row in t.iterrows():
            drift[int(px)].append((row.mean_cos, row.min_cos))
    fp = vision_fp32(ad.model, recs, mean, std)
    cmin, amax = min(x[0] for x in fp), max(x[1] for x in fp)
    print(f"VISION IDENTITY IN FP32 ({name}): cos min {cmin:.6f}, max abs diff {amax:.2e} over {len(fp)} frames, the layout at k = 9 "
          f"{'is' if same_layout else 'IS NOT'} the stock 81 cells: " + ("SAME" if cmin > 0.9999 and same_layout else "DIFFERS"), flush=True)
    print(f"FOVEA INPUTS ARE THE STOCK PIXELS ({name}): the 17-token layout's 42 px cells differ from the stock pixel_values by at most "
          f"{worst:.2e}, position ids {'the stock grid on every cell' if ids else 'DIFFER from the stock grid'}", flush=True)
    print(f"TOKEN DRIFT ({name}), each 17-token frame's tokens against the stock tokens of the same area: "
          + "; ".join(f"{px} px cells mean cos {np.mean([x[0] for x in v]):.3f} (lowest {np.min([x[1] for x in v]):.3f})"
                      for px, v in sorted(drift.items())), flush=True)
    return dict(n=n, differ=nd, max_prob_diff=mx, fp32_cos_min=cmin, fp32_max_abs=amax, fovea_input_max=worst, ids=ids,
                drift={px: float(np.mean([x[0] for x in v])) for px, v in drift.items()})


def pick_items(d, cap, keep, n):
    from accuracy_tables import common
    items = common.make_items(d["qs"], {v: common.cap_frames(f, cap) for v, f in d["uniform"].items()})
    return [it for it in items if keep(it)][:n]


def c3_cmd(a):
    from accuracy_tables import common
    from accuracy_tables.hpc.all350 import prepare, smolvlm256_checks
    cfg18 = common.load_config(CONFIG18)
    cfg18["run"]["prefetch_workers"] = WORKERS
    d18 = common.load_data(cfg18)
    with open(s18.questions_json()) as f:
        types = {q["video"]: q["type"] for q in json.load(f)}
    picked = pick_items(d18, 64, lambda it: types[it["video"]] == "action_recognition", a.questions)
    ad = load_model(cfg18, d18)
    out = {}
    out["hdepic18"] = c3_dataset("section 18 action_recognition", cfg18, d18, ad, picked, out_dir("c3", "hdepic18"))
    cfg14 = common.load_config(CONFIG14)
    cfg14["run"]["videos"] = smolvlm256_checks.pick_by_questions(prepare.question_counts(), a.questions)
    d14 = common.load_data(cfg14)
    from accuracy_tables import run
    run.check_parity(ad, d14, MODEL)
    out["egoeverything"] = c3_dataset("section 14 EgoEverything", cfg14, d14, ad, pick_items(d14, 64, lambda it: True, a.questions),
                                      out_dir("c3", "egoeverything"))
    with open(os.path.join(out_dir("c3"), "c3.json"), "w") as f:
        json.dump(out, f, default=float)
    print("C3 DONE (diagnostic)", flush=True)


# ---------------------------------------------------------------------------- C4
LADDER = (("a_stock_81", ("uniform", 384)), ("c_periphery_only_9", ("fovea", dict(k=0, gaze="gaze"))),
          ("d_foveated_17", ("fovea", dict(k=1, gaze="gaze"))), ("e1_shrunk_126px_9", ("uniform", 126)),
          ("e2_shrunk_168px_16", ("uniform", 168)), ("g_foveated_centre_17", ("fovea", dict(k=1, gaze="centre"))))


def accuracy_lines(path, label):
    recs = [json.loads(x) for x in open(path) if x.strip()]
    recs = [r for r in recs if r.get("error") is None]
    k, n = sum(r["correct"] for r in recs), len(recs)
    acc, lo, hi = wilson(k, n)
    tpf = np.mean([r["n_image_tokens"] / max(1, r["n_frames"]) for r in recs]) if recs else float("nan")
    fpq = np.mean([r["n_frames"] for r in recs]) if recs else float("nan")
    return f"C4 {label}: n {n}, accuracy {acc:.2f}% [{lo:.2f}, {hi:.2f}], {tpf:.1f} tokens a frame, {fpq:.1f} frames a question", (acc, lo, hi, n)


def c4_cmd(a):
    from accuracy_tables import common, flops_calibrate as fc, flops_run
    from accuracy_tables.models import build_messages
    from vlmeval import avgpool as ap
    cfg = common.load_config(CONFIG18)
    cfg["run"]["prefetch_workers"] = WORKERS
    d = common.load_data(cfg)
    with open(s18.questions_json()) as f:
        types = {q["video"]: q["type"] for q in json.load(f)}
    keep = lambda it: types[it["video"]] == "action_recognition"
    d9 = fc.Inputs(d)(9)                                                 # Uniform K' = 9's frames, as flops_run builds them
    items = [it for it in common.make_items(d["qs"], d9["uniform"]) if keep(it)]
    ad = load_model(cfg, d)
    folder = out_dir("c4")
    for name, cond in LADDER:
        ad.score(items, {name: cond}, os.path.join(folder, f"{name}.jsonl"))
    ap.run_questions(ad.avgpool_backend(), items, {"b_avgpool_4x4_16": dict(side=4)}, os.path.join(folder, "b_avgpool_4x4_16.jsonl"),
                     build_messages, prefetch_workers=ad.prefetch_workers())
    with open(os.path.join(results("hdepic_all_smolvlm_flops"), MODEL, "flops_plan.json")) as f:
        plan = json.load(f)
    gate = next(j for j in flops_run.reference_jobs(ad, cfg, d, plan) if j.get("name") == "budget 10%")
    gate_items = [it for it in gate["items"] if keep(it)]
    ad.score(gate_items, {"f_gate_frames_stock_81": ("uniform", 384)}, os.path.join(folder, "f_gate_frames_stock_81.jsonl"))
    qids = {it["qid"] for it in items}
    blind = load_labelled(results("hdepic_all_smolvlm_flops"))
    blind = blind[(blind.label == "blind") & blind.qid.isin(qids)]
    acc, lo, hi = wilson(int(blind.correct.sum()), len(blind))
    print(f"\n--- {DIAG} C4 on {len(items)} action_recognition questions, Uniform K' = 9's frames (f: the gate's frames at 10%)", flush=True)
    print(f"C4 blind (section 18's saved answers): n {len(blind)}, accuracy {acc:.2f}% [{lo:.2f}, {hi:.2f}]", flush=True)
    table = {"blind": (acc, lo, hi, len(blind))}
    for name in ["a_stock_81", "b_avgpool_4x4_16", *[n for n, _ in LADDER[1:]], "f_gate_frames_stock_81"]:
        line, got = accuracy_lines(os.path.join(folder, f"{name}.jsonl"), name + (" (rerun if C1 finds a frame bug)" if name.startswith("f_") else ""))
        print(line, flush=True)
        table[name] = got
    with open(os.path.join(folder, "c4.json"), "w") as f:
        json.dump(table, f, default=float)
    print("reading: (c) and (d) near Blind while (e1) is clearly above it: the foveated encoding path; (e1) near Blind too: the resolution "
          "itself; (e1) against (c) is the same pixels through two encodings", flush=True)
    print("C4 DONE (diagnostic)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("c1")
    s.add_argument("--questions", type=int, default=20)
    s.add_argument("--gaze-questions", type=int, default=5)
    sub.add_parser("c2")
    s = sub.add_parser("c3")
    s.add_argument("--questions", type=int, default=50)
    sub.add_parser("c4")
    a = ap.parse_args(argv)
    dict(c1=c1_cmd, c2=c2_cmd, c3=c3_cmd, c4=c4_cmd)[a.command](a)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)

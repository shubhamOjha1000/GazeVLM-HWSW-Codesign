"""Runbook section 18, step 4: the question manifest under D1 to D5, then the data check. CPU only, base environment.

    python -u -m accuracy_tables.hpc.hdepic.manifest18 --config accuracy_tables/hpc/config.hdepic18.yaml

Writes $GV_ROOT/data/hdepic/s18/questions.json (outside the repository: it holds HD-EPIC's question text, under its non-commercial licence):
one entry per planned question with qid, its clip as video, real_videos, group, type, hdepic_qid, question and options (verbatim, every
<TIME> and <BBOX> tag kept), answer (the letter of correct_idx), t0 (its last candidate's MP4 time), K, n_segments and exception. Then
loads the data as every job will (common.load_data) and checks it against the plan:
  every question once; each clip's frames, gate labels (K - 1), Uniform K and segment starts
  D2: the query frame in Uniform K, in ours' and the no-gate control's frames at every budget, and in Uniform K' at K' = 1, 2, 8, 16, 32
  D3: every frame's stamp is "<TIME hh:mm:ss.sss video n>" with its own MP4 time and video number, and the prompt reads it
  17 foveated tokens on every candidate; every frame file there
Then the stamp's tokens on SmolVLM's own tokenizer (the median per frame against section 14's "Frame at m:ss:"), the frames without gaze
per group, and data check OK.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

from accuracy_tables.hpc.hdepic import data, groups, s18

LETTERS = "ABCDE"
K_PRIME_PROBES = (1, 2, 8, 16, 32)


def read_plan():
    with open(s18.plan_json()) as f:
        return json.load(f)


def entries(plan, folder=None):
    """questions.json's entries from the plan and the groups' question files."""
    folder = folder or data.questions_dir()
    files, _ = groups.question_files(folder)
    raw, out = {}, []
    for q in plan["questions"]:
        key = (q["group"], q["type"])
        if key not in raw:
            with open(os.path.join(folder, files[q["group"]][q["type"]])) as f:
                raw[key] = json.load(f)
        r = raw[key][q["hdepic_qid"]] if isinstance(raw[key], dict) else raw[key][int(q["hdepic_qid"])]
        choices = list(r["choices"])
        if len(choices) != q["n_choices"] or not 0 <= int(r["correct_idx"]) < len(choices) or len(choices) > len(LETTERS):
            raise ValueError(f"{q['hdepic_qid']}: {len(choices)} choices, correct_idx {r['correct_idx']}")
        out.append(dict(qid=q["qid"], video=q["pseudo"], real_videos=q["videos"], group=q["group"], type=q["type"],
                        hdepic_qid=q["hdepic_qid"], question=r["question"], options=choices, answer=LETTERS[int(r["correct_idx"])],
                        t0=q["mp4_s"][-1], K=q["K"], n_segments=q["n_segments"], exception=q["exception"]))
    return out


def keys(frames):
    return {fr["key"][1] for fr in frames}


def data_problems(d, plan, cfg, fovea_tokens=17):
    """What common.load_data's result gets wrong against the plan -> (problems, frames without gaze per group)."""
    from accuracy_tables import common
    from accuracy_tables import flops_calibrate as fc
    from vlmeval import smolvlm_fovea as sm
    from vlmeval.qwen_mcq import build_messages
    found, missing = [], collections.Counter()
    want = {q["pseudo"]: q for q in plan["questions"]}
    if sorted(d["qs"].video) != sorted(want) or d["qs"].video.duplicated().any():
        found.append(f"{len(d['qs'])} questions loaded, {len(want)} planned")
    rows = {(b, g): common.budget_frames(cfg, d, "smolvlm", 81, b, g)[0] for b in cfg["budget"]["fractions"] for g in (True, False)}
    kp = {k: fc.with_uniform(d, k)["uniform"] for k in K_PRIME_PROBES}
    for v in d["videos"]:
        q = want.get(v)
        if q is None:
            found.append(f"{v}: not planned")
            continue
        if d["metas"][v]["n"] != q["K"] or len(d["uniform"][v]) != q["K"]:
            found.append(f"{v}: {d['metas'][v]['n']} frames and {len(d['uniform'][v])} Uniform frames, {q['K']} planned")
        if len(d["labels"][v]) != q["K"] - 1:
            found.append(f"{v}: {len(d['labels'][v])} gate labels for {q['K']} frames")
        if d["segment_starts"].get(v, []) != q["starts"]:
            found.append(f"{v}: segment starts {d['segment_starts'].get(v)} against the plan's {q['starts']}")
        for f in q["forced"]:
            where = [name for name, fr in [("Uniform K", d["uniform"][v])] + [(f"{'ours' if g else 'nogate'} {b:g}", r[v])
                                                                               for (b, g), r in rows.items()]
                     + [(f"Uniform K'={k}", u[v]) for k, u in kp.items()] if f not in keys(fr)]
            if where:
                found.append(f"{v}: the query frame {f} isn't in {where}")
        for f in range(q["K"]):
            fr = d["frame"](v, f)
            text = getattr(fr["t"], "stamp", None)
            if text != s18.stamp(q["mp4_s"][f], q["video_n"][f]) or not s18.STAMP.match(text or ""):
                found.append(f"{v} frame {f}: stamp {text!r}")
            if fr["gaze"] is None:
                missing[q["group"]] += 1
            g = fr["gaze"] if fr["gaze"] is not None else (0.5, 0.5)
            if len(sm.foveated_cells(*g, 1)) != fovea_tokens:
                found.append(f"{v} frame {f}: not {fovea_tokens} foveated tokens")
            if not os.path.exists(fr["path"]):
                found.append(f"{v} frame {f}: no {fr['path']}")
        if v == d["videos"][0]:
            r = d["qs"][d["qs"].video == v].iloc[0]
            msgs = build_messages(r.question, list(r.options), [(fr["path"], fr["t"]) for fr in d["uniform"][v]], 0)
            labels = [c["text"] for c in msgs[0]["content"] if c["type"] == "text" and c["text"].startswith("Frame at ")]
            if labels != [f"Frame at {s18.stamp(t, n)}:" for t, n in zip(q["mp4_s"], q["video_n"])]:
                found.append(f"{v}: the prompt's frame labels aren't the stamps")
    return found, missing


def stamp_tokens(d, model_id):
    """(median tokens of "Frame at <TIME ...>:", of section 14's "Frame at m:ss:", frames counted) on SmolVLM's own tokenizer."""
    from transformers import AutoTokenizer
    from vlmeval.qwen_mcq import fmt_time
    tok = AutoTokenizer.from_pretrained(model_id)
    pairs = [(t, n) for m in d["metas"].values() for t, n in zip(m["ts"], m["video_n"])]
    pairs = pairs[:: max(1, len(pairs) // 20000)]
    new = [len(tok(f"Frame at {s18.stamp(t, n)}:", add_special_tokens=False).input_ids) for t, n in pairs]
    old = [len(tok(f"Frame at {fmt_time(float(t))}:", add_special_tokens=False).input_ids) for t, _ in pairs]
    return float(np.median(new)), float(np.median(old)), len(pairs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    from accuracy_tables import common
    from vlmeval import smolvlm_fovea as sm
    out = s18.questions_json()
    data.refuse_inside_raw(out)
    plan = read_plan()
    qs = entries(plan)
    with open(out + ".tmp", "w") as f:
        json.dump(qs, f)
    os.replace(out + ".tmp", out)
    kept = collections.Counter((q["group"], q["type"]) for q in qs)
    dropped = collections.Counter((e["group"], e["type"]) for e in plan["excluded"])
    for g, t in sorted(set(kept) | set(dropped)):
        ks = [q["K"] for q in qs if (q["group"], q["type"]) == (g, t)]
        print(f"{g}/{t}: {kept[g, t]} of {kept[g, t] + dropped[g, t]} questions kept, K median {np.median(ks) if ks else float('nan'):g}",
              flush=True)
    for why, n in collections.Counter(e["reason"] for e in plan["excluded"] if not e["reason"].startswith("D5")).items():
        print(f"   excluded: {why}: {n}", flush=True)
    d5 = collections.Counter(e["group"] for e in plan["excluded"] if e["reason"].startswith("D5"))
    print(f"excluded: {len(plan['excluded'])} in all, {sum(d5.values())} of them D5's {dict(d5)}", flush=True)
    cfg = common.load_config(a.config)
    d = common.load_data(cfg)
    print(common.describe(d), flush=True)
    found, missing = data_problems(d, plan, cfg)
    frames = collections.Counter()
    for q in qs:
        frames[q["group"]] += q["K"]
    for g in groups.GROUPS:
        print(f"{g}: {missing[g]} of {frames[g]} candidate frames have no gaze (fovea at the centre, section 9's rule)", flush=True)
    new, old, n = stamp_tokens(d, sm.MODEL_CANDIDATES[0])
    print(f"STAMP TOKENS: \"Frame at <TIME hh:mm:ss.sss video n>:\" is a median {new:g} tokens a frame on SmolVLM's tokenizer, section 14's "
          f"\"Frame at m:ss:\" {old:g}: {new - old:+g} a frame (over {n} frames). D3's deviation; the prompt fit counts it", flush=True)
    for p in found[:40]:
        print("  ", p, flush=True)
    if found:
        sys.exit(f"DATA CHECK FAILED: {len(found)} problems. Stop and tell the user.")
    print(f"data check OK: {len(qs)} questions on {len(plan['videos'])} videos; every clip's frames, gate labels, Uniform K and segment "
          "starts; the query frame in every row; every frame's stamp; 17 foveated tokens on every candidate", flush=True)


if __name__ == "__main__":
    main()

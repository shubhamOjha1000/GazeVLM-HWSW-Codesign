"""DTD, the Differential Token Drop of TimeChat-Online (Yao et al., ACM MM 2025), on every VLM in vlmeval.

As the authors' code (github.com/yaolinli/TimeChat-Online, demo/qwen2_5_vl/modeling_qwen2_5_vl.py,
token_drop with method='feature'), causal over the frames in time order, on the LLM-level tokens (the
visual tokens entering the LLM, as there): the first frame keeps all its tokens; every later frame is
compared with the previous frame, position by position (cosine of the two tokens at the same position,
the previous frame's tokens as encoded, not what was kept of them), and drops
  absolute = True:  every token whose cosine is > threshold (their README: 0.5), or
  absolute = False: its int(threshold n) most similar tokens (threshold = the share dropped).
Kept tokens keep their positions and their order; a frame that keeps no token leaves the prompt with its
time label (in TimeChat-Online's single video block it simply vanishes).
The relative mode sets the budget exactly, (N + (T - 1)(N - int(threshold N))) / (T N) for T frames of N
tokens, so the tables use it; the absolute mode is the authors' default.

Tokens, positions and packing come from the StreamingTOM backends (vlmeval/streamingtom.py) without
their saliency, as for DyCoke: Qwen's M-RoPE gets every kept token at its own cell, Qwen3-VL's DeepStack
features keep the same tokens, Phi's 576 crop tokens get a row break after each row that has tokens (no
global view); InternVL, Gemma (projected tokens) and SmolVLM take the kept tokens in raster order.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from vlmeval import streamingtom as stm

THRESHOLD, ABSOLUTE = 0.5, True         # the authors' README: drop_threshold 0.5, drop_absolute True
DEFAULTS = dict(threshold=THRESHOLD, absolute=ABSOLUTE)


# ---------------------------------------------------------------------------- the method
def dropped(F_prev, F_t, threshold=THRESHOLD, absolute=ABSOLUTE):
    """(N,) bool: the tokens of F_t that DTD drops, given the previous frame's tokens F_prev (N, D)."""
    s = F.cosine_similarity(F_prev.float(), F_t.float(), dim=1)
    if absolute:
        return s > threshold
    out = torch.zeros(len(s), dtype=torch.bool, device=s.device)
    out[torch.topk(s, int(len(s) * threshold)).indices] = True
    return out


def dtd_plans(frames, threshold=THRESHOLD, absolute=ABSOLUTE):
    """DTD over (N, D) frame tokens in time order -> kept positions (sorted) per frame."""
    plans = []
    for f, F_t in enumerate(frames):
        if f == 0:
            plans.append(list(range(len(F_t))))
        else:
            plans.append(torch.nonzero(~dropped(frames[f - 1], F_t, threshold, absolute)).flatten().tolist())
    return plans


def retention(threshold, n_frames, n):
    """Share of the tokens the relative mode keeps over n_frames frames of n tokens."""
    return (n + (n_frames - 1) * (n - int(n * threshold))) / (n_frames * n)


# ---------------------------------------------------------------------------- per-model backends
class Qwen25(stm.Qwen25):
    saliency = False


class Qwen3(stm.Qwen3):
    saliency = False


class InternVL(stm.InternVL):
    saliency = False


class Gemma(stm.Gemma):
    saliency = False


class Phi(stm.Phi):
    saliency = False


class SmolVLM(stm.SmolVLM):
    saliency = False


# ---------------------------------------------------------------------------- evaluation
@torch.no_grad()
def run_questions(backend, items, conditions, out_path, build_messages, log_every=25):
    """Score questions on their frames with DTD-reduced tokens; resumable.

    conditions: {name: dict(threshold=..., absolute=True|False)}.
    items: as for the *_fovea.run_questions (sorted by video), each item's frames in time order. Each
    frame is encoded once per video; a video's reduced frames are made once per (condition, frame list)."""
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                done.add((r["condition"], r["qid"]))
    todo = [(it, c) for it in items for c in conditions if (c, it["qid"]) not in done]
    print(f"{len(done)} already scored, {len(todo)} to go -> {out_path}")

    states, toks, video, t_start, n_new = {}, {}, None, time.time(), 0
    with open(out_path, "a") as fh:
        for it, c in todo:
            if it["video"] != video:                         # new video: drop the old frames
                states.clear()
                toks.clear()
                torch.cuda.empty_cache()
                video = it["video"]
            arg = {**DEFAULTS, **conditions[c]}
            t0 = time.time()
            keys = tuple(fr["key"] for fr in it["frames"])
            if (c, keys) not in toks:
                for fr in it["frames"]:
                    if fr["key"] not in states:
                        states[fr["key"]] = backend.frame_state(fr)
                plans = dtd_plans([states[k]["tokens"] for k in keys], float(arg["threshold"]), bool(arg["absolute"]))
                kept_frames, per = [], []
                for fr, k, plan in zip(it["frames"], keys, plans):
                    if plan:
                        kept_frames.append(fr)
                        per.append(backend.pack(states[k], [(p, []) for p in plan], 0.0))
                toks[(c, keys)] = (kept_frames, per,
                                   sum(len(plan) for plan in plans) / max(1, sum(len(states[k]["tokens"]) for k in keys)))
            kept_frames, per, kept_share = toks[(c, keys)]
            try:
                p, n_tok = backend.score(dict(it, frames=kept_frames), per, build_messages)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                fh.write(json.dumps(dict(condition=c, qid=int(it["qid"]), video=it["video"],
                                         error="oom", n_frames=len(per))) + "\n")
                continue
            err = None if np.isfinite(p).all() else "nan"
            pred = "ABCDE"[int(np.nanargmax(p))] if np.isfinite(p).any() else "?"
            fh.write(json.dumps(dict(
                condition=c, qid=int(it["qid"]), video=it["video"], answer=it["answer"],
                pred=pred, correct=int(pred == it["answer"]), p=[round(float(v), 4) for v in p],
                n_frames=len(per), n_frames_dropped=len(keys) - len(per),
                n_image_tokens=int(sum(len(q[0]) for q in per)), kept_share=round(float(kept_share), 4),
                n_tokens=n_tok, dtd=dict(threshold=float(arg["threshold"]), absolute=bool(arg["absolute"])),
                sec=round(time.time() - t0, 3), error=err)) + "\n")
            fh.flush()
            n_new += 1
            if n_new % log_every == 0:
                rate = (time.time() - t_start) / n_new
                print(f"   {n_new}/{len(todo)}  {rate:.2f} s each  ~{rate * (len(todo) - n_new) / 60:.0f} min left"
                      f"  ({c}, {it['video']})", flush=True)
    print(f"done: {n_new} new in {(time.time() - t_start) / 60:.1f} min")

"""DyCoke (Tao et al., CVPR 2025), its temporal token merging (TTM) only, on every VLM in vlmeval.

TTM, as in the paper and the authors' code (github.com/KD-TAO/DyCoke, llava/model/llava_arch.py,
dycole_ttm): frames are taken in windows of 4. The first frame of a window keeps all its tokens; frame
2 keeps the int((1 - K) N) tokens least similar (cosine, same position) to frame 1, frame 4 likewise
against frame 3, and frame 3 against frame 1, on the post-projector tokens; K is the pruning rate.
Its second stage, KV-cache pruning during decoding, acts inside the LLM after the first answer token
and is disabled, as for every method of the accuracy tables (each is counted at the LLM input).

A 4-frame window keeps at least a quarter of the tokens (the first frames), above the tables' 5% and
10% budgets, so the protocol's longer-window variant is used: windows of W frames with the same rules,
each frame at an odd offset compared with the frame before it and each frame at an even offset with the
window's first frame (W = 4 is exactly DyCoke's TTM). A full window keeps
(N + (W - 1) int((1 - K) N)) / (W N) of the tokens, e.g. W = 32: K = 0.98 -> ~5%, K = 0.93 -> ~10%.

Three choices where the released code leaves gaps or has quirks: a pruned frame keeps at least one token
(so every frame keeps its place in the prompt); kept tokens stay in position (raster) order (theirs are in
order of similarity); a last, shorter window follows the same rules (the released code drops the last
frame when the frame count is odd, and its third-frame loop stops one window early, leaving the last
window's third frame whole).

Tokens, positions and packing come from the StreamingTOM backends (vlmeval/streamingtom.py) without
their saliency: Qwen's M-RoPE gets every kept token at its own cell, Qwen3-VL's DeepStack features keep
the same tokens, Phi's 576 crop tokens get a row break after each row that has tokens (no global view).
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from vlmeval import streamingtom as stm

WINDOW, PRUNE = 4, 0.7          # the paper's window and its main pruning rate K
DEFAULTS = dict(window=WINDOW, prune=PRUNE)


# ---------------------------------------------------------------------------- TTM
def keep_count(n, prune):
    """Tokens a pruned frame keeps: int((1 - K) N), at least one."""
    return max(1, int((1 - prune) * n))


def reference(j):
    """Offset, within its window, of the frame that the frame at offset j is compared with (None for
    the first frame, kept whole): odd offsets the frame before, even offsets the window's first."""
    return None if j == 0 else (j - 1 if j % 2 else 0)


def least_similar(F_ref, F_t, n_keep):
    """Positions (sorted) of the n_keep tokens of F_t least similar to the same positions of F_ref."""
    s = F.cosine_similarity(F_t.float(), F_ref.float(), dim=-1)
    return sorted(torch.topk(s, min(n_keep, len(s)), largest=False).indices.tolist())


def ttm_plans(frames, window=WINDOW, prune=PRUNE):
    """DyCoke's TTM over (N, D) frame tokens in time order -> kept positions per frame."""
    plans = []
    for w0 in range(0, len(frames), window):
        win = frames[w0:w0 + window]
        for j, F_t in enumerate(win):
            r = reference(j)
            plans.append(list(range(len(F_t))) if r is None else least_similar(win[r], F_t, keep_count(len(F_t), prune)))
    return plans


def retention(window, prune, n):
    """Share of the tokens a full window keeps."""
    return (n + (window - 1) * keep_count(n, prune)) / (window * n)


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
    """Score questions on their frames with TTM-reduced tokens; resumable.

    conditions: {name: dict(window=W, prune=K)}.
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
                plans = ttm_plans([states[k]["tokens"] for k in keys], int(arg["window"]), float(arg["prune"]))
                toks[(c, keys)] = ([backend.pack(states[k], [(p, []) for p in plan], 0.0) for k, plan in zip(keys, plans)],
                                   sum(len(plan) for plan in plans) / max(1, sum(len(states[k]["tokens"]) for k in keys)))
            per, kept_share = toks[(c, keys)]
            try:
                p, n_tok = backend.score(it, per, build_messages)
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
                n_frames=len(per), n_image_tokens=int(sum(len(q[0]) for q in per)),
                kept_share=round(float(kept_share), 4), n_tokens=n_tok,
                dycoke=dict(window=int(arg["window"]), prune=float(arg["prune"])),
                sec=round(time.time() - t0, 3), error=err)) + "\n")
            fh.flush()
            n_new += 1
            if n_new % log_every == 0:
                rate = (time.time() - t_start) / n_new
                print(f"   {n_new}/{len(todo)}  {rate:.2f} s each  ~{rate * (len(todo) - n_new) / 60:.0f} min left"
                      f"  ({c}, {it['video']})", flush=True)
    print(f"done: {n_new} new in {(time.time() - t_start) / 60:.1f} min")

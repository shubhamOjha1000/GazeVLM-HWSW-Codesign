"""The two-view GazeVLM ("GazeVLM: Gaze-Guided Vision-Language Models for Efficient and Robust
Inference", anonymous ICLR 2026 submission) on every VLM in vlmeval. Not Chen & Qi's GazeVLM
(vlmeval/gazevlm.py).

As the authors' code (github.com/anonymous-zzz/GazeVLM, their modified transformers: LLaVA-NeXT's
image_processing_llava_next.divide_to_patches and modeling_llava_next.token_sampling_circle /
token_sampling_grid; single-inference-sample.py: alpha = beta = 0.5), per frame:

GazeVLM-Pre, two views. The square frame of side S; half-size c = int(S / 2 * alpha) of the local view;
the gaze pixel (x, y) = floor(gaze * S); the window centre is shifted inside the frame (x < c -> c,
x + c >= S -> S - c), the local view = frame[x - c : x + c, y - c : y + c] (alpha = 0.5: a quarter of the
area); the global view = the whole frame (their crop of half-size S / 2). Each view is encoded on its
own at the model's stock input. Their order is kept: the local view's tokens, then the global view's.
GazeVLM-Post, token selection on each view's token grid (n x n): the frame's budget B (capped at both
views' tokens) is split B // 2 per view; per view, first a uniform grid of B_v - round(beta B_v) tokens (stride n / ceil(sqrt(k)), offset stride / 2), then the tokens nearest the
gaze token (floor of the gaze in that view x n; Euclidean distance on the grid) until the view has B_v
distinct tokens; kept in raster order (np.unique).
With selection on, the authors drop LLaVA-NeXT's row-break (newline) tokens, so no row breaks here
either (Phi-3.5 included).

Per model, tokens come from the StreamingTOM backends (vlmeval/streamingtom.py, saliency off): each
view through the model's own (parity-checked) encoder path; the local view is the crop of the cached
frame saved losslessly (an uncompressed BMP in /dev/shm), so it takes exactly the frame's path. Positions: the 1D-position LLMs take
the tokens in that order; Qwen2.5-VL / Qwen3-VL (M-RoPE) get each token at its true place in the frame,
local-view tokens as cells of the crop's finer size (alpha x a stock token), so both views sit in one
image of the frame; Qwen3-VL's DeepStack features are selected with the same tokens. Both views cost the
encoder: twice the stock encoder work per frame.
"""

import json
import math
import os
import shutil
import tempfile
import time

import numpy as np
import torch

from vlmeval import gazevlm as gz
from vlmeval import streamingtom as stm

ALPHA, BETA = 0.5, 0.5          # the authors' gaze region ratio (crop_alpha) and gaze token ratio (gaze_beta)
DEFAULTS = dict(alpha=ALPHA, beta=BETA)


# ---------------------------------------------------------------------------- the method
def crop_window(gaze, S, alpha=ALPHA):
    """The local view of a square S px frame: (r0, c0, side) in px, and the gaze inside it as fractions
    (row, col) of the view. gaze = (u, v) in [0, 1] (u = column, v = row). As theirs, the window is placed
    from the floored gaze pixel, the gaze inside it from the unfloored one."""
    c = int(S / 2 * alpha)
    gx, gy = gaze[1] * S, gaze[0] * S                                  # row, column (px)
    x, y = math.floor(gx), math.floor(gy)
    if 2 * c >= S:
        return (0, 0, S), (gx / S, gy / S)
    xa = c if x - c < 0 else S - c if x + c >= S else x
    ya = c if y - c < 0 else S - c if y + c >= S else y
    r0, c0 = xa - c, ya - c
    return (r0, c0, 2 * c), ((gx - r0) / (2 * c), (gy - c0) / (2 * c))


def gaze_token(frac, n):
    """The token of an n x n grid the gaze falls in (their floor of the gaze in token units)."""
    return min(max(int(frac[0] * n), 0), n - 1), min(max(int(frac[1] * n), 0), n - 1)


def token_sampling_grid(n, m, num_points):
    """The authors' token_sampling_grid: num_points (row, col) on a uniform grid over n x m tokens."""
    stride_n = n / np.ceil(np.sqrt(num_points))
    stride_m = m / np.ceil(np.sqrt(num_points))
    coords = []
    for i in range(int(np.ceil(n / stride_n))):
        for j in range(int(np.ceil(m / stride_m))):
            if len(coords) >= num_points:
                break
            coords.append((int(min(n - 1, i * stride_n + stride_n / 2)), int(min(m - 1, j * stride_m + stride_m / 2))))
    return coords


def token_sampling_circle(n, m, x, y, budget, beta=BETA):
    """The authors' token_sampling_circle: budget distinct tokens of an n x m grid, budget - round(beta
    budget) on the uniform grid, the rest the nearest to the gaze token (x, y) (stable order on distance
    ties; theirs is numpy's default sort), until budget are distinct. -> (k, 2) (row, col), raster order."""
    budget = min(int(budget), n * m)
    if budget <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    grid = np.indices((n, m)).reshape(2, -1).T
    order = np.argsort(np.linalg.norm(grid - np.array([x, y]), axis=1), kind="stable")
    sel = []
    if budget - int(np.round(beta * budget)) > 0:
        sel.extend(token_sampling_grid(n, m, budget - int(np.round(beta * budget))))
    k = budget - len(sel)
    if k > 0:
        sel.extend(tuple(grid[i]) for i in order[:k])
        while len(np.unique(np.asarray(sel), axis=0)) != budget:
            sel.append(tuple(grid[order[k]]))
            k += 1
    return np.unique(np.asarray(sel), axis=0)


def select(n, gaze_rc, budget, beta=BETA):
    """Flat raster indices of the selected tokens of an n x n view."""
    rc = token_sampling_circle(n, n, gaze_rc[0], gaze_rc[1], budget, beta)
    return (rc[:, 0] * n + rc[:, 1]).astype(np.int64)


def view_budget(B, n_tokens):
    """Tokens per view: the frame budget capped at both views' tokens, halved (their token_total_single)."""
    return min(int(B), 2 * n_tokens) // 2


# ---------------------------------------------------------------------------- per-model backends
class _Views:
    """Shared: per-frame selection from the two views' (N, D) raster tokens; pack_views -> the frame's
    tuple for score(). 1D-position models: the local view's tokens, then the global view's."""
    saliency = False

    def pack_views(self, local, glob, idx_l, idx_g, window, S):
        return (torch.cat([local["tokens"][idx_l], glob["tokens"][idx_g]]),)


class _Qwen:
    def _cells(self, n, idx_l, idx_g, window, S, px):
        tok = px / n
        r0, c0, side = window                                           # source px; frame px = source px * px / S
        sub = side * px / (S * n)                                       # a local-view token's size in frame px
        y0, x0 = r0 * px / S, c0 * px / S
        loc = [(y0 + (k // n) * sub, x0 + (k % n) * sub, sub) for k in np.asarray(idx_l).tolist()]
        glo = [((k // n) * tok, (k % n) * tok, tok) for k in np.asarray(idx_g).tolist()]
        return np.array(loc + glo, dtype=np.float64).reshape(-1, 3)


class Qwen25(_Views, _Qwen, stm.Qwen25):
    def pack_views(self, local, glob, idx_l, idx_g, window, S):
        tokens = torch.cat([local["tokens"][idx_l], glob["tokens"][idx_g]])
        return tokens, self._cells(self.side, idx_l, idx_g, window, S, self.px), self.px


class Qwen3(_Views, _Qwen, stm.Qwen3):
    def pack_views(self, local, glob, idx_l, idx_g, window, S):
        tokens = torch.cat([local["tokens"][idx_l], glob["tokens"][idx_g]])
        deep = [torch.cat([a[idx_l], b[idx_g]]) for a, b in zip(local["deep"], glob["deep"])]
        return tokens, deep, self._cells(self.side, idx_l, idx_g, window, S, self.q3.IMG), self.q3.IMG


class InternVL(_Views, stm.InternVL):
    pass


class Gemma(_Views, stm.Gemma):
    pass


class Phi(_Views, stm.Phi):
    pass


class SmolVLM(_Views, stm.SmolVLM):
    pass


# ---------------------------------------------------------------------------- evaluation
def _load_rgb(path):
    from PIL import Image
    return Image.open(path).convert("RGB")


@torch.no_grad()
def run_questions(backend, items, conditions, out_path, build_messages, log_every=25):
    """Score questions on their frames, each reduced to its two views' selected tokens; resumable.

    conditions: {name: dict(tokens=B (per frame), alpha=..., beta=...)}.
    items: as for the *_fovea.run_questions (sorted by video). A frame's global view is encoded once per
    video, its local view once per (frame, alpha)."""
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                done.add((r["condition"], r["qid"]))
    todo = [(it, c) for it in items for c in conditions if (c, it["qid"]) not in done]
    print(f"{len(done)} already scored, {len(todo)} to go -> {out_path}")

    # the crop goes to the encoder through the frame path; an uncompressed BMP in RAM (/dev/shm where it exists)
    # keeps that lossless and adds no PNG compression to the timings
    tmp = tempfile.mkdtemp(prefix="gazevlm2_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
    states, video, t_start, n_new = {}, None, time.time(), 0
    try:
        with open(out_path, "a") as fh:
            for it, c in todo:
                if it["video"] != video:                     # new video: drop the old frames
                    states.clear()
                    torch.cuda.empty_cache()
                    video = it["video"]
                arg = {**DEFAULTS, **conditions[c]}
                alpha, beta, B = float(arg["alpha"]), float(arg["beta"]), int(arg["tokens"])
                t0 = time.time()
                per, n_missing = [], 0
                for fr in it["frames"]:
                    pts, missing = gz.gaze_points(fr, "fixation")
                    n_missing += int(missing)
                    if ("g", fr["key"]) not in states:
                        states[("g", fr["key"])] = backend.frame_state(fr)
                        states[("S", fr["key"])] = _load_rgb(fr["path"]).size[0]
                    S = states[("S", fr["key"])]
                    window, frac_l = crop_window(pts[0], S, alpha)
                    if ("l", fr["key"], alpha) not in states:
                        r0, c0, side = window
                        crop_path = os.path.join(tmp, "crop.bmp")
                        _load_rgb(fr["path"]).crop((c0, r0, c0 + side, r0 + side)).save(crop_path)
                        states[("l", fr["key"], alpha)] = backend.frame_state(dict(fr, path=crop_path))
                    glob, local = states[("g", fr["key"])], states[("l", fr["key"], alpha)]
                    n = backend.side
                    bv = view_budget(B, len(glob["tokens"]))
                    frac_g = (pts[0][1], pts[0][0])                    # (row, col) of the whole frame
                    idx_l = select(n, gaze_token(frac_l, n), bv, beta)
                    idx_g = select(n, gaze_token(frac_g, n), bv, beta)
                    per.append(backend.pack_views(local, glob, torch.as_tensor(idx_l), torch.as_tensor(idx_g), window, S))
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
                    n_frames=len(per), n_encoder_images=2 * len(per),
                    n_image_tokens=int(sum(len(q[0]) for q in per)), n_gaze_missing=n_missing, n_tokens=n_tok,
                    gazevlm2=dict(tokens=B, alpha=alpha, beta=beta), sec=round(time.time() - t0, 3),
                    error=err)) + "\n")
                fh.flush()
                n_new += 1
                if n_new % log_every == 0:
                    rate = (time.time() - t_start) / n_new
                    print(f"   {n_new}/{len(todo)}  {rate:.2f} s each  ~{rate * (len(todo) - n_new) / 60:.0f} min left"
                          f"  ({c}, {it['video']})", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"done: {n_new} new in {(time.time() - t_start) / 60:.1f} min")

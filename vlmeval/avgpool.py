"""Uniform K + average pooling on every VLM in vlmeval: the accuracy tables' simplest reduction.

Each of the K uniform frames is encoded at the model's stock input, then its token grid is
average-pooled (adaptive pooling, as torch's adaptive_avg_pool2d) to s x s tokens, s chosen so that a
frame's tokens come closest to budget x the model's stock tokens per frame (`pool_side`). Pooling after
the encoder saves LLM tokens, not encoder compute.

Where each model pools (the place its own token grid is formed; the same as GazeVLM's thumbnail,
vlmeval/gazevlm.py, whose backends encode the frames):
  Qwen2.5-VL / Qwen3-VL  its 16 x 16 LLM tokens; M-RoPE: each pooled token at the centre of its bin's
                         cells; Qwen3-VL's DeepStack features pooled the same way
  InternVL3.5            its 16 x 16 LLM tokens (after pixel shuffle and projector), raster order
  Gemma-3                its 64 x 64 SigLIP patch grid, before its own RMSNorm + projection (its stock
                         256 tokens are 4 x 4 averages of that grid), raster order
  Phi-3.5-vision         its 24 x 24 crop-token grid of the 672 px view, before projection, a row break
                         after each pooled row; no separator and no global view (as for the other
                         baselines on Phi), so a frame is s (s + 1) tokens
  SmolVLM2               its 9 x 9 LLM tokens, raster order
The protocol's optional second row, resizing before the encoder (which saves encoder compute), is the
fovea modules' uniform-downscaling baseline and is not repeated here.
"""

import json
import os
import time

import numpy as np
import torch

from vlmeval import gazevlm as gz

EMPTY = (0, 0, 0, 0)            # no ROI rows / columns: GazeVLM's frame tokens reduce to the thumbnail


def tokens_per_frame(s, row_breaks=False):
    return s * s + (s if row_breaks else 0)


def pool_side(budget, stock, side, row_breaks=False):
    """The s x s grid (1 <= s <= side) whose tokens per frame come closest to budget x stock tokens
    (the smaller s on a tie)."""
    target = budget * stock
    return min(range(1, side + 1), key=lambda s: (abs(tokens_per_frame(s, row_breaks) - target), s))


# ---------------------------------------------------------------------------- per-model backends
class Qwen25(gz.Qwen25):
    def pooled(self, grid, s):
        return self.frame_tokens(grid, EMPTY, s)


class Qwen3(gz.Qwen3):
    def pooled(self, grid, s):
        return self.frame_tokens(grid, EMPTY, s)


class InternVL(gz.InternVL):
    def pooled(self, grid, s):
        return self.frame_tokens(grid, EMPTY, s)


class Gemma(gz.Gemma):
    def pooled(self, grid, s):
        return (self.gf.project(self.pj, gz.thumbnail(grid, s).reshape(-1, grid.shape[-1])),)


class Phi(gz.Phi):
    row_breaks = True

    def pooled(self, grid, s):
        return (self.img.img_projection(self.ph.grid_rows(gz.thumbnail(grid, s), self.img.sub_GN)),)


class SmolVLM(gz.SmolVLM):
    def pooled(self, grid, s):
        return self.frame_tokens(grid, EMPTY, s)


# ---------------------------------------------------------------------------- evaluation
@torch.no_grad()
def run_questions(backend, items, conditions, out_path, build_messages, log_every=25):
    """Score questions on their frames, each pooled to s x s tokens; resumable.

    conditions: {name: dict(side=s)}.
    items: as for the *_fovea.run_questions (sorted by video). Each frame is encoded once per video and
    pooled once per (side, frame)."""
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                done.add((r["condition"], r["qid"]))
    todo = [(it, c) for it in items for c in conditions if (c, it["qid"]) not in done]
    print(f"{len(done)} already scored, {len(todo)} to go -> {out_path}")

    grids, toks, video, t_start, n_new = {}, {}, None, time.time(), 0
    with open(out_path, "a") as fh:
        for it, c in todo:
            if it["video"] != video:                         # new video: drop the old frames
                grids.clear()
                toks.clear()
                torch.cuda.empty_cache()
                video = it["video"]
            s = int(conditions[c]["side"])
            t0 = time.time()
            per = []
            for fr in it["frames"]:
                if fr["key"] not in grids:
                    grids[fr["key"]] = backend.frame_grid(fr)
                if (s, fr["key"]) not in toks:
                    toks[(s, fr["key"])] = backend.pooled(grids[fr["key"]], s)
                per.append(toks[(s, fr["key"])])
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
                n_frames=len(per), n_image_tokens=int(sum(len(q[0]) for q in per)), n_tokens=n_tok,
                avgpool=dict(side=s), sec=round(time.time() - t0, 3), error=err)) + "\n")
            fh.flush()
            n_new += 1
            if n_new % log_every == 0:
                rate = (time.time() - t_start) / n_new
                print(f"   {n_new}/{len(todo)}  {rate:.2f} s each  ~{rate * (len(todo) - n_new) / 60:.0f} min left"
                      f"  ({c}, {it['video']})", flush=True)
    print(f"done: {n_new} new in {(time.time() - t_start) / 60:.1f} min")

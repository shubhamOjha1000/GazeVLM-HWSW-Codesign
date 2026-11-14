"""The sweep's frame representations through each model's own encoder and prompt (sweep10.py). Diagnostic dev work only.

Conditions {name: (kind, arg)}:
  ("res", side)           the whole frame resized to side x side px before encoding, one normal image a frame: at the stock side this
                          is the model's own stock input, exactly as its fovea module's 'uniform' condition makes it
  ("twoview", arg)        arg = dict(gaze='gaze' | 'centre', tile=True): the whole frame at the stock side (exactly the stock input), then
                          a tile of half the source's side (672 of 1344 px) around the gaze, slid inside the frame, resized to the stock
                          side as the model resizes a frame: 2x the stock detail. Two normal images a frame under the frame's one
                          "Frame at <TIME ...>:" label. A frame without gaze gets the centre (v1's fallback), counted per record.
                          tile=False: the whole frame alone through this code, for the identity check against Uniform.
Every image is a contiguous picture through the model's own encoder and position handling, never cell by cell:
  SmolVLM2-2.2B and -256M  the processor's own input (stock_input), downscaled for a smaller side as smolvlm_fovea.encode_uniform does,
                          then the model's own encoder and connector (encode_pixels); a tile is the PIL crop through the processor
  Qwen2.5-VL and Qwen3-VL  load_frame01 at 1344 px and resize01 to the side, as their run_questions do, then encode_cells on the side's
                          own uniform cells (each frame its own image: its 2-frame temporal patch holds the frame twice); M-RoPE positions
                          per image
Frames load on threads ahead of the GPU, the next questions' too (run_views's workers and ahead): only the timing changes.
"""
import json
import os
import time

import numpy as np

TILE_SHARE = 0.5
AHEAD = 2


def tile_box(gaze, W, H, share=TILE_SHARE):
    """(x0, y0, x1, y1) in source px: a square of share x the frame's side centred on the gaze, slid inside the frame (never padded)."""
    src = int(round(min(W, H) * share))
    gx, gy = gaze if gaze is not None else (0.5, 0.5)
    x0 = int(round(min(max(gx * W - src / 2, 0), W - src)))
    y0 = int(round(min(max(gy * H - src / 2, 0), H - src)))
    return x0, y0, x0 + src, y0 + src


def tile_gaze(fr, arg):
    """(gaze for the tile, True if the frame's gaze was missing and the centre stood in)."""
    if arg.get("gaze", "gaze") == "centre":
        return (0.5, 0.5), False
    g = fr.get("gaze")
    return (g, False) if g is not None else ((0.5, 0.5), True)


def n_views(cond):
    kind, arg = cond
    return 1 if kind == "res" or not arg.get("tile", True) else 2


def view_messages(build_messages, it, views, px):
    """The shared prompt (qwen_mcq.build_messages) with each frame's image followed by its other views as images: one label a frame."""
    msgs = build_messages(it["question"], it["options"], [(fr["path"], fr["t"]) for fr in it["frames"]], px)
    content = []
    for c in msgs[0]["content"]:
        content.append(c)
        if c["type"] == "image":
            content += [dict(c) for _ in range(views - 1)]
    return [dict(msgs[0], content=content)]


# ---------------------------------------------------------------------------- per model
class Smol:
    """SmolVLM2-2.2B (smolvlm_fovea) or -256M (smolvlm256_fovea)."""

    def __init__(self, ad):
        from vlmeval import smolvlm256_fovea as s2
        from vlmeval import smolvlm_fovea as sm
        self.sm, self.m = sm, (s2 if ad.key == "smolvlm256" else sm)
        self.model, self.proc, self.letter_ids = ad.model, ad.proc, ad.letter_ids
        self.ip = ad.proc.image_processor
        self.mean, self.std = self.ip.image_mean, self.ip.image_std
        self.stock_side, self.px = self.m.IMG, self.m.IMG * self.m.IMG

    def stock(self, pil):
        return self.m.stock_input(self.ip, pil)

    def load(self, fr, cond):
        """CPU: ([one input per view], gaze missing)."""
        from PIL import Image
        from vlmeval import qwen_fovea as qf
        kind, arg = cond
        with Image.open(fr["path"]) as im:
            pil = im.convert("RGB")
        rec = self.stock(pil)
        if kind == "res":
            if arg == self.m.IMG:
                return [rec["pv"]], False
            return [self.m.normalize(qf.resize01(rec["x01"], arg), self.mean, self.std)], False
        g, missing = tile_gaze(fr, arg)
        views = [rec["pv"]]
        if arg.get("tile", True):
            views.append(self.stock(pil.crop(tile_box(g, *pil.size)))["pv"])
        return views, missing

    def encode(self, view):
        return (self.m.encode_pixels(self.model, view),)

    def build(self, msgs, enc):
        return self.m.build_inputs(self.model, self.proc, msgs, [e[0] for e in enc])

    def probs(self, inputs):
        return self.sm.letter_probs(self.model, inputs, self.letter_ids)

    def prompt_len(self, msgs, image_tokens):
        return int(self.m.prompt_ids(self.proc, msgs, list(image_tokens), self.m.TOKENIZE["add_special_tokens"]).shape[1])


class Qwen25:
    """Qwen2.5-VL (qwen_fovea): 28 px tokens, window attention."""

    def __init__(self, ad):
        from vlmeval import qwen_fovea as qf
        self.qf = qf
        self.model, self.proc, self.letter_ids = ad.model, ad.proc, ad.letter_ids
        ip = ad.proc.image_processor
        self.mean, self.std = ip.image_mean, ip.image_std
        self.vis = qf.visual_of(ad.model)
        qf.check_encoder(self.vis)
        self.stock_side, self.px = ad.SIDE, qf.IMG * qf.IMG

    def load(self, fr, cond):
        qf = self.qf
        kind, arg = cond
        x = qf.load_frame01(fr["path"], qf.IMG)
        if kind == "res":
            return [(qf.resize01(x, arg), arg)], False
        g, missing = tile_gaze(fr, arg)
        S = self.stock_side
        views = [(qf.resize01(x, S), S)]
        if arg.get("tile", True):
            x0, y0, x1, y1 = tile_box(g, x.shape[-1], x.shape[-2])
            views.append((qf.resize01(x[:, y0:y1, x0:x1].contiguous(), S), S))
        return views, missing

    def cells(self, side):
        return self.qf.uniform_cells(side)

    def encode(self, view):
        img, side = view
        cells = self.cells(side)
        return (self.qf.encode_cells(self.vis, img, cells, side, self.mean, self.std), cells, side)

    def build(self, msgs, enc):
        return self.qf.build_inputs(self.model, self.proc, msgs, [e[0] for e in enc], [e[1] for e in enc], [e[2] for e in enc])

    def probs(self, inputs):
        return self.qf.letter_probs(self.model, inputs, self.letter_ids)

    def prompt_len(self, msgs, image_tokens):
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        text = self.qf.expand_image_pads(text, [int(t) for t in image_tokens])
        return int(self.proc.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1])


class Qwen3(Qwen25):
    """Qwen3-VL (qwen3_fovea): 32 px tokens, full attention, DeepStack features beside each image's tokens."""

    def __init__(self, ad):
        from vlmeval import qwen_fovea as qf
        from vlmeval import qwen3_fovea as q3
        self.qf, self.q3 = qf, q3
        self.model, self.proc, self.letter_ids = ad.model, ad.proc, ad.letter_ids
        ip = ad.proc.image_processor
        self.mean, self.std = ip.image_mean, ip.image_std
        self.vis = q3.parts(ad.model)[0]
        q3.check_encoder(self.vis)
        self.stock_side, self.px = q3.IMG, q3.IMG * q3.IMG

    def cells(self, side):
        return self.q3.uniform_cells(side)

    def encode(self, view):
        img, side = view
        cells = self.cells(side)
        t, d = self.q3.encode_cells(self.vis, img, cells, side, self.mean, self.std)
        return (t, d, cells, side)

    def build(self, msgs, enc):
        return self.q3.build_inputs(self.model, self.proc, msgs, [e[0] for e in enc], [e[1] for e in enc], [e[2] for e in enc],
                                    [e[3] for e in enc])

    def probs(self, inputs):
        return self.q3.letter_probs(self.model, inputs, self.letter_ids)


def backend(ad):
    return {"smolvlm": Smol, "smolvlm256": Smol, "qwen2.5": Qwen25, "qwen3": Qwen3}[ad.key](ad)


# ---------------------------------------------------------------------------- scoring
def prompt_len(be, build_messages, question, options, frames, views):
    """Tokens of this code's prompt for these frames with no image tokens (the prompt fit: + image tokens per image)."""
    msgs = view_messages(build_messages, dict(question=question, options=list(options), frames=frames), views, be.px)
    return be.prompt_len(msgs, [0] * (len(frames) * views))


def run_views(be, items, conditions, out_path, workers=0, ahead=AHEAD, log_every=25):
    """Scores items under conditions {name: ('res', side) | ('twoview', arg)}; resumable, one record a (condition, question). With
    workers, each question's frames load on threads while the GPU encodes the question before (up to `ahead` questions ahead)."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from accuracy_tables.models import build_messages
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                done.add((r["condition"], r["qid"]))
    todo = [(it, c) for c in conditions for it in items if (c, it["qid"]) not in done]
    print(f"{len(done)} already scored, {len(todo)} to go -> {out_path}", flush=True)
    pool = ThreadPoolExecutor(max_workers=workers) if workers else None
    futures, t_start, n_new = {}, time.time(), 0
    try:
        with open(out_path, "a") as fh, torch.no_grad():
            for k, (it, c) in enumerate(todo):
                cond = conditions[c]
                if pool is not None:
                    for j in range(k, min(len(todo), k + 1 + ahead)):
                        if j not in futures:
                            futures[j] = [pool.submit(be.load, fr, conditions[todo[j][1]]) for fr in todo[j][0]["frames"]]
                    loaded = [f.result() for f in futures.pop(k)]
                else:
                    loaded = [be.load(fr, cond) for fr in it["frames"]]
                t0 = time.time()
                nv = n_views(cond)
                try:
                    enc = [be.encode(v) for views, _ in loaded for v in views]
                    inputs, ids = be.build(view_messages(build_messages, it, nv, be.px), enc)
                    p = be.probs(inputs)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    fh.write(json.dumps(dict(condition=c, qid=int(it["qid"]), video=it["video"], error="oom", n_frames=len(loaded))) + "\n")
                    continue
                err = None if np.isfinite(p).all() else "nan"
                pred = "ABCDE"[int(np.nanargmax(p))] if np.isfinite(p).any() else "?"
                fh.write(json.dumps(dict(
                    condition=c, qid=int(it["qid"]), video=it["video"], answer=it["answer"], pred=pred, correct=int(pred == it["answer"]),
                    p=[round(float(v), 4) for v in p], n_frames=len(loaded), n_views=nv, n_image_tokens=int(sum(len(e[0]) for e in enc)),
                    n_gaze_missing=int(sum(m for _, m in loaded)), n_tokens=int(ids.shape[1]), sec=round(time.time() - t0, 3),
                    error=err)) + "\n")
                fh.flush()
                n_new += 1
                del enc, inputs
                if n_new % log_every == 0:
                    rate = (time.time() - t_start) / n_new
                    print(f"   {n_new}/{len(todo)}  {rate:.2f} s each  ~{rate * (len(todo) - n_new) / 60:.0f} min left  ({c})", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    print(f"done: {n_new} new in {(time.time() - t_start) / 60:.1f} min", flush=True)

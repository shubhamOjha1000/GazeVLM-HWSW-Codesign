"""GazeVLM (Chen & Qi, arXiv 2509.16476), the gaze-crop baseline, on every VLM in vlmeval.

The method (single images, Qwen2.5-VL in the paper): the user's gaze trace is smoothed with a
Gaussian into a heatmap G over the image (sum 1); pixels are taken in descending order of G until
a fraction rho of the gaze mass is covered; the bounding box of those pixels, grown to at least
R_min, is the region of interest (ROI). The ROI crop goes to the VLM at high resolution with a
global thumbnail of the whole image (28 x 28 px), as two images, global view first.

Token-map version (the accuracy-tables protocol). A native-resolution crop saves tokens only on
the Qwen models; on the fixed-input models (InternVL's tile, Phi's crops, SmolVLM's 384 px image,
Gemma's 256 tokens per image) a crop plus a thumbnail costs more than the frame itself. So on all
six models each frame is encoded once at the model's stock input, the ROI is mapped onto its token
grid (28 px per token for Qwen2.5-VL / InternVL / Phi, 32 px Qwen3-VL, 42 px SmolVLM, 56 px
Gemma), the tokens the ROI touches are kept, and the whole grid average-pooled to 2 x 2 tokens is
the thumbnail. Per frame: [thumbnail, 4 tokens] + [ROI tokens, raster order]. Positions: Qwen's
M-RoPE gets each ROI token at its own (row, column) and each thumbnail token at the centre of the
area it pools; the 1D-position LLMs take the tokens in that order. Model specifics:

    Qwen3-VL   each token's DeepStack features are kept / pooled the same way
    Gemma-3    pooling happens where Gemma pools, on its 64 x 64 SigLIP patch grid before its
               projector: ROI tokens are its own 4 x 4-patch tokens, thumbnail tokens 32 x 32
    Phi-3.5    the 24 x 24 crop-token grid (before projection) is the frame's grid, and Phi's own
               layout is kept: the ROI rows, each with a row break, the separator, then the
               thumbnail rows as the global view (Phi's 156-token global view is dropped: alone it
               is 21% of a frame's 757 tokens, above the budgets compared)

GazeVLM is image-only, so it runs per frame on the Uniform K frames. Its gaze per frame is our
fixation estimate (vlmeval.spatial.gaze_fixation), or optionally the raw gaze samples of the
preceding second; rho is the budget knob. The authors' code is not public: SIGMA, R_MIN and the
heatmap grid are our choices, recorded with every result.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

HEAT = 224          # the heatmap grid (cells per side) the ROI is found on, independent of the model
SIGMA = 0.1         # std of the Gaussian per gaze point, fraction of the frame side
R_MIN = 0.125       # smallest ROI side, fraction of the frame side (56 px on a 448 px frame, the
                    # smallest image Qwen2.5-VL's processor makes: 2 x 2 tokens)
THUMB = 2           # the global thumbnail: THUMB x THUMB tokens
DEFAULTS = dict(sigma=SIGMA, r_min=R_MIN, thumb=THUMB, gaze="fixation")


# ---------------------------------------------------------------------------- the method
def gaze_heatmap(points, n=HEAT, sigma=SIGMA):
    """(n, n) heatmap, rows = y, of gaze points (u, v) in [0, 1] (u right, v down): one isotropic
    Gaussian of std sigma (fraction of the frame side) per point, normalized to sum 1 over the
    frame."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    c = (np.arange(n) + 0.5) / n
    gx = np.exp(-((c[None, :] - pts[:, :1]) ** 2) / (2 * sigma ** 2))       # (P, n) along x
    gy = np.exp(-((c[None, :] - pts[:, 1:2]) ** 2) / (2 * sigma ** 2))      # (P, n) along y
    G = np.einsum("py,px->yx", gy, gx)
    return G / G.sum()


def mass_box(G, rho):
    """Bounding box (x0, y0, x1, y1), fractions of the frame, of the smallest set of heatmap cells,
    taken in descending order of G, whose mass reaches rho."""
    n = G.shape[0]
    flat = G.ravel()
    order = np.argsort(-flat, kind="stable")
    k = min(int(np.searchsorted(np.cumsum(flat[order]), rho)) + 1, flat.size)
    ys, xs = np.unravel_index(order[:k], G.shape)
    return xs.min() / n, ys.min() / n, (xs.max() + 1) / n, (ys.max() + 1) / n


def grow_box(box, r_min=R_MIN):
    """The box grown about its centre to at least r_min per side, moved back inside the frame."""
    def side(a, b):
        if b - a >= r_min:
            return a, b
        m = (a + b) / 2
        a, b = m - r_min / 2, m + r_min / 2
        if a < 0:
            a, b = 0.0, r_min
        if b > 1:
            a, b = 1.0 - r_min, 1.0
        return a, b
    x0, y0, x1, y1 = box
    (x0, x1), (y0, y1) = side(x0, x1), side(y0, y1)
    return x0, y0, x1, y1


def roi_box(points, rho, sigma=SIGMA, r_min=R_MIN):
    """GazeVLM's ROI for a frame: the rho-mass box of the gaze heatmap, at least r_min per side."""
    return grow_box(mass_box(gaze_heatmap(points, sigma=sigma), rho), r_min)


def box_on_grid(box, g):
    """Token rows [r0, r1) and columns [c0, c1) of a g x g token grid that the box touches."""
    x0, y0, x1, y1 = box
    eps = 1e-6
    c0, r0 = int(np.floor(x0 * g + eps)), int(np.floor(y0 * g + eps))
    c1, r1 = int(np.ceil(x1 * g - eps)), int(np.ceil(y1 * g - eps))
    c0, r0 = min(max(c0, 0), g - 1), min(max(r0, 0), g - 1)
    return r0, max(min(r1, g), r0 + 1), c0, max(min(c1, g), c0 + 1)


def thumb_bins(g, t=THUMB):
    """[lo, hi) of the t bins that adaptive average pooling uses along g cells."""
    return [(int(np.floor(i * g / t)), int(np.ceil((i + 1) * g / t))) for i in range(t)]


def layout(g, rc, t=THUMB):
    """A frame's tokens as cells (y, x, size) in token units: the t x t thumbnail bins (raster;
    each its top-left corner and height, a square when t divides g), then the ROI tokens rc =
    (r0, r1, c0, c1) in raster order."""
    bins = thumb_bins(g, t)
    thumb = [(a, c, b - a) for (a, b) in bins for (c, _) in bins]
    r0, r1, c0, c1 = rc
    return np.array(thumb + [(r, c, 1) for r in range(r0, r1) for c in range(c0, c1)], dtype=np.int64)


def thumbnail(grid, t=THUMB):
    """(g, g, D) -> (t, t, D): the grid average-pooled (adaptive pooling)."""
    return F.adaptive_avg_pool2d(grid.permute(2, 0, 1).unsqueeze(0).float(), t)[0].permute(1, 2, 0).to(grid.dtype)


def pick(grid, rc, t=THUMB):
    """(g, g, D) token grid -> (t * t + n_roi, D): the thumbnail (raster), then the ROI tokens."""
    r0, r1, c0, c1 = rc
    D = grid.shape[-1]
    return torch.cat([thumbnail(grid, t).reshape(-1, D), grid[r0:r1, c0:c1].reshape(-1, D)])


def phi_sequences(grid, rc, sub_gn, t=THUMB):
    """Phi's two parts before projection from its (24, 24, C) crop-token grid: the ROI tokens with
    a row break after each row (Phi's HD part) and the t x t thumbnail with its row breaks (in place
    of Phi's global view); phi_fovea.image_embeds puts the separator between them."""
    from vlmeval import phi_fovea as ph
    r0, r1, c0, c1 = rc
    return ph.grid_rows(grid[r0:r1, c0:c1], sub_gn), ph.grid_rows(thumbnail(grid, t), sub_gn)


def gaze_points(fr, source="fixation"):
    """(gaze points of a frame for its heatmap, True if it had none and the centre is used).
    source: 'fixation' = the frame's one fixation estimate; 'trace' = its raw samples of the
    preceding second (fr['gaze_trace']), falling back to the fixation."""
    if source == "trace" and fr.get("gaze_trace"):
        return list(fr["gaze_trace"]), False
    if fr.get("gaze") is not None:
        return [fr["gaze"]], False
    return [(0.5, 0.5)], True


# ---------------------------------------------------------------------------- per-model backends
class Backend:
    """Glue to one model's vlmeval module. frame_grid(fr) encodes a frame at the model's stock
    input, once per frame and video; frame_tokens(grid, rc, t) gives the frame's GazeVLM tokens
    for ROI token rows / columns rc on its side x side grid, a tuple whose first item is the
    (N, D) LLM-input tokens; score(item, per_frame, build_messages) -> (probabilities over A-E,
    prompt length in tokens)."""
    side = 16
    max_pixels = 448 * 448

    def _messages(self, it, build_messages):
        return build_messages(it["question"], it["options"], [(fr["path"], fr["t"]) for fr in it["frames"]],
                              self.max_pixels)


class Qwen25(Backend):
    """Qwen2.5-VL at its stock 448 px frame: 16 x 16 tokens of 28 px, M-RoPE positions from cells."""
    def __init__(self, model, proc, letter_ids):
        from vlmeval import qwen_fovea as qf
        self.qf, self.model, self.proc, self.letter_ids = qf, model, proc, letter_ids
        self.vis = qf.visual_of(model)
        qf.check_encoder(self.vis)
        self.mean, self.std = proc.image_processor.image_mean, proc.image_processor.image_std
        self.px, self.max_pixels = 448, qf.IMG * qf.IMG
        self.side = self.px // qf.BASE

    def frame_grid(self, fr):
        qf = self.qf
        img = qf.resize01(qf.load_frame01(fr["path"], qf.IMG), self.px)
        return qf.encode_cells(self.vis, img, qf.uniform_cells(self.px), self.px, self.mean, self.std
                               ).view(self.side, self.side, -1)

    def frame_tokens(self, grid, rc, t=THUMB):
        return pick(grid, rc, t), layout(self.side, rc, t) * self.qf.BASE, self.px

    def score(self, it, per, build_messages):
        inputs, ids = self.qf.build_inputs(self.model, self.proc, self._messages(it, build_messages),
                                           [p[0] for p in per], [p[1] for p in per], [p[2] for p in per])
        return self.qf.letter_probs(self.model, inputs, self.letter_ids), int(ids.shape[1])


class Qwen3(Backend):
    """Qwen3-VL at its 512 px frame: 16 x 16 tokens of 32 px, DeepStack features alongside."""
    def __init__(self, model, proc, letter_ids):
        from vlmeval import qwen_fovea as qf, qwen3_fovea as q3
        self.qf, self.q3, self.model, self.proc, self.letter_ids = qf, q3, model, proc, letter_ids
        self.vis = q3.parts(model)[0]
        q3.check_encoder(self.vis)
        self.mean, self.std = proc.image_processor.image_mean, proc.image_processor.image_std
        self.side, self.max_pixels = q3.IMG // q3.BASE, qf.IMG * qf.IMG

    def frame_grid(self, fr):
        qf, q3, s = self.qf, self.q3, self.side
        img = qf.resize01(qf.load_frame01(fr["path"], qf.IMG), q3.IMG)
        tok, deep = q3.encode_cells(self.vis, img, q3.uniform_cells(q3.IMG), q3.IMG, self.mean, self.std)
        return tok.view(s, s, -1), [d.view(s, s, -1) for d in deep]

    def frame_tokens(self, grid, rc, t=THUMB):
        tok, deep = grid
        return pick(tok, rc, t), [pick(d, rc, t) for d in deep], layout(self.side, rc, t) * self.q3.BASE, self.q3.IMG

    def score(self, it, per, build_messages):
        inputs, ids = self.q3.build_inputs(self.model, self.proc, self._messages(it, build_messages),
                                           [p[0] for p in per], [p[1] for p in per], [p[2] for p in per],
                                           [p[3] for p in per])
        return self.q3.letter_probs(self.model, inputs, self.letter_ids), int(ids.shape[1])


class InternVL(Backend):
    """InternVL3.5 at its 448 px tile: 16 x 16 tokens of 28 px (after pixel shuffle + projector)."""
    def __init__(self, model, proc, letter_ids):
        from vlmeval import qwen_fovea as qf, internvl_fovea as iv
        self.qf, self.iv, self.model, self.proc, self.letter_ids = qf, iv, model, proc, letter_ids
        self.mean, self.std = proc.image_processor.image_mean, proc.image_processor.image_std
        self.side, self.max_pixels = iv.TILE // iv.BASE, iv.TILE * iv.TILE

    def frame_grid(self, fr):
        img = self.qf.load_frame01(fr["path"], self.qf.IMG)
        return self.iv.encode_uniform(self.model, img, self.iv.TILE, self.mean, self.std).view(self.side, self.side, -1)

    def frame_tokens(self, grid, rc, t=THUMB):
        return (pick(grid, rc, t),)

    def score(self, it, per, build_messages):
        inputs, ids = self.iv.build_inputs(self.model, self.proc, self._messages(it, build_messages), [p[0] for p in per])
        return self.iv.letter_probs(self.model, inputs, self.letter_ids), int(ids.shape[1])


class Gemma(Backend):
    """Gemma-3 at 896 px: tokens are 4 x 4 averages of the 64 x 64 SigLIP patch grid, projected."""
    def __init__(self, model, proc, letter_ids):
        from vlmeval import qwen_fovea as qf, gemma_fovea as gf
        self.qf, self.gf, self.model, self.proc, self.letter_ids = qf, gf, model, proc, letter_ids
        self.vt, self.pj = gf.parts(model)
        self.mean, self.std = proc.image_processor.image_mean, proc.image_processor.image_std
        self.side, self.max_pixels = gf.GRID // gf.BASE, gf.G_IMG * gf.G_IMG

    def frame_grid(self, fr):
        qf, gf = self.qf, self.gf
        img = qf.resize01(qf.load_frame01(fr["path"], qf.IMG), gf.G_IMG)
        return gf.vision_grids(self.vt, [img], self.mean, self.std)[0]               # (64, 64, C) patches

    def frame_tokens(self, grid, rc, t=THUMB):
        cells = layout(self.side, rc, t) * self.gf.BASE                              # token units -> patches
        return (self.gf.project(self.pj, self.gf.pool_cells(grid, cells)),)

    def score(self, it, per, build_messages):
        inputs, ids = self.gf.build_inputs(self.model, self.proc, self._messages(it, build_messages), [p[0] for p in per])
        return self.gf.letter_probs(self.model, inputs, self.letter_ids), int(ids.shape[1])


class Phi(Backend):
    """Phi-3.5-vision: the 24 x 24 crop-token grid of its 672 px view (28 px tokens, before projection)."""
    def __init__(self, img, llm, tok, letter_ids):
        from vlmeval import phi_fovea as ph
        self.ph, self.img, self.llm, self.tok, self.letter_ids = ph, img, llm, tok, letter_ids
        self.side, self.max_pixels = ph.GRID, ph.CROP * ph.CROP

    def frame_grid(self, fr):
        hd01, _ = self.ph.views(self.ph.load_pil(fr["path"]))
        return self.ph.crop_grid(self.img, hd01)

    def frame_tokens(self, grid, rc, t=THUMB):
        return (self.ph.image_embeds(self.img, *phi_sequences(grid, rc, self.img.sub_GN, t)),)

    def score(self, it, per, build_messages):
        inputs, ids = self.ph.build_inputs(self.llm, self.tok, self._messages(it, build_messages), [p[0] for p in per])
        return self.ph.letter_probs(self.llm, inputs, self.letter_ids), int(ids.shape[1])


class SmolVLM(Backend):
    """SmolVLM2 at its processor's 384 px image: 9 x 9 tokens of 42 px."""
    def __init__(self, model, proc, letter_ids):
        from vlmeval import smolvlm_fovea as sm
        self.sm, self.model, self.proc, self.letter_ids = sm, model, proc, letter_ids
        self.ip = proc.image_processor
        self.mean, self.std = self.ip.image_mean, self.ip.image_std
        self.side, self.max_pixels = sm.GRID, sm.IMG * sm.IMG

    def frame_grid(self, fr):
        from PIL import Image
        rec = self.sm.stock_input(self.ip, Image.open(fr["path"]).convert("RGB"))
        return self.sm.encode_uniform(self.model, rec, self.sm.IMG, self.mean, self.std).view(self.side, self.side, -1)

    def frame_tokens(self, grid, rc, t=THUMB):
        return (pick(grid, rc, t),)

    def score(self, it, per, build_messages):
        inputs, ids = self.sm.build_inputs(self.model, self.proc, self._messages(it, build_messages), [p[0] for p in per])
        return self.sm.letter_probs(self.model, inputs, self.letter_ids), int(ids.shape[1])


# ---------------------------------------------------------------------------- evaluation
@torch.no_grad()
def run_questions(backend, items, conditions, out_path, build_messages, log_every=25):
    """Score questions on their frames with GazeVLM tokens; resumable.

    conditions: {name: dict(rho=..., and optionally sigma, r_min, thumb, gaze='fixation' | 'trace')}.
    items: dicts with qid, video, question, options, answer, frames (key, path, t, gaze, and
    gaze_trace for gaze='trace'), sorted by video: each frame is encoded once per video and its
    tokens made once per (condition, frame)."""
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
            arg = {**DEFAULTS, **conditions[c]}
            t0 = time.time()
            per = []
            for fr in it["frames"]:
                key = (c, fr["key"])
                if key not in toks:
                    if fr["key"] not in grids:
                        grids[fr["key"]] = backend.frame_grid(fr)
                    pts, missing = gaze_points(fr, arg["gaze"])
                    box = roi_box(pts, arg["rho"], arg["sigma"], arg["r_min"])
                    rc = box_on_grid(box, backend.side)
                    toks[key] = (backend.frame_tokens(grids[fr["key"]], rc, arg["thumb"]), box, rc, missing)
                per.append(toks[key])
            try:
                p, n_tok = backend.score(it, [q[0] for q in per], build_messages)
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
                n_frames=len(per), n_image_tokens=int(sum(len(q[0][0]) for q in per)),
                n_roi_tokens=int(sum((q[2][1] - q[2][0]) * (q[2][3] - q[2][2]) for q in per)),
                roi_area=round(float(np.mean([(q[1][2] - q[1][0]) * (q[1][3] - q[1][1]) for q in per])), 4) if per else 0.0,
                n_gaze_missing=int(sum(q[3] for q in per)), n_tokens=n_tok,
                gazevlm={k: arg[k] for k in ("rho", "sigma", "r_min", "thumb", "gaze")},
                sec=round(time.time() - t0, 3), error=err)) + "\n")
            fh.flush()
            n_new += 1
            if n_new % log_every == 0:
                rate = (time.time() - t_start) / n_new
                print(f"   {n_new}/{len(todo)}  {rate:.2f} s each  ~{rate * (len(todo) - n_new) / 60:.0f} min left"
                      f"  ({c}, {it['video']})", flush=True)
    print(f"done: {n_new} new in {(time.time() - t_start) / 60:.1f} min")

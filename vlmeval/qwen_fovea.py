"""Gaze-foveated tokenization inside Qwen2.5-VL's own vision encoder (no training).

Qwen2.5-VL cuts an image into 14 px patches, runs its ViT (window attention in 112 px
windows, full attention in a few layers, 2D RoPE on the patch grid) and merges each 2x2
group of patches into one LLM token. So one token = one 28 px cell.

Foveation here keeps that unit and changes what a cell covers:

    fovea    28 px cells at full resolution           (exactly what Qwen normally sees)
    ring     56 px cells shrunk to 28 px              (one token for 4x the area)
    outside  112 px cells shrunk to 28 px             (one token for 16x the area)

on a 1344 px frame (12 x 112; the AEA 1408 px frame resized). The layout is the spatial
module's quadtree rule (apt-vlm `foveation.py`, re-implemented in `vlmeval.spatial`), with
28 px as the base patch: a cell of size 2^i * 28 is kept if the eccentricity of its centre
from the gaze is at least ring i-1 (FOVI rings), else split into four.

Each cell becomes 4 patches of 14 px (its 28 px image split 2 x 2), so Qwen's patch
embedding and merger are used unchanged. What has to change is WHERE each patch is:

    2D RoPE    a shrunk cell's 4 patches get the coordinates of the centres of the cell's
               four quadrants on the 14 px grid (fractional for 56 / 112 px cells); 28 px
               cells get exactly Qwen's integer coordinates. RoPE is a function of the
               coordinate, so no table has to be interpolated (unlike APT's learned
               position embedding).
    windows    each cell's patches join the 112 px attention window that contains the
               cell (cells are aligned to their size, so they never straddle windows).
    LLM        the M-RoPE positions of the image tokens are the cell centres on the
               28 px token grid; text after the image continues as after a full-size
               1344 px image.

The encoder is run part by part (patch embedding, blocks, merger: `encode_cells`) with the
layout's RoPE coordinates and window groups, so nothing depends on how a given transformers
version writes the encoder's forward(); the LLM input is `inputs_embeds` + explicit 3D
`position_ids`. With a layout of only 28 px cells both reduce to Qwen's own computation, which
the parity checks verify against the stock forward (`tests/test_qwen_fovea.py` on a tiny
random model, and `parity_report` on the real one).

Training-free assumption to keep in mind: Qwen never saw fractional patch positions or
shrunk content next to full-resolution content. The parity checks only prove the plumbing;
the keyframe evaluation measures whether the model copes.
"""

import inspect
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from vlmeval import spatial as sp

PATCH, BASE, WINDOW = 14, 28, 112
IMG = 1344
IMAGE_PAD = "<|image_pad|>"


# ---------------------------------------------------------------------------- layouts
def foveated_cells(gu, gv, a=40.0, img=IMG, num_scales=3, fov=sp.FOV, R_deg=None):
    """Cells (y, x, size) in pixels, top-left corners, raster order, for gaze (gu, gv) in
    [0, 1]. Sizes 28 * 2^i, i < num_scales; the largest must fit a 112 px window.
    Rings: FOVI recipe with fall-off a and maximum eccentricity R_deg (default fov / 2, as in
    the spatial module); a larger R_deg pushes every ring out (more tokens)."""
    sizes = [BASE * 2 ** i for i in range(num_scales)]
    if sizes[-1] > WINDOW:
        raise ValueError(f"largest cell {sizes[-1]} px exceeds Qwen's {WINDOW} px attention window")
    if img % sizes[-1]:
        raise ValueError(f"image size {img} is not a multiple of the largest cell {sizes[-1]}")
    rings = sp.ring_radii_deg(R_deg or fov / 2, a, num_scales)
    n = img // BASE
    covered = np.zeros((n, n), bool)
    cells = []
    for i in range(num_scales - 1, 0, -1):
        k = 2 ** i
        g = n // k
        c = (np.arange(g) + 0.5) * BASE * k
        X, Y = np.meshgrid(c, c)
        keep = sp.ecc_deg(X, Y, gu * img, gv * img, img, fov) >= rings[i - 1]
        keep &= ~covered.reshape(g, k, g, k).any(axis=(1, 3))
        cells += [(r * k * BASE, q * k * BASE, k * BASE) for r, q in np.argwhere(keep)]
        covered |= np.repeat(np.repeat(keep, k, 0), k, 1)
    cells += [(r * BASE, q * BASE, BASE) for r, q in np.argwhere(~covered)]
    return sort_raster(np.array(cells, dtype=np.int64))


def uniform_cells(img, base=BASE):
    """All base px cells of an img x img frame: Qwen's normal tokenization (28 px cells for
    Qwen2.5-VL; models with another token cell pass their own)."""
    if img % base:
        raise ValueError(f"image size {img} is not a multiple of {base}")
    r = np.arange(img // base) * base
    Y, X = np.meshgrid(r, r, indexing="ij")
    return np.stack([Y.ravel(), X.ravel(), np.full(Y.size, base)], 1).astype(np.int64)


def sort_raster(cells):
    """Order cells by top-left corner, row-major (Qwen's token order for a uniform grid)."""
    return cells[np.lexsort((cells[:, 1], cells[:, 0]))]


def same_tokens_size(n_tokens, base=BASE):
    """Side (px, multiple of base) of the uniform frame whose token count is closest to n."""
    return base * max(1, int(round(np.sqrt(n_tokens))))


# ---------------------------------------------------------------------------- patches
def cells_to_patches(img01, cells, mean, std, temporal=2, patch=PATCH):
    """img01: (3, H, W) float in [0, 1]. Returns (4N, 3 * temporal * patch * patch) patches in
    Qwen's processor layout: each cell resized to 2 x 2 patches, cells in the given order, each
    cell's patches row-major, features ordered (channel, time, row, column), the image repeated
    over time. patch = 14 px for Qwen2.5-VL (28 px cells)."""
    N = len(cells)
    base = 2 * patch
    crops = torch.empty(N, 3, base, base, dtype=torch.float32)
    for s in np.unique(cells[:, 2]):
        idx = np.nonzero(cells[:, 2] == s)[0]
        c = torch.stack([img01[:, y:y + s, x:x + s] for y, x, _ in cells[idx]]).float()
        if s != base:
            c = F.interpolate(c, size=(base, base), mode="bicubic", antialias=True,
                              align_corners=False).clamp(0, 1)
        crops[torch.as_tensor(idx)] = c
    m = torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    sd = torch.as_tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    crops = (crops - m) / sd
    p = crops.reshape(N, 3, 2, patch, 2, patch).permute(0, 2, 4, 1, 3, 5)   # N, mh, mw, C, py, px
    p = p.unsqueeze(4).expand(N, 2, 2, 3, temporal, patch, patch)
    return p.reshape(N * 4, 3 * temporal * patch * patch)


def patch_positions(cells, patch=PATCH):
    """(h, w) of each patch on the patch grid, float: the centre of the cell quadrant the
    patch shows, in Qwen's convention that patch index i covers [patch i, patch (i + 1)).
    Integer (= Qwen's own) for cells of 2 x 2 patches (28 px on Qwen2.5-VL's 14 px grid)."""
    c = torch.as_tensor(cells, dtype=torch.float64)
    y, x, s = c[:, :1], c[:, 1:2], c[:, 2:]
    qi = torch.tensor([[0.5, 0.5, 1.5, 1.5]], dtype=torch.float64)
    qj = torch.tensor([[0.5, 1.5, 0.5, 1.5]], dtype=torch.float64)
    h = (y + qi * s / 2 - patch / 2) / patch
    w = (x + qj * s / 2 - patch / 2) / patch
    return h.reshape(-1).float(), w.reshape(-1).float()


def window_groups(cells, img):
    """Qwen's window attention for a layout: (order of cells grouped by their 112 px window,
    cumulative patch counts per window starting at 0). Windows are counted row-major over
    ceil(img / 112) columns, so partial windows at the edge match Qwen's padded ones."""
    nw = -(-img // WINDOW)
    wid = (cells[:, 0] // WINDOW) * nw + cells[:, 1] // WINDOW
    order = np.argsort(wid, kind="stable")
    counts = np.bincount(wid, minlength=nw * nw)
    counts = counts[counts > 0]
    return torch.as_tensor(order, dtype=torch.long), [0] + (np.cumsum(counts) * 4).tolist()


# ---------------------------------------------------------------------------- vision encoder
def visual_of(model):
    vis = getattr(model, "visual", None)
    return vis if vis is not None else model.model.visual


def check_encoder(vis):
    """The parts encode_cells runs itself, so it does not depend on how this transformers
    version writes the encoder's forward(): patch_embed, blocks, merger, and blocks that take
    cu_seqlens plus RoPE as position_embeddings (cos, sin) or rotary_pos_emb."""
    for name in ("patch_embed", "blocks", "merger"):
        if not hasattr(vis, name):
            raise RuntimeError(f"{type(vis).__name__} has no {name!r}: this transformers version "
                               f"needs a different encoder loop")
    _rope_kw(vis)


def _rope_kw(vis):
    """Which keyword the blocks take RoPE by: position_embeddings (cos, sin), as Qwen2.5-VL's
    encoder passes it, or the older rotary_pos_emb."""
    params = inspect.signature(type(vis.blocks[0]).forward).parameters
    for name in ("position_embeddings", "rotary_pos_emb"):
        if name in params:
            return name
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):    # wrapped forward(*args, **kwargs)
        return "position_embeddings"
    raise RuntimeError(f"{type(vis.blocks[0]).__name__}.forward{tuple(params)}: no way to pass RoPE")


def _as_tensor(out):
    if torch.is_tensor(out):
        return out
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        return pooled
    if isinstance(out, (tuple, list)):
        return out[0]
    return out.last_hidden_state


def _inv_freq(vis):
    """Qwen's vision RoPE frequencies: the buffer if present, else the same formula
    (dim = head_dim / 2, theta = 10000)."""
    inv = getattr(getattr(vis, "rotary_pos_emb", None), "inv_freq", None)
    if inv is not None:
        return inv.float()
    cfg = vis.config
    dim = cfg.hidden_size // cfg.num_heads // 2
    return 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))


def _full_attention_blocks(vis):
    idx = getattr(vis, "fullatt_block_indexes", None)
    return set(idx if idx is not None else vis.config.fullatt_block_indexes)


def _run_blocks(vis, x, rot, cu_window, cu_full):
    """Qwen's ViT blocks: window attention in most, full attention in fullatt_block_indexes."""
    emb = torch.cat([rot, rot], dim=-1)
    pos = (emb.cos(), emb.sin())
    full = _full_attention_blocks(vis)
    rope = _rope_kw(vis)
    for i, blk in enumerate(vis.blocks):
        kw = dict(cu_seqlens=cu_full if i in full else cu_window)
        kw[rope] = pos if rope == "position_embeddings" else rot
        out = blk(x, **kw)
        x = out[0] if isinstance(out, (tuple, list)) else out
    return x


@torch.no_grad()
def encode_cells(vis, img01, cells, img, mean, std):
    """Qwen vision features (one row per cell, in `cells` order) for any cell layout.

    Runs the encoder's own parts in Qwen2.5-VL's order: patch embedding -> regroup the cells
    by attention window -> blocks (window / full attention, 2D RoPE) -> merger (2 x 2 patches
    -> one token) -> back to the input order. Only the RoPE coordinates and the window groups
    come from the layout; with all 28 px cells both equal Qwen's own (checked against the
    stock forward in tests/test_qwen_fovea.py and parity_report)."""
    check_encoder(vis)
    dev = next(vis.parameters()).device
    dtype = next(vis.parameters()).dtype
    N = len(cells)
    x = vis.patch_embed(cells_to_patches(img01, cells, mean, std).to(dev, dtype))     # (4N, D)
    h, w = patch_positions(cells)
    inv = _inv_freq(vis).to(dev)
    rot = torch.cat([torch.outer(h.to(dev), inv), torch.outer(w.to(dev), inv)], dim=-1)
    order, cu = window_groups(cells, img)
    order = order.to(dev)
    x = x.view(N, 4, -1)[order].reshape(4 * N, -1)
    rot = rot.view(N, 4, -1)[order].reshape(4 * N, -1)
    cu_window = torch.unique_consecutive(torch.tensor(cu, dtype=torch.int32, device=dev))
    cu_full = torch.tensor([0, 4 * N], dtype=torch.int32, device=dev)
    x = vis.merger(_run_blocks(vis, x, rot, cu_window, cu_full))                       # (N, out)
    return x[torch.argsort(order)]


@torch.no_grad()
def encode_stock(vis, img01, mean, std):
    """Qwen's own forward (the reference) for a frame whose side is a multiple of 28."""
    img = img01.shape[-1]
    dev = next(vis.parameters()).device
    dtype = next(vis.parameters()).dtype
    patches = cells_to_patches(img01, uniform_cells(img), mean, std).to(dev, dtype)
    grid = torch.tensor([[1, img // PATCH, img // PATCH]], device=dev)
    return _as_tensor(vis(patches, grid_thw=grid))


# ---------------------------------------------------------------------------- LLM input
def expand_image_pads(text, n_tokens):
    """One <|image_pad|> per image in the chat template -> n_tokens[i] copies, as the
    processor does from image_grid_thw."""
    parts = text.split(IMAGE_PAD)
    if len(parts) - 1 != len(n_tokens):
        raise ValueError(f"{len(parts) - 1} image placeholders for {len(n_tokens)} images")
    return parts[0] + "".join(IMAGE_PAD * n + p for n, p in zip(n_tokens, parts[1:]))


def mrope_positions(input_ids, image_token_id, cells_list, img_list, base=BASE):
    """(3, 1, L) float M-RoPE positions (time, height, width) in Qwen2.5-VL's scheme (also
    Qwen3-VL's for images): text advances all three by 1; an image's tokens share the current
    start and add the token row / column; text after an image starts at start + grid size.
    Image token row / column = the cell centre on the base px token grid (integer for base px
    cells)."""
    ids = input_ids[0].tolist()
    out, cur, i, k = [], 0.0, 0, 0
    while i < len(ids):
        if ids[i] == image_token_id:
            c = np.asarray(cells_list[k], dtype=np.float64)
            s = c[:, 2] / base
            hh = c[:, 0] / base + (s - 1) / 2
            ww = c[:, 1] / base + (s - 1) / 2
            out += [(cur, cur + a, cur + b) for a, b in zip(hh, ww)]
            cur += img_list[k] // base
            i += len(c)
            k += 1
        else:
            out.append((cur, cur, cur))
            cur += 1
            i += 1
    if k != len(cells_list):
        raise ValueError(f"found {k} images in the prompt, expected {len(cells_list)}")
    return torch.tensor(out, dtype=torch.float32).T.unsqueeze(1)


@torch.no_grad()
def build_inputs(model, proc, messages, feats_list, cells_list, img_list):
    """inputs_embeds, position_ids, attention_mask for a prompt whose images are given as
    precomputed vision features (one row per cell) with their layouts."""
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = expand_image_pads(text, [len(c) for c in cells_list])
    ids = proc.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    img_id = proc.tokenizer.convert_tokens_to_ids(IMAGE_PAD)
    emb_layer = model.get_input_embeddings()
    ids = ids.to(emb_layer.weight.device)
    emb = emb_layer(ids)
    mask = ids[0] == img_id
    feats = torch.cat(feats_list).to(emb.dtype)
    if int(mask.sum()) != len(feats):
        raise ValueError(f"{int(mask.sum())} image tokens in the prompt, {len(feats)} features")
    emb[0, mask] = feats
    pos = mrope_positions(ids.cpu(), img_id, cells_list, img_list).to(emb.device)
    return dict(inputs_embeds=emb, position_ids=pos,
                attention_mask=torch.ones_like(ids)), ids


@torch.no_grad()
def letter_probs(model, inputs, letter_ids, all_positions=False):
    """Softmax over A-E of the raw next-token logits (no generation-time penalties, so it
    differs slightly from qwen_mcq.score, which goes through generate()).
    all_positions=True runs the LM head over the whole sequence, the shape the stock forward
    computes (parity checks); otherwise only over the last position (logits_to_keep=1): the full
    logits are sequence x vocabulary, e.g. 7.5 GB in bf16 for 96 frames on a 152k vocabulary."""
    kw = {} if all_positions else dict(logits_to_keep=1)
    logits = model(**inputs, use_cache=False, **kw).logits[0, -1, letter_ids].float()
    return torch.softmax(logits, -1).cpu().numpy()


# ---------------------------------------------------------------------------- frames
def load_frame01(path, size=IMG):
    """(3, size, size) float in [0, 1] from a cached frame (resized with antialiasing if
    the cache has a different size)."""
    import cv2
    im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
    if t.shape[-1] != size or t.shape[-2] != size:
        t = F.interpolate(t[None], size=(size, size), mode="bicubic", antialias=True,
                          align_corners=False)[0].clamp(0, 1)
    return t


def resize01(img01, size):
    if img01.shape[-1] == size:
        return img01
    return F.interpolate(img01[None], size=(size, size), mode="bicubic", antialias=True,
                         align_corners=False)[0].clamp(0, 1)


def to_uint8_pil(img01):
    from PIL import Image
    return Image.fromarray((img01.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------- checks on the real model
@torch.no_grad()
def parity_report(model, proc, letter_ids, img01, question, options, build_messages, size=448):
    """Foveation plumbing with a uniform layout vs Qwen's own pipeline on one real frame.

    1. vision: our encoder path (overrides active, all 28 px cells) vs the stock encoder
    2. positions: our M-RoPE vs the model's get_rope_index (when it can be called)
    3. answer: letter probabilities, ours vs processor + model forward on the same image
    Returns a dict of the measured differences."""
    vis = visual_of(model)
    check_encoder(vis)
    ip = proc.image_processor
    mean, std = ip.image_mean, ip.image_std
    x = to_uint8_pil(resize01(img01, size))                # the exact image both paths see
    x01 = torch.from_numpy(np.asarray(x)).permute(2, 0, 1).float() / 255.0
    cells = uniform_cells(size)

    f_ours = encode_cells(vis, x01, cells, size, mean, std).float()
    f_stock = encode_stock(vis, x01, mean, std).float()
    cos_v = F.cosine_similarity(f_ours, f_stock, dim=-1)

    msgs = build_messages(question, options, [("frame.jpg", 0.0)], size * size)
    inputs, ids = build_inputs(model, proc, msgs, [f_ours.to(f_stock.dtype)], [cells], [size])
    p_ours = letter_probs(model, inputs, letter_ids, all_positions=True)

    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    stock = proc(text=[text], images=[x], return_tensors="pt").to(model.device)
    if int(stock["image_grid_thw"][0, 1]) * PATCH != size:
        raise RuntimeError(f"the processor resized the image: grid {stock['image_grid_thw'].tolist()}")
    pv_ours = cells_to_patches(x01, cells, mean, std)
    pix_diff = (pv_ours - stock["pixel_values"].float().cpu()).abs().max().item()
    p_stock = torch.softmax(model(**stock, use_cache=False).logits[0, -1, letter_ids].float(), -1).cpu().numpy()

    rep = dict(pixel_max_abs_diff=pix_diff,
               vision_cos_min=cos_v.min().item(), vision_max_abs_diff=(f_ours - f_stock).abs().max().item(),
               tokens_ids_equal=bool(torch.equal(ids.cpu(), stock["input_ids"].cpu())),
               probs_max_abs_diff=float(np.abs(p_ours - p_stock).max()),
               same_answer=bool(p_ours.argmax() == p_stock.argmax()))
    rope = getattr(model, "get_rope_index", None) or getattr(getattr(model, "model", None), "get_rope_index", None)
    if rope is not None:
        try:
            pos_stock = rope(input_ids=stock["input_ids"], image_grid_thw=stock["image_grid_thw"],
                             attention_mask=stock["attention_mask"])[0]
            rep["positions_max_abs_diff"] = (inputs["position_ids"].float().cpu()
                                             - pos_stock.float().cpu()).abs().max().item()
        except Exception as ex:                          # signature differs across versions
            rep["positions_max_abs_diff"] = f"not checked ({type(ex).__name__})"
    return rep


@torch.no_grad()
def fovea_feature_drift(vis, img01, cells, mean, std):
    """How far foveated features move from full-resolution ones, per cell size: cosine of
    each cell's token with the mean of the full-resolution (all 28 px) tokens it covers.
    28 px cells see the same pixels, so only the changed context moves them."""
    img = img01.shape[-1]
    full = encode_stock(vis, img01, mean, std).float()
    n = img // BASE
    grid = full.view(n, n, -1)
    fov = encode_cells(vis, img01, cells, img, mean, std).float()
    rows = []
    for (y, x, s), f in zip(cells, fov):
        y0, x0, k = y // BASE, x // BASE, s // BASE
        ref = grid[y0:y0 + k, x0:x0 + k].reshape(-1, grid.shape[-1]).mean(0)
        rows.append((int(s), F.cosine_similarity(f, ref, dim=0).item()))
    import pandas as pd
    return pd.DataFrame(rows, columns=["cell_px", "cos_to_full_res"]).groupby("cell_px").agg(
        cells=("cos_to_full_res", "size"), mean_cos=("cos_to_full_res", "mean"),
        min_cos=("cos_to_full_res", "min"))


# ---------------------------------------------------------------------------- evaluation
def frame_layout(conditions, cond, fr):
    """(cells, frame side px, gaze used or None) for one frame under one condition.

    fr: dict with gaze (u, v) or None and gaze_other (u, v) or None.
    conditions: {name: (kind, arg)} with kind in
        'uniform'        arg = side in px (multiple of 28): Qwen's normal tokenization
        'uniform_match'  arg = name of a 'fovea' condition: a uniform frame with the same
                         token count as that condition gives THIS frame
        'fovea'          arg = dict(a=..., gaze='gaze' | 'centre' | 'other', and optionally
                         img=frame side (default 1344), R=max eccentricity (default fov / 2),
                         num_scales=3)
    A missing gaze falls back to the image centre (counted in the results)."""
    kind, arg = conditions[cond]
    if kind == "uniform":
        return uniform_cells(arg), arg, None
    if kind == "uniform_match":
        cells, _, _ = frame_layout(conditions, arg, fr)
        side = same_tokens_size(len(cells))
        return uniform_cells(side), side, None
    src = arg["gaze"]
    g = (0.5, 0.5) if src == "centre" else fr.get("gaze" if src == "gaze" else "gaze_other")
    used = g
    if g is None:
        g = (0.5, 0.5)
    img = arg.get("img", IMG)
    cells = foveated_cells(*g, a=arg["a"], img=img, num_scales=arg.get("num_scales", 3),
                           R_deg=arg.get("R"))
    return cells, img, used


@torch.no_grad()
def run_questions(model, proc, letter_ids, items, conditions, out_path, build_messages,
                  log_every=25):
    """Score multiple-choice questions whose frames are tokenized per condition; resumable.

    items: dicts with qid, video, question, options, answer, and frames: a list of dicts
           (key, path to the 1344 px frame, t seconds, gaze, gaze_other). Items of the same
           video should be adjacent: each frame's vision features are computed once per
           video and condition, then reused by every question of that video.
    """
    vis = visual_of(model)
    check_encoder(vis)
    ip = proc.image_processor
    mean, std = ip.image_mean, ip.image_std
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                done.add((r["condition"], r["qid"]))
    todo = [(it, c) for it in items for c in conditions if (c, it["qid"]) not in done]
    print(f"{len(done)} already scored, {len(todo)} to go -> {out_path}")

    feats, video, t_start, n_new = {}, None, time.time(), 0
    with open(out_path, "a") as fh:
        for it, c in todo:
            if it["video"] != video:                      # new video: drop the old features
                feats.clear()
                torch.cuda.empty_cache()
                video = it["video"]
            t0 = time.time()
            per = []
            for fr in it["frames"]:
                key = (c, fr["key"])
                if key not in feats:
                    cells, size, used = frame_layout(conditions, c, fr)
                    img01 = resize01(load_frame01(fr["path"], IMG), size)
                    feats[key] = (encode_cells(vis, img01, cells, size, mean, std), cells, size,
                                  conditions[c][0] == "fovea" and used is None)
                per.append(feats[key])
            try:
                msgs = build_messages(it["question"], it["options"],
                                      [(fr["path"], fr["t"]) for fr in it["frames"]], IMG * IMG)
                inputs, ids = build_inputs(model, proc, msgs, [p[0] for p in per],
                                           [p[1] for p in per], [p[2] for p in per])
                p = letter_probs(model, inputs, letter_ids)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                fh.write(json.dumps(dict(condition=c, qid=int(it["qid"]), video=it["video"],
                                         error="oom", n_frames=len(per))) + "\n")
                continue
            err = None if np.isfinite(p).all() else "nan"
            pred = "ABCDE"[int(np.nanargmax(p))] if np.isfinite(p).any() else "?"
            sizes = np.concatenate([q[1][:, 2] for q in per]) if per else np.zeros(0, int)
            fh.write(json.dumps(dict(
                condition=c, qid=int(it["qid"]), video=it["video"], answer=it["answer"],
                pred=pred, correct=int(pred == it["answer"]), p=[round(float(v), 4) for v in p],
                n_frames=len(per), n_image_tokens=int(sum(len(q[1]) for q in per)),
                cells_by_size={str(s): int((sizes == s).sum()) for s in np.unique(sizes)},
                frame_px=sorted({int(q[2]) for q in per}), n_gaze_missing=int(sum(q[3] for q in per)),
                n_tokens=int(ids.shape[1]), sec=round(time.time() - t0, 3), error=err)) + "\n")
            fh.flush()
            n_new += 1
            if n_new % log_every == 0:
                rate = (time.time() - t_start) / n_new
                print(f"   {n_new}/{len(todo)}  {rate:.2f} s each  ~{rate * (len(todo) - n_new) / 60:.0f} min left"
                      f"  ({c}, {it['video']})", flush=True)
    print(f"done: {n_new} new in {(time.time() - t_start) / 60:.1f} min")

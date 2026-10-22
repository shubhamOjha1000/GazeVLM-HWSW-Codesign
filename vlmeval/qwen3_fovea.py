"""Gaze-foveated tokenization inside Qwen3-VL-4B's own vision encoder (no training).

Qwen3-VL cuts an image into 16 px patches, adds a learned 48 x 48 position table resampled
(bilinear) to the image's patch grid, runs its ViT (full attention, 2D RoPE on the patch grid)
and merges each 2 x 2 group of patches into one LLM token: one token = one 32 px cell.
DeepStack: the features after three intermediate blocks (5, 11, 17 on the 4B model) are merged
the same way and added to the image tokens' hidden states after the LLM's first three layers.

Foveation keeps that unit, as for Qwen2.5-VL (vlmeval/qwen_fovea.py), before the encoder:

    fovea      32 px cells, full resolution    (exactly Qwen3-VL's own patches)
    ring       64 px cells shrunk to 32 px     (one token for 4x the area)
    periphery  128 px cells shrunk to 32 px    (one token for 16x the area)

on a 512 px frame: 16 x 16 = 256 tokens at stock, as on the other three models. The layout is
qwen_fovea.foveated_cells(img=448) scaled by 32 / 28: Qwen3-VL's token grid on 512 px is
Qwen2.5-VL's on 448 px, so the same rings give the same cells and token counts on all four.

Each cell becomes 2 x 2 patches of 16 px, placed at the centres of the cell's quadrants on the
32 x 32 patch grid (integer = Qwen3-VL's own for 32 px cells, fractional for shrunk ones):

    position table  read there with Qwen3-VL's own bilinear resampling arithmetic
    2D RoPE         at the same coordinates
    attention       full attention over the frame's patches (Qwen3-VL has no windows)
    DeepStack       a cell's 4 patches are consecutive, so every DeepStack merger gives one
                    feature per cell, added at that cell's token
    LLM             M-RoPE positions of the image tokens = the cell centres on the 16 x 16 token
                    grid; the LLM runs through the model's own text model with inputs_embeds,
                    explicit 3D position_ids and the DeepStack features, then its LM head

The encoder is run part by part (patch embedding, position table, blocks, DeepStack mergers,
merger), so nothing depends on how a transformers version writes its forward(). With an
all-32 px layout every step reduces to Qwen3-VL's own computation, which the parity checks
verify (tests/test_qwen3_fovea.py on tiny random models, parity_report on the real one).
"""

import json
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from vlmeval import qwen_fovea as qf

MODEL_CANDIDATES = ("Qwen/Qwen3-VL-4B-Instruct",)
PATCH, BASE, IMG = 16, 32, 512


# ---------------------------------------------------------------------------- layouts
def foveated_cells(gu, gv, a=40.0, R_deg=None, num_scales=3):
    """Cells (y, x, size) in px on the 512 px frame for gaze (gu, gv) in [0, 1]: the 448 px
    layout of the other models (qwen_fovea.foveated_cells) scaled by 32 / 28, so sizes are
    32 / 64 / 128 px and the cells and token counts are the same as theirs."""
    c = qf.foveated_cells(gu, gv, a=a, img=IMG // BASE * qf.BASE, num_scales=num_scales, R_deg=R_deg)
    return c * BASE // qf.BASE


def uniform_cells(side):
    """All 32 px cells of a side x side frame: Qwen3-VL's own tokenization."""
    return qf.uniform_cells(side, base=BASE)


def same_tokens_size(n_tokens):
    """Side (px, multiple of 32) of the uniform frame whose token count is closest to n."""
    return qf.same_tokens_size(n_tokens, base=BASE)


# ---------------------------------------------------------------------------- vision encoder
def parts(model):
    """(vision encoder, text model, LM head) of Qwen3VLForConditionalGeneration."""
    inner = getattr(model, "model", model)
    vis = getattr(inner, "visual", None)
    lm = getattr(inner, "language_model", None)
    if vis is None or lm is None:
        raise RuntimeError(f"{type(model).__name__} has no visual / language_model")
    return vis, lm, model.lm_head


def check_encoder(vis):
    """The parts encode_cells runs itself, so it does not depend on how this transformers
    version writes the encoder's forward()."""
    for name in ("patch_embed", "pos_embed", "blocks", "merger", "deepstack_merger_list",
                 "deepstack_visual_indexes"):
        if not hasattr(vis, name):
            raise RuntimeError(f"{type(vis).__name__} has no {name!r}: this transformers version "
                               f"needs a different encoder loop")


def _axis_taps(p, grid, side):
    """Qwen3-VL's bilinear (align_corners) resampling of a side-long position-table axis to a
    grid-long patch axis, read at patch coordinates p (fractional allowed): two table indices
    and their weights per coordinate, with the float32 arithmetic of transformers' own
    resampling, so integer coordinates give exactly its values."""
    src = p.float() * (side - 1) / max(grid - 1, 1)
    floor = torch.floor(src)
    offsets = torch.arange(0, 2, device=p.device)
    taps = (floor.long()[:, None] + offsets).clamp(0, side - 1)
    dist = (src[:, None] - floor[:, None] - offsets).abs()
    return taps, (1 - dist).clamp(min=0)


def sample_positions(vis, h, w, grid):
    """Learned position embeddings (N, C) at patch coordinates (h, w) of a grid x grid patch
    grid: the side x side table resampled to the grid as Qwen3-VL does, read bilinearly."""
    side = int(round(vis.pos_embed.num_embeddings ** 0.5))
    ht, hw = _axis_taps(h, grid, side)
    wt, ww = _axis_taps(w, grid, side)
    idx = (ht[:, :, None] * side + wt[:, None, :]).reshape(-1, 4)
    wgt = (hw[:, :, None] * ww[:, None, :]).reshape(-1, 4)
    dev = vis.pos_embed.weight.device
    return (vis.pos_embed(idx.to(dev)) * wgt.to(dev)[:, :, None]).sum(1)


def _rope(vis, h, w):
    """(cos, sin) of Qwen3-VL's 2D vision RoPE at patch coordinates (h, w): the h frequencies,
    then the w frequencies, twice over the head dimension (its own values at integers)."""
    inv = qf._inv_freq(vis).to(h.device)
    rot = torch.cat([torch.outer(h, inv), torch.outer(w, inv)], dim=-1)
    emb = torch.cat([rot, rot], dim=-1)
    return emb.cos(), emb.sin()


def _run_blocks(vis, x, pos, cu):
    """Qwen3-VL's ViT blocks (full attention within cu_seqlens), with the DeepStack mergers
    after the blocks in deepstack_visual_indexes: (last hidden states, [DeepStack features])."""
    idx = list(vis.deepstack_visual_indexes)
    deep = []
    for i, blk in enumerate(vis.blocks):
        out = blk(x, cu_seqlens=cu, position_embeddings=pos)
        x = out[0] if isinstance(out, (tuple, list)) else out
        if i in idx:
            deep.append(vis.deepstack_merger_list[idx.index(i)](x))
    return x, deep


@torch.no_grad()
def encode_cells(vis, img01, cells, img, mean, std):
    """Qwen3-VL image tokens (N, D) and DeepStack features [(N, D)] * 3, one row per cell in
    `cells` order, for any 32/64/128 px layout of an img x img frame (img sets the patch grid
    the position table is resampled to: img / 16 patches per side).

    Runs the encoder's own parts in Qwen3-VL's order: patch embedding + position table ->
    blocks (full attention, 2D RoPE; DeepStack mergers after their blocks) -> merger (2 x 2
    patches -> one token). Cells go through in raster order (Qwen3-VL's own token order for a
    uniform grid) and come back in the given order."""
    check_encoder(vis)
    p = next(vis.parameters())
    order = np.lexsort((cells[:, 1], cells[:, 0]))
    c = cells[order]
    x = vis.patch_embed(qf.cells_to_patches(img01, c, mean, std, patch=PATCH).to(p.device, p.dtype))
    h, w = qf.patch_positions(c, patch=PATCH)
    h, w = h.to(p.device), w.to(p.device)
    x = x + sample_positions(vis, h, w, img // PATCH).to(x.dtype)
    cu = torch.tensor([0, len(x)], dtype=torch.int32, device=p.device)
    x, deep = _run_blocks(vis, x, _rope(vis, h, w), cu)
    back = torch.as_tensor(np.argsort(order), device=p.device)
    return vis.merger(x)[back], [d[back] for d in deep]


def _vision_out(out):
    """(image tokens, [DeepStack features]) from the encoder's forward: an output object with
    pooler_output and deepstack_features (transformers 5) or the tuple (tokens, features)."""
    if hasattr(out, "deepstack_features"):
        return out.pooler_output, list(out.deepstack_features)
    out = tuple(out)
    return (out[0], list(out[1])) if len(out) == 2 else (out[1], list(out[-1]))


@torch.no_grad()
def encode_stock(vis, img01, mean, std):
    """Qwen3-VL's own encoder forward (the reference) on our patches of a frame whose side is a
    multiple of 32: (image tokens, DeepStack features)."""
    side = img01.shape[-1]
    p = next(vis.parameters())
    patches = qf.cells_to_patches(img01, uniform_cells(side), mean, std, patch=PATCH).to(p.device, p.dtype)
    grid = torch.tensor([[1, side // PATCH, side // PATCH]], device=p.device)
    return _vision_out(vis(patches, grid_thw=grid))


# ---------------------------------------------------------------------------- LLM input
@torch.no_grad()
def inputs_from_ids(model, ids, feats_list, deep_list, cells_list, img_list):
    """Text-model inputs for token ids whose image tokens are given as precomputed rows (one per
    cell) with their layouts: inputs_embeds, position_ids (3D M-RoPE at the cell centres),
    attention_mask, and the DeepStack features at the image tokens (visual_pos_masks)."""
    img_id = model.config.image_token_id
    emb_layer = model.get_input_embeddings()
    ids = ids.to(emb_layer.weight.device)
    emb = emb_layer(ids)
    mask = ids == img_id
    n = int(mask.sum())
    if n != sum(len(f) for f in feats_list):
        raise ValueError(f"{n} image tokens in the prompt, {sum(len(f) for f in feats_list)} features")
    deep = None
    if n:
        emb[mask] = torch.cat(feats_list).to(emb.device, emb.dtype)
        deep = [torch.cat([d[k] for d in deep_list]).to(emb.device, emb.dtype)
                for k in range(len(deep_list[0]))]
    pos = qf.mrope_positions(ids.cpu(), img_id, cells_list, img_list, base=BASE).to(emb.device)
    return dict(inputs_embeds=emb, position_ids=pos, attention_mask=torch.ones_like(ids),
                visual_pos_masks=mask if n else None, deepstack_visual_embeds=deep)


@torch.no_grad()
def build_inputs(model, proc, messages, feats_list, deep_list, cells_list, img_list):
    """inputs_from_ids for a chat prompt: one <|image_pad|> per image -> one per cell."""
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = qf.expand_image_pads(text, [len(c) for c in cells_list])
    ids = proc.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    return inputs_from_ids(model, ids, feats_list, deep_list, cells_list, img_list), ids


@torch.no_grad()
def next_token_logits(model, inputs, all_positions=False):
    """Next-token logits through Qwen3-VL's own text model (DeepStack included) and LM head.
    all_positions runs the head over the whole sequence, the shape the stock forward computes
    (same rounding; parity), else only over the last position (memory)."""
    _, lm, head = parts(model)
    out = lm(**inputs, use_cache=False)
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    h = h if all_positions else h[:, -1:]
    return head(h.to(head.weight.device))[0, -1]


@torch.no_grad()
def letter_probs(model, inputs, letter_ids, all_positions=False):
    """Softmax over A-E of the raw next-token logits."""
    logits = next_token_logits(model, inputs, all_positions)[letter_ids].float()
    return torch.softmax(logits, -1).cpu().numpy()


def load_model(model_id):
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as Model
    except ImportError:
        from transformers import AutoModelForImageTextToText as Model
    kw = dict(device_map="auto", attn_implementation="sdpa")
    try:
        model = Model.from_pretrained(model_id, dtype=torch.bfloat16, **kw)
    except TypeError:
        model = Model.from_pretrained(model_id, torch_dtype=torch.bfloat16, **kw)
    proc = AutoProcessor.from_pretrained(model_id)
    ids = []
    for L in "ABCDE":
        t = proc.tokenizer.encode(L, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"letter {L!r} is not a single token: {t}")
        ids.append(t[0])
    return model.eval(), proc, ids


# ---------------------------------------------------------------------------- checks on the real model
@torch.no_grad()
def parity_report(model, proc, letter_ids, img01, question, options, build_messages, size=IMG):
    """Our foveation path with an all-32 px layout vs Qwen3-VL's own pipeline, on one real frame.

      vision  our part-by-part encoder vs the stock encoder forward on the same patches: image
              tokens and DeepStack features, in fp32 (TF32 off, exact attention) and in bf16
      pixels  our patches vs the processor's pixel_values
      LLM     our prompt + inputs_embeds + M-RoPE + DeepStack carrying Qwen3-VL's own image
              tokens (from the processor's pixels) vs the processor's one-call API + model:
              token ids, positions (when get_rope_index can be called), answer probabilities
      ours    our whole path in bf16 vs the stock model (expected identical)"""
    vis, _, _ = parts(model)
    check_encoder(vis)
    ip = proc.image_processor
    mean, std = ip.image_mean, ip.image_std
    x = qf.to_uint8_pil(qf.resize01(img01, size))             # the exact image both paths see
    x01 = torch.from_numpy(np.array(x)).permute(2, 0, 1).float() / 255.0
    cells = uniform_cells(size)
    t16, d16 = encode_cells(vis, x01, cells, size, mean, std)
    s16, sd16 = encode_stock(vis, x01, mean, std)

    dtype = next(vis.parameters()).dtype
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        exact_attention = sdpa_kernel(SDPBackend.MATH)
    except Exception:
        exact_attention = nullcontext()
    try:                                                     # vision parity in true fp32
        vis.float()
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        with exact_attention:
            t32, d32 = encode_cells(vis, x01, cells, size, mean, std)
            s32, sd32 = encode_stock(vis, x01, mean, std)
    finally:
        vis.to(dtype)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32

    msgs = build_messages(question, options, [("frame.jpg", 0.0)], size * size)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    stock = proc(text=[text], images=[x], return_tensors="pt").to(model.device)
    grid = stock["image_grid_thw"]
    if int(grid[0, 1]) * PATCH != size or int(grid[0, 2]) * PATCH != size:
        raise RuntimeError(f"the processor resized the image: grid {grid.tolist()}")
    pv_ours = qf.cells_to_patches(x01, cells, mean, std, patch=PATCH)
    pix = (pv_ours - stock["pixel_values"].float().cpu()).abs().max().item()
    p_stock = torch.softmax(model(**stock, use_cache=False).logits[0, -1, letter_ids].float(), -1).cpu().numpy()

    st, sdeep = _vision_out(vis(stock["pixel_values"].to(dtype), grid_thw=grid))   # Qwen3-VL's own tokens
    inputs, ids = build_inputs(model, proc, msgs, [st], [sdeep], [cells], [size])
    p_llm = letter_probs(model, inputs, letter_ids, all_positions=True)
    inputs16, _ = build_inputs(model, proc, msgs, [t16], [d16], [cells], [size])
    p_ours = letter_probs(model, inputs16, letter_ids, all_positions=True)

    cos = lambda a, b: F.cosine_similarity(a.float(), b.float(), dim=-1).min().item()
    mx = lambda a, b: (a.float() - b.float()).abs().max().item()
    rep = dict(pixel_max_abs_diff=pix,
               vision_fp32_cos_min=cos(t32, s32), vision_fp32_max_abs_diff=mx(t32, s32),
               deepstack_fp32_cos_min=min(cos(a, b) for a, b in zip(d32, sd32)),
               vision_bf16_max_abs_diff=mx(t16, s16),
               deepstack_bf16_max_abs_diff=max(mx(a, b) for a, b in zip(d16, sd16)),
               tokens_ids_equal=bool(torch.equal(ids.cpu(), stock["input_ids"].cpu())),
               llm_probs_max_abs_diff=float(np.abs(p_llm - p_stock).max()),
               llm_same_answer=bool(p_llm.argmax() == p_stock.argmax()),
               ours_bf16_probs_max_abs_diff=float(np.abs(p_ours - p_stock).max()),
               ours_bf16_same_answer=bool(p_ours.argmax() == p_stock.argmax()))
    rope = getattr(getattr(model, "model", model), "get_rope_index", None)
    if rope is not None:
        kw = dict(input_ids=stock["input_ids"], image_grid_thw=grid, attention_mask=stock["attention_mask"])
        tries = ([dict(kw, mm_token_type_ids=stock["mm_token_type_ids"])] if "mm_token_type_ids" in stock
                 else []) + [kw]
        for kw_ in tries:                                    # the signature differs across versions
            try:
                pos_stock = rope(**kw_)[0]
                rep["positions_max_abs_diff"] = (inputs["position_ids"].float().cpu()
                                                 - pos_stock.float().cpu()).abs().max().item()
                break
            except Exception as ex:
                rep["positions_max_abs_diff"] = f"not checked ({type(ex).__name__})"
    return rep


@torch.no_grad()
def fovea_feature_drift(vis, img01, cells, mean, std):
    """Cosine of each foveated token with the mean of Qwen3-VL's own tokens over the same area
    (all-32 px layout of the same frame), per cell size."""
    import pandas as pd
    side = img01.shape[-1]
    n = side // BASE
    full = encode_stock(vis, img01, mean, std)[0].float().view(n, n, -1)
    fov = encode_cells(vis, img01, cells, side, mean, std)[0].float()
    rows = []
    for (y, x, s), f in zip(cells, fov):
        y0, x0, k = y // BASE, x // BASE, s // BASE
        ref = full[y0:y0 + k, x0:x0 + k].reshape(-1, full.shape[-1]).mean(0)
        rows.append((int(s), F.cosine_similarity(f, ref, dim=0).item()))
    return pd.DataFrame(rows, columns=["cell_px", "cos_to_stock"]).groupby("cell_px").agg(
        cells=("cos_to_stock", "size"), mean_cos=("cos_to_stock", "mean"), min_cos=("cos_to_stock", "min"))


# ---------------------------------------------------------------------------- evaluation
def frame_layout(conditions, cond, fr):
    """(cells, frame side px, gaze used or None) for one frame under one condition.

    conditions: {name: (kind, arg)} with kind in
        'uniform'        arg = side px (multiple of 32): Qwen3-VL's own input at that size
                         (512 = stock, 256 tokens)
        'fovea'          arg = dict(a=..., R=..., gaze='gaze' | 'centre', num_scales=3), on the
                         512 px frame
        'uniform_match'  arg = name of a 'fovea' condition: Qwen3-VL's own input downscaled to
                         the side whose token grid is closest to that condition's count here
    A missing gaze falls back to the image centre (counted in the results)."""
    kind, arg = conditions[cond]
    if kind == "uniform":
        return uniform_cells(arg), arg, None
    if kind == "uniform_match":
        cells, _, _ = frame_layout(conditions, arg, fr)
        side = same_tokens_size(len(cells))
        return uniform_cells(side), side, None
    g = (0.5, 0.5) if arg["gaze"] == "centre" else fr.get("gaze")
    used = g
    if g is None:
        g = (0.5, 0.5)
    return foveated_cells(*g, a=arg["a"], R_deg=arg.get("R"), num_scales=arg.get("num_scales", 3)), IMG, used


@torch.no_grad()
def run_questions(model, proc, letter_ids, items, conditions, out_path, build_messages, log_every=25):
    """Score questions on their frames under each condition; resumable.

    items: dicts with qid, video, question, options, answer and frames (key, path to the 1344 px
    frame, t seconds, gaze). Work runs video by video and, within a video, condition by
    condition: each frame's tokens and DeepStack features (4x the memory of the tokens alone)
    are computed once per (video, condition) and reused by every question of that video."""
    vis, _, _ = parts(model)
    check_encoder(vis)
    ip = proc.image_processor
    mean, std = ip.image_mean, ip.image_std
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                done.add((r["condition"], r["qid"]))
    todo = [(it, c) for c in conditions for it in items if (c, it["qid"]) not in done]
    todo.sort(key=lambda x: x[0]["video"])                   # stable: video, condition, question
    print(f"{len(done)} already scored, {len(todo)} to go -> {out_path}")

    toks, imgs, block, video, t_start, n_new = {}, {}, None, None, time.time(), 0
    with open(out_path, "a") as fh:
        for it, c in todo:
            if it["video"] != video:
                imgs.clear()
                video = it["video"]
            if (video, c) != block:
                toks.clear()
                torch.cuda.empty_cache()
                block = (video, c)
            t0 = time.time()
            per = []
            for fr in it["frames"]:
                if fr["key"] not in toks:
                    if fr["key"] not in imgs:
                        imgs[fr["key"]] = qf.load_frame01(fr["path"], qf.IMG)
                    cells, side, used = frame_layout(conditions, c, fr)
                    t_, d_ = encode_cells(vis, qf.resize01(imgs[fr["key"]], side), cells, side, mean, std)
                    toks[fr["key"]] = (t_, d_, cells, side, conditions[c][0] == "fovea" and used is None)
                per.append(toks[fr["key"]])
            try:
                msgs = build_messages(it["question"], it["options"],
                                      [(fr["path"], fr["t"]) for fr in it["frames"]], IMG * IMG)
                inputs, ids = build_inputs(model, proc, msgs, [q[0] for q in per], [q[1] for q in per],
                                           [q[2] for q in per], [q[3] for q in per])
                p = letter_probs(model, inputs, letter_ids)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                fh.write(json.dumps(dict(condition=c, qid=int(it["qid"]), video=it["video"],
                                         error="oom", n_frames=len(per))) + "\n")
                continue
            err = None if np.isfinite(p).all() else "nan"
            pred = "ABCDE"[int(np.nanargmax(p))] if np.isfinite(p).any() else "?"
            sizes = np.concatenate([q[2][:, 2] for q in per] or [np.zeros(0, int)])
            fh.write(json.dumps(dict(
                condition=c, qid=int(it["qid"]), video=it["video"], answer=it["answer"],
                pred=pred, correct=int(pred == it["answer"]), p=[round(float(v), 4) for v in p],
                n_frames=len(per), n_image_tokens=int(sum(len(q[2]) for q in per)),
                cells_by_size={str(int(s)): int((sizes == s).sum()) for s in np.unique(sizes)},
                frame_px=sorted({int(q[3]) for q in per}), n_gaze_missing=int(sum(q[4] for q in per)),
                n_tokens=int(ids.shape[1]), sec=round(time.time() - t0, 3), error=err)) + "\n")
            fh.flush()
            n_new += 1
            if n_new % log_every == 0:
                rate = (time.time() - t_start) / n_new
                print(f"   {n_new}/{len(todo)}  {rate:.2f} s each  ~{rate * (len(todo) - n_new) / 60:.0f} min left"
                      f"  ({c}, {it['video']})", flush=True)
    print(f"done: {n_new} new in {(time.time() - t_start) / 60:.1f} min")

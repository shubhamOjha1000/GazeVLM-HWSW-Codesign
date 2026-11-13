"""Gaze-foveated tokenization inside SmolVLM2-256M-Video-Instruct's own vision encoder (no training), runbook section 16.

The same port as vlmeval/smolvlm_fovea.py (SmolVLM2-2.2B), on the 256M's own geometry. Its config and processor (pinned at
REVISION) give:

    vision    SigLIP base on a 512 px image: 16 px patches, a 32 x 32 grid, learned position table, full attention
    connector pixel shuffle by scale_factor 4 (each 4 x 4 block of patch features -> one token), then a projection
    LLM       SmolLM2-135M (30 layers, 9 query heads against 3 key/value heads of 64), 1D positions, 8,192 positions
    one token = one 64 px cell, 8 x 8 = 64 tokens per image; video frames are one 512 px image each, without tiling

check_geometry compares every one of these with the loaded model and processor and stops on any difference.

The fovea keeps that unit, before the encoder, in the model's own 4x steps, as the 2.2B's port uses its own 3x (approved by the user,
2026-11-13):

    fovea      64 px cells, full resolution     (exactly the 256M's own tokens)
    periphery  256 px cells shrunk to 64 px     (one token for 16x the area)

On the 2 x 2 grid of 256 px cells, the k nearest the gaze (eccentricity of the cell centre) stay at full resolution: 4 + 15 k tokens,
k = 1 / 2 / 3 -> 19 / 34 / 49 of 64, the same for every gaze. The budget rule and the no-gate control use k = 1, the smallest, as on
the 2.2B (17 of 81). Each cell becomes 4 x 4 patches of 16 px. A shrunk cell's patch takes the position of the patch at offset
(k - 1) // 2 inside the 4 x 4 block of patches it shows, the 2.2B port's rule. On the 2.2B that is the block's centre patch; on a
4 x 4 block there is no centre patch, so it is the one just up and left of the centre. Positions are SmolVLM's own position ids
(captured from its embedding forward). Each cell's 4 x 4 patch features go through the model's own pixel shuffle and projection.

Everything that doesn't depend on the geometry (the prompt, the image markers, the answer letters, the merge of image features, the
answer's probabilities) is the 2.2B module's own code, imported, not copied.
"""

import json
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from vlmeval import qwen_fovea as qf
from vlmeval import smolvlm_fovea as sm
from vlmeval import spatial as sp

MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
REVISION = "9cda5c165b28c68a52e9bb542d2b89f0f6963709"            # pinned; the download job prints the snapshot it got
DOWNLOAD_PATTERNS = ("*.json", "*.safetensors", "*.txt")           # the weights, configs and tokenizer, not the ONNX exports
# From the pinned config.json and preprocessor_config.json; check_geometry compares them with the loaded model and processor.
IMG, PATCH, SHUFFLE = 512, 16, 4         # vision_config.image_size, vision_config.patch_size, scale_factor
BASE = PATCH * SHUFFLE                   # 64 px: one token
GRID = IMG // BASE                       # 8 tokens per side
TOP = BASE * SHUFFLE                     # 256 px: a periphery cell, SHUFFLE x SHUFFLE tokens shrunk to one
CELLS = IMG // TOP                       # 2 x 2 periphery cells
STOCK_TOKENS = GRID * GRID               # 64
FAKE, GLOBAL, IMAGE = sm.FAKE, sm.GLOBAL, sm.IMAGE

# the geometry-free parts, as the 2.2B runs them
parts, normalize, encode_pixels, image_markers, expand_images = sm.parts, sm.normalize, sm.encode_pixels, sm.image_markers, sm.expand_images
prompt_ids, inputs_from_ids, letter_probs, letter_token_ids = sm.prompt_ids, sm.inputs_from_ids, sm.letter_probs, sm.letter_token_ids
fix_stock_merge, _encode, _first_mismatch, _image_features = sm.fix_stock_merge, sm._encode, sm._first_mismatch, sm._image_features


# ---------------------------------------------------------------------------- the geometry
def geometry_problems(model_config, image_processor, processor=None):
    """Where the loaded model or processor differs from this module's geometry -> list of messages."""
    v = model_config.vision_config
    got = dict(image_size=int(v.image_size), patch_size=int(v.patch_size), scale_factor=int(model_config.scale_factor),
               processor_image=int(dict(getattr(image_processor, "max_image_size", {}) or {}).get("longest_edge", -1)))
    want = dict(image_size=IMG, patch_size=PATCH, scale_factor=SHUFFLE, processor_image=IMG)
    out = [f"{k}: {got[k]} in the model, {want[k]} here" for k in want if got[k] != want[k]]
    if (IMG // PATCH) % SHUFFLE:
        out.append(f"{IMG // PATCH} patches per side don't split into {SHUFFLE} x {SHUFFLE} blocks")
    seq = getattr(processor, "image_seq_len", None)
    if seq is not None and int(seq) != STOCK_TOKENS:
        out.append(f"the processor's image_seq_len is {seq}, {STOCK_TOKENS} here")
    return out


def check_geometry(model, proc):
    found = geometry_problems(model.config, proc.image_processor, proc)
    if found:
        raise RuntimeError("SmolVLM2-256M's geometry isn't this module's: " + "; ".join(found) + ". Stop and tell the user.")


# ---------------------------------------------------------------------------- layouts
def foveated_cells(gu, gv, k):
    """Cells (y, x, size) in px on the 512 px frame for gaze (gu, gv) in [0, 1]: the 2 x 2 grid of 256 px cells with the k nearest
    the gaze split into 4 x 4 cells of 64 px. Raster order."""
    split = TOP // BASE
    c = (np.arange(CELLS) + 0.5) * TOP
    X, Y = np.meshgrid(c, c)
    ecc = sp.ecc_deg(X, Y, gu * IMG, gv * IMG, IMG, sp.FOV).ravel()
    fine = set(np.argsort(ecc, kind="stable")[:k].tolist())
    cells = []
    for i in range(CELLS * CELLS):
        r, q = divmod(i, CELLS)
        if i in fine:
            cells += [(r * TOP + a * BASE, q * TOP + b * BASE, BASE) for a in range(split) for b in range(split)]
        else:
            cells.append((r * TOP, q * TOP, TOP))
    return qf.sort_raster(np.array(cells, dtype=np.int64))


def tokens_at(k):
    """Tokens of the k-cell fovea, whatever the gaze."""
    return CELLS * CELLS - k + k * (TOP // BASE) ** 2


def uniform_cells(side=IMG):
    """All 64 px cells of a side x side frame (512: the 256M's own 64 tokens)."""
    return qf.uniform_cells(side, base=BASE)


def same_tokens_size(n_tokens):
    return qf.same_tokens_size(n_tokens, base=BASE)


def cells_to_patches(img01, cells, mean, std):
    """(16 N, 3, 16, 16) normalized patches: each cell resized to 64 px (bicubic, antialiased) and cut into 4 x 4 patches, row-major,
    cells in the given order."""
    N = len(cells)
    crops = torch.empty(N, 3, BASE, BASE, dtype=torch.float32)
    for s in np.unique(cells[:, 2]):
        idx = np.nonzero(cells[:, 2] == s)[0]
        c = torch.stack([img01[:, y:y + s, x:x + s] for y, x, _ in cells[idx]]).float()
        if s != BASE:
            c = F.interpolate(c, size=(BASE, BASE), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
        crops[torch.as_tensor(idx)] = c
    m = torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    sd = torch.as_tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    crops = (crops - m) / sd
    S = SHUFFLE
    return crops.reshape(N, 3, S, PATCH, S, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(N * S * S, 3, PATCH, PATCH)


def patch_positions(cells):
    """(h, w) integer patch-grid positions of each cell's 4 x 4 patches (row-major): the patch itself for a 64 px cell; for a larger
    cell the patch at offset (k - 1) // 2 inside the k x k block each patch shows (k = 4 for a 256 px cell)."""
    c = np.asarray(cells, dtype=np.int64)
    k = (c[:, 2] // BASE)[:, None]
    a = np.repeat(np.arange(SHUFFLE), SHUFFLE)[None]
    b = np.tile(np.arange(SHUFFLE), SHUFFLE)[None]
    h = c[:, :1] // PATCH + a * k + (k - 1) // 2
    w = c[:, 1:2] // PATCH + b * k + (k - 1) // 2
    return h.ravel(), w.ravel()


# ---------------------------------------------------------------------------- the encoder
_POS = {}


def grid_position_ids(emb, n, dtype, device):
    """The position ids SmolVLM's own embeddings use for an n x n patch grid, captured from its forward: (n, n)."""
    key = (n, dtype, str(device))
    if key not in _POS:
        got = {}
        hook = emb.position_embedding.register_forward_hook(lambda m, i, o: got.__setitem__("ids", i[0]))
        try:
            emb(pixel_values=torch.zeros(1, 3, n * PATCH, n * PATCH, dtype=dtype, device=device),
                patch_attention_mask=torch.ones(1, n, n, dtype=torch.bool, device=device))
        finally:
            hook.remove()
        _POS[key] = got["ids"].reshape(n, n)
    return _POS[key]


@torch.no_grad()
def encode_cells(model, img01, cells, mean, std):
    """LLM tokens (N, D), one per cell in `cells` order, for any 64 / 256 px layout: patches embedded with SmolVLM's own patch
    embedding and position ids, in raster order of their positions through the encoder and post-layernorm, then each cell's 4 x 4
    patch features through its own pixel shuffle and projection."""
    vis, connector = parts(model)
    p = next(vis.parameters())
    emb = vis.embeddings
    n = len(cells)
    x = emb.patch_embedding(cells_to_patches(img01, cells, mean, std).to(p.device, p.dtype)).flatten(1)
    h, w = patch_positions(cells)
    table = grid_position_ids(emb, emb.num_patches_per_side, p.dtype, p.device)
    x = x + emb.position_embedding(table[torch.as_tensor(h, device=p.device), torch.as_tensor(w, device=p.device)])
    order = torch.as_tensor(np.argsort(h * 4096 + w, kind="stable"), device=p.device)    # raster order
    out = vis.post_layernorm(_encode(vis, x[order][None]))[0]
    feats = torch.empty_like(out)
    feats[order] = out                                                                    # cell order
    t = connector.pixel_shuffle(feats.reshape(n, SHUFFLE * SHUFFLE, -1), connector.scale_factor)
    return connector.modality_projection(t)[:, 0]


@torch.no_grad()
def encode_uniform(model, rec, side, mean, std):
    """SmolVLM's own path on its 512 px input (side = 512: the stock tokens) or on that image downscaled to side (a multiple of 64)."""
    if side == IMG:
        return encode_pixels(model, rec["pv"])
    return encode_pixels(model, normalize(qf.resize01(rec["x01"], side), mean, std))


def stock_input(ip, pil):
    """SmolVLM's own 512 px input for a frame (its processor, no tiling): dict(pv = normalized (3, 512, 512), x01 = the same image
    back in [0, 1] for cutting cells)."""
    pv = ip(images=[[pil]], do_image_splitting=False, return_tensors="pt")["pixel_values"][0, 0].float()
    if pv.shape[-1] != IMG or pv.shape[-2] != IMG:
        raise RuntimeError(f"the processor gave a {tuple(pv.shape)} image, expected {IMG} px")
    m = torch.as_tensor(ip.image_mean, dtype=pv.dtype).view(3, 1, 1)
    s = torch.as_tensor(ip.image_std, dtype=pv.dtype).view(3, 1, 1)
    return dict(pv=pv, x01=pv * s + m)


# ---------------------------------------------------------------------------- conditions
def _fovea_layout(arg, fr):
    g = (0.5, 0.5) if arg["gaze"] == "centre" else fr.get("gaze")
    used = g
    if g is None:
        g = (0.5, 0.5)
    return foveated_cells(*g, arg["k"]), used


def frame_layout(conditions, cond, fr):
    """(cells, frame side px, gaze used or None) of one frame under one condition, as frame_tokens encodes it (the FLOPs plan)."""
    kind, arg = conditions[cond]
    if kind == "uniform":
        return uniform_cells(arg), arg, None
    if kind == "uniform_match":
        side = same_tokens_size(len(_fovea_layout(conditions[arg][1], fr)[0]))
        return uniform_cells(side), side, None
    cells, used = _fovea_layout(arg, fr)
    return cells, IMG, used


def frame_tokens(model, conditions, cond, fr, rec, mean, std):
    """(LLM tokens (N, D), cells or None, frame side px, gaze used) for one frame. conditions as in smolvlm_fovea.frame_tokens:
    'uniform' (side px), 'fovea' (dict(k=..., gaze='gaze' | 'centre'): k of the 4 256 px cells nearest the gaze at full resolution),
    'uniform_match'. A missing gaze falls back to the image centre (counted in the results)."""
    kind, arg = conditions[cond]
    if kind == "uniform":
        return encode_uniform(model, rec, arg, mean, std), None, arg, None
    if kind == "uniform_match":
        side = same_tokens_size(len(_fovea_layout(conditions[arg][1], fr)[0]))
        return encode_uniform(model, rec, side, mean, std), None, side, None
    cells, used = _fovea_layout(arg, fr)
    return encode_cells(model, rec["x01"], cells, mean, std), cells, IMG, used


# ---------------------------------------------------------------------------- LLM input
TOKENIZE = dict(add_special_tokens=None)     # set by this model's parity_report, as the 2.2B's sets its own


@torch.no_grad()
def build_inputs(model, proc, messages, feats_list):
    ids = prompt_ids(proc, messages, [len(f) for f in feats_list], TOKENIZE["add_special_tokens"])
    return inputs_from_ids(model, ids, feats_list), ids


def load_model(model_id=MODEL_ID, revision=REVISION):
    from transformers import AutoProcessor
    fix_stock_merge()                          # the stock forward is the parity reference
    try:
        from transformers import SmolVLMForConditionalGeneration as Model
    except ImportError:
        from transformers import AutoModelForImageTextToText as Model
    kw = dict(device_map="auto", attn_implementation="sdpa", revision=revision)
    dtype = torch.float16 if os.environ.get("VLMEVAL_DTYPE") == "float16" else torch.bfloat16
    try:
        model = Model.from_pretrained(model_id, dtype=dtype, **kw)
    except TypeError:
        model = Model.from_pretrained(model_id, torch_dtype=dtype, **kw)
    proc = AutoProcessor.from_pretrained(model_id, revision=revision)
    check_geometry(model, proc)
    return model.eval(), proc, letter_token_ids(proc.tokenizer)


# ---------------------------------------------------------------------------- the check on the real model
@torch.no_grad()
def parity_report(model, proc, letter_ids, pil, question, options, build_messages):
    """smolvlm_fovea.parity_report on this model's geometry: our path with an all-64 px layout against SmolVLM's own pipeline on one
    real frame, the vision side in fp32 (TF32 off, exact attention), the LLM side with the processor's one-call API."""
    merge_fixed = fix_stock_merge()
    ip = proc.image_processor
    mean, std = ip.image_mean, ip.image_std
    vis, connector = parts(model)
    rec = stock_input(ip, pil)
    cells = uniform_cells()
    f_cells16 = encode_cells(model, rec["x01"], cells, mean, std).float()
    f_pix16 = encode_pixels(model, rec["pv"]).float()

    dtype = next(vis.parameters()).dtype
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        exact_attention = sdpa_kernel(SDPBackend.MATH)
    except Exception:
        exact_attention = nullcontext()
    try:
        vis.float(); connector.float()
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        with exact_attention:
            f_cells32 = encode_cells(model, rec["x01"], cells, mean, std).float()
            f_pix32 = encode_pixels(model, rec["pv"]).float()
    finally:
        vis.to(dtype); connector.to(dtype)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    cos = lambda a, b: F.cosine_similarity(a, b, dim=-1).min().item()

    msgs = build_messages(question, options, [("frame.jpg", 0.0)], IMG * IMG)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    stock = proc(text=[text], images=[[pil]], do_image_splitting=False, return_tensors="pt").to(model.device)
    sid = stock["input_ids"][0].cpu()
    matched = [v for v in (None, True, False) if torch.equal(prompt_ids(proc, msgs, [STOCK_TOKENS], v)[0], sid)]
    if matched:
        TOKENIZE["add_special_tokens"] = matched[0]
    pix = (normalize(rec["x01"], mean, std) - stock["pixel_values"][0, 0].float().cpu()).abs().max().item()
    f_stock = _image_features(model.model.get_image_features(
        pixel_values=stock["pixel_values"].to(dtype), pixel_attention_mask=stock.get("pixel_attention_mask")))
    f_stock = f_stock.reshape(-1, f_stock.shape[-1])

    p_stock = torch.softmax(model(**stock, use_cache=False).logits[0, -1, letter_ids].float(), -1).cpu().numpy()
    inputs, ids = build_inputs(model, proc, msgs, [f_stock])
    p_llm = letter_probs(model, inputs, letter_ids, all_positions=True)
    inputs16, _ = build_inputs(model, proc, msgs, [f_cells16])
    p_ours = letter_probs(model, inputs16, letter_ids, all_positions=True)
    same_ids = bool(torch.equal(ids[0].cpu(), sid))
    return dict(stock_merge="in order (5.20's replaced)" if merge_fixed else "transformers' own",
                geometry=f"{IMG} px, {PATCH} px patches, pixel shuffle {SHUFFLE}: {STOCK_TOKENS} tokens per frame",
                images_in_stock=int(stock["pixel_values"].shape[1]),
                tokens_per_image_in_stock=int(f_stock.shape[0]),
                tokenization=f"add_special_tokens={matched[0]}" if matched else "NO VARIANT MATCHED",
                pixel_max_abs_diff=pix,
                own_path_vs_get_image_features=(f_pix16 - f_stock.float()).abs().max().item(),
                vision_fp32_cos_min=cos(f_cells32, f_pix32),
                vision_fp32_max_abs_diff=(f_cells32 - f_pix32).abs().max().item(),
                tokens_ids_equal=same_ids,
                first_token_mismatch="-" if same_ids else _first_mismatch(proc.tokenizer, ids[0].cpu(), sid),
                llm_probs_max_abs_diff=float(np.abs(p_llm - p_stock).max()),
                llm_same_answer=bool(p_llm.argmax() == p_stock.argmax()),
                info_bf16_ours_cos_min=cos(f_cells16, f_pix16),
                info_bf16_smolvlm_own_cos_min_vs_fp32=cos(f_pix16, f_pix32),
                info_bf16_ours_probs_max_abs_diff=float(np.abs(p_ours - p_stock).max()),
                info_bf16_ours_same_answer=bool(p_ours.argmax() == p_stock.argmax()))


# ---------------------------------------------------------------------------- evaluation
@torch.no_grad()
def load_stock_inputs(ip, frames, recs, workers):
    """smolvlm_fovea.load_stock_inputs with this model's stock_input: each frame's input made on `workers` threads, exactly what
    stock_input gives it; only the timing changes."""
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image
    todo = [fr for fr in frames if fr["key"] not in recs]

    def load(fr):
        with Image.open(fr["path"]) as im:
            return stock_input(ip, im.convert("RGB"))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fr, rec in zip(todo, pool.map(load, todo)):
            recs[fr["key"]] = rec


def run_questions(model, proc, letter_ids, items, conditions, out_path, build_messages, log_every=25, prefetch_workers=0):
    """smolvlm_fovea.run_questions on this model's geometry, with the same records: resumable, video by video and within a video
    condition by condition; each frame's input is made once per video and its tokens once per (video, condition)."""
    from PIL import Image
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

    toks, recs, block, video, t_start, n_new = {}, {}, None, None, time.time(), 0
    with open(out_path, "a") as fh:
        for it, c in todo:
            if it["video"] != video:
                recs.clear()
                video = it["video"]
            if (video, c) != block:
                toks.clear()
                torch.cuda.empty_cache()
                block = (video, c)
                if prefetch_workers:
                    load_stock_inputs(ip, it["frames"], recs, prefetch_workers)
            t0 = time.time()
            per = []
            for fr in it["frames"]:
                if fr["key"] not in toks:
                    if fr["key"] not in recs:
                        recs[fr["key"]] = stock_input(ip, Image.open(fr["path"]).convert("RGB"))
                    f, cells, side, used = frame_tokens(model, conditions, c, fr, recs[fr["key"]], mean, std)
                    toks[fr["key"]] = (f, cells, side, conditions[c][0] == "fovea" and used is None)
                per.append(toks[fr["key"]])
            try:
                msgs = build_messages(it["question"], it["options"], [(fr["path"], fr["t"]) for fr in it["frames"]], IMG * IMG)
                inputs, ids = build_inputs(model, proc, msgs, [q[0] for q in per])
                p = letter_probs(model, inputs, letter_ids)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                fh.write(json.dumps(dict(condition=c, qid=int(it["qid"]), video=it["video"], error="oom", n_frames=len(per))) + "\n")
                continue
            err = None if np.isfinite(p).all() else "nan"
            pred = "ABCDE"[int(np.nanargmax(p))] if np.isfinite(p).any() else "?"
            sizes = np.concatenate([q[1][:, 2] for q in per if q[1] is not None] or [np.zeros(0, int)])
            fh.write(json.dumps(dict(
                condition=c, qid=int(it["qid"]), video=it["video"], answer=it["answer"],
                pred=pred, correct=int(pred == it["answer"]), p=[round(float(v), 4) for v in p],
                n_frames=len(per), n_image_tokens=int(sum(len(q[0]) for q in per)),
                cells_by_size={str(int(s)): int((sizes == s).sum()) for s in np.unique(sizes)},
                frame_px=sorted({int(q[2]) for q in per}), n_gaze_missing=int(sum(q[3] for q in per)),
                n_tokens=int(ids.shape[1]), sec=round(time.time() - t0, 3), error=err)) + "\n")
            fh.flush()
            n_new += 1
            if n_new % log_every == 0:
                rate = (time.time() - t_start) / n_new
                print(f"   {n_new}/{len(todo)}  {rate:.2f} s each  ~{rate * (len(todo) - n_new) / 60:.0f} min left"
                      f"  ({c}, {it['video']})", flush=True)
    print(f"done: {n_new} new in {(time.time() - t_start) / 60:.1f} min")

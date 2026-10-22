"""Gaze-foveated tokenization inside SmolVLM2-2.2B-Instruct's own vision encoder (no training).

SmolVLM2 encodes a 384 px image with SigLIP (14 px patches: a 27 x 27 grid over 378 px, learned
position table looked up through bucketed coordinates, full attention); its connector merges each
3 x 3 block of patch features into one token by pixel shuffle (channel concatenation) and projects
it into the LLM (SmolLM2-1.7B, 1D positions): 81 tokens per image, one token = one 42 px cell. For
video frames SmolVLM2 uses one 384 px image per frame, without tiling; so does this port.

Foveation keeps that unit, before the encoder, in SmolVLM's own 3x steps:

    fovea      42 px cells, full resolution    (exactly SmolVLM's own patches)
    periphery  126 px cells shrunk to 42 px    (one token for 9x the area)

On the 3 x 3 grid of 126 px cells, the K cells nearest the gaze (eccentricity of the cell centre)
stay at full resolution: 9 + 8 K tokens, K = 1 / 3 / 5 / 7 -> 17 / 33 / 49 / 65 of 81 (about the
shares of the other models' four budgets; their FOVI rings would split one cell or all nine on
a grid this coarse). Each cell becomes 3 x 3 patches of 14 px; a shrunk cell's patches take the
positions of the centre patches of the 3 x 3 blocks they show, which are grid positions, so
SmolVLM's own position ids are used throughout (captured from its embedding forward, rounding
included). The encoder has full attention, so the variable-length sequence needs nothing else;
each cell's 3 x 3 patch features go through SmolVLM's own pixel shuffle and projection.

Every condition starts from SmolVLM's own 384 px input (its processor's resizing of the frame);
the uniform baseline at equal tokens is that image downscaled through SmolVLM's own encoder,
which buckets its position table for smaller grids natively.
"""

import json
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from vlmeval import qwen_fovea as qf
from vlmeval import spatial as sp

MODEL_CANDIDATES = ("HuggingFaceTB/SmolVLM2-2.2B-Instruct",)
IMG, PATCH, BASE, TOP = 384, 14, 42, 126
GRID = 9                                       # 42 px tokens per side (27 patches / 3)
FAKE, GLOBAL, IMAGE = "<fake_token_around_image>", "<global-img>", "<image>"


# ---------------------------------------------------------------------------- layouts
def foveated_cells(gu, gv, k):
    """Cells (y, x, size) in px on the 384 px frame for gaze (gu, gv) in [0, 1]: the 3 x 3 grid of
    126 px cells with the k nearest the gaze split into 3 x 3 cells of 42 px. Raster order."""
    c = (np.arange(3) + 0.5) * TOP
    X, Y = np.meshgrid(c, c)
    ecc = sp.ecc_deg(X, Y, gu * IMG, gv * IMG, IMG, sp.FOV).ravel()
    fine = set(np.argsort(ecc, kind="stable")[:k].tolist())
    cells = []
    for i in range(9):
        r, q = divmod(i, 3)
        if i in fine:
            cells += [(r * TOP + a * BASE, q * TOP + b * BASE, BASE) for a in range(3) for b in range(3)]
        else:
            cells.append((r * TOP, q * TOP, TOP))
    return qf.sort_raster(np.array(cells, dtype=np.int64))


def uniform_cells():
    """All 42 px cells of the 378 px area SigLIP tokenizes: SmolVLM's own 81 tokens."""
    return qf.uniform_cells(GRID * BASE, base=BASE)


def same_tokens_size(n_tokens):
    """Side (px, multiple of 42) of the uniform frame whose token count is closest to n."""
    return qf.same_tokens_size(n_tokens, base=BASE)


def cells_to_patches(img01, cells, mean, std):
    """(9N, 3, 14, 14) normalized patches: each cell resized to 42 px (bicubic, antialiased) and
    cut into 3 x 3 patches, row-major, cells in the given order."""
    N = len(cells)
    crops = torch.empty(N, 3, BASE, BASE, dtype=torch.float32)
    for s in np.unique(cells[:, 2]):
        idx = np.nonzero(cells[:, 2] == s)[0]
        c = torch.stack([img01[:, y:y + s, x:x + s] for y, x, _ in cells[idx]]).float()
        if s != BASE:
            c = F.interpolate(c, size=(BASE, BASE), mode="bicubic", antialias=True,
                              align_corners=False).clamp(0, 1)
        crops[torch.as_tensor(idx)] = c
    m = torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    sd = torch.as_tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    crops = (crops - m) / sd
    return crops.reshape(N, 3, 3, PATCH, 3, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(N * 9, 3, PATCH, PATCH)


def patch_positions(cells):
    """(h, w) integer patch-grid positions of each cell's 3 x 3 patches (row-major): the patch
    itself for a 42 px cell; for a larger cell the centre patch of the block each patch shows."""
    c = np.asarray(cells, dtype=np.int64)
    k = (c[:, 2] // BASE)[:, None]
    a = np.repeat(np.arange(3), 3)[None]
    b = np.tile(np.arange(3), 3)[None]
    h = c[:, :1] // PATCH + a * k + (k - 1) // 2
    w = c[:, 1:2] // PATCH + b * k + (k - 1) // 2
    return h.ravel(), w.ravel()


# ---------------------------------------------------------------------------- model parts
def parts(model):
    """(vision transformer, connector) of SmolVLMForConditionalGeneration."""
    inner = getattr(model, "model", model)
    return inner.vision_model, inner.connector


def _merge_in_order(self, input_ids, inputs_embeds, image_hidden_states):
    """SmolVLM's merge of the image features into the prompt as in transformers <= 5.19 and on main:
    the k-th <image> token of the batch gets the k-th feature row."""
    if input_ids is None:
        mask = (inputs_embeds == self.get_input_embeddings()(torch.full(
            (), self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device))).all(-1)
    else:
        mask = input_ids == self.config.image_token_id
    feats = image_hidden_states.reshape(-1, inputs_embeds.shape[-1]).to(inputs_embeds.device, inputs_embeds.dtype)
    if int(mask.sum()) != len(feats):
        raise ValueError(f"{int(mask.sum())} <image> tokens, {len(feats)} image feature rows")
    return inputs_embeds.masked_scatter(mask.unsqueeze(-1).expand_as(inputs_embeds), feats)


def fix_stock_merge():
    """transformers 5.20 merges SmolVLM's image features in blocks of the vision patch size (14),
    which fails for 81 tokens per image; that version's SmolVLMModel gets the in-order merge of the
    versions before and after (the same result wherever theirs works). True if it is in use."""
    import inspect
    from transformers.models.smolvlm import modeling_smolvlm as ms
    merge = getattr(ms.SmolVLMModel, "inputs_merger", None)
    if merge is not None and merge is not _merge_in_order:
        try:
            src = inspect.getsource(merge)
        except (OSError, TypeError):
            src = ""
        if "vision_config.patch_size" in src:
            ms.SmolVLMModel.inputs_merger = _merge_in_order
    return getattr(ms.SmolVLMModel, "inputs_merger", None) is _merge_in_order


_POS = {}


def grid_position_ids(emb, n, dtype, device):
    """The position ids SmolVLM's own embeddings use for an n x n patch grid, captured from its
    forward (so its bucketing, including the rounding of coordinates to the pixel dtype, is
    reproduced exactly): (n, n)."""
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


def _encode(vis, x):
    out = vis.encoder(inputs_embeds=x)
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


@torch.no_grad()
def encode_cells(model, img01, cells, mean, std):
    """LLM tokens (N, D), one per cell in `cells` order, for any 42/126 px layout of the 384 px
    frame: patches embedded with SmolVLM's own patch embedding and position ids, in raster order
    of their positions through the encoder and post-layernorm, then each cell's 3 x 3 patch
    features through SmolVLM's pixel shuffle and projection."""
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
    t = connector.pixel_shuffle(feats.reshape(n, 9, -1), connector.scale_factor)        # (n, 1, 9C)
    return connector.modality_projection(t)[:, 0]


@torch.no_grad()
def encode_pixels(model, pv):
    """SmolVLM's own encoder + connector on normalized pixels (3, H, W): (H // 42)^2 tokens."""
    vis, connector = parts(model)
    p = next(vis.parameters())
    out = vis(pixel_values=pv[None].to(p.device, p.dtype))
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return connector(h)[0]


def normalize(x01, mean, std):
    return (x01 - torch.as_tensor(mean, dtype=x01.dtype).view(3, 1, 1)) / torch.as_tensor(std, dtype=x01.dtype).view(3, 1, 1)


@torch.no_grad()
def encode_uniform(model, rec, side, mean, std):
    """SmolVLM's own path on its 384 px input (side = 384: the stock tokens) or on that image
    downscaled to side (a multiple of 42)."""
    if side == IMG:
        return encode_pixels(model, rec["pv"])
    return encode_pixels(model, normalize(qf.resize01(rec["x01"], side), mean, std))


def stock_input(ip, pil):
    """SmolVLM's own 384 px input for a frame (its processor, no tiling): dict(pv = normalized
    (3, 384, 384), x01 = the same image back in [0, 1] for cutting cells)."""
    pv = ip(images=[[pil]], do_image_splitting=False, return_tensors="pt")["pixel_values"][0, 0].float()
    if pv.shape[-1] != IMG or pv.shape[-2] != IMG:
        raise RuntimeError(f"the processor gave a {tuple(pv.shape)} image, expected 384 px")
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


def frame_tokens(model, conditions, cond, fr, rec, mean, std):
    """(LLM tokens (N, D), cells or None, frame side px, gaze used) for one frame.

    conditions: {name: (kind, arg)} with kind in
        'uniform'        arg = side px: SmolVLM's own input at that size (384 = stock, 81 tokens)
        'fovea'          arg = dict(k=..., gaze='gaze' | 'centre'): k of the 9 126 px cells nearest
                         the gaze at full resolution
        'uniform_match'  arg = name of a 'fovea' condition: SmolVLM's own input downscaled to the
                         side whose token grid is closest to that condition's count
    A missing gaze falls back to the image centre (counted in the results)."""
    kind, arg = conditions[cond]
    if kind == "uniform":
        return encode_uniform(model, rec, arg, mean, std), None, arg, None
    if kind == "uniform_match":
        side = same_tokens_size(len(_fovea_layout(conditions[arg][1], fr)[0]))
        return encode_uniform(model, rec, side, mean, std), None, side, None
    cells, used = _fovea_layout(arg, fr)
    return encode_cells(model, rec["x01"], cells, mean, std), cells, IMG, used


# ---------------------------------------------------------------------------- LLM input
# How the prompt is tokenized; parity_report sets it to the variant that reproduces SmolVLM's own
# processor exactly.
TOKENIZE = dict(add_special_tokens=None)


def image_markers(proc):
    return (getattr(proc, "fake_image_token", FAKE), getattr(proc, "global_image_token", GLOBAL),
            getattr(proc, "image_token", IMAGE))


def expand_images(proc, text, n_tokens):
    """Each <image> placeholder -> <fake_token_around_image><global-img> + n <image> +
    <fake_token_around_image> (the processor's single-image prompt, 81 per image at stock)."""
    fake, glb, img = image_markers(proc)
    parts_ = text.split(img)
    if len(parts_) - 1 != len(n_tokens):
        raise ValueError(f"{len(parts_) - 1} image placeholders in the prompt, {len(n_tokens)} images")
    return parts_[0] + "".join(fake + glb + img * n + fake + p for n, p in zip(n_tokens, parts_[1:]))


def prompt_ids(proc, messages, n_tokens, add_special_tokens=None):
    tok = proc.tokenizer
    text = expand_images(proc, proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                         n_tokens)
    bos = tok.bos_token or ""
    add = (not (bos and text.startswith(bos))) if add_special_tokens is None else add_special_tokens
    return tok(text, return_tensors="pt", add_special_tokens=add).input_ids


@torch.no_grad()
def build_inputs(model, proc, messages, feats_list):
    ids = prompt_ids(proc, messages, [len(f) for f in feats_list], TOKENIZE["add_special_tokens"])
    return inputs_from_ids(model, ids, feats_list), ids


@torch.no_grad()
def inputs_from_ids(model, ids, feats_list):
    """inputs_embeds + attention_mask for token ids whose <image> tokens get the given rows."""
    img_id = model.config.image_token_id
    emb_layer = model.get_input_embeddings()
    ids = ids.to(emb_layer.weight.device)
    mask = ids == img_id
    emb = emb_layer(ids)
    if int(mask.sum()) != sum(len(f) for f in feats_list):
        raise ValueError(f"{int(mask.sum())} image tokens in the prompt, {sum(len(f) for f in feats_list)} features")
    if feats_list:
        emb[mask] = torch.cat(feats_list).to(emb.device, emb.dtype)
    return dict(inputs_embeds=emb, attention_mask=torch.ones_like(ids))


@torch.no_grad()
def letter_probs(model, inputs, letter_ids):
    logits = model(**inputs, use_cache=False).logits[0, -1, letter_ids].float()
    return torch.softmax(logits, -1).cpu().numpy()


def letter_token_ids(tok, prefix="Assistant:", sep=" "):
    """Token id of each answer letter as SmolVLM writes it after the generation prompt
    ("Assistant:" + " A", the format of its assistant turns)."""
    base = tok(prefix, add_special_tokens=False).input_ids
    ids = []
    for L in "ABCDE":
        full = tok(prefix + sep + L, add_special_tokens=False).input_ids
        if full[:len(base)] != base or len(full) != len(base) + 1:
            raise ValueError(f"letter {L!r} after {prefix!r} is not one extra token: {base} -> {full}")
        ids.append(full[-1])
    return ids


def load_model(model_id):
    from transformers import AutoProcessor
    fix_stock_merge()                          # the stock forward is the parity reference
    try:
        from transformers import SmolVLMForConditionalGeneration as Model
    except ImportError:
        from transformers import AutoModelForImageTextToText as Model
    kw = dict(device_map="auto", attn_implementation="sdpa")
    try:
        model = Model.from_pretrained(model_id, dtype=torch.bfloat16, **kw)
    except TypeError:
        model = Model.from_pretrained(model_id, torch_dtype=torch.bfloat16, **kw)
    proc = AutoProcessor.from_pretrained(model_id)
    return model.eval(), proc, letter_token_ids(proc.tokenizer)


# ---------------------------------------------------------------------------- checks on the real model
def _first_mismatch(tok, a, b, ctx=4):
    a, b = a.tolist(), b.tolist()
    n = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    show = lambda x: repr(tok.decode(x[max(0, n - ctx):n + ctx]))
    return f"at token {n} (lengths {len(a)} vs {len(b)}): ours {show(a)} | stock {show(b)}"


def _image_features(out):
    """get_image_features' tokens as one tensor: it returns a tensor, an output with pooler_output,
    or (transformers 5.20) an output whose pooler_output is a list, one (n, D) tensor per sample."""
    f = out if torch.is_tensor(out) else getattr(out, "pooler_output", None)
    if f is None:
        f = out[0]
    if isinstance(f, (list, tuple)):
        f = torch.cat([x.reshape(-1, x.shape[-1]) for x in f])
    return f


@torch.no_grad()
def parity_report(model, proc, letter_ids, pil, question, options, build_messages):
    """Our foveation path with an all-42 px layout vs SmolVLM's own pipeline, on one real frame.

      vision  in fp32 (TF32 off, exact attention): all-42 px layout through our path vs SmolVLM's
              own encoder + connector on the processor's pixels (encode_pixels, itself checked
              against get_image_features in bf16); bf16 deviation reported for information
      LLM     our prompt + inputs_embeds carrying SmolVLM's own image tokens vs the processor's
              one-call API + model: token ids and answer probabilities"""
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
    try:                                                     # vision parity in true fp32
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
    matched = [v for v in (None, True, False) if torch.equal(prompt_ids(proc, msgs, [81], v)[0], sid)]
    if matched:
        TOKENIZE["add_special_tokens"] = matched[0]
    pix = (normalize(rec["x01"], mean, std) - stock["pixel_values"][0, 0].float().cpu()).abs().max().item()
    f_stock = _image_features(model.model.get_image_features(
        pixel_values=stock["pixel_values"].to(dtype), pixel_attention_mask=stock.get("pixel_attention_mask")))
    f_stock = f_stock.reshape(-1, f_stock.shape[-1])

    p_stock = torch.softmax(model(**stock, use_cache=False).logits[0, -1, letter_ids].float(), -1).cpu().numpy()
    inputs, ids = build_inputs(model, proc, msgs, [f_stock])            # LLM plumbing, stock image tokens
    p_llm = letter_probs(model, inputs, letter_ids)
    inputs16, _ = build_inputs(model, proc, msgs, [f_cells16])           # our whole path in bf16 (info)
    p_ours = letter_probs(model, inputs16, letter_ids)
    same_ids = bool(torch.equal(ids[0].cpu(), sid))
    return dict(stock_merge="in order (5.20's replaced)" if merge_fixed else "transformers' own",
                images_in_stock=int(stock["pixel_values"].shape[1]),
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


@torch.no_grad()
def fovea_feature_drift(model, rec, cells, mean, std):
    """Cosine of each foveated token with the mean of SmolVLM's own tokens over the same area."""
    import pandas as pd
    full = encode_pixels(model, rec["pv"]).float().view(GRID, GRID, -1)
    fov = encode_cells(model, rec["x01"], cells, mean, std).float()
    rows = []
    for (y, x, s), f in zip(cells, fov):
        y0, x0, k = y // BASE, x // BASE, s // BASE
        ref = full[y0:y0 + k, x0:x0 + k].reshape(-1, full.shape[-1]).mean(0)
        rows.append((int(s), F.cosine_similarity(f, ref, dim=0).item()))
    return pd.DataFrame(rows, columns=["cell_px", "cos_to_stock"]).groupby("cell_px").agg(
        cells=("cos_to_stock", "size"), mean_cos=("cos_to_stock", "mean"), min_cos=("cos_to_stock", "min"))


# ---------------------------------------------------------------------------- evaluation
@torch.no_grad()
def run_questions(model, proc, letter_ids, items, conditions, out_path, build_messages, log_every=25):
    """Score questions on their frames under each condition; resumable. Work runs video by video
    and, within a video, condition by condition; each frame's SmolVLM input is made once per video
    and its tokens once per (video, condition)."""
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
                msgs = build_messages(it["question"], it["options"],
                                      [(fr["path"], fr["t"]) for fr in it["frames"]], IMG * IMG)
                inputs, ids = build_inputs(model, proc, msgs, [q[0] for q in per])
                p = letter_probs(model, inputs, letter_ids)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                fh.write(json.dumps(dict(condition=c, qid=int(it["qid"]), video=it["video"],
                                         error="oom", n_frames=len(per))) + "\n")
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

"""Gaze-foveated tokenization inside InternVL3.5-4B's own vision encoder (no training).

InternVL encodes a 448 px tile with InternViT (14 px patches, 32 x 32 grid, learned absolute
position embeddings, full attention), drops the CLS token, merges each 2 x 2 block of patch
features into one token by pixel shuffle (channel concatenation), and projects it with an MLP
into the LLM (Qwen3): 256 tokens per tile, one token = one 28 px cell.

Foveation keeps that unit, as for Qwen2.5-VL (vlmeval/qwen_fovea.py), before the encoder:

    fovea      28 px cells, full resolution    (exactly InternVL's own patches)
    ring       56 px cells shrunk to 28 px     (one token for 4x the area)
    periphery  112 px cells shrunk to 28 px    (one token for 16x the area)

Each cell becomes 2 x 2 patches of 14 px. Their position embeddings are the learned 32 x 32
table sampled (bilinear) at the centres of the cell's quadrants: InternVL's own values for
28 px cells, interpolated for shrunk ones. The encoder has full attention and no windows, so
the variable-length sequence (CLS + 4 patches per cell) needs nothing else. After the
encoder, each cell's 4 patch features go through InternVL's own pixel shuffle and projector
(placed as a 2 x 2 block of a dummy grid, so the stock functions are used unchanged).

Layout: InternVL's 16 x 16 token grid on a 448 px tile is Qwen's on 448 px, so the cells are
qwen_fovea.foveated_cells(img=448): same rings, same token counts as the other two models.
Uniform baseline at equal tokens: InternVL's own path on the frame downscaled to the side
whose token grid matches (its position embeddings interpolated as InternVL does natively).
The LLM uses 1D positions, so only the number of image tokens changes in the prompt.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from vlmeval import qwen_fovea as qf

MODEL_CANDIDATES = ("OpenGVLab/InternVL3_5-4B-HF", "OpenGVLab/InternVL3_5-4B-hf")
TILE, PATCH, BASE = 448, 14, 28
IMG_START, IMG_END, IMG_CONTEXT = "<img>", "</img>", "<IMG_CONTEXT>"


# ---------------------------------------------------------------------------- model parts
def parts(model):
    """(vision tower, projector, pixel_shuffle function, downsample ratio, feature layer)."""
    def get(name):
        m = getattr(model, name, None)
        return m if m is not None else getattr(model.model, name)
    shuffle = getattr(model, "pixel_shuffle", None) or model.model.pixel_shuffle
    cfg = model.config
    return (get("vision_tower"), get("multi_modal_projector"), shuffle,
            getattr(cfg, "downsample_ratio", 0.5), getattr(cfg, "vision_feature_layer", -1))


def _as_tensor(out):
    if torch.is_tensor(out):
        return out
    for name in ("last_hidden_state", "pooler_output"):
        t = getattr(out, name, None)
        if t is not None:
            return t
    return out[0]


def _image_features(out):
    """get_image_features() output -> the projected LLM tokens (newer versions return an object
    whose last_hidden_state is the raw encoder output and pooler_output the tokens)."""
    if torch.is_tensor(out):
        return out
    pooled = getattr(out, "pooler_output", None)
    return pooled if pooled is not None else _as_tensor(out)


def _pos_table(vt):
    """(cls position embedding (1, C), patch table (1, C, g, g))."""
    pe = vt.embeddings.position_embeddings
    g = int(round((pe.shape[1] - 1) ** 0.5))
    return pe[:, 0], pe[:, 1:].reshape(1, g, g, -1).permute(0, 3, 1, 2)


def sample_positions(vt, h, w):
    """Learned patch position embeddings at fractional grid coordinates (h, w) (patch index
    units on the native 32 x 32 grid), bilinear; exact at integer coordinates."""
    _, table = _pos_table(vt)
    g = table.shape[-1]
    xy = torch.stack([w / (g - 1) * 2 - 1, h / (g - 1) * 2 - 1], dim=-1).to(table.device)
    out = F.grid_sample(table.float(), xy.view(1, 1, -1, 2).float(), mode="bilinear", align_corners=True)
    return out[0, :, 0].T.to(table.dtype)                                    # (n, C)


def _encoder_features(vt, x, layer):
    """InternViT encoder on an embedded sequence (1, L, C) -> features as InternVL uses them:
    the last layer through the final norm (layer -1), else that layer's hidden states."""
    if layer == -1:
        h = _as_tensor(vt.encoder(x))
        norm = getattr(vt, "layernorm", None)
        return norm(h) if norm is not None else h
    return vt.encoder(x, output_hidden_states=True).hidden_states[layer]


def tokens_from_cells(model, sub):
    """Patch features grouped per cell (4N, C), each cell's 4 patches row-major -> LLM tokens
    (N, D) through InternVL's own pixel shuffle and projector. Each cell is written as a 2 x 2
    block of a dummy 32 x 32 grid; the shuffle merges exactly those blocks."""
    _, proj, shuffle, ratio, _ = parts(model)
    n_cells = len(sub) // 4
    side = TILE // BASE                                       # 16 cells per side
    per = side * side
    out = []
    for i in range(0, n_cells, per):
        s = sub[4 * i:4 * (i + per)]
        n = len(s) // 4
        block = torch.zeros(per, 2, 2, s.shape[-1], device=s.device, dtype=s.dtype)
        block[:n] = s.view(n, 2, 2, -1)
        grid = block.view(side, side, 2, 2, -1).permute(0, 2, 1, 3, 4).reshape(2 * side, 2 * side, -1)
        v = shuffle(grid.unsqueeze(0), scale_factor=ratio)
        v = v.reshape(1, -1, v.shape[-1])
        out.append(_as_tensor(proj(v))[0, :n])
    return torch.cat(out)


# ---------------------------------------------------------------------------- encoding
@torch.no_grad()
def encode_cells(model, img01, cells, mean, std):
    """LLM tokens (N, D), one per cell, for any 28/56/112 px layout of a 448 px frame."""
    vt, _, _, _, layer = parts(model)
    p = next(vt.parameters())
    n = len(cells)
    patches = qf.cells_to_patches(img01, cells, mean, std, temporal=1).view(4 * n, 3, PATCH, PATCH)
    emb = vt.embeddings
    x = emb.patch_embeddings.projection(patches.to(p.device, p.dtype)).flatten(1)       # (4N, C)
    h, w = qf.patch_positions(cells)
    x = x + sample_positions(vt, h, w).to(x.dtype)
    cls_pos, _ = _pos_table(vt)
    cls = (emb.cls_token.reshape(1, -1) + cls_pos).to(x.dtype)
    feats = _encoder_features(vt, torch.cat([cls, x]).unsqueeze(0), layer)[0, 1:]     # drop CLS
    return tokens_from_cells(model, feats)


@torch.no_grad()
def encode_uniform(model, img01, side, mean, std):
    """InternVL's own path on the frame resized to side x side px (side multiple of 28):
    (side / 28)^2 tokens. At 448 this is exactly InternVL's input."""
    vt, proj, shuffle, ratio, layer = parts(model)
    p = next(vt.parameters())
    m = torch.as_tensor(mean, dtype=torch.float32).view(3, 1, 1)
    s = torch.as_tensor(std, dtype=torch.float32).view(3, 1, 1)
    pv = ((qf.resize01(img01, side) - m) / s).unsqueeze(0).to(p.device, p.dtype)
    out = vt(pixel_values=pv, output_hidden_states=layer != -1)
    h = out.last_hidden_state if layer == -1 else out.hidden_states[layer]
    h = h[:, 1:]
    g = int(round(h.shape[1] ** 0.5))
    v = shuffle(h.reshape(1, g, g, -1), scale_factor=ratio)
    return _as_tensor(proj(v.reshape(1, -1, v.shape[-1])))[0]


# ---------------------------------------------------------------------------- conditions
def frame_tokens(model, conditions, cond, fr, img01, mean, std):
    """(LLM tokens (N, D), cells or None, frame side px, gaze used) for one frame.

    conditions: {name: (kind, arg)} with kind in
        'uniform'        arg = side px: InternVL's own input at that size (448 = stock, 256 tokens)
        'fovea'          arg = dict(a=..., R=..., gaze='gaze' | 'centre', num_scales=3)
        'uniform_match'  arg = name of a 'fovea' condition: InternVL's own input downscaled to
                         the side whose token grid is closest to that condition's count here
    A missing gaze falls back to the image centre (counted in the results)."""
    kind, arg = conditions[cond]
    if kind == "uniform":
        return encode_uniform(model, img01, arg, mean, std), None, arg, None
    if kind == "uniform_match":
        n = len(_fovea_layout(conditions[arg][1], fr)[0])
        side = qf.same_tokens_size(n)
        return encode_uniform(model, img01, side, mean, std), None, side, None
    cells, used = _fovea_layout(arg, fr)
    return encode_cells(model, qf.resize01(img01, TILE), cells, mean, std), cells, TILE, used


def _fovea_layout(arg, fr):
    g = (0.5, 0.5) if arg["gaze"] == "centre" else fr.get("gaze")
    used = g
    if g is None:
        g = (0.5, 0.5)
    return qf.foveated_cells(*g, a=arg["a"], img=TILE, R_deg=arg.get("R"),
                             num_scales=arg.get("num_scales", 3)), used


# ---------------------------------------------------------------------------- LLM input
# How the prompt is tokenized; parity_report sets it to the variant that reproduces InternVL's
# own processor exactly.
TOKENIZE = dict(add_special_tokens=None)


def image_markers(proc):
    return (getattr(proc, "start_image_token", IMG_START), getattr(proc, "end_image_token", IMG_END),
            getattr(proc, "image_token", IMG_CONTEXT))


def expand_images(proc, text, n_tokens):
    """Each image placeholder -> <img> + n <IMG_CONTEXT> + </img> (the processor uses 256 per tile)."""
    start, end, ctx = image_markers(proc)
    parts_ = text.split(ctx)
    if len(parts_) - 1 != len(n_tokens):
        raise ValueError(f"{len(parts_) - 1} image placeholders in the prompt, {len(n_tokens)} images")
    return parts_[0] + "".join(start + ctx * n + end + p for n, p in zip(n_tokens, parts_[1:]))


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
    ctx_id = getattr(model.config, "image_token_id", None) or proc.tokenizer.convert_tokens_to_ids(IMG_CONTEXT)
    emb_layer = model.get_input_embeddings()
    ids = ids.to(emb_layer.weight.device)
    mask = ids == ctx_id
    emb = emb_layer(ids)
    feats = torch.cat(feats_list).to(emb.dtype)
    if int(mask.sum()) != len(feats):
        raise ValueError(f"{int(mask.sum())} image tokens in the prompt, {len(feats)} features")
    emb[mask] = feats
    return dict(inputs_embeds=emb, attention_mask=torch.ones_like(ids)), ids


@torch.no_grad()
def letter_probs(model, inputs, letter_ids):
    logits = model(**inputs, use_cache=False).logits[0, -1, letter_ids].float()
    return torch.softmax(logits, -1).cpu().numpy()


def load_model(model_id):
    from transformers import AutoProcessor
    try:
        from transformers import InternVLForConditionalGeneration as Model
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
def _first_mismatch(tok, a, b, ctx=4):
    a, b = a.tolist(), b.tolist()
    n = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    show = lambda x: repr(tok.decode(x[max(0, n - ctx):n + ctx]))
    return f"at token {n} (lengths {len(a)} vs {len(b)}): ours {show(a)} | stock {show(b)}"


@torch.no_grad()
def parity_report(model, proc, letter_ids, img01, question, options, build_messages):
    """Our foveation path with an all-28 px layout at 448 px vs InternVL's own pipeline (the
    processor's one-call API on a message holding the image): pixels, vision tokens, token
    ids, answer probabilities. Also checks our uniform path against the same reference."""
    ip = proc.image_processor
    mean, std = ip.image_mean, ip.image_std
    x = qf.to_uint8_pil(qf.resize01(img01, TILE))
    x01 = torch.from_numpy(np.array(x)).permute(2, 0, 1).float() / 255.0
    f_cells = encode_cells(model, x01, qf.uniform_cells(TILE), mean, std).float()
    f_unif = encode_uniform(model, x01, TILE, mean, std).float()

    msgs = build_messages(question, options, [("frame.jpg", 0.0)], TILE * TILE)
    msgs_pil = [dict(m, content=[{"type": "image", "image": x} if c.get("type") == "image" else c
                                 for c in m["content"]]) for m in msgs]
    stock = proc.apply_chat_template(msgs_pil, add_generation_prompt=True, tokenize=True,
                                     return_dict=True, return_tensors="pt").to(model.device)
    sid = stock["input_ids"][0].cpu()
    matched = [v for v in (None, True, False) if torch.equal(prompt_ids(proc, msgs, [256], v)[0], sid)]
    if matched:
        TOKENIZE["add_special_tokens"] = matched[0]

    pv = stock["pixel_values"]
    m = torch.as_tensor(mean).view(3, 1, 1)
    s = torch.as_tensor(std).view(3, 1, 1)
    pix = ((x01 - m) / s - pv[0].float().cpu()).abs().max().item()
    cfg = model.config
    f_stock = _image_features(model.get_image_features(
        pixel_values=pv.to(next(model.parameters()).dtype),
        vision_feature_layer=getattr(cfg, "vision_feature_layer", -1),
        vision_feature_select_strategy=getattr(cfg, "vision_feature_select_strategy", "default")))
    f_stock = f_stock.reshape(-1, f_stock.shape[-1]).float()

    inputs, ids = build_inputs(model, proc, msgs, [f_cells])
    p_ours = letter_probs(model, inputs, letter_ids)
    p_stock = torch.softmax(model(**stock, use_cache=False).logits[0, -1, letter_ids].float(), -1).cpu().numpy()
    same_ids = bool(torch.equal(ids[0].cpu(), sid))
    return dict(tiles_in_stock=int(pv.shape[0]),
                tokenization=f"add_special_tokens={matched[0]}" if matched else "NO VARIANT MATCHED",
                pixel_max_abs_diff=pix,
                vision_cells_cos_min=F.cosine_similarity(f_cells, f_stock, dim=-1).min().item(),
                vision_cells_max_abs_diff=(f_cells - f_stock).abs().max().item(),
                vision_uniform_max_abs_diff=(f_unif - f_stock).abs().max().item(),
                tokens_ids_equal=same_ids,
                first_token_mismatch="-" if same_ids else _first_mismatch(proc.tokenizer, ids[0].cpu(), sid),
                probs_max_abs_diff=float(np.abs(p_ours - p_stock).max()),
                same_answer=bool(p_ours.argmax() == p_stock.argmax()),
                probs_ours=np.round(p_ours, 3).tolist(), probs_stock=np.round(p_stock, 3).tolist())


@torch.no_grad()
def fovea_feature_drift(model, img01, cells, mean, std):
    """Cosine of each foveated token with the mean of InternVL's own tokens over the same area."""
    import pandas as pd
    full = encode_uniform(model, img01, TILE, mean, std).float().view(16, 16, -1)
    fov = encode_cells(model, img01, cells, mean, std).float()
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
    """Score questions on their frames under each condition; resumable. Tokens are cached per
    (condition, frame) within a video and reused by every question of that video."""
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

    toks, imgs, video, t_start, n_new = {}, {}, None, time.time(), 0
    with open(out_path, "a") as fh:
        for it, c in todo:
            if it["video"] != video:
                toks.clear(); imgs.clear()
                torch.cuda.empty_cache()
                video = it["video"]
            t0 = time.time()
            per = []
            for fr in it["frames"]:
                key = (c, fr["key"])
                if key not in toks:
                    if fr["key"] not in imgs:
                        imgs[fr["key"]] = qf.load_frame01(fr["path"], qf.IMG)
                    f, cells, side, used = frame_tokens(model, conditions, c, fr, imgs[fr["key"]], mean, std)
                    toks[key] = (f, cells, side, conditions[c][0] == "fovea" and used is None)
                per.append(toks[key])
            try:
                msgs = build_messages(it["question"], it["options"],
                                      [(fr["path"], fr["t"]) for fr in it["frames"]], TILE * TILE)
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

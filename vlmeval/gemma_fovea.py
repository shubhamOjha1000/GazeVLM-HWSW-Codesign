"""Gaze-foveated visual tokens for Gemma-3-4B, by pooling after the vision encoder (no training).

Gemma 3 encodes a frame with SigLIP-So400m at 896 px (64 x 64 patches of 14 px), averages each
4 x 4 block of patch features into one token (16 x 16 = 256 tokens), then applies RMSNorm and a
linear projection into the LLM. The averaging has no weights, so foveation is done there:

    fovea      4 x 4 patches  (56 px)   one token  = exactly Gemma's own token, unchanged
    ring       8 x 8 patches  (112 px)  one token  = the mean of 4 of Gemma's tokens
    periphery  16 x 16 patches (224 px) one token  = the mean of 16 of Gemma's tokens

The encoder still sees the whole 896 px frame, so the saving is in LLM tokens (the dominant
cost with ~28 frames per question), and nothing in the fovea drifts.

Layout: Gemma's 16 x 16 token grid on 896 px has the same geometry as Qwen's on 448 px, so the
cells are qwen_fovea.foveated_cells(img=448) scaled to patch units (same rings, same token
counts as the Qwen runs). Uniform baseline at equal tokens: adaptive average pooling of the
64 x 64 patch grid to s x s tokens, through the same norm + projection.

The LLM input is inputs_embeds (image tokens replaced by these features) with token_type_ids
marking image tokens, as Gemma's processor does, so the bidirectional attention inside each
image is kept. parity_report checks the stock 4 x 4 layout against Gemma's own pipeline.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from vlmeval import qwen_fovea as qf

MODEL_ID = "google/gemma-3-4b-it"
G_IMG, G_PATCH, GRID, BASE = 896, 14, 64, 4          # frame px, patch px, patches per side, patches per token
QWEN_IMG = 448                                       # the Qwen frame with the same 16 x 16 token grid
BOI, EOI, SOFT = "<start_of_image>", "<end_of_image>", "<image_soft_token>"


# ---------------------------------------------------------------------------- layouts (patch units)
def fovea_cells(gu, gv, a, R_deg=None, num_scales=3):
    """Cells (y, x, size) in patches on the 64 x 64 grid: sizes 4 / 8 / 16."""
    c = qf.foveated_cells(gu, gv, a=a, img=QWEN_IMG, R_deg=R_deg, num_scales=num_scales)
    return c * (G_IMG // QWEN_IMG) // G_PATCH       # 448 px -> 896 px -> patches (28 px -> 4)


def stock_cells():
    """Gemma's own tokens: all 4 x 4 cells, raster order (256)."""
    r = np.arange(GRID // BASE) * BASE
    Y, X = np.meshgrid(r, r, indexing="ij")
    return np.stack([Y.ravel(), X.ravel(), np.full(Y.size, BASE)], 1).astype(np.int64)


# ---------------------------------------------------------------------------- pooling
def pool_cells(grid, cells):
    """grid: (64, 64, C) patch features -> (N, C): the mean over each cell, in cells order."""
    g = grid.permute(2, 0, 1).unsqueeze(0).float()                 # 1, C, 64, 64
    out = torch.empty(len(cells), grid.shape[-1], device=grid.device, dtype=torch.float32)
    for k in np.unique(cells[:, 2]):
        idx = np.nonzero(cells[:, 2] == k)[0]
        pooled = F.avg_pool2d(g, int(k))[0]                         # C, 64/k, 64/k
        r = torch.as_tensor(cells[idx, 0] // k, device=grid.device)
        c = torch.as_tensor(cells[idx, 1] // k, device=grid.device)
        out[torch.as_tensor(idx, device=grid.device)] = pooled[:, r, c].T
    return out.to(grid.dtype)


def pool_uniform(grid, side):
    """Uniform baseline: the 64 x 64 grid averaged to side x side tokens, raster order."""
    g = grid.permute(2, 0, 1).unsqueeze(0).float()
    return F.adaptive_avg_pool2d(g, side)[0].flatten(1).T.to(grid.dtype)


def project(proj, feats):
    """Gemma's own RMSNorm + projection for any number of pooled features (N, C) -> (N, D).

    The stock projector takes a 64 x 64 patch grid and averages 4 x 4 blocks; filling each block
    with one pooled feature makes its averaging return that feature exactly, so the projector is
    used as is, whatever its internals are in this transformers version."""
    tokens = (GRID // BASE) ** 2
    out = []
    for i in range(0, len(feats), tokens):
        f = feats[i:i + tokens]
        block = torch.zeros(tokens, f.shape[-1], device=f.device, dtype=f.dtype)
        block[:len(f)] = f
        side = GRID // BASE
        grid = block.view(side, side, -1).repeat_interleave(BASE, 0).repeat_interleave(BASE, 1)
        out.append(_as_tensor(proj(grid.reshape(1, GRID * GRID, -1)))[0, :len(f)])
    return torch.cat(out)


def _as_tensor(out):
    if torch.is_tensor(out):
        return out
    for name in ("pooler_output", "last_hidden_state"):
        t = getattr(out, name, None)
        if t is not None:
            return t
    return out[0]


# ---------------------------------------------------------------------------- model parts
def parts(model):
    """(vision tower, multi-modal projector) across transformers versions."""
    def get(name):
        m = getattr(model, name, None)
        return m if m is not None else getattr(model.model, name)
    return get("vision_tower"), get("multi_modal_projector")


@torch.no_grad()
def vision_grids(vt, imgs01, mean, std, batch=8):
    """SigLIP patch features for a list of (3, 896, 896) frames in [0, 1] -> list of (64, 64, C)."""
    p = next(vt.parameters())
    m = torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    s = torch.as_tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(imgs01), batch):
        x = ((torch.stack(imgs01[i:i + batch]) - m) / s).to(p.device, p.dtype)
        h = vt(pixel_values=x).last_hidden_state                   # B, 4096, C
        out += [hh.view(GRID, GRID, -1) for hh in h]
    return out


# ---------------------------------------------------------------------------- conditions
def frame_tokens(conditions, cond, fr, grid, proj):
    """(LLM features (N, D), cells or None, gaze used) for one frame under one condition.

    conditions: {name: (kind, arg)} with kind in
        'stock'          Gemma's own 4 x 4 pooling (256 tokens)
        'fovea'          arg = dict(a=..., R=..., gaze='gaze' | 'centre', num_scales=3)
        'uniform_match'  arg = name of a 'fovea' condition: uniform pooling to the side whose
                         s x s is closest to that condition's token count for THIS frame
    A missing gaze falls back to the image centre (counted in the results)."""
    kind, arg = conditions[cond]
    if kind == "stock":
        cells = stock_cells()
        return project(proj, pool_cells(grid, cells)), cells, None
    if kind == "uniform_match":
        n = len(_fovea_layout(conditions[arg][1], fr)[0])
        side = max(1, int(round(np.sqrt(n))))
        return project(proj, pool_uniform(grid, side)), None, None
    cells, used = _fovea_layout(arg, fr)
    return project(proj, pool_cells(grid, cells)), cells, used


def _fovea_layout(arg, fr):
    g = (0.5, 0.5) if arg["gaze"] == "centre" else fr.get("gaze")
    used = g
    if g is None:
        g = (0.5, 0.5)
    return fovea_cells(*g, a=arg["a"], R_deg=arg.get("R"), num_scales=arg.get("num_scales", 3)), used


# ---------------------------------------------------------------------------- LLM input
def prompt_text(proc, messages):
    """Chat template text, without a leading <bos> when the tokenizer adds one itself."""
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tok = proc.tokenizer
    bos = tok.bos_token
    if bos and text.startswith(bos) and tok("x").input_ids[:1] == [tok.bos_token_id]:
        text = text[len(bos):]
    return text


def image_affixes(proc):
    """Text around an image's soft tokens, as Gemma's processor writes it."""
    soft = getattr(proc, "image_token", SOFT)
    full = getattr(proc, "full_image_sequence", None)
    if full and soft in full:
        return full[:full.index(soft)], full[full.rindex(soft) + len(soft):], soft
    return "\n\n" + getattr(proc, "boi_token", BOI), getattr(proc, "eoi_token", EOI) + "\n\n", soft


def expand_images(proc, text, n_tokens):
    """Each <start_of_image> -> prefix + n soft tokens + suffix (the processor uses 256)."""
    boi = getattr(proc, "boi_token", BOI)
    pre, suf, soft = image_affixes(proc)
    parts_ = text.split(boi)
    if len(parts_) - 1 != len(n_tokens):
        raise ValueError(f"{len(parts_) - 1} images in the prompt, {len(n_tokens)} given")
    return parts_[0] + "".join(pre + soft * n + suf + p for n, p in zip(n_tokens, parts_[1:]))


@torch.no_grad()
def build_inputs(model, proc, messages, feats_list):
    """inputs_embeds (image tokens replaced by our features), attention mask, token_type_ids."""
    text = expand_images(proc, prompt_text(proc, messages), [len(f) for f in feats_list])
    ids = proc.tokenizer(text, return_tensors="pt").input_ids
    soft_id = getattr(model.config, "image_token_id", None) or proc.tokenizer.convert_tokens_to_ids(SOFT)
    emb_layer = model.get_input_embeddings()
    ids = ids.to(emb_layer.weight.device)
    mask = ids == soft_id
    safe = ids.masked_fill(mask, 0)                     # the soft-token id may be outside the table
    emb = emb_layer(safe)
    feats = torch.cat(feats_list).to(emb.dtype)
    if int(mask.sum()) != len(feats):
        raise ValueError(f"{int(mask.sum())} image tokens in the prompt, {len(feats)} features")
    emb[mask] = feats
    return dict(inputs_embeds=emb, attention_mask=torch.ones_like(ids),
                token_type_ids=mask.long()), ids


@torch.no_grad()
def letter_probs(model, inputs, letter_ids):
    logits = model(**inputs, use_cache=False).logits[0, -1, letter_ids].float()
    return torch.softmax(logits, -1).cpu().numpy()


def load_model(model_id=MODEL_ID):
    from transformers import AutoProcessor
    try:
        from transformers import Gemma3ForConditionalGeneration as Model
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
def parity_report(model, proc, letter_ids, img01, question, options, build_messages):
    """Our path with Gemma's own 4 x 4 layout vs Gemma's processor + model on one real frame:
    pixels, vision features, token ids, answer probabilities."""
    vt, pj = parts(model)
    ip = proc.image_processor
    x = qf.to_uint8_pil(qf.resize01(img01, G_IMG))
    x01 = torch.from_numpy(np.array(x)).permute(2, 0, 1).float() / 255.0
    grid = vision_grids(vt, [x01], ip.image_mean, ip.image_std)[0]
    f_ours = project(pj, pool_cells(grid, stock_cells())).float()

    msgs = build_messages(question, options, [("frame.jpg", 0.0)], G_IMG * G_IMG)
    inputs, ids = build_inputs(model, proc, msgs, [f_ours])
    p_ours = letter_probs(model, inputs, letter_ids)

    text = prompt_text(proc, msgs)
    stock = proc(text=[text], images=[x], return_tensors="pt").to(model.device)
    pv = stock["pixel_values"]
    m = torch.as_tensor(ip.image_mean).view(3, 1, 1)
    s = torch.as_tensor(ip.image_std).view(3, 1, 1)
    pix = ((x01 - m) / s - pv[0].float().cpu()).abs().max().item()
    f_stock = _as_tensor(model.get_image_features(pixel_values=pv.to(next(vt.parameters()).dtype)))
    f_stock = f_stock.reshape(-1, f_stock.shape[-1]).float()
    cos = F.cosine_similarity(f_ours, f_stock, dim=-1)
    p_stock = torch.softmax(model(**stock, use_cache=False).logits[0, -1, letter_ids].float(), -1).cpu().numpy()
    return dict(pixel_max_abs_diff=pix, vision_cos_min=cos.min().item(),
                vision_max_abs_diff=(f_ours - f_stock).abs().max().item(),
                tokens_ids_equal=bool(torch.equal(ids.cpu(), stock["input_ids"].cpu())),
                probs_max_abs_diff=float(np.abs(p_ours - p_stock).max()),
                same_answer=bool(p_ours.argmax() == p_stock.argmax()))


# ---------------------------------------------------------------------------- evaluation
@torch.no_grad()
def run_questions(model, proc, letter_ids, items, conditions, out_path, build_messages, log_every=25):
    """Score multiple-choice questions on their frames under each condition; resumable.

    items as in qwen_fovea.run_questions (frames carry key, path, t, gaze). The SigLIP features
    of a frame are computed once per video and shared by every condition (pooling happens after
    the encoder); pooled + projected tokens are cached per (condition, frame)."""
    vt, pj = parts(model)
    ip = proc.image_processor
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
            if it["video"] != video:
                grids.clear(); toks.clear()
                torch.cuda.empty_cache()
                video = it["video"]
                frs = it["frames"]
                imgs = [qf.resize01(qf.load_frame01(fr["path"], qf.IMG), G_IMG) for fr in frs]
                for fr, g in zip(frs, vision_grids(vt, imgs, ip.image_mean, ip.image_std)):
                    grids[fr["key"]] = g
            t0 = time.time()
            per = []
            for fr in it["frames"]:
                key = (c, fr["key"])
                if key not in toks:
                    f, cells, used = frame_tokens(conditions, c, fr, grids[fr["key"]], pj)
                    toks[key] = (f, cells, conditions[c][0] == "fovea" and used is None)
                per.append(toks[key])
            try:
                msgs = build_messages(it["question"], it["options"],
                                      [(fr["path"], fr["t"]) for fr in it["frames"]], G_IMG * G_IMG)
                inputs, ids = build_inputs(model, proc, msgs, [p[0] for p in per])
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
                cells_by_size={str(int(s) * G_PATCH): int((sizes == s).sum()) for s in np.unique(sizes)},
                frame_px=[G_IMG], n_gaze_missing=int(sum(q[2] for q in per)),
                n_tokens=int(ids.shape[1]), sec=round(time.time() - t0, 3), error=err)) + "\n")
            fh.flush()
            n_new += 1
            if n_new % log_every == 0:
                rate = (time.time() - t_start) / n_new
                print(f"   {n_new}/{len(todo)}  {rate:.2f} s each  ~{rate * (len(todo) - n_new) / 60:.0f} min left"
                      f"  ({c}, {it['video']})", flush=True)
    print(f"done: {n_new} new in {(time.time() - t_start) / 60:.1f} min")

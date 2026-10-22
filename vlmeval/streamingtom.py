"""StreamingTOM (Chen et al., arXiv 2510.18269), its Causal Temporal Reduction (CTR) only, on every VLM
in vlmeval.

CTR, as in the paper and the authors' code (github.com/YIGE24/StreamingTOM, streamingtom/modules/ctr.py):
frames arrive in order and each is reduced, from its N post-projector tokens, to exactly G tokens using
only the previous frame. Token i is static if the cosine of its features with token i of the previous
frame exceeds tau (0.9), dynamic otherwise (every token of the first frame is dynamic). The budget is
split in proportion, G_s = floor(G |S| / N) and G_d = G - G_s (each capped by its set's size, the rest
moved to the other). Dynamic tokens: the G_d with the highest saliency, the attention each token
receives in the vision encoder's last layer averaged over heads and queries. Static tokens:
density-peak clustering (DPC-KNN, k = 7 neighbours, distances scaled by 1 / sqrt(d); see dpc_centres
for the one place the released code differs) picks G_s
centres; every other static token joins its nearest centre, and a merged token is
beta * centre + (1 - beta) * mean(other members), beta = 0.6, at the centre's position. The G tokens
go to the LLM in position (raster) order. Its second stage, 4-bit kv-cache memory with retrieval,
acts inside the LLM and is disabled here, as for every method of the accuracy tables (each is counted
at the LLM input).

Per model: the tokens are the model's own LLM-input tokens at its stock input (the grids of
vlmeval/gazevlm.py), the saliency is the encoder's attention in the layer whose output the model uses
(Phi-3.5: CLIP's second-to-last layer; InternVL: its feature layer), patch scores averaged over the
patches of each token (2 x 2 for Qwen / InternVL / Phi, 4 x 4 Gemma, 3 x 3 SmolVLM). Positions: Qwen's
M-RoPE gets each token at its own cell (a merged token at its centre's); Qwen3-VL's DeepStack features
are reduced with the same plan; Phi gets a row break after each token row that has tokens and neither
its separator nor its 156-token global view (alone above the budgets compared).

The saliency is read without changing any model: while one frame is encoded, transformers' shared
attention registry entry for the model's implementation (sdpa) is wrapped so that, for the one target
attention module only, softmax(q k^T * scale) is accumulated from the q, k it receives (after RoPE),
then the stock function runs as usual.
"""

import json
import os
import time
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from vlmeval import gazevlm as gz

TAU, KNN, BETA = 0.9, 7, 0.6          # the authors' defaults (README: CTR_SIMILARITY_THRESHOLD, CTR_K, CTR_BETA)
DEFAULTS = dict(tau=TAU, knn=KNN, beta=BETA, released_dpc=False)


# ---------------------------------------------------------------------------- saliency
def received_attention(q, k, mask=None, scaling=None, chunk=1024):
    """(B, H, Lq, d) queries, (B, Hk, Lk, d) keys -> (B, Lk): the attention each key receives,
    softmax(q k^T * scale + mask) averaged over heads and queries (chunked over queries)."""
    B, H, Lq, d = q.shape
    if k.shape[1] != H:
        k = k.repeat_interleave(H // k.shape[1], dim=1)
    scale = d ** -0.5 if scaling is None else scaling
    kt = k.float().transpose(-1, -2)
    out = torch.zeros(B, k.shape[2], device=q.device, dtype=torch.float32)
    for s in range(0, Lq, chunk):
        a = torch.matmul(q[:, :, s:s + chunk].float(), kt) * scale
        if mask is not None:
            m = mask[..., s:s + chunk, :]
            a = a.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else a + m.float()
        out += a.softmax(-1).sum(dim=(1, 2))
    return out / (H * Lq)


def cls_attention(q, k, mask=None, scaling=None):
    """[CLS] attention (B, Lk): softmax of the query at position 0 over all keys, averaged over heads."""
    return received_attention(q[:, :, :1], k, None if mask is None else mask[..., :1, :], scaling)


def pseudo_cls_attention(q, k, mask=None, scaling=None, pool=None):
    """Pseudo-[CLS] attention (B, N) for encoders without a [CLS] token (FastVID's recipe for Qwen-VL):
    the mean query over all positions against the keys averaged per LLM token by `pool` ((B, H, L, d)
    -> (B, H, N, d)), softmax over the N tokens, averaged over heads."""
    return received_attention(q.float().mean(2, keepdim=True), pool(k.float()), None, scaling)


def group_pool(g):
    """Key pooling over consecutive groups of g positions (Qwen: a token's 2 x 2 patches are adjacent)."""
    return lambda k: k.unflatten(2, (-1, g)).mean(3)


def block_pool(side, b):
    """Key pooling over the b x b blocks of a side x side raster patch grid (SigLIP patches -> tokens)."""
    return lambda k: k.unflatten(2, (side // b, b, side // b, b)).mean((3, 5)).flatten(2, 3)


@contextmanager
def capture_saliency(target, score=None):
    """Inside: every call of the attention module `target` through transformers' attention registry
    appends score(query, key, mask, scaling) to the yielded list (default: the received attention,
    (B, L)). Other modules are untouched."""
    from transformers.modeling_utils import AttentionInterface
    score = score or received_attention
    impl = target.config._attn_implementation
    mapping = AttentionInterface._global_mapping
    orig = mapping.get(impl)
    if orig is None:
        raise RuntimeError(f"attention implementation {impl!r} is not dispatched through the registry "
                           f"(load the model with attn_implementation='sdpa')")
    got = []

    def wrapped(module, query, key, value, attention_mask=None, *args, **kwargs):
        if module is target:
            got.append(score(query, key, attention_mask, kwargs.get("scaling")))
        return orig(module, query, key, value, attention_mask, *args, **kwargs)

    mapping[impl] = wrapped
    try:
        yield got
    finally:
        mapping[impl] = orig


def _kind(backend):
    """The backend's saliency: 'received' (StreamingTOM, the default), 'cls' (FastVID: [CLS] attention,
    pseudo-[CLS] on encoders without one) or None (DyCoke: tokens only)."""
    return getattr(backend, "saliency", "received")


def _capture(backend, target, cls_score):
    """capture_saliency for the backend's kind (cls_score = its [CLS] / pseudo-[CLS] score), or nothing."""
    kind = _kind(backend)
    if not kind:
        return nullcontext([])
    return capture_saliency(target, cls_score if kind == "cls" else received_attention)


def encoder_layers(m):
    """The ModuleList of encoder layers of a vision tower, across transformers versions."""
    for path in ("encoder.layers", "vision_model.encoder.layers", "encoder.layer", "vision_model.encoder.layer"):
        obj = m
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            return obj
        except AttributeError:
            continue
    raise AttributeError(f"no encoder layers found in {type(m).__name__}")


def pool_saliency(s, side, k):
    """(side * side,) patch saliency, raster -> ((side // k) ** 2,): the mean over k x k patches."""
    return F.avg_pool2d(s.float().view(1, 1, side, side), k).flatten()


# ---------------------------------------------------------------------------- CTR
def dpc_centres(x, n, knn=KNN, released=False, seed=0):
    """Indices (sorted) of the n density-peak centres of the rows of x (DPC-KNN, as the authors'
    _dpc_cluster): density from the k nearest distances (self included), distance to the nearest
    denser point (the densest point: the largest distance), centres = largest density x distance;
    ties broken by a tiny seeded noise (theirs is random). released=True compares densities the way
    the released code does (distance to the nearest LESS dense point: its mask is the transpose of
    Chat-UniVi's batched DPC-KNN, which it follows otherwise)."""
    m = len(x)
    if m <= 1 or n >= m:
        return torch.arange(m, device=x.device)
    d = torch.cdist(x.float(), x.float()) / (x.shape[-1] ** 0.5)
    near = torch.topk(d, min(knn, m - 1), dim=-1, largest=False).values
    rho = (-(near ** 2).mean(-1)).exp()
    rho = rho + torch.rand(m, generator=torch.Generator().manual_seed(seed)).to(rho.device) * 1e-6
    denser = rho[None, :] > rho[:, None]                       # [i, j]: j is denser than i
    delta = torch.where(denser.T if released else denser, d, d.max()).min(-1).values
    return torch.sort(torch.topk(rho * delta, n).indices).values


def ctr_plan(F_t, F_prev, sal, G, tau=TAU, knn=KNN, released_dpc=False):
    """CTR for one frame -> (plan, number of static tokens). F_t, F_prev: (N, D) token features of this
    frame and the previous one (None for the first frame); sal: (N,) saliency; G tokens kept.
    plan: [(position, members)] sorted by position, members = the other static tokens merged into the
    centre at that position ([] for a kept token)."""
    N = len(F_t)
    G = min(int(G), N)
    if F_prev is None:
        static = torch.zeros(N, dtype=torch.bool, device=F_t.device)
    else:
        static = F.cosine_similarity(F_t.float(), F_prev.float(), dim=-1) > tau
    S = torch.nonzero(static).flatten()
    D = torch.nonzero(~static).flatten()
    g_s = int(G * len(S) / N)
    g_d = G - g_s
    if g_s > len(S):
        g_s, g_d = len(S), G - len(S)
    if g_d > len(D):
        g_d, g_s = len(D), G - len(D)
    plan = []
    if g_d > 0:
        top = torch.topk(sal[D].float(), g_d).indices
        plan += [(int(i), []) for i in D[top]]
    if g_s > 0:
        xs = F_t[S]
        cen = dpc_centres(xs, g_s, knn, released_dpc)
        rest = torch.ones(len(S), dtype=torch.bool, device=F_t.device)
        rest[cen] = False
        others = torch.nonzero(rest).flatten()
        if len(others):
            d = torch.cdist(xs[others].float(), xs[cen].float())
            near = d.argmin(1)
        for j, c in enumerate(cen):
            members = [int(S[o]) for o in others[near == j]] if len(others) else []
            plan.append((int(S[c]), members))
    plan.sort(key=lambda p: p[0])
    return plan, int(len(S))


def apply_plan(X, plan, beta=BETA):
    """(N, ...) per-token features -> (len(plan), ...): kept tokens as they are, a merged token
    beta * centre + (1 - beta) * mean(its other members)."""
    out = []
    for pos, members in plan:
        c = X[pos]
        if members:
            idx = torch.as_tensor(members, device=X.device)
            c = (beta * c.float() + (1 - beta) * X[idx].float().mean(0)).to(X.dtype)
        out.append(c)
    return torch.stack(out)


# ---------------------------------------------------------------------------- per-model backends
class Qwen25(gz.Qwen25):
    """Qwen2.5-VL: 256 tokens of 28 px; saliency from its last ViT block (a full-attention block)."""
    def frame_state(self, fr):
        qf = self.qf
        img = qf.resize01(qf.load_frame01(fr["path"], qf.IMG), self.px)
        cells = qf.uniform_cells(self.px)
        pseudo = lambda q, k, m, s: pseudo_cls_attention(q, k, m, s, group_pool(4))
        with _capture(self, self.vis.blocks[-1].attn, pseudo) as got:
            tok = qf.encode_cells(self.vis, img, cells, self.px, self.mean, self.std)
        st = dict(tokens=tok)
        if got:
            order, _ = qf.window_groups(cells, self.px)           # the encoder's cell order (4 patches each)
            seq = torch.cat([g[0] for g in got])
            if _kind(self) == "received":
                seq = seq.view(-1, 4).mean(1)
            st["sal"] = torch.empty_like(seq)
            st["sal"][order.to(seq.device)] = seq
        return st

    def pack(self, st, plan, beta):
        cells = self.qf.uniform_cells(self.px)[[p for p, _ in plan]]
        return apply_plan(st["tokens"], plan, beta), cells, self.px


class Qwen3(gz.Qwen3):
    """Qwen3-VL: 256 tokens of 32 px plus DeepStack features; saliency from its last ViT block."""
    def frame_state(self, fr):
        qf, q3 = self.qf, self.q3
        img = qf.resize01(qf.load_frame01(fr["path"], qf.IMG), q3.IMG)
        cells = q3.uniform_cells(q3.IMG)
        pseudo = lambda q, k, m, s: pseudo_cls_attention(q, k, m, s, group_pool(4))
        with _capture(self, self.vis.blocks[-1].attn, pseudo) as got:
            tok, deep = q3.encode_cells(self.vis, img, cells, q3.IMG, self.mean, self.std)
        st = dict(tokens=tok, deep=deep)
        if got:
            order = torch.as_tensor(np.lexsort((cells[:, 1], cells[:, 0])))   # the encoder's (raster) order
            seq = torch.cat([g[0] for g in got])
            if _kind(self) == "received":
                seq = seq.view(-1, 4).mean(1)
            st["sal"] = torch.empty_like(seq)
            st["sal"][order.to(seq.device)] = seq
        return st

    def pack(self, st, plan, beta):
        cells = self.q3.uniform_cells(self.q3.IMG)[[p for p, _ in plan]]
        return (apply_plan(st["tokens"], plan, beta), [apply_plan(d, plan, beta) for d in st["deep"]],
                cells, self.q3.IMG)


class InternVL(gz.InternVL):
    """InternVL3.5: 256 tokens of 28 px; saliency from InternViT's feature layer (CLS dropped)."""
    def frame_state(self, fr):
        vt, _, _, _, layer = self.iv.parts(self.model)
        img = self.qf.load_frame01(fr["path"], self.qf.IMG)
        with _capture(self, encoder_layers(vt)[layer].attention, cls_attention) as got:
            tok = self.iv.encode_uniform(self.model, img, self.iv.TILE, self.mean, self.std)
        st = dict(tokens=tok)
        if got:
            s = got[0][0, 1:]
            st["sal"] = pool_saliency(s, int(round(len(s) ** 0.5)), 2)
        return st

    def pack(self, st, plan, beta):
        return (apply_plan(st["tokens"], plan, beta),)


class Gemma(gz.Gemma):
    """Gemma-3: its 256 projected tokens (4 x 4 patch averages); saliency from SigLIP's last layer."""
    def frame_state(self, fr):
        qf, gf = self.qf, self.gf
        img = qf.resize01(qf.load_frame01(fr["path"], qf.IMG), gf.G_IMG)
        pseudo = lambda q, k, m, s: pseudo_cls_attention(q, k, m, s, block_pool(gf.GRID, gf.BASE))
        with _capture(self, encoder_layers(self.vt)[-1].self_attn, pseudo) as got:
            grid = gf.vision_grids(self.vt, [img], self.mean, self.std)[0]
        st = dict(tokens=gf.project(self.pj, gf.pool_cells(grid, gf.stock_cells())))
        if got:
            st["sal"] = got[0][0] if _kind(self) == "cls" else pool_saliency(got[0][0], gf.GRID, gf.BASE)
        return st

    def pack(self, st, plan, beta):
        return (apply_plan(st["tokens"], plan, beta),)


class Phi(gz.Phi):
    """Phi-3.5-vision: its 576 crop tokens of the 672 px view, projected; saliency from CLIP's feature
    layer (the second-to-last) on the 4 crops; a row break after each token row that has tokens."""
    def frame_state(self, fr):
        ph, img = self.ph, self.img
        hd01, _ = ph.views(ph.load_pil(fr["path"]))
        with _capture(self, img.clip.encoder.layers[-2].self_attn, cls_attention) as got:
            grid = ph.crop_grid(img, hd01)                                       # (24, 24, 4096)
        st = dict(tokens=img.img_projection(grid.reshape(-1, grid.shape[-1])))
        if got:
            s = got[0][:, 1:]                                                    # 4 crops, CLS dropped
            p = int(round(s.shape[1] ** 0.5))
            s = F.avg_pool2d(s.float().view(-1, 1, p, p), 2)                     # (4, 1, 12, 12)
            st["sal"] = ph.arrange_crops(s.permute(0, 2, 3, 1)).reshape(-1)
        return st

    def pack(self, st, plan, beta):
        x = apply_plan(st["tokens"], plan, beta)
        br = self.img.img_projection(self.img.sub_GN.reshape(1, -1).to(x.dtype))
        rows = np.array([p // self.side for p, _ in plan])
        out = []
        for r in np.unique(rows):
            out += [x[torch.as_tensor(np.nonzero(rows == r)[0], device=x.device)], br]
        return (torch.cat(out),)


class SmolVLM(gz.SmolVLM):
    """SmolVLM2: 81 tokens of 42 px; saliency from its SigLIP's last layer (3 x 3 patches per token)."""
    def frame_state(self, fr):
        from PIL import Image
        sm = self.sm
        vis, _ = sm.parts(self.model)
        rec = sm.stock_input(self.ip, Image.open(fr["path"]).convert("RGB"))
        side = sm.IMG // sm.PATCH                                                # 27 patches per side
        pseudo = lambda q, k, m, s: pseudo_cls_attention(q, k, m, s, block_pool(side, 3))
        with _capture(self, encoder_layers(vis)[-1].self_attn, pseudo) as got:
            tok = sm.encode_uniform(self.model, rec, sm.IMG, self.mean, self.std)
        st = dict(tokens=tok)
        if got:
            st["sal"] = got[0][0] if _kind(self) == "cls" else pool_saliency(got[0][0], side, 3)
        return st

    def pack(self, st, plan, beta):
        return (apply_plan(st["tokens"], plan, beta),)


# ---------------------------------------------------------------------------- evaluation
def budget(ratio, n_tokens):
    """G tokens per frame for a ratio of the frame's N selectable tokens (at least 1)."""
    return max(1, int(round(ratio * n_tokens)))


@torch.no_grad()
def run_questions(backend, items, conditions, out_path, build_messages, log_every=25):
    """Score questions on their frames with CTR-reduced tokens; resumable.

    conditions: {name: dict(ratio=G / N, and optionally tau, knn, beta, released_dpc)}.
    items: as for the *_fovea.run_questions (sorted by video), each item's frames in time order: a
    frame is reduced from its own tokens and its predecessor's only (causal). Each frame is encoded once
    per video; its tokens are made once per (condition, predecessor, frame)."""
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                done.add((r["condition"], r["qid"]))
    todo = [(it, c) for it in items for c in conditions if (c, it["qid"]) not in done]
    print(f"{len(done)} already scored, {len(todo)} to go -> {out_path}")

    states, toks, video, t_start, n_new = {}, {}, None, time.time(), 0
    with open(out_path, "a") as fh:
        for it, c in todo:
            if it["video"] != video:                         # new video: drop the old frames
                states.clear()
                toks.clear()
                torch.cuda.empty_cache()
                video = it["video"]
            arg = {**DEFAULTS, **conditions[c]}
            t0 = time.time()
            per, prev = [], None
            for fr in it["frames"]:
                key = (c, prev, fr["key"])
                if fr["key"] not in states:
                    states[fr["key"]] = backend.frame_state(fr)
                    if "sal" not in states[fr["key"]]:
                        raise RuntimeError(f"{type(backend).__name__}: no saliency captured (its target attention "
                                           f"layer was not called through the attention registry)")
                if key not in toks:
                    st = states[fr["key"]]
                    F_prev = states[prev]["tokens"] if prev is not None else None
                    plan, n_static = ctr_plan(st["tokens"], F_prev, st["sal"], budget(arg["ratio"], len(st["tokens"])),
                                              arg["tau"], arg["knn"], arg["released_dpc"])
                    toks[key] = (backend.pack(st, plan, arg["beta"]), n_static / len(st["tokens"]),
                                 sum(1 for _, m in plan if m))
                per.append(toks[key])
                prev = fr["key"]
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
                static_share=round(float(np.mean([q[1] for q in per])), 4) if per else 0.0,
                n_merged_tokens=int(sum(q[2] for q in per)), n_tokens=n_tok,
                streamingtom={k: arg[k] for k in ("ratio", "tau", "knn", "beta", "released_dpc")},
                sec=round(time.time() - t0, 3), error=err)) + "\n")
            fh.flush()
            n_new += 1
            if n_new % log_every == 0:
                rate = (time.time() - t_start) / n_new
                print(f"   {n_new}/{len(todo)}  {rate:.2f} s each  ~{rate * (len(todo) - n_new) / 60:.0f} min left"
                      f"  ({c}, {it['video']})", flush=True)
    print(f"done: {n_new} new in {(time.time() - t_start) / 60:.1f} min")

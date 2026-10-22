"""FastVID (Shen et al., NeurIPS 2025) on every VLM in vlmeval.

As in the paper and the authors' Qwen2.5-VL implementation (github.com/LunarShen/FastVID,
fastvid_qwen25vl/.../modeling_qwen2_5_vl.py), over all K frames of a question (offline):

Dynamic temporal segmentation (DySeg). Each frame's global feature is the mean of its LLM tokens;
t_i = cos(g_i, g_i+1). The video is cut at the c - 1 lowest transitions and at every t_i < tau.

Density spatiotemporal pruning (STPrune), per segment of P frames: it keeps r P N tokens, a share d as
context (density-based token merging, DTM) and the rest as salient tokens (attention-based token
selection, ATS). Inside a segment, a frame nearly identical to the one before it (cosine of the global
features >= ignore) gets no tokens of its own; the salient budget is spread over the remaining (valid)
frames, the context budget over anchor frames, every p-th valid frame counted from the segment's end,
at most N / 2 each.
  ATS: a frame's salient tokens are those with the highest [CLS] attention (a pseudo-[CLS], the mean
  query, against keys averaged per LLM token, where the encoder has no [CLS] token), in the vision
  encoder's last (feature) layer.
  DTM: an anchor frame's context tokens are density peaks (DPC-KNN, 4 neighbours) among its non-salient
  tokens; every other token of the segment, salient ones and this frame's anchors excepted, joins its most
  similar anchor (cosine), and an anchor with n members becomes w a + (1 - w) mean(members),
  w = max(1 / (n + 1), beta). Anchors keep their own positions.
The kept tokens go to the LLM frame by frame in position (raster) order; a frame left without tokens is
left out of the prompt with its time label (FastVID's video block simply loses it).

Hyperparameters: the paper's c = 8, tau = 0.9, d = 0.4, p = 4, beta = 0.6 (the authors' Qwen2.5-VL run
used tau = 0.84) and the Qwen2.5-VL code's near-duplicate threshold ignore = 0.95; r is the budget.

Per model ([CLS] / pseudo-[CLS] attention and tokens come from the backends of vlmeval/streamingtom.py
with saliency = 'cls'): the real [CLS] token for InternViT and Phi's CLIP (feature layer, patch
attention averaged per token), a pseudo-[CLS] for Qwen2.5-VL, Qwen3-VL (keys averaged over each token's
2 x 2 patches), Gemma (4 x 4) and SmolVLM (3 x 3). Qwen's M-RoPE gets every kept token at its own cell,
Qwen3-VL's DeepStack features are kept and merged with the same tokens and weights, Phi's 576 crop tokens
get a row break after each row that has tokens (no global view).
"""

import json
import os
import time

import numpy as np
import torch

from vlmeval import streamingtom as stm

C, TAU, IGNORE, D, P, BETA, KNN = 8, 0.9, 0.95, 0.4, 4, 0.6, 4
DEFAULTS = dict(c=C, tau=TAU, ignore=IGNORE, d=D, p=P, beta=BETA)


# ---------------------------------------------------------------------------- the method
def _cuts_to_sizes(cuts, n):
    """Boundaries after frames `cuts` (sorted) -> segment sizes over n frames."""
    sizes, prev = [], -1
    for i in cuts:
        sizes.append(i - prev)
        prev = i
    sizes.append(n - 1 - prev)
    return sizes


def _transitions(G):
    g = G.float() / G.float().norm(dim=1, keepdim=True)
    return (g[:-1] * g[1:]).sum(1)


def segments(G, c=C, tau=TAU):
    """DySeg: (F, D) global frame features -> segment sizes (cut at the c - 1 lowest transitions and
    at every transition below tau)."""
    n = len(G)
    if n <= 1:
        return [n]
    t = _transitions(G)
    cuts = set(torch.topk(t, min(c - 1, len(t)), largest=False).indices.tolist()) if c > 1 else set()
    cuts |= set(torch.nonzero(t < tau).flatten().tolist())
    return _cuts_to_sizes(sorted(cuts), n)


def budgets(G_seg, N, r, d=D, p=P, ignore=IGNORE):
    """Salient and context token counts per frame of one segment (the authors' Qwen2.5-VL allocation)."""
    n = len(G_seg)
    seg_retain = int(N * r) * n
    seg_context = max(int(seg_retain * d), 1)
    seg_salient = max(seg_retain - seg_context, 0)
    if n == 1:
        return [seg_salient], [min(seg_context, N - seg_salient)]
    sub = _cuts_to_sizes(sorted(torch.nonzero(_transitions(G_seg) < ignore).flatten().tolist()), n)
    V = len(sub)                                                         # valid frames: the last of each run
    sal_v = [seg_salient // V + (1 if i < seg_salient % V else 0) for i in range(V)]
    T = (V + p - 1) // p
    ctx_t = [seg_context // T + (1 if i < seg_context % T else 0) for i in range(T)]
    ctx_v, t = [], 0
    for i in range(V):
        if i % p == 0:
            ctx_v.append(min(ctx_t[t], N // 2))
            t += 1
        else:
            ctx_v.append(0)
    ctx_v.reverse()                                                      # anchor frames from the segment's end
    sal, ctx = [], []
    for j, size in enumerate(sub):
        sal += [0] * (size - 1)
        ctx += [0] * (size - 1)
        ctx.append(min(ctx_v[j], N // 2))
        sal.append(min(sal_v[j], N - ctx[-1]))
    return sal, ctx


def merge_rows(srcs, Xn, others, anchors, beta=BETA):
    """DTM's anchor-centric aggregation: every token in `others` (flat indices) joins the anchor (flat
    indices) it is most similar to (cosine, Xn = normalized tokens), and an anchor with n members becomes
    w a + (1 - w) mean(members), w = max(1 / (n + 1), beta). -> merged anchor rows of each tensor in srcs."""
    one_hot = torch.zeros(len(others), len(anchors), device=Xn.device)
    if len(others):
        one_hot[torch.arange(len(others), device=Xn.device), (Xn[others] @ Xn[anchors].T).argmax(1)] = 1
    count = one_hot.sum(0)[:, None]
    w = (1 / (count + 1)).clamp(min=beta)
    out = []
    for src in srcs:
        agg = (one_hot.T @ src[others].float()) / count.clamp(min=1)
        out.append((w * src[anchors].float() + (1 - w) * agg).to(src.dtype))
    return out


def stprune(feats, sals, n_sal, n_ctx, beta=BETA, knn=KNN, extra=()):
    """STPrune for one segment. feats: per frame (N, D) tokens; sals: per frame (N,) [CLS] attention;
    n_sal / n_ctx: per-frame counts; extra: per frame lists of (N, D') features merged with the same
    assignment (Qwen3-VL's DeepStack). -> per frame (kept positions sorted, {anchor position: [merged
    token, merged extra...]})."""
    n, N = len(feats), len(feats[0])
    dev = feats[0].device
    salient, anchors = [], []
    for f in range(n):
        s = torch.topk(sals[f].float(), n_sal[f]).indices.to(dev) if n_sal[f] > 0 else torch.zeros(0, dtype=torch.long, device=dev)
        a = torch.zeros(0, dtype=torch.long, device=dev)
        if n_ctx[f] > 0:
            rest = torch.ones(N, dtype=torch.bool, device=dev)
            rest[s] = False
            rest = torch.nonzero(rest).flatten()
            a = rest[stm.dpc_centres(feats[f][rest], n_ctx[f], knn)]
        salient.append(s.to(dev))
        anchors.append(a)
    X = torch.cat(feats)                                                 # (n N, D), frame-major
    Xn = X.float() / X.float().norm(dim=-1, keepdim=True)
    extras = [torch.cat([extra[f][k] for f in range(n)]) for k in range(len(extra[0]))] if extra else []
    sal_flat = torch.cat([s + f * N for f, s in enumerate(salient)])
    merged = [dict() for _ in range(n)]
    for f, a in enumerate(anchors):
        if not len(a):
            continue
        a_flat = a + f * N
        keep = torch.ones(n * N, dtype=torch.bool, device=dev)
        keep[sal_flat] = False
        keep[a_flat] = False
        others = torch.nonzero(keep).flatten()
        for rows in merge_rows([X] + extras, Xn, others, a_flat, beta):
            for j, pos in enumerate(a.tolist()):
                merged[f].setdefault(pos, []).append(rows[j])
    out = []
    for f in range(n):
        kept = sorted(set(salient[f].tolist()) | set(anchors[f].tolist()))
        out.append((kept, merged[f]))
    return out


def fastvid_plans(states, r, c=C, tau=TAU, ignore=IGNORE, d=D, p=P, beta=BETA, knn=KNN):
    """FastVID over one question's frames. states: per frame dict(tokens=(N, D), sal=(N,), and optionally
    deep=[(N, D')...]). -> per frame (kept positions, {anchor position: merged rows}), and the segment sizes."""
    G = torch.stack([st["tokens"].float().mean(0) for st in states])
    sizes = segments(G, c, tau)
    out, f0 = [], 0
    for size in sizes:
        seg = states[f0:f0 + size]
        N = len(seg[0]["tokens"])
        n_sal, n_ctx = budgets(G[f0:f0 + size], N, r, d, p, ignore)
        extra = [st["deep"] for st in seg] if "deep" in seg[0] else ()
        out += stprune([st["tokens"] for st in seg], [st["sal"] for st in seg], n_sal, n_ctx, beta, knn, extra)
        f0 += size
    return out, sizes


def apply_merges(st, merged):
    """A frame's state with its anchor rows replaced by their merged values (tokens, then DeepStack)."""
    if not merged:
        return st
    out = dict(st, tokens=st["tokens"].clone())
    if "deep" in st:
        out["deep"] = [d.clone() for d in st["deep"]]
    for pos, rows in merged.items():
        out["tokens"][pos] = rows[0]
        for k, row in enumerate(rows[1:]):
            out["deep"][k][pos] = row
    return out


# ---------------------------------------------------------------------------- per-model backends
class Qwen25(stm.Qwen25):
    saliency = "cls"


class Qwen3(stm.Qwen3):
    saliency = "cls"


class InternVL(stm.InternVL):
    saliency = "cls"


class Gemma(stm.Gemma):
    saliency = "cls"


class Phi(stm.Phi):
    saliency = "cls"


class SmolVLM(stm.SmolVLM):
    saliency = "cls"


# ---------------------------------------------------------------------------- evaluation
@torch.no_grad()
def run_questions(backend, items, conditions, out_path, build_messages, log_every=25):
    """Score questions on their frames with FastVID-reduced tokens; resumable.

    conditions: {name: dict(r=..., and optionally c, tau, ignore, d, p, beta)}.
    items: as for the *_fovea.run_questions (sorted by video), each item's frames in time order. Each
    frame is encoded once per video; a video's reduced frames are made once per (condition, frame list)."""
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
            keys = tuple(fr["key"] for fr in it["frames"])
            if (c, keys) not in toks:
                for fr in it["frames"]:
                    if fr["key"] not in states:
                        states[fr["key"]] = backend.frame_state(fr)
                        if "sal" not in states[fr["key"]]:
                            raise RuntimeError(f"{type(backend).__name__}: no [CLS] attention captured")
                plans, sizes = fastvid_plans([states[k] for k in keys], float(arg["r"]), int(arg["c"]),
                                             float(arg["tau"]), float(arg["ignore"]), float(arg["d"]),
                                             int(arg["p"]), float(arg["beta"]))
                kept_frames, per = [], []
                for fr, k, (kept, merged) in zip(it["frames"], keys, plans):
                    if kept:
                        kept_frames.append(fr)
                        per.append(backend.pack(apply_merges(states[k], merged), [(q, []) for q in kept], 0.0))
                toks[(c, keys)] = (kept_frames, per, len(sizes))
            kept_frames, per, n_segments = toks[(c, keys)]
            try:
                p, n_tok = backend.score(dict(it, frames=kept_frames), per, build_messages)
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
                n_frames=len(per), n_frames_dropped=len(keys) - len(per),
                n_image_tokens=int(sum(len(q[0]) for q in per)), n_segments=n_segments, n_tokens=n_tok,
                fastvid={k: arg[k] for k in ("r", "c", "tau", "ignore", "d", "p", "beta")},
                sec=round(time.time() - t0, 3), error=err)) + "\n")
            fh.flush()
            n_new += 1
            if n_new % log_every == 0:
                rate = (time.time() - t_start) / n_new
                print(f"   {n_new}/{len(todo)}  {rate:.2f} s each  ~{rate * (len(todo) - n_new) / 60:.0f} min left"
                      f"  ({c}, {it['video']})", flush=True)
    print(f"done: {n_new} new in {(time.time() - t_start) / 60:.1f} min")

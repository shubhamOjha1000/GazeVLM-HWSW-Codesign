"""PruneVid (Huang, Zhou, Chen, Wang & Han, ACL Findings 2025) on every VLM in vlmeval: its
spatial-temporal token merging, before the LLM.

As the authors' code (github.com/Visual-AI/PruneVid, models/pllava/modeling_pllava.py:
merge_frames_dynamic, cluster_dpc_knn, refine_clusters, compute_cluster_vectors; settings from its
README), over all K frames of a question (offline), on the LLM-level tokens (after the projector, as there):

Temporal segmentation. The frames are clustered by DPC-KNN (k = 7) on their mean token into
int(gamma T) clusters (gamma = 0.25, temporal_segment_ratio); the labels are then made contiguous
(refine_clusters: each cluster keeps its longest runs, its other runs, and clusters whose runs are all
one frame long, take the label of the longer neighbouring run, the left one on a tie) and every run is
a segment.
Static / dynamic tokens. In a segment of W frames the token at position i is static if the mean pairwise
cosine of its W copies is > tau (tau = 0.8); a one-frame segment has none (the released code divides 0 by
0 there, and the NaN fails the test). Static tokens are averaged over the segment (one per position);
dynamic tokens stay per frame.
Spatial merging. The segment's static tokens, and each frame's dynamic tokens, are clustered by DPC-KNN
(k = 7) into int(beta n) clusters (beta = 0.5, cluster_ratio) when there are more than 14 of them; a
cluster becomes the mean of its tokens.
PruneVid's second stage, question-guided pruning at LLM layer 10 (alpha), is not used: the answer is
the first token's logits, computed before it acts (the accuracy tables' protocol, as UTOPI did).

Packing. PruneVid's PLLaVA feeds one flat sequence (per segment its static tokens, then each frame's
dynamic tokens). Here a segment's merged static tokens go with its first frame, every frame keeps its
merged dynamic tokens, and each merged token takes its cluster centre's position (Qwen's M-RoPE: the
centre's cell at that frame's time; the others: position (raster) order). Static and dynamic positions
are disjoint, so no position repeats within a frame. A frame left without tokens (all static, not the
first of its segment) leaves the prompt with its time label (`n_frames_dropped`).

Per model, the tokens come from the backends of vlmeval/streamingtom.py with saliency = False (no
attention needed), as for DyCoke: Qwen2.5-VL, Qwen3-VL (DeepStack features averaged with the same
segments and clusters), InternVL, Gemma (projected tokens) and SmolVLM their LLM tokens; Phi its 576
crop tokens with a row break after each row that has tokens (no global view).
Budget: beta (and tau) set the share of tokens kept; it depends on the content and is measured per
question (`n_image_tokens`).
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as nnf

from vlmeval import streamingtom as stm
from vlmeval.fastvid import apply_merges

GAMMA, TAU, BETA, KNN, MIN_TOKENS = 0.25, 0.8, 0.5, 7, 14
DEFAULTS = dict(gamma=GAMMA, tau=TAU, beta=BETA, knn=KNN, min_tokens=MIN_TOKENS)


# ---------------------------------------------------------------------------- the method
def cluster(x, n, knn=KNN):
    """DPC-KNN as the authors' cluster_dpc_knn: the n density-peak centres of the rows of x (indices,
    sorted; stm.dpc_centres is the same density, distance and score) and each row's cluster (index into
    the centres): its nearest centre, a centre its own."""
    centres = stm.dpc_centres(x, n, knn)
    labels = torch.cdist(x.float(), x[centres].float()).argmin(1)
    labels[centres] = torch.arange(len(centres), device=x.device)
    return centres, labels


def refine_clusters(labels):
    """The authors' refine_clusters for one video: per-frame cluster labels -> labels whose runs are
    the segments. Each cluster keeps its longest run(s); its other runs, and every run of a cluster whose
    runs are all one frame long, are relabelled with the neighbouring run that is longer (left on a
    tie; 0 if there is none)."""
    lab = [int(v) for v in labels]
    T, out = len(lab), list(lab)
    for c in sorted(set(lab)):
        idx = [i for i, v in enumerate(lab) if v == c]
        runs, start, prev = [], idx[0], idx[0]
        for i in idx[1:]:
            if i == prev + 1:
                prev = i
            else:
                runs.append((start, prev))
                start = prev = i
        runs.append((start, prev))
        longest = max(e - s + 1 for s, e in runs)
        for s, e in runs:
            if longest == 1 or e - s + 1 != longest:
                out[s:e + 1] = [-1] * (e - s + 1)
    i = 0
    while i < T:
        if out[i] != -1:
            i += 1
            continue
        s = i
        while i < T and out[i] == -1:
            i += 1
        e = i - 1
        left = right = None
        n_left = n_right = 0
        if s > 0:
            left, j = out[s - 1], s - 1
            while j >= 0 and out[j] == left:
                j -= 1
            n_left = s - j - 1
        if e < T - 1:
            right, j = out[e + 1], e + 1
            while j < T and out[j] == right:
                j += 1
            n_right = j - e - 1
        new = left if n_left > n_right else right if n_right > n_left else (left if left is not None else right)
        out[s:e + 1] = [0 if new is None else new] * (e - s + 1)
    return out


def run_lengths(labels):
    sizes, run = [], 1
    for a, b in zip(labels[:-1], labels[1:]):
        if a == b:
            run += 1
        else:
            sizes.append(run)
            run = 1
    return sizes + [run]


def segments(G, gamma=GAMMA, knn=KNN):
    """(T, D) mean frame tokens -> segment sizes: int(gamma T) DPC-KNN clusters (at least one), made
    contiguous by refine_clusters."""
    T = len(G)
    if T <= 1:
        return [T]
    _, labels = cluster(G, min(max(int(T * gamma), 1), T), knn)
    return run_lengths(refine_clusters(labels.tolist()))


def static_positions(X, tau=TAU):
    """(W, N, D) a segment's tokens -> (N,) bool: static where the mean pairwise cosine of the W copies of
    a position is > tau, computed as the authors' (sum of the W x W cosines - W) / (W (W - 1))."""
    W = len(X)
    if W < 2:
        return torch.zeros(X.shape[1], dtype=torch.bool, device=X.device)
    Xn = nnf.normalize(X.float(), dim=-1)
    sim = torch.einsum("wnc,tnc->wtn", Xn, Xn)
    return (sim.sum(1) - 1).sum(0) / (W * (W - 1)) > tau


def spatial_merge(srcs, beta=BETA, knn=KNN, min_tokens=MIN_TOKENS):
    """One set of tokens, srcs = [(m, D) tokens, (m, D') extra features...]: when m > min_tokens,
    int(beta m) DPC-KNN clusters on the tokens (at least one), each cluster the mean of its rows (extra
    features with the same clusters). -> (centre indices, sorted; the rows of each src, one per centre)."""
    x = srcs[0]
    m = len(x)
    if m <= min_tokens:
        return torch.arange(m, device=x.device), list(srcs)
    centres, labels = cluster(x, max(int(m * beta), 1), knn)
    one_hot = nnf.one_hot(labels, len(centres)).float()
    count = one_hot.sum(0)[:, None]                                     # >= 1: a centre is its own member
    return centres, [((one_hot.T @ s.float()) / count).to(s.dtype) for s in srcs]


def prunevid_plans(states, gamma=GAMMA, tau=TAU, beta=BETA, knn=KNN, min_tokens=MIN_TOKENS):
    """PruneVid over one question's frames. states: per frame dict(tokens=(N, D), optionally
    deep=[(N, D')...]). -> per frame {position: [merged token, merged deep...]} (empty: the frame has no
    tokens), the segment sizes and the share of the frames' tokens that were static."""
    G = torch.stack([st["tokens"].float().mean(0) for st in states])
    sizes = segments(G, gamma, knn)
    out, f0, n_static = [], 0, 0
    for W in sizes:
        seg = states[f0:f0 + W]
        X = torch.stack([st["tokens"] for st in seg])                  # (W, N, D)
        deep = [torch.stack([st["deep"][k] for st in seg]) for k in range(len(seg[0]["deep"]))] if "deep" in seg[0] else []
        static = static_positions(X, tau)
        sp, dp = torch.nonzero(static).flatten(), torch.nonzero(~static).flatten()
        n_static += len(sp) * W
        rows = [dict() for _ in range(W)]
        if len(sp):                                                     # averaged over time, with the first frame
            mean = lambda A: A[:, sp].float().mean(0).to(A.dtype)
            c, merged = spatial_merge([mean(X)] + [mean(d) for d in deep], beta, knn, min_tokens)
            for j, pos in enumerate(sp[c].tolist()):
                rows[0][pos] = [m[j] for m in merged]
        if len(dp):
            for f in range(W):
                c, merged = spatial_merge([X[f, dp]] + [d[f, dp] for d in deep], beta, knn, min_tokens)
                for j, pos in enumerate(dp[c].tolist()):
                    rows[f][pos] = [m[j] for m in merged]
        out += rows
        f0 += W
    return out, sizes, n_static / sum(len(st["tokens"]) for st in states)


# ---------------------------------------------------------------------------- per-model backends
class Qwen25(stm.Qwen25):
    saliency = False


class Qwen3(stm.Qwen3):
    saliency = False


class InternVL(stm.InternVL):
    saliency = False


class Gemma(stm.Gemma):
    saliency = False


class Phi(stm.Phi):
    saliency = False


class SmolVLM(stm.SmolVLM):
    saliency = False


# ---------------------------------------------------------------------------- evaluation
@torch.no_grad()
def run_questions(backend, items, conditions, out_path, build_messages, log_every=25):
    """Score questions on their frames with PruneVid-merged tokens; resumable.

    conditions: {name: dict(tau=..., beta=..., and optionally gamma, knn, min_tokens)}.
    items: as for the *_fovea.run_questions (sorted by video), each item's frames in time order. Each
    frame is encoded once per video; a video's merged frames are made once per (condition, frame list)."""
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
                plans, sizes, static_share = prunevid_plans(
                    [states[k] for k in keys], float(arg["gamma"]), float(arg["tau"]), float(arg["beta"]),
                    int(arg["knn"]), int(arg["min_tokens"]))
                kept_frames, per = [], []
                for fr, k, rows in zip(it["frames"], keys, plans):
                    if rows:
                        kept_frames.append(fr)
                        per.append(backend.pack(apply_merges(states[k], rows), [(q, []) for q in sorted(rows)], 0.0))
                toks[(c, keys)] = (kept_frames, per, len(sizes), static_share)
            kept_frames, per, n_segments, static_share = toks[(c, keys)]
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
                n_image_tokens=int(sum(len(q[0]) for q in per)), n_segments=n_segments,
                static_share=round(float(static_share), 4), n_tokens=n_tok,
                prunevid={k: arg[k] for k in ("gamma", "tau", "beta", "knn", "min_tokens")},
                sec=round(time.time() - t0, 3), error=err)) + "\n")
            fh.flush()
            n_new += 1
            if n_new % log_every == 0:
                rate = (time.time() - t_start) / n_new
                print(f"   {n_new}/{len(todo)}  {rate:.2f} s each  ~{rate * (len(todo) - n_new) / 60:.0f} min left"
                      f"  ({c}, {it['video']})", flush=True)
    print(f"done: {n_new} new in {(time.time() - t_start) / 60:.1f} min")

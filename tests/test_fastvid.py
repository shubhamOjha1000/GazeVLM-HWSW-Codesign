"""FastVID baseline (vlmeval/fastvid.py). CPU only, no models, no data.

  1. [CLS] / pseudo-[CLS] attention and the key pooling per encoder
  2. DySeg: cuts at the c - 1 lowest transitions and below tau
  3. budgets: the authors' allocation (single frame, segment, near-duplicate frames, the N / 2 cap)
  4. DTM's merge: w a + (1 - w) mean(members), w = max(1 / (n + 1), beta)
  5. STPrune / whole plans: salient + anchor tokens per frame, positions kept
  6. run_questions: frames left without tokens leave the prompt, records, resumable
"""
import json

import numpy as np
import pytest
import torch

from vlmeval import fastvid as fv
from vlmeval import streamingtom as stm


# ---------------------------------------------------------------------------- attention
def test_cls_and_pseudo_cls_attention():
    g = torch.Generator().manual_seed(0)
    q, k = torch.randn(1, 2, 5, 4, generator=g), torch.randn(1, 2, 5, 4, generator=g)
    a = stm.cls_attention(q, k)
    ref = (torch.matmul(q[:, :, :1], k.transpose(-1, -2)) * 0.5).softmax(-1).mean(dim=(1, 2))
    assert torch.allclose(a, ref, atol=1e-6) and a.shape == (1, 5)
    k4 = torch.randn(1, 2, 8, 4, generator=g)                                   # 2 tokens of 4 patches each
    q4 = torch.randn(1, 2, 8, 4, generator=g)
    p = stm.pseudo_cls_attention(q4, k4, pool=stm.group_pool(4))
    kbar = k4.view(1, 2, 2, 4, 4).mean(3)
    ref = (torch.matmul(q4.mean(2, keepdim=True), kbar.transpose(-1, -2)) * 0.5).softmax(-1).mean(dim=(1, 2))
    assert torch.allclose(p, ref, atol=1e-6) and p.shape == (1, 2)


def test_block_pool_averages_square_blocks():
    k = torch.arange(36.0).view(1, 1, 36, 1)                                     # a 6 x 6 raster grid
    out = stm.block_pool(6, 3)(k)[0, 0, :, 0]
    grid = torch.arange(36.0).view(6, 6)
    assert torch.allclose(out, torch.stack([grid[:3, :3].mean(), grid[:3, 3:].mean(),
                                            grid[3:, :3].mean(), grid[3:, 3:].mean()]))


# ---------------------------------------------------------------------------- DySeg and budgets
def _globals(groups, d=16, seed=0):
    """Global features: frames of group i all along direction i (plus a little noise)."""
    g = torch.Generator().manual_seed(seed)
    rows = [torch.eye(d)[i] + 0.01 * torch.randn(d, generator=g) for i, n in enumerate(groups) for _ in range(n)]
    return torch.stack(rows)


def test_dyseg_cuts():
    G = _globals([4, 3, 3])
    assert fv.segments(G, c=1, tau=0.9) == [4, 3, 3]                           # only the two real transitions
    sizes = fv.segments(_globals([10]), c=3, tau=0.0)                          # one scene: the c - 1 lowest
    assert len(sizes) == 3 and sum(sizes) == 10
    assert fv.segments(G[:1]) == [1]


def test_budgets_single_frame_and_segment():
    assert fv.budgets(_globals([1]), 100, 0.1) == ([6], [4])                   # 10 kept: 4 context, 6 salient
    sal, ctx = fv.budgets(_globals([1, 1, 1, 1]), 100, 0.1)                    # 4 distinct frames, 40 kept
    assert sal == [6, 6, 6, 6] and ctx == [0, 0, 0, 16]                        # the anchor frame is the last


def test_budgets_give_near_duplicates_nothing():
    G = _globals([1, 1])
    G = torch.stack([G[0], G[0], G[0], G[1]])                                  # frames 1, 2 repeat frame 0
    sal, ctx = fv.budgets(G, 100, 0.1)
    assert sal == [0, 0, 12, 12] and ctx == [0, 0, 0, 16]


def test_budgets_cap_context_at_half_a_frame():
    sal, ctx = fv.budgets(_globals([1] * 6), 8, 0.5)                           # 24 kept, 9 context over 2 anchor frames
    assert ctx == [0, 4, 0, 0, 0, 4] and sal == [3, 3, 3, 2, 2, 2]


# ---------------------------------------------------------------------------- DTM and plans
def test_merge_rows_weights():
    X = torch.tensor([[1.0, 0], [0, 1], [0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [5.0, 5.0]])
    Xn = X / X.norm(dim=-1, keepdim=True)
    rows, = fv.merge_rows([X], Xn, torch.tensor([2, 3, 4]), torch.tensor([0, 1]), beta=0.6)
    assert torch.allclose(rows[0], 0.6 * X[0] + 0.4 * X[[2, 3]].mean(0))
    assert torch.allclose(rows[1], 0.6 * X[1] + 0.4 * X[4])
    alone, = fv.merge_rows([X], Xn, torch.tensor([2]), torch.tensor([0, 5]), beta=0.6)
    assert torch.allclose(alone[1], X[5])                                     # no members: unchanged


def _states(n_frames, N=8, d=16, seed=1, same=False):
    g = torch.Generator().manual_seed(seed)
    out = []
    for f in range(n_frames):
        base = torch.eye(d)[0 if same else f % d]
        tok = base + (0.0 if same else 0.1) * torch.randn(N, d, generator=g)
        out.append(dict(tokens=tok, sal=torch.arange(float(N)) if f % 2 else torch.arange(float(N)).flip(0)))
    return out


def test_plans_one_frame_segments():
    plans, sizes = fv.fastvid_plans(_states(6), r=0.5)                         # c = 8 > 5 transitions: all cut
    assert sizes == [1] * 6
    for f, (kept, merged) in enumerate(plans):
        top = [5, 6, 7] if f % 2 else [0, 1, 2]                                # 3 salient by attention
        assert len(kept) == 4 and set(top) <= set(kept) and len(merged) == 1    # + 1 merged anchor


def test_plans_one_segment_anchor_frames():
    plans, sizes = fv.fastvid_plans(_states(6), r=0.5, c=1, tau=-2.0)          # no cuts: one segment
    assert sizes == [6]
    assert [len(k) for k, _ in plans] == [3, 7, 3, 2, 2, 6]
    assert [len(m) for _, m in plans] == [0, 4, 0, 0, 0, 4]


def test_apply_merges_replaces_rows():
    st = dict(tokens=torch.zeros(4, 2), deep=[torch.zeros(4, 3)])
    out = fv.apply_merges(st, {2: [torch.ones(2), torch.full((3,), 2.0)]})
    assert torch.equal(out["tokens"][2], torch.ones(2)) and torch.equal(out["deep"][0][2], torch.full((3,), 2.0))
    assert torch.equal(st["tokens"], torch.zeros(4, 2))                        # the cached state is untouched


# ---------------------------------------------------------------------------- the run loop
class FakeBackend:
    def __init__(self, same):
        self.encoded, self.same = [], same

    def frame_state(self, fr):
        self.encoded.append(fr["key"])
        return _states(4, same=self.same)[fr["key"][1]]

    def pack(self, st, plan, beta):
        return (stm.apply_plan(st["tokens"], plan, beta),)

    def score(self, it, per, build_messages):
        assert len(it["frames"]) == len(per)
        p = np.full(5, 0.1)
        p["ABCDE".index(it["answer"])] = 0.6
        return p, 10 + sum(len(f[0]) for f in per)


def _items():
    fr = lambda v, k: dict(key=(v, k), path=f"{v}_{k}.jpg", t=float(k))
    return [dict(qid=q, video=v, question="q", options=list("abcde"), answer="E", frames=[fr(v, k) for k in range(4)])
            for q, v in ((1, "a"), (2, "a"), (3, "b"))]


def test_run_questions_drops_empty_frames_and_resumes(tmp_path):
    out = str(tmp_path / "fastvid.jsonl")
    conds = {"r=0.5": dict(r=0.5, c=1, tau=-2.0)}
    be = FakeBackend(same=True)                                                 # 4 identical frames
    fv.run_questions(be, _items(), conds, out, build_messages=None)
    rows = [json.loads(l) for l in open(out)]
    assert len(rows) == 3 and all(r["correct"] == 1 and r["error"] is None for r in rows)
    assert sorted(be.encoded) == sorted({(v, k) for v in "ab" for k in range(4)})
    r = rows[0]                                                                 # all budget on the last frame
    assert r["n_frames"] == 1 and r["n_frames_dropped"] == 3 and r["n_image_tokens"] == 8 and r["n_segments"] == 1
    assert r["fastvid"]["r"] == 0.5 and r["fastvid"]["d"] == fv.D
    fv.run_questions(FakeBackend(same=True), _items(), conds, out, build_messages=None)
    assert len(open(out).read().splitlines()) == 3

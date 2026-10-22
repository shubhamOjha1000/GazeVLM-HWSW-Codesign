"""Uniform K + average pooling (vlmeval/avgpool.py). CPU only, no models, no data.

  1. pool_side: the s x s grid closest to the budget, per model (Phi's row breaks counted)
  2. the pooled tokens: adaptive average pooling, Qwen positions at bin centres, Gemma before its
     projector, Phi with a row break per pooled row
  3. run_questions: each frame encoded once, pooled once per side, records, resumable
"""
import json
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from vlmeval import avgpool as ap
from vlmeval import gazevlm as gz
from vlmeval import phi_fovea as ph


def test_pool_side_per_model():
    assert ap.pool_side(0.05, 256, 16) == 4 and ap.pool_side(0.1, 256, 16) == 5      # 16 / 25 of 256
    assert ap.pool_side(0.05, 757, 24, True) == 6 and ap.pool_side(0.1, 757, 24, True) == 8   # Phi: 42 / 72 of 757
    assert ap.pool_side(0.05, 81, 9) == 2 and ap.pool_side(0.1, 81, 9) == 3          # SmolVLM: 4 / 9 of 81
    assert ap.pool_side(0.0, 256, 16) == 1 and ap.pool_side(2.0, 256, 16) == 16


def test_empty_roi_is_the_pooled_grid():
    grid = torch.randn(16, 16, 6, generator=torch.Generator().manual_seed(0))
    ref = F.adaptive_avg_pool2d(grid.permute(2, 0, 1)[None], 5)[0].permute(1, 2, 0).reshape(-1, 6)
    assert torch.allclose(gz.pick(grid, ap.EMPTY, 5), ref, atol=1e-6)
    cells = gz.layout(16, ap.EMPTY, 5)                                         # bins 0-4, 3-7, 6-10, 9-13, 12-16
    assert len(cells) == 25 and set(cells[:, 2]) == {4}
    assert cells[:5, 1].tolist() == [0, 3, 6, 9, 12] and cells[::5, 0].tolist() == [0, 3, 6, 9, 12]


def test_gemma_pools_the_patch_grid_before_its_projector():
    be = object.__new__(ap.Gemma)
    be.gf, be.pj = SimpleNamespace(project=lambda pj, f: f * 2), None         # stands in for norm + projection
    grid = torch.randn(64, 64, 3, generator=torch.Generator().manual_seed(1))
    out, = be.pooled(grid, 4)
    assert torch.allclose(out, 2 * grid.view(4, 16, 4, 16, 3).mean((1, 3)).reshape(-1, 3), atol=1e-6)


def test_phi_row_break_after_each_pooled_row():
    be = object.__new__(ap.Phi)
    br = torch.full((4,), 9.0)
    be.ph, be.img = ph, SimpleNamespace(sub_GN=br, img_projection=lambda x: x)
    grid = torch.randn(24, 24, 4, generator=torch.Generator().manual_seed(2))
    out, = be.pooled(grid, 2)
    assert out.shape == (ap.tokens_per_frame(2, True), 4)                      # 2 x (2 + 1)
    pooled = grid.view(2, 12, 2, 12, 4).mean((1, 3))
    assert torch.allclose(out[0], pooled[0, 0], atol=1e-6) and torch.allclose(out[4], pooled[1, 1], atol=1e-6)
    assert torch.equal(out[2], br) and torch.equal(out[5], br)


# ---------------------------------------------------------------------------- the run loop
class FakeBackend:
    def __init__(self):
        self.encoded, self.pooled_calls = [], []

    def frame_grid(self, fr):
        self.encoded.append(fr["key"])
        return torch.ones(16, 16, 2)

    def pooled(self, grid, s):
        self.pooled_calls.append(s)
        return (gz.thumbnail(grid, s).reshape(-1, 2),)

    def score(self, it, per, build_messages):
        p = np.full(5, 0.1)
        p["ABCDE".index(it["answer"])] = 0.6
        return p, 10 + sum(len(f[0]) for f in per)


def _items():
    fr = lambda v, k: dict(key=(v, k), path=f"{v}_{k}.jpg", t=float(k))
    return [dict(qid=q, video=v, question="q", options=list("abcde"), answer="E", frames=[fr(v, k) for k in range(4)])
            for q, v in ((1, "a"), (2, "a"), (3, "b"))]


def test_run_questions_encodes_once_and_resumes(tmp_path):
    out = str(tmp_path / "avgpool.jsonl")
    conds = {"4x4": dict(side=4), "5x5": dict(side=5)}
    be = FakeBackend()
    ap.run_questions(be, _items(), conds, out, build_messages=None)
    rows = [json.loads(l) for l in open(out)]
    assert len(rows) == 6 and all(r["correct"] == 1 and r["error"] is None for r in rows)
    assert sorted(be.encoded) == sorted({(v, k) for v in "ab" for k in range(4)})   # once per frame
    assert len(be.pooled_calls) == 2 * 8                                        # once per (side, frame)
    r4 = next(r for r in rows if r["condition"] == "4x4")
    assert r4["n_frames"] == 4 and r4["n_image_tokens"] == 4 * 16 and r4["avgpool"] == dict(side=4)
    ap.run_questions(FakeBackend(), _items(), conds, out, build_messages=None)
    assert len(open(out).read().splitlines()) == 6

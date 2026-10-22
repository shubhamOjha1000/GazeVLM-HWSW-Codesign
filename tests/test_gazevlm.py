"""GazeVLM baseline (vlmeval/gazevlm.py). CPU only, no models, no data.

  1. heatmap: normalized, peaked at the gaze
  2. rho-mass box: holds the gaze, grows with rho, matches the Gaussian's mass radius
  3. minimum ROI side, box -> token rows / columns
  4. tokens: 2 x 2 thumbnail = adaptive average pool, then the ROI tokens in raster order
  5. Qwen: M-RoPE positions put ROI tokens on their own cells and thumbnail tokens at bin centres
  6. Gemma: the thumbnail is the mean of a 32 x 32 patch quadrant, ROI tokens are Gemma's own
  7. Phi: ROI rows with row breaks, thumbnail rows with row breaks
  8. run_questions: one record per (condition, question), frames encoded once per video, resumable
"""
import json

import numpy as np
import pytest
import torch

from vlmeval import gazevlm as gz


# ---------------------------------------------------------------------------- the method
def test_heatmap_is_normalized_and_peaks_at_the_gaze():
    G = gz.gaze_heatmap([(0.3, 0.7)])
    assert G.shape == (gz.HEAT, gz.HEAT) and G.sum() == pytest.approx(1.0)
    y, x = np.unravel_index(G.argmax(), G.shape)
    assert abs((x + 0.5) / gz.HEAT - 0.3) < 1 / gz.HEAT and abs((y + 0.5) / gz.HEAT - 0.7) < 1 / gz.HEAT
    two = gz.gaze_heatmap([(0.2, 0.2), (0.8, 0.8)])
    h = gz.HEAT // 2
    assert two.sum() == pytest.approx(1.0) and two[:h, :h].sum() == pytest.approx(two[h:, h:].sum(), rel=1e-9)
    assert two[:h, :h].sum() > 0.49


@pytest.mark.parametrize("g", [(0.5, 0.5), (0.1, 0.9), (0.97, 0.03)])
def test_mass_box_holds_the_gaze_and_grows_with_rho(g):
    G = gz.gaze_heatmap([g])
    prev = 0.0
    for rho in (0.05, 0.1, 0.3, 0.5, 0.8, 0.95):
        x0, y0, x1, y1 = gz.mass_box(G, rho)
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
        assert x0 <= g[0] <= x1 and y0 <= g[1] <= y1
        area = (x1 - x0) * (y1 - y0)
        assert area >= prev
        prev = area


@pytest.mark.parametrize("rho", [0.2, 0.5, 0.8])
def test_mass_box_matches_the_gaussian_mass_radius(rho):
    """One gaze point away from the borders: the set is a disk of radius sigma sqrt(-2 ln(1 - rho))."""
    x0, y0, x1, y1 = gz.mass_box(gz.gaze_heatmap([(0.5, 0.5)], sigma=0.1), rho)
    r = 0.1 * np.sqrt(-2 * np.log(1 - rho))
    assert x1 - x0 == pytest.approx(2 * r, abs=3 / gz.HEAT) and y1 - y0 == pytest.approx(2 * r, abs=3 / gz.HEAT)


def test_grow_box_reaches_the_minimum_inside_the_frame():
    assert gz.grow_box((0.4, 0.4, 0.45, 0.42), 0.125) == pytest.approx((0.3625, 0.3475, 0.4875, 0.4725))
    assert gz.grow_box((0.0, 0.95, 0.02, 1.0), 0.125) == pytest.approx((0.0, 0.875, 0.125, 1.0))
    assert gz.grow_box((0.1, 0.1, 0.6, 0.7), 0.125) == (0.1, 0.1, 0.6, 0.7)


def test_box_on_grid():
    assert gz.box_on_grid((0.25, 0.5, 0.5, 0.75), 16) == (8, 12, 4, 8)          # exact token edges
    assert gz.box_on_grid((0.26, 0.51, 0.49, 0.74), 16) == (8, 12, 4, 8)        # inside them
    assert gz.box_on_grid((0.24, 0.49, 0.51, 0.76), 16) == (7, 13, 3, 9)        # touching neighbours
    assert gz.box_on_grid((0.5, 0.5, 0.5, 0.5), 9) == (4, 5, 4, 5)              # never empty
    assert gz.box_on_grid((0.0, 0.0, 1.0, 1.0), 24) == (0, 24, 0, 24)


def test_roi_box_budget_follows_rho():
    """ROI tokens on a 16 x 16 grid rise with rho; small rho stops at the minimum side (2-3 tokens)."""
    n = []
    for rho in (0.05, 0.2, 0.5, 0.8):
        r0, r1, c0, c1 = gz.box_on_grid(gz.roi_box([(0.5, 0.5)], rho), 16)
        n.append((r1 - r0) * (c1 - c0))
    assert n == sorted(n) and n[0] <= 9 and n[-1] > 16


# ---------------------------------------------------------------------------- tokens
def test_thumb_bins_and_layout():
    assert gz.thumb_bins(16) == [(0, 8), (8, 16)] and gz.thumb_bins(9) == [(0, 5), (4, 9)]
    cells = gz.layout(16, (3, 5, 6, 9))
    assert cells[:4].tolist() == [[0, 0, 8], [0, 8, 8], [8, 0, 8], [8, 8, 8]]
    assert cells[4:].tolist() == [[r, c, 1] for r in (3, 4) for c in (6, 7, 8)]


@pytest.mark.parametrize("g", [16, 9, 24])
def test_pick_is_thumbnail_then_roi(g):
    grid = torch.randn(g, g, 5, generator=torch.Generator().manual_seed(g))
    rc = (1, 4, 2, 5)
    out = gz.pick(grid, rc)
    ref = torch.nn.functional.adaptive_avg_pool2d(grid.permute(2, 0, 1)[None], 2)[0].flatten(1).T
    assert out.shape == (4 + 9, 5)
    assert torch.allclose(out[:4], ref, atol=1e-6) and torch.equal(out[4:], grid[1:4, 2:5].reshape(-1, 5))


def test_qwen_positions_roi_on_its_cells_thumbnail_at_bin_centres():
    from vlmeval import qwen_fovea as qf
    rc = (3, 5, 6, 8)
    cells = gz.layout(16, rc) * qf.BASE
    P = 99
    ids = torch.tensor([[1, 2] + [P] * len(cells) + [3]])
    pos = qf.mrope_positions(ids, P, [cells], [448])[:, 0].T            # (L, 3): time, row, column
    img = pos[2:2 + len(cells)]
    assert torch.all(img[:, 0] == 2)                                     # the image's start
    assert img[:4, 1:].tolist() == [[5.5, 5.5], [5.5, 13.5], [13.5, 5.5], [13.5, 13.5]]
    assert img[4:, 1:].tolist() == [[2 + r, 2 + c] for r in (3, 4) for c in (6, 7)]
    assert pos[-1].tolist() == [18, 18, 18]                              # text resumes after the 16-token grid


def test_gemma_cells_pool_the_patch_grid():
    from vlmeval import gemma_fovea as gf
    grid = torch.randn(gf.GRID, gf.GRID, 3, generator=torch.Generator().manual_seed(0))
    rc = (2, 4, 10, 11)
    pooled = gf.pool_cells(grid, gz.layout(16, rc) * gf.BASE)
    assert pooled.shape == (4 + 2, 3)
    assert torch.allclose(pooled[1], grid[:32, 32:].reshape(-1, 3).mean(0), atol=1e-5)
    stock = gf.pool_cells(grid, gf.stock_cells()).view(16, 16, 3)
    assert torch.allclose(pooled[4:], stock[2:4, 10:11].reshape(-1, 3), atol=1e-5)


def test_phi_sequences_keep_phis_row_breaks():
    grid = torch.randn(24, 24, 6, generator=torch.Generator().manual_seed(1))
    sub = torch.full((1, 1, 1, 6), 7.0)
    roi, thumb = gz.phi_sequences(grid, (5, 8, 10, 14), sub)
    assert roi.shape == (3 * (4 + 1), 6) and thumb.shape == (2 * (2 + 1), 6)
    assert torch.equal(roi[:4], grid[5, 10:14]) and torch.all(roi[4] == 7) and torch.equal(roi[5:9], grid[6, 10:14])
    assert torch.all(thumb[2] == 7) and torch.all(thumb[5] == 7)
    assert torch.allclose(thumb[0], grid[:12, :12].reshape(-1, 6).mean(0), atol=1e-5)


def test_gaze_points_sources():
    fr = dict(gaze=(0.2, 0.3), gaze_trace=[(0.1, 0.1), (0.2, 0.2)])
    assert gz.gaze_points(fr) == ([(0.2, 0.3)], False)
    assert gz.gaze_points(fr, "trace") == ([(0.1, 0.1), (0.2, 0.2)], False)
    assert gz.gaze_points(dict(gaze=(0.2, 0.3), gaze_trace=[]), "trace") == ([(0.2, 0.3)], False)
    assert gz.gaze_points(dict(gaze=None)) == ([(0.5, 0.5)], True)


# ---------------------------------------------------------------------------- the run loop
class FakeBackend(gz.Backend):
    side = 8

    def __init__(self):
        self.encoded = []

    def frame_grid(self, fr):
        self.encoded.append(fr["key"])
        return torch.full((8, 8, 2), float(fr["key"][1]))

    def frame_tokens(self, grid, rc, t=gz.THUMB):
        return (gz.pick(grid, rc, t),)

    def score(self, it, per, build_messages):
        p = np.full(5, 0.1)
        p["ABCDE".index(it["answer"])] = 0.6
        return p, 10 + sum(len(f[0]) for f in per)


def _items():
    fr = lambda v, k: dict(key=(v, k), path=f"{v}_{k}.jpg", t=float(k), gaze=(0.5, 0.5) if k else None)
    return [dict(qid=q, video=v, question="q", options=list("abcde"), answer="B", frames=[fr(v, k) for k in range(3)])
            for q, v in ((1, "a"), (2, "a"), (3, "b"))]


def test_run_questions_records_caches_and_resumes(tmp_path):
    out = str(tmp_path / "gazevlm.jsonl")
    conds = {"rho 0.1": dict(rho=0.1), "rho 0.5": dict(rho=0.5)}
    be = FakeBackend()
    gz.run_questions(be, _items(), conds, out, build_messages=None)
    rows = [json.loads(l) for l in open(out)]
    assert len(rows) == 6 and all(r["correct"] == 1 and r["error"] is None for r in rows)
    assert sorted(be.encoded) == sorted({(v, k) for v in "ab" for k in range(3)})   # once per frame
    r = rows[0]
    assert r["n_frames"] == 3 and r["n_gaze_missing"] == 1
    assert r["n_image_tokens"] == 3 * 4 + r["n_roi_tokens"] and r["n_tokens"] == 10 + r["n_image_tokens"]
    assert r["gazevlm"] == dict(rho=0.1, sigma=gz.SIGMA, r_min=gz.R_MIN, thumb=gz.THUMB, gaze="fixation")
    gz.run_questions(FakeBackend(), _items(), conds, out, build_messages=None)
    assert len(open(out).read().splitlines()) == 6

"""Foveated pooling for Gemma-3-4B (vlmeval/gemma_fovea.py). CPU only, no downloads.

  1. cells cover the 64 x 64 patch grid exactly once, sizes 4 / 8 / 16 patches, aligned
  2. token counts equal the Qwen 448 px layouts for the same gaze and rings
  3. pooling with Gemma's own 4 x 4 cells equals Gemma's 4 x 4 average pooling
  4. a big cell's token is the mean of the Gemma tokens it covers
  5. the projector trick (fill each 4 x 4 block with one feature) reproduces the stock
     Gemma3MultiModalProjector exactly (tiny random projector from transformers' own class)
  6. fovea (4 x 4) tokens are EXACTLY Gemma's own tokens at the same place: nothing drifts
  7. condition layouts (stock / fovea / uniform at the same tokens) and prompt expansion
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")

from vlmeval import gemma_fovea as gf
from vlmeval import qwen_fovea as qf

GAZES = [(0.5, 0.5), (0.1, 0.1), (0.8, 0.3), (0.97, 0.95), (0.3, 0.75)]
PICKS = [(70, 160), (180, 40), (180, 160), (240, 160)]       # (R, a) used in the notebooks
C = 8


def rand_grid(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(gf.GRID, gf.GRID, C, generator=g)


@pytest.fixture(scope="module")
def projector():
    try:
        from transformers import Gemma3Config
        from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector
        cfg = Gemma3Config(
            text_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=8, vocab_size=64),
            vision_config=dict(hidden_size=C, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
                               image_size=896, patch_size=14),
            mm_tokens_per_image=256)
        p = Gemma3MultiModalProjector(cfg).float().eval()
    except Exception as ex:
        pytest.skip(f"cannot build a Gemma3MultiModalProjector here: {ex}")
    torch.manual_seed(0)
    with torch.no_grad():
        for w in p.parameters():
            w.normal_()
    return p


# ---------------------------------------------------------------------------- layout
@pytest.mark.parametrize("g", GAZES)
@pytest.mark.parametrize("R, a", PICKS)
def test_cells_cover_the_grid_once(g, R, a):
    cells = gf.fovea_cells(*g, a=a, R_deg=R)
    cnt = np.zeros((gf.GRID, gf.GRID), int)
    for y, x, s in cells:
        assert s in (4, 8, 16) and y % s == 0 and x % s == 0
        cnt[y:y + s, x:x + s] += 1
    assert (cnt == 1).all()
    assert len(cells) == len(qf.foveated_cells(*g, a=a, img=448, R_deg=R))


def test_stock_cells():
    cells = gf.stock_cells()
    assert len(cells) == 256 and (cells[:, 2] == 4).all()
    assert cells[1].tolist() == [0, 4, 4] and cells[16].tolist() == [4, 0, 4]      # raster order


# ---------------------------------------------------------------------------- pooling
def test_stock_pooling_is_gemmas():
    grid = rand_grid()
    ours = gf.pool_cells(grid, gf.stock_cells())
    ref = F.avg_pool2d(grid.permute(2, 0, 1)[None], 4)[0].flatten(1).T
    assert torch.allclose(ours, ref, atol=1e-6)


def test_big_cell_is_the_mean_of_its_gemma_tokens():
    grid = rand_grid(1)
    stock = gf.pool_cells(grid, gf.stock_cells()).view(16, 16, C)
    big = gf.pool_cells(grid, np.array([[8, 16, 8], [32, 48, 16]]))
    assert torch.allclose(big[0], stock[2:4, 4:6].reshape(-1, C).mean(0), atol=1e-6)
    assert torch.allclose(big[1], stock[8:12, 12:16].reshape(-1, C).mean(0), atol=1e-6)


def test_uniform_pooling():
    grid = rand_grid(2)
    assert gf.pool_uniform(grid, 7).shape == (49, C)
    assert torch.allclose(gf.pool_uniform(grid, 16), gf.pool_cells(grid, gf.stock_cells()), atol=1e-6)


# ---------------------------------------------------------------------------- projector
def test_projector_trick_reproduces_gemma(projector):
    grid = rand_grid(3)
    stock = projector(grid.reshape(1, -1, C))[0]                        # Gemma: 4 x 4 pool, norm, project
    ours = gf.project(projector, gf.pool_cells(grid, gf.stock_cells()))
    assert ours.shape == stock.shape == (256, 16)
    assert torch.allclose(ours, stock, atol=1e-5, rtol=1e-4)
    few = gf.project(projector, gf.pool_cells(grid, gf.stock_cells())[:37])   # any count works
    assert torch.allclose(few, stock[:37], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("R, a", PICKS)
def test_fovea_tokens_are_gemmas_own_tokens(projector, R, a):
    grid = rand_grid(4)
    stock = projector(grid.reshape(1, -1, C))[0].view(16, 16, -1)
    cells = gf.fovea_cells(0.4, 0.6, a=a, R_deg=R)
    ours = gf.project(projector, gf.pool_cells(grid, cells))
    fine = np.nonzero(cells[:, 2] == 4)[0]
    assert len(fine) > 0
    ref = stock[torch.as_tensor(cells[fine, 0] // 4), torch.as_tensor(cells[fine, 1] // 4)]
    assert torch.allclose(ours[torch.as_tensor(fine)], ref, atol=1e-5, rtol=1e-4)


def test_frame_tokens_conditions(projector):
    grid = rand_grid(5)
    conds = {"stock": ("stock", None), "f": ("fovea", dict(a=40, R=180, gaze="gaze")),
             "ctr": ("fovea", dict(a=40, R=180, gaze="centre")), "m": ("uniform_match", "f")}
    fr = dict(gaze=(0.3, 0.7))
    s, s_cells, _ = gf.frame_tokens(conds, "stock", fr, grid, projector)
    f, f_cells, used = gf.frame_tokens(conds, "f", fr, grid, projector)
    m, m_cells, _ = gf.frame_tokens(conds, "m", fr, grid, projector)
    assert len(s) == 256 and used == (0.3, 0.7) and len(f) == len(f_cells) < 256
    side = int(round(np.sqrt(len(f))))
    assert m_cells is None and len(m) == side * side
    _, miss_cells, used = gf.frame_tokens(conds, "f", dict(gaze=None), grid, projector)
    assert used is None and np.array_equal(miss_cells, gf.frame_tokens(conds, "ctr", fr, grid, projector)[1])


def test_expand_images():
    class P:
        boi_token, eoi_token, image_token = gf.BOI, gf.EOI, gf.SOFT
        full_image_sequence = "\n\n" + gf.BOI + gf.SOFT * 256 + gf.EOI + "\n\n"
    t = "a" + gf.BOI + "b" + gf.BOI + "c"
    out = gf.expand_images(P(), t, [2, 3])
    assert out == ("a\n\n" + gf.BOI + gf.SOFT * 2 + gf.EOI + "\n\nb\n\n" + gf.BOI + gf.SOFT * 3
                   + gf.EOI + "\n\nc")
    with pytest.raises(ValueError):
        gf.expand_images(P(), t, [2])

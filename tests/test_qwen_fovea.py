"""Foveated tokenization for Qwen2.5-VL (vlmeval/qwen_fovea.py). CPU only, no downloads.

Layout
  1. every 28 px cell of the frame is covered exactly once; cells are 28/56/112 px and
     aligned to their own size (so none straddles a 112 px attention window)
  2. the cell under the gaze is a 28 px (full-resolution) cell
  3. tokens grow with the fall-off a and stay between 144 (all 112 px) and 2304 (all 28 px)
  4. token counts equal the spatial screen's (vlmeval/spatial.py) for the same gaze and a
Plumbing, against a tiny random-weight Qwen2.5-VL vision encoder built from transformers'
own classes (so the reference is Qwen's real code, not a re-implementation)
  5. patch positions: 28 px cells get exactly Qwen's integer RoPE coordinates; 56 px cells
     get their quadrant centres
  6. patches equal Qwen's image processor output for the same image
  7. window groups equal Qwen's get_window_index (including partial edge windows)
  8. PARITY: a layout of only 28 px cells, run through the overrides, reproduces the stock
     encoder's features
  9. a foveated layout runs, gives one token per cell, and each token follows its cell when
     the cells are reordered (no bookkeeping mix-ups)
 10. LLM M-RoPE positions follow Qwen2.5-VL's rule for text / image / text
"""
import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from vlmeval import qwen_fovea as qf
from vlmeval import spatial as sp

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
GAZES = [(0.5, 0.5), (0.1, 0.1), (0.8, 0.3), (0.97, 0.95), (0.3, 0.75)]


def rand_img(size, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(3, size, size, generator=g) * 255).round() / 255   # exact uint8 values


@pytest.fixture(scope="module")
def vis():
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
    torch.manual_seed(0)
    cfg = Qwen2_5_VLVisionConfig(depth=4, hidden_size=64, intermediate_size=96, num_heads=2,
                                 out_hidden_size=32, fullatt_block_indexes=[1, 3], window_size=112,
                                 patch_size=14, spatial_merge_size=2, temporal_patch_size=2)
    cfg._attn_implementation = "eager"
    m = Qwen2_5_VisionTransformerPretrainedModel(cfg).float().eval()
    qf.check_encoder(m)
    return m


# ---------------------------------------------------------------------------- layout
@pytest.mark.parametrize("g", GAZES)
@pytest.mark.parametrize("a", [10, 40, 160])
def test_layout_covers_every_cell_once(g, a):
    cells = qf.foveated_cells(*g, a=a)
    n = qf.IMG // qf.BASE
    cnt = np.zeros((n, n), int)
    for y, x, s in cells:
        assert s in (28, 56, 112) and y % s == 0 and x % s == 0
        cnt[y // 28:(y + s) // 28, x // 28:(x + s) // 28] += 1
    assert (cnt == 1).all()


@pytest.mark.parametrize("g", GAZES)
@pytest.mark.parametrize("a", [10, 40])
def test_gaze_cell_is_full_resolution(g, a):
    cells = qf.foveated_cells(*g, a=a)
    gx, gy = g[0] * qf.IMG, g[1] * qf.IMG
    assert [s for y, x, s in cells if y <= gy < y + s and x <= gx < x + s] == [28]


def test_tokens_grow_with_a_and_stay_in_bounds():
    n = [len(qf.foveated_cells(0.5, 0.5, a=a)) for a in (5, 10, 20, 40, 80, 160, 320)]
    assert all(x <= y for x, y in zip(n, n[1:])), n
    assert 144 <= n[0] < n[-1] <= 2304, n


@pytest.mark.parametrize("g", GAZES)
def test_layout_matches_spatial_screen(g):
    for a in (10, 40, 160):
        d = sp.density_map(*g, a, num_scales=3, img=qf.IMG, p=qf.BASE)
        assert len(qf.foveated_cells(*g, a=a)) == round(d.sum())


def test_largest_cell_must_fit_a_window():
    with pytest.raises(ValueError):
        qf.foveated_cells(0.5, 0.5, num_scales=4)


# ---------------------------------------------------------------------------- plumbing
def test_patch_positions():
    size = 448
    h, w = qf.patch_positions(qf.uniform_cells(size))
    n = size // 14
    hp = torch.arange(n).unsqueeze(1).expand(-1, n).reshape(n // 2, 2, n // 2, 2).permute(0, 2, 1, 3).flatten()
    wp = torch.arange(n).unsqueeze(0).expand(n, -1).reshape(n // 2, 2, n // 2, 2).permute(0, 2, 1, 3).flatten()
    assert torch.equal(h, hp.float()) and torch.equal(w, wp.float())
    # a 56 px cell at (56, 112): quadrant centres on the 14 px grid
    h, w = qf.patch_positions(np.array([[56, 112, 56]]))
    assert torch.allclose(h, torch.tensor([4.5, 4.5, 6.5, 6.5])) and torch.allclose(w, torch.tensor([8.5, 10.5, 8.5, 10.5]))


def test_patches_match_qwen_image_processor():
    try:
        from transformers import Qwen2VLImageProcessor
    except ImportError:
        pytest.skip("Qwen2VLImageProcessor not available")
    ip = Qwen2VLImageProcessor()
    img = rand_img(448)
    pv = torch.as_tensor(ip(images=[qf.to_uint8_pil(img)], return_tensors="pt")["pixel_values"]).float()
    ours = qf.cells_to_patches(img, qf.uniform_cells(448), ip.image_mean, ip.image_std)
    assert pv.shape == ours.shape
    assert torch.allclose(pv, ours, atol=1e-4), (pv - ours).abs().max()


@pytest.mark.parametrize("size", [448, 392, 1344])
def test_window_groups_match_qwen(vis, size):
    order, cu = qf.window_groups(qf.uniform_cells(size), size)
    so, scu = vis.get_window_index(torch.tensor([[1, size // 14, size // 14]]))
    scu = torch.unique_consecutive(torch.as_tensor(scu)).tolist()
    assert torch.equal(order, torch.as_tensor(so).long()) and cu == scu


@pytest.mark.parametrize("size", [448, 392])
def test_uniform_layout_reproduces_qwen_encoder(vis, size):
    img = rand_img(size)
    cells = qf.uniform_cells(size)
    ours = qf.encode_cells(vis, img, cells, size, MEAN, STD)
    stock = qf.encode_stock(vis, img, MEAN, STD)
    assert ours.shape == stock.shape == (len(cells), 32)
    assert torch.allclose(ours, stock, atol=1e-5, rtol=1e-4), (ours - stock).abs().max()
    # the hooks are removed afterwards: the class methods are back
    assert "rot_pos_emb" not in vars(vis) and "get_window_index" not in vars(vis)


def test_foveated_tokens_follow_their_cells(vis):
    img = rand_img(qf.IMG, 1)
    cells = qf.foveated_cells(0.3, 0.6, a=40)
    out = qf.encode_cells(vis, img, cells, qf.IMG, MEAN, STD)
    assert out.shape == (len(cells), 32) and torch.isfinite(out).all()
    perm = np.random.default_rng(0).permutation(len(cells))
    out_p = qf.encode_cells(vis, img, cells[perm], qf.IMG, MEAN, STD)
    assert torch.allclose(out_p, out[torch.as_tensor(perm)], atol=1e-5)


def test_mrope_positions():
    IMG_ID = 9
    # uniform 112 px image = 4 x 4 tokens between two text tokens on each side
    ids = torch.tensor([[1, 2] + [IMG_ID] * 16 + [3, 4]])
    pos = qf.mrope_positions(ids, IMG_ID, [qf.uniform_cells(112)], [112])[:, 0]
    assert pos.shape == (3, 20)
    assert pos[:, :2].tolist() == [[0, 1]] * 3
    r, c = np.divmod(np.arange(16), 4)
    assert pos[0, 2:18].tolist() == [2] * 16
    assert pos[1, 2:18].tolist() == (2 + r).tolist() and pos[2, 2:18].tolist() == (2 + c).tolist()
    assert pos[:, 18:].tolist() == [[6, 7]] * 3              # start + grid size (4)
    # one 112 px cell = one token at the centre of its 4 x 4 token block
    pos = qf.mrope_positions(torch.tensor([[1, IMG_ID, 3]]), IMG_ID,
                             [np.array([[0, 0, 112]])], [112])[:, 0]
    assert pos[:, 1].tolist() == [1, 2.5, 2.5] and pos[:, 2].tolist() == [5, 5, 5]


def test_expand_image_pads():
    t = "a<|image_pad|>b<|image_pad|>c"
    assert qf.expand_image_pads(t, [2, 3]) == "a" + "<|image_pad|>" * 2 + "b" + "<|image_pad|>" * 3 + "c"
    with pytest.raises(ValueError):
        qf.expand_image_pads(t, [2])

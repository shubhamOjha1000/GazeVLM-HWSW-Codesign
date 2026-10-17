"""Foveated tokenization for InternVL3.5 (vlmeval/internvl_fovea.py). CPU only, no downloads.

Against a tiny random InternViT built from transformers' own InternVLVisionModel:
  1. position embeddings sampled at integer grid points are InternVL's own; at half-way points
     they are the mean of the neighbours
  2. PARITY: an all-28 px layout through our pre-encoder path (patch embedding, sampled
     positions, CLS, encoder, norm, per-cell pixel shuffle) equals InternVL's own vision forward
     + pixel shuffle + projector on the same 448 px frame
  3. a foveated layout gives one finite token per cell, and each token follows its cell when the
     cells are reordered (full attention: no bookkeeping mix-ups)
  4. the uniform path at a smaller side gives (side / 28)^2 tokens
  5. condition layouts (uniform / fovea / uniform at the same tokens) and prompt expansion
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers")

from vlmeval import internvl_fovea as iv
from vlmeval import qwen_fovea as qf

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
C, D = 32, 24


def _hf_pixel_shuffle(x, scale_factor=0.5):
    """transformers' InternVLModel.pixel_shuffle, used if the real one cannot be called here."""
    b, w, h, c = x.size()
    x = x.view(b, w, int(h * scale_factor), int(c / scale_factor)).permute(0, 2, 1, 3).contiguous()
    x = x.view(b, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor ** 2)))
    return x.permute(0, 2, 1, 3).contiguous()


class TinyInternVL(nn.Module):
    """The pieces internvl_fovea.parts() reads, around a real (tiny, random) InternViT."""

    def __init__(self, vt):
        super().__init__()
        self.vision_tower = vt
        self.multi_modal_projector = nn.Sequential(nn.LayerNorm(4 * C), nn.Linear(4 * C, D), nn.GELU(),
                                                   nn.Linear(D, D))
        self.config = SimpleNamespace(downsample_ratio=0.5, vision_feature_layer=-1)

    def pixel_shuffle(self, x, scale_factor=0.5):
        try:
            from transformers.models.internvl.modeling_internvl import InternVLModel
            return InternVLModel.pixel_shuffle(self, x, scale_factor=scale_factor)
        except Exception:
            return _hf_pixel_shuffle(x, scale_factor)


@pytest.fixture(scope="module")
def model():
    try:
        from transformers.models.internvl.configuration_internvl import InternVLVisionConfig
        from transformers.models.internvl.modeling_internvl import InternVLVisionModel
        cfg = InternVLVisionConfig(hidden_size=C, num_hidden_layers=2, num_attention_heads=2,
                                   intermediate_size=64, image_size=[448, 448], patch_size=[14, 14],
                                   use_absolute_position_embeddings=True)
        cfg._attn_implementation = "eager"
        vt = InternVLVisionModel(cfg).float().eval()
    except Exception as ex:
        pytest.skip(f"cannot build an InternVLVisionModel here: {ex}")
    torch.manual_seed(0)
    m = TinyInternVL(vt).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.2)
    return m


def rand_img(size, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(3, size, size, generator=g) * 255).round() / 255


# ---------------------------------------------------------------------------- positions
def test_sampled_positions(model):
    vt = model.vision_tower
    _, table = iv._pos_table(vt)                                     # 1, C, 32, 32
    h = torch.tensor([0., 5., 31., 7.5])
    w = torch.tensor([0., 9., 31., 3.0])
    got = iv.sample_positions(vt, h, w)
    assert torch.allclose(got[0], table[0, :, 0, 0], atol=1e-5)
    assert torch.allclose(got[1], table[0, :, 5, 9], atol=1e-5)
    assert torch.allclose(got[2], table[0, :, 31, 31], atol=1e-5)
    assert torch.allclose(got[3], (table[0, :, 7, 3] + table[0, :, 8, 3]) / 2, atol=1e-5)


# ---------------------------------------------------------------------------- parity
def test_all_28px_layout_equals_internvl(model):
    img = rand_img(448)
    ours = iv.encode_cells(model, img, qf.uniform_cells(448), MEAN, STD)
    stock = iv.encode_uniform(model, img, 448, MEAN, STD)          # InternVL's own vision forward
    assert ours.shape == stock.shape == (256, D)
    assert torch.allclose(ours, stock, atol=1e-4, rtol=1e-4), (ours - stock).abs().max()


def test_foveated_tokens_follow_their_cells(model):
    img = rand_img(448, 1)
    cells = qf.foveated_cells(0.3, 0.6, a=40, img=448, R_deg=180)
    out = iv.encode_cells(model, img, cells, MEAN, STD)
    assert out.shape == (len(cells), D) and torch.isfinite(out).all() and len(cells) < 256
    perm = np.random.default_rng(0).permutation(len(cells))
    out_p = iv.encode_cells(model, img, cells[perm], MEAN, STD)
    assert torch.allclose(out_p, out[torch.as_tensor(perm)], atol=1e-4, rtol=1e-4)


def test_uniform_smaller_side(model):
    out = iv.encode_uniform(model, rand_img(448, 2), 196, MEAN, STD)
    assert out.shape == (49, D) and torch.isfinite(out).all()


# ---------------------------------------------------------------------------- conditions / prompt
def test_frame_tokens_conditions(model):
    img = rand_img(1344, 3)
    conds = {"u448": ("uniform", 448), "f": ("fovea", dict(a=40, R=180, gaze="gaze")),
             "ctr": ("fovea", dict(a=40, R=180, gaze="centre")), "m": ("uniform_match", "f")}
    fr = dict(gaze=(0.3, 0.7))
    u, _, side, _ = iv.frame_tokens(model, conds, "u448", fr, img, MEAN, STD)
    f, cells, fside, used = iv.frame_tokens(model, conds, "f", fr, img, MEAN, STD)
    m, _, mside, _ = iv.frame_tokens(model, conds, "m", fr, img, MEAN, STD)
    assert len(u) == 256 and side == 448 and fside == 448 and used == (0.3, 0.7) and len(f) == len(cells)
    assert mside == qf.same_tokens_size(len(cells)) and len(m) == (mside // 28) ** 2
    _, miss, _, used = iv.frame_tokens(model, conds, "f", dict(gaze=None), img, MEAN, STD)
    assert used is None and np.array_equal(miss, iv.frame_tokens(model, conds, "ctr", fr, img, MEAN, STD)[1])


def test_expand_images():
    class P:
        start_image_token, end_image_token, image_token = iv.IMG_START, iv.IMG_END, iv.IMG_CONTEXT
    t = "a" + iv.IMG_CONTEXT + "b" + iv.IMG_CONTEXT + "c"
    out = iv.expand_images(P(), t, [2, 3])
    assert out == ("a" + iv.IMG_START + iv.IMG_CONTEXT * 2 + iv.IMG_END + "b" + iv.IMG_START
                   + iv.IMG_CONTEXT * 3 + iv.IMG_END + "c")
    with pytest.raises(ValueError):
        iv.expand_images(P(), t, [2])

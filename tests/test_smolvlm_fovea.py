"""Foveated tokenization for SmolVLM2 (vlmeval/smolvlm_fovea.py). CPU only, no downloads.

Layout
  1. the k 126 px cells (of the 3 x 3 grid) nearest the gaze are split into 42 px cells:
     9 + 8 k tokens covering the 378 px area SigLIP tokenizes once, the gaze cell at full resolution
Plumbing, against a tiny random SmolVLM built from transformers' own classes (384 px, 14 px
patches: the real 27 x 27 grid)
  2. patch positions: 42 px cells get their own grid positions, 126 px cells the centre patches
  3. patches equal the image cut into SigLIP's 14 px patches
  4. PARITY: an all-42 px layout through our path (SmolVLM's own position ids, encoder, pixel
     shuffle and projection) equals SmolVLM's own encoder + connector on the image
  5. a foveated layout gives one finite token per cell, each following its cell when reordered
  6. the uniform path at a smaller side gives (side / 42)^2 tokens
  7. LLM PARITY: our inputs_embeds (image tokens from our path) give the stock forward's logits
  8. prompt expansion and condition layouts
"""
import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from vlmeval import smolvlm_fovea as sm

MEAN = STD = [0.5, 0.5, 0.5]


def rand_img(size, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(3, size, size, generator=g) * 255).round() / 255


def rec_of(img):
    return dict(x01=img, pv=sm.normalize(img, MEAN, STD))


@pytest.fixture(scope="module")
def model():
    try:
        from transformers.models.smolvlm.configuration_smolvlm import SmolVLMConfig
        from transformers.models.smolvlm.modeling_smolvlm import SmolVLMForConditionalGeneration
        cfg = SmolVLMConfig(
            vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                               image_size=384, patch_size=14),
            text_config=dict(model_type="llama", hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                             max_position_embeddings=1024),
            scale_factor=3, image_token_id=100)
        m = SmolVLMForConditionalGeneration(cfg).float().eval()
    except Exception as ex:
        pytest.skip(f"cannot build a tiny SmolVLM here: {ex}")
    torch.manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.2)
    return m


# ---------------------------------------------------------------------------- layout
@pytest.mark.parametrize("g", [(0.5, 0.5), (0.1, 0.1), (0.8, 0.3), (0.97, 0.5)])
@pytest.mark.parametrize("k", [0, 1, 3, 5, 7, 9])
def test_layout(g, k):
    cells = sm.foveated_cells(*g, k)
    assert len(cells) == 9 + 8 * k
    cnt = np.zeros((9, 9), int)
    for y, x, s in cells:
        assert s in (42, 126) and y % s == 0 and x % s == 0
        cnt[y // 42:(y + s) // 42, x // 42:(x + s) // 42] += 1
    assert (cnt == 1).all()
    if k:
        gx, gy = g[0] * 384, g[1] * 384
        assert [s for y, x, s in cells if y <= gy < y + s and x <= gx < x + s] == [42]


# ---------------------------------------------------------------------------- plumbing
def test_patch_positions():
    h, w = sm.patch_positions(sm.uniform_cells())
    assert sorted(zip(h.tolist(), w.tolist())) == [(i, j) for i in range(27) for j in range(27)]
    h, w = sm.patch_positions(np.array([[126, 0, 126]]))       # centre patches of its 3 x 3 blocks
    assert h.tolist() == [10, 10, 10, 13, 13, 13, 16, 16, 16] and w.tolist() == [1, 4, 7] * 3


def test_patches_are_the_image_cut_into_14px_patches():
    img = rand_img(384)
    p = sm.cells_to_patches(img, sm.uniform_cells(), MEAN, STD)
    h, w = sm.patch_positions(sm.uniform_cells())
    norm = sm.normalize(img, MEAN, STD)
    want = torch.stack([norm[:, i * 14:(i + 1) * 14, j * 14:(j + 1) * 14] for i, j in zip(h, w)])
    assert p.shape == (729, 3, 14, 14) and torch.allclose(p, want, atol=1e-6)


def test_all_42px_layout_equals_smolvlm(model):
    img = rand_img(384, 1)
    ours = sm.encode_cells(model, img, sm.uniform_cells(), MEAN, STD)
    stock = sm.encode_pixels(model, sm.normalize(img, MEAN, STD))
    assert ours.shape == stock.shape == (81, 48)
    assert torch.allclose(ours, stock, atol=1e-4, rtol=1e-4), (ours - stock).abs().max()


def test_foveated_tokens_follow_their_cells(model):
    img = rand_img(384, 2)
    cells = sm.foveated_cells(0.3, 0.6, 3)
    out = sm.encode_cells(model, img, cells, MEAN, STD)
    assert out.shape == (33, 48) and torch.isfinite(out).all()
    perm = np.random.default_rng(0).permutation(len(cells))
    out_p = sm.encode_cells(model, img, cells[perm], MEAN, STD)
    assert torch.allclose(out_p, out[torch.as_tensor(perm)], atol=1e-5)


@pytest.mark.parametrize("side", [168, 252, 336])
def test_uniform_smaller_side(model, side):
    out = sm.encode_uniform(model, rec_of(rand_img(384, 3)), side, MEAN, STD)
    assert out.shape == ((side // 42) ** 2, 48) and torch.isfinite(out).all()


def test_llm_inputs_reproduce_smolvlm_forward(model):
    img = rand_img(384, 4)
    P = model.config.image_token_id
    ids = torch.tensor([[5, 6, 7, 8] + [P] * 81 + [7, 9, 10]])
    with torch.no_grad():
        stock = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                      pixel_values=sm.normalize(img, MEAN, STD)[None, None], use_cache=False).logits[0]
        inputs = sm.inputs_from_ids(model, ids, [sm.encode_cells(model, img, sm.uniform_cells(), MEAN, STD)])
        ours = model(**inputs, use_cache=False).logits[0]
    assert ours.shape == stock.shape
    assert torch.allclose(ours, stock, atol=1e-4, rtol=1e-4), (ours - stock).abs().max()


# ---------------------------------------------------------------------------- prompt / conditions
def test_expand_images():
    class P:
        fake_image_token, global_image_token, image_token = sm.FAKE, sm.GLOBAL, sm.IMAGE
    t = "a" + sm.IMAGE + "b" + sm.IMAGE + "c"
    out = sm.expand_images(P(), t, [2, 3])
    assert out == ("a" + sm.FAKE + sm.GLOBAL + sm.IMAGE * 2 + sm.FAKE + "b"
                   + sm.FAKE + sm.GLOBAL + sm.IMAGE * 3 + sm.FAKE + "c")
    with pytest.raises(ValueError):
        sm.expand_images(P(), t, [2])


def test_frame_tokens_conditions(model):
    rec = rec_of(rand_img(384, 5))
    conds = {"stock": ("uniform", 384), "f": ("fovea", dict(k=3, gaze="gaze")),
             "ctr": ("fovea", dict(k=3, gaze="centre")), "m": ("uniform_match", "f")}
    fr = dict(gaze=(0.3, 0.7))
    t, cells, side, used = sm.frame_tokens(model, conds, "stock", fr, rec, MEAN, STD)
    assert len(t) == 81 and cells is None and side == 384
    t, cells, side, used = sm.frame_tokens(model, conds, "f", fr, rec, MEAN, STD)
    assert len(t) == len(cells) == 33 and used == (0.3, 0.7)
    t, _, side, _ = sm.frame_tokens(model, conds, "m", fr, rec, MEAN, STD)
    assert side == sm.same_tokens_size(33) and len(t) == (side // 42) ** 2
    _, miss, _, used = sm.frame_tokens(model, conds, "f", dict(gaze=None), rec, MEAN, STD)
    assert used is None and np.array_equal(miss, sm.frame_tokens(model, conds, "ctr", fr, rec, MEAN, STD)[1])

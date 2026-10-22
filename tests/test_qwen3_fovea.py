"""Foveated tokenization for Qwen3-VL (vlmeval/qwen3_fovea.py). CPU only, no downloads.

Layout
  1. the 512 px layout is the other models' 448 px layout scaled by 32 / 28: 32/64/128 px cells,
     aligned to their size, covering the frame once, same token count
Plumbing, against tiny random-weight models built from transformers' own Qwen3-VL classes
  2. patch positions of an all-32 px layout = Qwen3-VL's own (block-major order, integers)
  3. patches equal the Qwen2-VL image processor's output with Qwen3-VL's settings
  4. position table: integer coordinates give Qwen3-VL's own resampled table; fractional ones a
     bilinear read of the table (checked against grid_sample)
  5. PARITY: an all-32 px layout through our part-by-part encoder reproduces the stock encoder's
     tokens and DeepStack features, at 512 px and at smaller sides
  6. a foveated layout gives one finite token and DeepStack feature per cell, each following its
     cell when the cells are reordered
  7. LLM PARITY: our inputs_embeds + M-RoPE positions + DeepStack features, through the model's
     own text model and LM head, reproduce the stock forward's logits on a two-image prompt
  8. M-RoPE positions with 32 px cells; condition layouts
"""
import inspect

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from vlmeval import qwen_fovea as qf
from vlmeval import qwen3_fovea as q3

MEAN = STD = [0.5, 0.5, 0.5]
GAZES = [(0.5, 0.5), (0.1, 0.1), (0.8, 0.3), (0.97, 0.95)]
VIS = dict(depth=3, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=48,
           num_position_embeddings=64, deepstack_visual_indexes=[0, 1], patch_size=16,
           spatial_merge_size=2, temporal_patch_size=2)


def rand_img(size, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(3, size, size, generator=g) * 255).round() / 255   # exact uint8 values


def _randomize(m, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.2)
    return m


@pytest.fixture(scope="module")
def vis():
    try:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        cfg = Qwen3VLVisionConfig(**VIS)
        cfg._attn_implementation = "eager"
        m = Qwen3VLVisionModel(cfg).float().eval()
    except Exception as ex:
        pytest.skip(f"cannot build a Qwen3VLVisionModel here: {ex}")
    q3.check_encoder(m)
    return _randomize(m)


@pytest.fixture(scope="module")
def tiny_model():
    """A tiny Qwen3VLForConditionalGeneration: 3 text layers, DeepStack into the first two."""
    try:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
    except Exception as ex:
        pytest.skip(f"no Qwen3-VL in this transformers: {ex}")
    text = dict(vocab_size=96, hidden_size=48, intermediate_size=96, num_hidden_layers=3,
                num_attention_heads=4, num_key_value_heads=2, head_dim=12, max_position_embeddings=4096)
    rope = dict(rope_type="default", mrope_section=[2, 2, 2], mrope_interleaved=True)
    last = None
    for kw in (dict(rope_parameters=dict(rope, rope_theta=10000.0)),      # transformers 5
               dict(rope_scaling=rope, rope_theta=10000.0)):              # 4.57
        try:
            cfg = Qwen3VLConfig(text_config=dict(text, **kw), vision_config=dict(VIS), image_token_id=90,
                                video_token_id=91, vision_start_token_id=92, vision_end_token_id=93)
            m = Qwen3VLForConditionalGeneration(cfg).float().eval()
            sec = list(getattr(m.model.language_model.rotary_emb, "mrope_section", []))
            if sec != [2, 2, 2]:
                raise ValueError(f"mrope_section {sec}")
            return _randomize(m, 1)
        except Exception as ex:
            last = ex
    pytest.skip(f"cannot build a tiny Qwen3-VL here: {last}")


# ---------------------------------------------------------------------------- layout
@pytest.mark.parametrize("g", GAZES)
@pytest.mark.parametrize("R", [None, 70, 180, 240])
def test_layout_is_the_448_layout_scaled(g, R):
    cells = q3.foveated_cells(*g, a=40, R_deg=R)
    ref = qf.foveated_cells(*g, a=40, img=448, R_deg=R)
    assert np.array_equal(cells, ref * 32 // 28) and len(cells) == len(ref)
    cnt = np.zeros((16, 16), int)
    for y, x, s in cells:
        assert s in (32, 64, 128) and y % s == 0 and x % s == 0
        cnt[y // 32:(y + s) // 32, x // 32:(x + s) // 32] += 1
    assert (cnt == 1).all()


# ---------------------------------------------------------------------------- plumbing
def test_patch_positions_are_qwen3s_own():
    h, w = qf.patch_positions(q3.uniform_cells(512), patch=16)
    n = 32
    hp = torch.arange(n).unsqueeze(1).expand(-1, n).reshape(n // 2, 2, n // 2, 2).permute(0, 2, 1, 3).flatten()
    wp = torch.arange(n).unsqueeze(0).expand(n, -1).reshape(n // 2, 2, n // 2, 2).permute(0, 2, 1, 3).flatten()
    assert torch.equal(h, hp.float()) and torch.equal(w, wp.float())
    # a 64 px cell at (64, 128): quadrant centres on the 16 px grid
    h4, w4 = qf.patch_positions(np.array([[64, 128, 64]]), patch=16)
    assert torch.allclose(h4, torch.tensor([4.5, 4.5, 6.5, 6.5])) and torch.allclose(w4, torch.tensor([8.5, 10.5, 8.5, 10.5]))
    try:
        from transformers.vision_utils import get_vision_position_ids
    except ImportError:
        return
    pid = get_vision_position_ids(torch.tensor([[1, n, n]]), 2)
    assert torch.equal(h, pid[:, 0].float()) and torch.equal(w, pid[:, 1].float())


def test_patches_match_qwen3_image_processor():
    try:
        from transformers import Qwen2VLImageProcessor
        ip = Qwen2VLImageProcessor(patch_size=16, merge_size=2, temporal_patch_size=2,
                                   image_mean=MEAN, image_std=STD)
    except Exception as ex:
        pytest.skip(f"Qwen2VLImageProcessor not available: {ex}")
    img = rand_img(512)
    out = ip(images=[qf.to_uint8_pil(img)], return_tensors="pt")
    assert torch.as_tensor(out["image_grid_thw"]).tolist() == [[1, 32, 32]]
    pv = torch.as_tensor(out["pixel_values"]).float()
    ours = qf.cells_to_patches(img, q3.uniform_cells(512), MEAN, STD, patch=16)
    assert pv.shape == ours.shape
    assert torch.allclose(pv, ours, atol=1e-4), (pv - ours).abs().max()


def _stock_table(vis, grid):
    """Qwen3-VL's own position table resampled to a grid x grid patch grid, in its patch order."""
    thw = torch.tensor([[1, grid, grid]])
    if hasattr(vis, "fast_pos_embed_interpolate"):                           # transformers 4.57
        return vis.fast_pos_embed_interpolate(thw)
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights
    idx, w = get_vision_interpolation_indices_and_weights(
        thw, num_grid_per_side=vis.num_grid_per_side, mode="bilinear", align_corners=True, spatial_merge_size=2)
    return (vis.pos_embed(idx) * w[:, :, None]).sum(1)


@pytest.mark.parametrize("grid", [32, 14, 20])
def test_position_table_is_qwen3s_own(vis, grid):
    h, w = qf.patch_positions(q3.uniform_cells(grid * 16), patch=16)
    ours = q3.sample_positions(vis, h, w, grid)
    try:
        stock = _stock_table(vis, grid)
    except Exception as ex:
        pytest.skip(f"no stock position-table resampling to compare with: {ex}")
    assert ours.shape == stock.shape
    assert torch.allclose(ours, stock, atol=1e-5), (ours - stock).abs().max()


def test_position_table_fractional_is_bilinear(vis):
    side = int(round(vis.pos_embed.num_embeddings ** 0.5))
    table = vis.pos_embed.weight.detach().T.reshape(1, -1, side, side)
    grid = 32
    h = torch.tensor([0.5, 3.25, 30.5, 7.0])
    w = torch.tensor([1.5, 0.0, 29.5, 12.75])
    ours = q3.sample_positions(vis, h, w, grid)
    xy = torch.stack([w / (grid - 1) * 2 - 1, h / (grid - 1) * 2 - 1], -1).view(1, 1, -1, 2)
    ref = torch.nn.functional.grid_sample(table, xy, mode="bilinear", align_corners=True)[0, :, 0].T
    assert torch.allclose(ours, ref, atol=1e-5), (ours - ref).abs().max()


@pytest.mark.parametrize("side", [512, 224, 320])
def test_uniform_layout_reproduces_qwen3_encoder(vis, side):
    img = rand_img(side, 1)
    t, d = q3.encode_cells(vis, img, q3.uniform_cells(side), side, MEAN, STD)
    st, sd = q3.encode_stock(vis, img, MEAN, STD)
    n = (side // 32) ** 2
    assert t.shape == st.shape == (n, 48) and len(d) == len(sd) == 2
    assert torch.allclose(t, st, atol=1e-5, rtol=1e-4), (t - st).abs().max()
    for a, b in zip(d, sd):
        assert a.shape == (n, 48) and torch.allclose(a, b, atol=1e-5, rtol=1e-4), (a - b).abs().max()


def test_foveated_tokens_follow_their_cells(vis):
    img = rand_img(512, 2)
    cells = q3.foveated_cells(0.3, 0.6, a=40, R_deg=180)
    t, d = q3.encode_cells(vis, img, cells, 512, MEAN, STD)
    assert t.shape == (len(cells), 48) and len(cells) < 256 and torch.isfinite(t).all()
    assert len(d) == 2 and all(x.shape == t.shape and torch.isfinite(x).all() for x in d)
    perm = np.random.default_rng(0).permutation(len(cells))
    tp, dp = q3.encode_cells(vis, img, cells[perm], 512, MEAN, STD)
    pt = torch.as_tensor(perm)
    assert torch.equal(tp, t[pt]) and all(torch.equal(a, b[pt]) for a, b in zip(dp, d))


def test_llm_inputs_reproduce_qwen3_forward(tiny_model):
    m = tiny_model
    vis, lm, head = q3.parts(m)
    sides = [64, 96]                                    # two images: 2 x 2 and 3 x 3 tokens
    imgs = [rand_img(s, 3 + i) for i, s in enumerate(sides)]
    cells = [q3.uniform_cells(s) for s in sides]
    S, E, P = m.config.vision_start_token_id, m.config.vision_end_token_id, m.config.image_token_id
    ids = torch.tensor([[5, 6, 7, S] + [P] * len(cells[0]) + [E, 8, 9, S] + [P] * len(cells[1]) + [E, 10, 11, 12]])
    pv = torch.cat([qf.cells_to_patches(im, c, MEAN, STD, patch=16) for im, c in zip(imgs, cells)])
    grid = torch.tensor([[1, s // 16, s // 16] for s in sides])
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pv, image_grid_thw=grid)
    if "mm_token_type_ids" in inspect.signature(type(m).forward).parameters:
        kw["mm_token_type_ids"] = (ids == P).int()
    with torch.no_grad():
        stock = m(**kw, use_cache=False).logits[0]
        enc = [q3.encode_cells(vis, im, c, s, MEAN, STD) for im, c, s in zip(imgs, cells, sides)]
        inputs = q3.inputs_from_ids(m, ids, [e[0] for e in enc], [e[1] for e in enc], cells, sides)
        out = lm(**inputs, use_cache=False)
        ours = head(out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0])[0]
    assert ours.shape == stock.shape
    assert torch.allclose(ours, stock, atol=1e-4, rtol=1e-4), (ours - stock).abs().max()
    assert torch.allclose(q3.next_token_logits(m, inputs), stock[-1], atol=1e-4, rtol=1e-4)


def test_mrope_positions_with_32px_cells():
    IMG_ID = 9
    # uniform 128 px image = 4 x 4 tokens between two text tokens on each side
    ids = torch.tensor([[1, 2] + [IMG_ID] * 16 + [3, 4]])
    pos = qf.mrope_positions(ids, IMG_ID, [q3.uniform_cells(128)], [128], base=32)[:, 0]
    r, c = np.divmod(np.arange(16), 4)
    assert pos[:, :2].tolist() == [[0, 1]] * 3
    assert pos[0, 2:18].tolist() == [2] * 16
    assert pos[1, 2:18].tolist() == (2 + r).tolist() and pos[2, 2:18].tolist() == (2 + c).tolist()
    assert pos[:, 18:].tolist() == [[6, 7]] * 3              # start + grid size (4)
    # one 128 px cell = one token at the centre of its 4 x 4 token block
    pos = qf.mrope_positions(torch.tensor([[1, IMG_ID, 3]]), IMG_ID, [np.array([[0, 0, 128]])], [128],
                             base=32)[:, 0]
    assert pos[:, 1].tolist() == [1, 2.5, 2.5] and pos[:, 2].tolist() == [5, 5, 5]


def test_frame_layout_conditions():
    conds = {"stock": ("uniform", 512), "f": ("fovea", dict(a=40, R=180, gaze="gaze")),
             "ctr": ("fovea", dict(a=40, R=180, gaze="centre")), "m": ("uniform_match", "f")}
    fr = dict(gaze=(0.3, 0.7))
    cells, side, used = q3.frame_layout(conds, "stock", fr)
    assert side == 512 and len(cells) == 256 and used is None
    cells, side, used = q3.frame_layout(conds, "f", fr)
    assert side == q3.IMG and used == (0.3, 0.7)
    assert np.array_equal(cells, q3.foveated_cells(0.3, 0.7, a=40, R_deg=180))
    m_cells, m_side, _ = q3.frame_layout(conds, "m", fr)
    assert m_side == q3.same_tokens_size(len(cells)) and len(m_cells) == (m_side // 32) ** 2
    miss, _, used = q3.frame_layout(conds, "f", dict(gaze=None))
    assert used is None and np.array_equal(miss, q3.frame_layout(conds, "ctr", fr)[0])

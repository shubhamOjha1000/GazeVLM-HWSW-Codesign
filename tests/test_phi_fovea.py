"""Foveated tokenization for Phi-3.5-vision (vlmeval/phi_fovea.py). CPU only, no downloads.

Against a tiny random CLIP built from transformers' own CLIPVisionModel (336 px, 14 px patches:
the real 24 x 24 grid) inside phi_fovea.PhiImage:
  1. layouts on the 672 px view cover it once and never straddle a 336 px crop
  2. merge_2x2 / crop arrangement / row breaks equal Phi's own reshape_hd_patches_2x2merge and
     add_image_newline (copied from its remote code)
  3. checkpoint names (CLIP nested as vision_model before transformers 5) load onto this CLIP
  4. position table: exact at integer patch coordinates, bilinear in between
  5. PARITY: an all-28 px layout through our cells path equals CLIP's own forward on the 4 crops
     (Phi's feature layer -2, CLS dropped, 2 x 2 merged)
  6. a foveated layout gives one finite token per cell, each following its cell when reordered;
     its HD sequence has one row break per token row
  7. the uniform path keeps Phi's crop structure: 4 k^2 crop tokens
  8. Phi's prompt: image tags in its chat template; the processor's chunked tokenization (a BOS
     per chunk, -k ids for image k)
"""
import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from vlmeval import phi_fovea as ph
from vlmeval import qwen_fovea as qf

CLIP_TINY = dict(ph.CLIP_L336, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=2)


def rand_img(size, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(3, size, size, generator=g) * 255).round() / 255


def tiny():
    return ph.PhiImage(hidden=24, clip_cfg=CLIP_TINY).float().eval()


@pytest.fixture(scope="module")
def img():
    try:
        m = tiny()
    except Exception as ex:
        pytest.skip(f"cannot build a tiny CLIP here: {ex}")
    torch.manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.2)
    return m


# ---------------------------------------------------------------------------- layout and HD assembly
@pytest.mark.parametrize("g", [(0.5, 0.5), (0.1, 0.9), (0.73, 0.31)])
@pytest.mark.parametrize("R", [70, 180, 240])
def test_layout_covers_the_view_and_respects_crops(g, R):
    cells = ph.foveated_cells(*g, a=40, R_deg=R)
    cnt = np.zeros((24, 24), int)
    for y, x, s in cells:
        assert s in (28, 56, 112) and y % s == 0 and x % s == 0
        assert y // 336 == (y + s - 1) // 336 and x // 336 == (x + s - 1) // 336
        cnt[y // 28:(y + s) // 28, x // 28:(x + s) // 28] += 1
    assert (cnt == 1).all() and len(cells) < 576


def _phi_reshape(image_features, h_crop, w_crop):          # Phi3ImageEmbedding.reshape_hd_patches_2x2merge
    N, L, C = image_features.shape
    num_images = N // (h_crop * w_crop)
    H = int(L ** 0.5)
    return (image_features.reshape(N, H, H, C).reshape(N, H // 2, 2, H // 2, 2, C).permute(0, 1, 3, 2, 4, 5)
            .reshape(N, -1, 4 * C).reshape(num_images, h_crop, w_crop, H // 2, H // 2, -1)
            .permute(0, 1, 3, 2, 4, 5).reshape(num_images, h_crop * H // 2, w_crop * H // 2, 4 * C))


def _phi_newline(image_features_hd, sub_GN):               # Phi3ImageEmbedding.add_image_newline
    num_images, h, w, hid_dim = image_features_hd.shape
    newline = sub_GN.expand(num_images, h, -1, -1)
    return torch.cat([image_features_hd, newline], dim=2).reshape(num_images, -1, hid_dim)


def test_hd_assembly_matches_phi():
    f = torch.randn(4, 576, 8)                              # 4 crops of 24 x 24 patch features
    sub = torch.randn(1, 1, 1, 32)
    ref = _phi_newline(_phi_reshape(f, 2, 2), sub)[0]
    grid = ph.arrange_crops(ph.merge_2x2(f))
    assert grid.shape == (24, 24, 32) and torch.equal(ph.grid_rows(grid, sub), ref)
    assert torch.equal(ph.hd_rows(grid.reshape(-1, 32), qf.uniform_cells(ph.HD), sub), ref)
    glb = torch.randn(1, 576, 8)                            # the global view: a 1 x 1 "crop grid"
    assert torch.equal(ph.grid_rows(ph.merge_2x2(glb)[0], sub), _phi_newline(_phi_reshape(glb, 1, 1), sub)[0])


def test_checkpoint_names_load(img):
    nested = hasattr(img.img_processor, "vision_model")
    ckpt = {}
    for k, v in img.state_dict().items():
        if k.endswith("position_ids"):
            continue
        if not nested and k.startswith("img_processor."):       # the checkpoint's (pre-5) names
            k = "img_processor.vision_model." + k[len("img_processor."):]
        ckpt[k] = v.clone()
    fresh = ph.load_image_weights(tiny(), ckpt)
    got, want = fresh.state_dict(), img.state_dict()
    assert all(torch.equal(got[k], want[k]) for k in want if not k.endswith("position_ids"))


# ---------------------------------------------------------------------------- CLIP plumbing
def test_position_table_exact_and_bilinear(img):
    clip = img.clip
    table = clip.embeddings.position_embedding.weight[1:].view(24, 24, -1)
    h = torch.tensor([0., 5., 23., 7.5])
    w = torch.tensor([0., 9., 23., 3.25])
    got = ph.sample_positions(clip, h, w)
    assert torch.equal(got[0], table[0, 0]) and torch.equal(got[1], table[5, 9]) and torch.equal(got[2], table[23, 23])
    want = 0.5 * (0.75 * table[7, 3] + 0.25 * table[7, 4]) + 0.5 * (0.75 * table[8, 3] + 0.25 * table[8, 4])
    assert torch.allclose(got[3], want, atol=1e-6)


def test_all_28px_layout_equals_clip_on_the_crops(img):
    hd = rand_img(ph.HD)
    ours = ph.encode_cells(img, hd, qf.uniform_cells(ph.HD))
    stock = ph.crop_grid(img, hd).reshape(-1, ours.shape[-1])
    assert ours.shape == stock.shape == (576, 128)
    assert torch.allclose(ours, stock, atol=1e-4, rtol=1e-4), (ours - stock).abs().max()


def test_foveated_tokens_follow_their_cells(img):
    hd = rand_img(ph.HD, 1)
    cells = ph.foveated_cells(0.3, 0.6, a=40, R_deg=180)
    out = ph.encode_cells(img, hd, cells)
    assert out.shape == (len(cells), 128) and torch.isfinite(out).all()
    perm = np.random.default_rng(0).permutation(len(cells))
    out_p = ph.encode_cells(img, hd, cells[perm])
    assert torch.allclose(out_p, out[torch.as_tensor(perm)], atol=1e-5)
    seq = ph.hd_rows(out, cells, img.sub_GN)
    assert len(seq) == len(cells) + len(np.unique(cells[:, 0] // 28))


@pytest.mark.parametrize("k", [12, 7, 5])
def test_uniform_keeps_the_crop_structure(img, k):
    g = ph.crop_grid(img, rand_img(ph.HD, 2), k)
    assert g.shape == (2 * k, 2 * k, 128) and torch.isfinite(g).all()
    assert ph.same_tokens_k(4 * k * k) == k


def test_frame_tokens_conditions(img):
    hd = rand_img(ph.HD, 3)
    glb_seq = torch.randn(156, 128)
    conds = {"stock": ("stock", None), "f": ("fovea", dict(a=40, R=180, gaze="gaze")),
             "ctr": ("fovea", dict(a=40, R=180, gaze="centre")), "m": ("uniform_match", "f")}
    fr = dict(gaze=(0.3, 0.7))
    e, cells, n, used = ph.frame_tokens(img, conds, "stock", fr, hd, glb_seq)
    assert cells is None and n == 576 and e.shape == (576 + 24 + 1 + 156, 24)
    e, cells, n, used = ph.frame_tokens(img, conds, "f", fr, hd, glb_seq)
    rows = len(np.unique(cells[:, 0] // 28))
    assert used == (0.3, 0.7) and n == len(cells) and e.shape[0] == n + rows + 1 + 156
    e, _, n_m, _ = ph.frame_tokens(img, conds, "m", fr, hd, glb_seq)
    k = ph.same_tokens_k(len(cells))
    assert n_m == 4 * k * k and e.shape[0] == n_m + 2 * k + 1 + 156
    _, miss, _, used = ph.frame_tokens(img, conds, "f", dict(gaze=None), hd, glb_seq)
    assert used is None and np.array_equal(miss, ph.frame_tokens(img, conds, "ctr", fr, hd, glb_seq)[1])


# ---------------------------------------------------------------------------- prompt
class FakeTok:
    """Character-level tokenizer with a BOS: enough to check Phi's chunked tokenization."""
    bos = 1

    def __call__(self, text, add_special_tokens=True):
        ids = ([self.bos] if add_special_tokens else []) + [100 + ord(c) % 50 for c in text]
        return type("Enc", (), {"input_ids": ids})()

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):   # Phi's template
        return "".join(f"<|{m['role']}|>\n{m['content']}<|end|>\n" for m in msgs) + "<|assistant|>\n"


def test_prompt_tags_and_chunked_tokenization():
    tok = FakeTok()
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "intro"}, {"type": "text", "text": "Frame at 0:00:"},
        {"type": "image", "image": "a.jpg"}, {"type": "text", "text": "Frame at 0:05:"},
        {"type": "image", "image": "b.jpg"}, {"type": "text", "text": "Q?"}]}]
    text = ph.phi_text(tok, msgs)
    assert text == ("<|user|>\nintro\nFrame at 0:00:\n<|image_1|>\nFrame at 0:05:\n<|image_2|>\nQ?"
                    "<|end|>\n<|assistant|>\n")
    ids = ph.prompt_ids(tok, text, [3, 2])[0].tolist()
    chunks = ph.IMAGE_TAG.split(text)
    want = tok(chunks[0]).input_ids + [-1] * 3 + tok(chunks[1]).input_ids + [-2] * 2 + tok(chunks[2]).input_ids
    assert ids == want and ids.count(1) == 3                 # a BOS per chunk, as Phi's processor


def test_letter_ids_follow_the_generation_prompt():
    tok = FakeTok()
    gen = tok.apply_chat_template([{"role": "user", "content": "x"}])
    assert ph.letter_token_ids(tok, gen) == [100 + ord(L) % 50 for L in "ABCDE"]

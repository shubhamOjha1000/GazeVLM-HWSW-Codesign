"""StreamingTOM CTR baseline (vlmeval/streamingtom.py). CPU only, no downloads, no data.

  1. received attention = softmax(q k^T * scale) averaged over heads and queries (chunked, GQA, masks)
  2. capture_saliency on a tiny transformers CLIP encoder: the sdpa call of exactly the target layer,
     equal to the attention recomputed from that layer's own q / k projections
  3. DPC-KNN centres: one per well-separated blob
  4. CTR plan: first frame all dynamic (top-G saliency), a repeated frame all static (G merged
     tokens covering it), the proportional budget split, exactly G tokens in position order
  5. merged token = beta * centre + (1 - beta) * mean of its other members
  6. Phi packing: a row break after each token row that has tokens; Qwen packing: cells of the plan
  7. run_questions: causal, frames encoded once per video, one record per (condition, question), resumable
"""
import json

import numpy as np
import pytest
import torch

from vlmeval import streamingtom as stm


# ---------------------------------------------------------------------------- saliency
def _reference(q, k, scale, mask=None):
    a = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if mask is not None:
        a = a.masked_fill(~mask, float("-inf"))
    return a.softmax(-1).mean(dim=(1, 2))


def test_received_attention_matches_the_softmax_mean():
    g = torch.Generator().manual_seed(0)
    q, k = torch.randn(2, 4, 37, 8, generator=g), torch.randn(2, 4, 37, 8, generator=g)
    ref = _reference(q, k, 8 ** -0.5)
    assert torch.allclose(stm.received_attention(q, k, chunk=5), ref, atol=1e-6)
    assert torch.allclose(stm.received_attention(q, k).sum(-1), torch.ones(2), atol=1e-5)
    k2 = torch.randn(2, 2, 37, 8, generator=g)                                       # grouped kv heads
    assert torch.allclose(stm.received_attention(q, k2, scaling=0.3), _reference(q, k2.repeat_interleave(2, 1), 0.3),
                          atol=1e-6)
    mask = torch.ones(2, 1, 37, 37, dtype=torch.bool)
    mask[..., -3:] = False                                                           # three keys masked out
    out = stm.received_attention(q, k, mask, chunk=7)
    assert torch.allclose(out, _reference(q, k, 8 ** -0.5, mask), atol=1e-6) and torch.all(out[:, -3:] == 0)


@pytest.fixture(scope="module")
def clip():
    try:
        from transformers import CLIPVisionConfig, CLIPVisionModel
        cfg = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=2,
                               image_size=56, patch_size=14)
        cfg._attn_implementation = "sdpa"
        m = CLIPVisionModel(cfg).eval()
    except Exception as ex:
        pytest.skip(f"cannot build a tiny CLIP here: {ex}")
    return m


def test_capture_saliency_reads_exactly_the_target_layer(clip):
    layers = stm.encoder_layers(getattr(clip, "vision_model", clip))
    target = layers[-2].self_attn
    seen = {}

    def grab(module, args, kwargs):
        seen["h"] = kwargs.get("hidden_states", args[0] if args else None)

    hook = target.register_forward_pre_hook(grab, with_kwargs=True)
    x = torch.randn(2, 3, 56, 56, generator=torch.Generator().manual_seed(1))
    try:
        with torch.no_grad(), stm.capture_saliency(target) as got:
            clip(pixel_values=x)
    finally:
        hook.remove()
    assert len(got) == 1 and got[0].shape == (2, 17)                     # one call: CLS + 4 x 4 patches
    h = seen["h"]
    H = target.num_heads
    q = target.q_proj(h).view(2, 17, H, -1).transpose(1, 2)
    k = target.k_proj(h).view(2, 17, H, -1).transpose(1, 2)
    assert torch.allclose(got[0], _reference(q, k, target.scale), atol=1e-5)
    with torch.no_grad(), stm.capture_saliency(layers[0].self_attn) as other:   # registry restored, reusable
        clip(pixel_values=x)
    assert len(other) == 1 and not torch.allclose(other[0], got[0])


# ---------------------------------------------------------------------------- CTR
def test_dpc_centres_one_per_blob():
    offs = torch.tensor([[0, 0], [0.1, 0], [-0.1, 0], [0, 0.1], [0, -0.1]])
    x = torch.cat([offs + torch.tensor(c) for c in ([0.0, 0.0], [10.0, 0.0], [0.0, 10.0])])
    cen = stm.dpc_centres(x, 3, knn=3)
    assert sorted((cen // 5).tolist()) == [0, 1, 2] and torch.equal(cen, torch.sort(cen).values)
    assert torch.equal(stm.dpc_centres(x, 20), torch.arange(15))


def _frame(seed, n=16, d=6):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed))


def _check_plan(plan, G, n):
    pos = [p for p, _ in plan]
    used = pos + [m for _, ms in plan for m in ms]
    assert len(plan) == G and pos == sorted(pos) and len(set(used)) == len(used) and all(0 <= u < n for u in used)


def test_first_frame_is_all_dynamic_top_saliency():
    F_t = _frame(0)
    sal = torch.arange(16.0)
    plan, n_static = stm.ctr_plan(F_t, None, sal, 4)
    assert n_static == 0 and plan == [(12, []), (13, []), (14, []), (15, [])]


def test_repeated_frame_is_all_static_and_merged():
    F_t = _frame(1)
    plan, n_static = stm.ctr_plan(F_t, F_t.clone(), torch.rand(16), 4, knn=3)
    _check_plan(plan, 4, 16)
    assert n_static == 16 and sum(1 + len(m) for _, m in plan) == 16          # every token in some merged token


def test_budget_split_follows_the_static_share():
    F_prev = _frame(2)
    F_t = F_prev.clone()
    F_t[:6] = -F_prev[:6]                                                      # 6 dynamic, 10 static
    sal = torch.zeros(16)
    sal[:6] = torch.tensor([5.0, 1, 4, 2, 3, 0])
    plan, n_static = stm.ctr_plan(F_t, F_prev, sal, 8, knn=3)
    _check_plan(plan, 8, 16)
    dynamic = [p for p, m in plan if p < 6]
    static = [(p, m) for p, m in plan if p >= 6]
    assert n_static == 10 and dynamic == [0, 2, 4]                 # G_d = 8 - floor(8 * 10 / 16) = 3: top saliency
    assert len(static) == 5 and sum(1 + len(m) for _, m in static) == 10   # G_s = 5 groups covering the static tokens


def test_apply_plan_merges_with_beta():
    X = torch.arange(12.0).view(6, 2)
    out = stm.apply_plan(X, [(0, []), (2, [3, 5])], beta=0.6)
    assert torch.equal(out[0], X[0])
    assert torch.allclose(out[1], 0.6 * X[2] + 0.4 * X[[3, 5]].mean(0))


# ---------------------------------------------------------------------------- packing
def test_phi_pack_adds_a_row_break_per_row():
    from vlmeval import phi_fovea as ph
    cfg = dict(ph.CLIP_L336, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2)
    try:
        img = ph.PhiImage(hidden=24, clip_cfg=cfg).float().eval()
    except Exception as ex:
        pytest.skip(f"cannot build a tiny Phi image module here: {ex}")
    torch.nn.init.normal_(img.sub_GN)
    b = stm.Phi.__new__(stm.Phi)
    b.ph, b.img, b.side = ph, img, ph.GRID
    tokens = torch.randn(ph.GRID * ph.GRID, 24)
    plan = [(3, []), (5, []), (24 * 2 + 1, [24 * 2 + 2]), (24 * 7, [])]          # rows 0, 0, 2, 7
    with torch.no_grad():
        out, = b.pack(dict(tokens=tokens), plan, 0.6)
        br = img.img_projection(img.sub_GN.reshape(1, -1))[0]
    assert out.shape == (4 + 3, 24)
    assert torch.equal(out[0], tokens[3]) and torch.equal(out[1], tokens[5])
    assert all(torch.allclose(out[i], br) for i in (2, 4, 6))


def test_qwen_pack_cells_follow_the_plan():
    from vlmeval import qwen_fovea as qf
    b = stm.Qwen25.__new__(stm.Qwen25)
    b.qf, b.px = qf, 448
    tokens = torch.randn(256, 8)
    plan = [(0, []), (17, [18, 33]), (255, [])]
    feats, cells, px = b.pack(dict(tokens=tokens), plan, 0.6)
    assert px == 448 and feats.shape == (3, 8)
    assert cells.tolist() == [[0, 0, 28], [28, 28, 28], [15 * 28, 15 * 28, 28]]


# ---------------------------------------------------------------------------- the run loop
class FakeBackend:
    def __init__(self):
        self.encoded = []
        base = _frame(7, n=8, d=4)
        self.frames = {0: base, 1: base.clone(), 2: -base}                   # static, then all dynamic

    def frame_state(self, fr):
        self.encoded.append(fr["key"])
        return dict(tokens=self.frames[fr["key"][1]], sal=torch.arange(8.0))

    def pack(self, st, plan, beta):
        return (stm.apply_plan(st["tokens"], plan, beta),)

    def score(self, it, per, build_messages):
        p = np.full(5, 0.1)
        p["ABCDE".index(it["answer"])] = 0.6
        return p, 10 + sum(len(f[0]) for f in per)


def _items():
    fr = lambda v, k: dict(key=(v, k), path=f"{v}_{k}.jpg", t=float(k))
    return [dict(qid=q, video=v, question="q", options=list("abcde"), answer="C", frames=[fr(v, k) for k in range(3)])
            for q, v in ((1, "a"), (2, "a"), (3, "b"))]


def test_run_questions_is_causal_cached_and_resumable(tmp_path):
    out = str(tmp_path / "streamingtom.jsonl")
    conds = {"G=0.25N": dict(ratio=0.25), "G=0.5N": dict(ratio=0.5)}
    be = FakeBackend()
    stm.run_questions(be, _items(), conds, out, build_messages=None)
    rows = [json.loads(l) for l in open(out)]
    assert len(rows) == 6 and all(r["correct"] == 1 and r["error"] is None for r in rows)
    assert sorted(be.encoded) == sorted({(v, k) for v in "ab" for k in range(3)})   # once per frame and video
    r = next(r for r in rows if r["condition"] == "G=0.5N")
    assert r["n_image_tokens"] == 3 * 4 and r["n_tokens"] == 10 + 12
    assert r["static_share"] == pytest.approx(1 / 3, abs=1e-4) and r["n_merged_tokens"] >= 1   # recorded to 4 decimals
    assert r["streamingtom"] == dict(ratio=0.5, tau=stm.TAU, knn=stm.KNN, beta=stm.BETA, released_dpc=False)
    stm.run_questions(FakeBackend(), _items(), conds, out, build_messages=None)
    assert len(open(out).read().splitlines()) == 6

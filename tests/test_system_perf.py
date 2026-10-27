"""System-performance measurement (system_perf/). CPU only, no data.

  1. question sampling across videos; frames encoded per method
  2. the probe's hooks on a decoder stack; the batch expansion of its prefill inputs
  3. decode_benchmark: KV-cache greedy decoding from the captured prefill inputs = recomputing every step
  4. the frame cache; the report's budgets, speed-ups, savings and overheads
"""
import pandas as pd
import pytest
import torch
from PIL import Image

from system_perf import measure as ms
from system_perf import probe as pr
from system_perf import report as rp


def test_sample_questions_spread_over_videos():
    qs = pd.DataFrame(dict(video=["a"] * 5 + ["b"] * 2 + ["c"], qid=list(range(8))))
    q = ms.sample_questions(qs, 6, seed=0)
    by = qs.set_index("qid").video
    assert len(set(q)) == 6 and sorted(by[q].value_counts().tolist()) == [1, 2, 3]
    assert [by[x] for x in q[:3]] == ["a", "b", "c"] and q == ms.sample_questions(qs, 6, seed=0)


def test_frames_encoded():
    it = dict(frames=[1, 2, 3])
    assert ms.frames_encoded("uniform", it) == 3 and ms.frames_encoded("gazevlm2", it) == 6
    assert ms.frames_encoded("blind", dict(frames=[])) == 0


class Dec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

    def forward(self, inputs_embeds=None, position_ids=None, **kw):
        return self.layers[0](inputs_embeds)


def test_probe_marks_the_first_call_and_keeps_its_inputs():
    dec, x = Dec(), torch.randn(1, 3, 4)
    with pr.Probe(dec) as p:
        dec(inputs_embeds=x, position_ids=torch.arange(3)[None])
        dec(inputs_embeds=x[:, :1])
    assert p.calls == 2 and p.kwargs["inputs_embeds"] is x and p.t_start <= p.t_end
    assert not dec._forward_hooks and not dec._forward_pre_hooks                   # removed on exit


def test_expand_batch():
    kw = dict(inputs_embeds=torch.zeros(1, 5, 4), position_ids=torch.zeros(3, 1, 5), attention_mask=torch.ones(1, 5),
              visual_pos_masks=torch.zeros(1, 5, dtype=torch.bool), deepstack_visual_embeds=[torch.zeros(2, 4)], use_cache=False)
    out = pr.expand_batch(kw, 3)
    assert out["inputs_embeds"].shape == (3, 5, 4) and out["position_ids"].shape == (3, 3, 5)
    assert out["visual_pos_masks"].shape == (3, 5) and out["deepstack_visual_embeds"][0].shape == (6, 4)
    assert "attention_mask" not in out and out["use_cache"] is False
    assert pr.expand_batch(dict(position_ids=torch.zeros(1, 5)), 2)["position_ids"].shape == (2, 5)


@pytest.mark.parametrize("with_positions", [False, True])
def test_decode_benchmark_equals_greedy_without_cache(with_positions):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=64)
    m = LlamaForCausalLM(cfg).eval()
    ids = torch.tensor([[3, 7, 11, 2, 9]])
    dec = pr.find_decoder(m)
    assert dec is m.model
    with torch.no_grad():
        kw = dict(inputs_embeds=dec.embed_tokens(ids))
        if with_positions:
            kw["position_ids"] = torch.arange(5)[None]
        d = pr.decode_benchmark(m, dec, kw, n_tokens=6)
        seq, ref = ids, []
        for _ in range(6):
            t = int(m(input_ids=seq).logits[0, -1].argmax())
            ref.append(t)
            seq = torch.cat([seq, torch.tensor([[t]])], 1)
    assert d["tokens"] == ref and d["tps"] > 0


def test_frame_cache(tmp_path):
    root, other = tmp_path / "frames", tmp_path / "other"
    root.mkdir()
    other.mkdir()
    p, q = str(root / "f.png"), str(other / "g.png")
    Image.new("RGB", (8, 8), (10, 20, 30)).save(p)
    Image.new("RGB", (8, 8), (0, 0, 0)).save(q)
    with pr.FrameCache(str(root)) as c:
        c.preload([p])
        a = Image.open(p)
        a.putpixel((0, 0), (255, 255, 255))                                          # a copy: the cache is untouched
        assert Image.open(p).getpixel((0, 0)) == (10, 20, 30) and list(c.images) == [p]
        Image.open(q)
        assert list(c.images) == [p]                                                 # paths outside root: not cached
    assert not c.images and Image.open is not None


def test_report_budgets_speedups_and_overheads():
    rows = []
    for qid in (1, 2):
        base = dict(qid=qid, video="v", error=None, n_frames=96, decode_s=1.0, energy_j=10.0, power_w=100.0,
                    decode_energy_j=5.0, prefill_s=0.5)
        rows.append(dict(base, row="uniform", condition="U", frames_encoded=96, n_image_tokens=1000, ttft_s=2.0, ve_s=1.0,
                         tps=10.0, ve_peak_gb=1.0, ld_peak_gb=4.0, overall_gb=4.0))
        rows.append(dict(base, row="ours", condition="O", frames_encoded=47, n_image_tokens=100, ttft_s=1.0, ve_s=0.5,
                         tps=20.0, ve_peak_gb=0.5, ld_peak_gb=1.0, overall_gb=1.0))
        rows.append(dict(base, row="dtd", condition="D", frames_encoded=96, n_image_tokens=100, ttft_s=1.5, ve_s=1.1,
                         tps=18.0, ve_peak_gb=1.0, ld_peak_gb=1.0, overall_gb=1.0))
    flops = pd.DataFrame([dict(row="uniform", condition="U", ve_gflops=100.0, prefill_gflops=50.0),
                          dict(row="ours", condition="O", ve_gflops=50.0, prefill_gflops=5.0),
                          dict(row="dtd", condition="D", ve_gflops=101.0, prefill_gflops=5.0)])
    gate = {"v": dict(decisions=50, models=1, flops=72_000 * 50, ms=3.0)}
    acc = pd.DataFrame(dict(condition=["O"], accuracy=[44.0])).set_index("condition")
    t = rp.table(pd.DataFrame(rows), flops, gate, acc).set_index("row")
    assert t.loc["ours", "budget_pct"] == pytest.approx(10.0) and t.loc["ours", "ttft_speedup"] == pytest.approx(2.0)
    assert t.loc["ours", "tps_speedup"] == pytest.approx(2.0) and t.loc["ours", "overall_saving_pct"] == pytest.approx(75.0)
    assert t.loc["ours", "overhead_mflops"] == pytest.approx(3.6) and t.loc["ours", "overhead_ms"] == pytest.approx(3.0)
    assert t.loc["dtd", "overhead_mflops"] == pytest.approx(1000.0) and t.loc["dtd", "overhead_ms"] == pytest.approx(100.0)
    assert t.loc["ours", "accuracy"] == 44.0 and t.loc["ours", "ve_gflops"] == 50.0
    assert rp.latex_rows(t.reset_index(), "Qwen2.5-VL-3B").count(r"\\") == 3

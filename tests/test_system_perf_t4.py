"""system_perf/t4.py: the T4 notebook's runner, which converts Qwen3-VL to fp16 in its own process only. CPU only, no model."""
import pytest
import torch

from accuracy_tables.hpc.all350 import shards
from system_perf import report, t4
from vlmeval import qwen3_fovea as q3


def fake_load(model_id):
    m = torch.nn.Linear(2, 2).to(torch.bfloat16)
    m.register_buffer("inv_freq", torch.tensor([1.0, 0.9, 1.3e-7]), persistent=False)        # float32, like the rotary's
    m.register_buffer("scale", torch.ones(2, dtype=torch.bfloat16))
    m.register_buffer("ids", torch.arange(3))
    return m, "proc", [1, 2]


def test_qwen3_is_converted_to_fp16_once(monkeypatch):
    monkeypatch.setattr(q3, "load_model", fake_load)
    t4.fp16_qwen3()
    model, proc, letters = q3.load_model("x")
    assert all(p.dtype == torch.float16 for p in model.parameters()) and proc == "proc" and letters == [1, 2]
    assert model.scale.dtype == torch.float16 and model.ids.dtype == torch.int64
    assert model.inv_freq.dtype == torch.float32 and model.inv_freq.tolist() == torch.tensor([1.0, 0.9, 1.3e-7]).tolist()   # not rounded
    wrapped = q3.load_model
    t4.fp16_qwen3()
    assert q3.load_model is wrapped                                     # not wrapped twice


@pytest.mark.parametrize("bf16", [False, True])
def test_a_step_runs_with_its_own_arguments_and_fp16_only_without_bf16(bf16, monkeypatch):
    monkeypatch.setattr(q3, "load_model", fake_load)
    monkeypatch.setattr(t4, "has_bf16", lambda: bf16)
    got = []
    monkeypatch.setattr(report, "main", lambda argv: got.append(argv))
    t4.main(["system_perf.report", "--results", "r", "--model", "qwen3"])
    assert got == [["--results", "r", "--model", "qwen3"]]
    assert getattr(q3.load_model, "fp16", False) is (not bf16)


def test_other_steps_are_refused():
    with pytest.raises(SystemExit, match="usage"):
        t4.main(["accuracy_tables.hpc.all350.flops_shards", "merge"])
    with pytest.raises(SystemExit, match="usage"):
        t4.main([])


def test_the_runner_is_outside_the_code_the_cluster_shards_fingerprint():
    assert "system_perf" not in shards.CODE_FOLDERS

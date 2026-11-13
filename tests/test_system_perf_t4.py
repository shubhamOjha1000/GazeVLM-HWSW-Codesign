"""system_perf/t4.py: the T4 notebook's runner, which converts Qwen3-VL to fp16 in its own process only. CPU only, no model."""
import pytest
import torch

from accuracy_tables.hpc.all350 import shards
from system_perf import report, t4
from vlmeval import qwen3_fovea as q3


def fake_load(model_id):
    return torch.nn.Linear(2, 2).to(torch.bfloat16), "proc", [1, 2]


def test_qwen3_is_converted_to_fp16_once(monkeypatch):
    monkeypatch.setattr(q3, "load_model", fake_load)
    t4.fp16_qwen3()
    model, proc, letters = q3.load_model("x")
    assert next(model.parameters()).dtype == torch.float16 and proc == "proc" and letters == [1, 2]
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

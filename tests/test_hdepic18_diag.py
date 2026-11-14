"""Runbook section 18, diagnostic C1 to C4: the pieces that run without a GPU (Wilson intervals, prompt collapsing, the fovea's box, the
answer agreement, the fovea's input check, the record comparison, the ladder's layouts) and the jobs. CPU only, fake data."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from accuracy_tables.hpc.hdepic import diag18

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def job(name):
    with open(os.path.join(REPO, "accuracy_tables", "hpc", name)) as f:
        return f.read()


def test_wilson_intervals():
    acc, lo, hi = diag18.wilson(50, 100)
    assert acc == pytest.approx(50) and lo == pytest.approx(40.38, abs=0.01) and hi == pytest.approx(59.62, abs=0.01)
    assert all(np.isnan(x) for x in diag18.wilson(0, 0))


def test_prompts_collapse_their_image_tokens():
    text = "a<image><image><image>b<image>c"
    assert diag18.collapse(text) == "a<image x3>b<image x1>c" and diag18.strip_counts(diag18.collapse(text)) == "a<image>b<image>c"


def test_the_fovea_box_in_source_pixels():
    assert diag18.fovea_box(None, 384) == (126, 126, 252, 252)                   # no gaze: the centre cell
    assert diag18.fovea_box((0.1, 0.1), 1344) == (0, 0, 441, 441)                # 126 px of 384, at 1344 px


def test_answer_agreement_and_which_side_budget_takes():
    rows = []
    for q in range(10):
        uni, blind = "A", ("A" if q < 4 else "B")
        rows += [dict(qid=q, label="uniform", pred=uni), dict(qid=q, label="blind", pred=blind),
                 dict(qid=q, label="budget 10%", pred=blind), dict(qid=q, label="nogate 10%", pred="C")]
    pair, letters, side, n = diag18.agreement(pd.DataFrame(rows))
    assert n == 10 and pair.loc["budget 10%", "blind"] == pytest.approx(100) and pair.loc["budget 10%", "uniform"] == pytest.approx(40)
    assert side == dict(n=6, with_blind=pytest.approx(100), with_uniform=pytest.approx(0))
    assert letters.loc["nogate 10%", "C"] == pytest.approx(100) and "gazevlm 10%" not in pair


def test_the_fovea_inputs_check_finds_the_stock_pixels():
    torch = pytest.importorskip("torch")
    from vlmeval import smolvlm_fovea as sm
    x01 = torch.rand(3, sm.IMG, sm.IMG)
    rec = dict(x01=x01, pv=(x01 - 0.5) / 0.5)
    worst, ids = diag18.fovea_inputs(rec, sm.foveated_cells(0.7, 0.2, 1), [0.5] * 3, [0.5] * 3)
    assert worst < 1e-5 and ids
    bad = dict(rec, pv=rec["pv"].flip(-1))
    assert diag18.fovea_inputs(bad, sm.foveated_cells(0.7, 0.2, 1), [0.5] * 3, [0.5] * 3)[0] > 0.1


def test_the_ladder_layouts():
    pytest.importorskip("torch")
    from vlmeval import smolvlm_fovea as sm
    assert len(sm.foveated_cells(0.5, 0.5, 0)) == 9 and len(sm.foveated_cells(0.5, 0.5, 1)) == 17
    assert np.array_equal(np.asarray(sm.foveated_cells(0.3, 0.7, 9)), np.asarray(sm.uniform_cells()))     # C3's identity layout
    sizes = {name: cond[1] for name, cond in diag18.LADDER if cond[0] == "uniform"}
    assert {s: (s // sm.BASE) ** 2 for s in sizes.values()} == {384: 81, 126: 9, 168: 16}


def test_two_record_files_are_compared_by_question(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("".join(json.dumps(dict(qid=q, pred="A", p=[0.6, 0.1, 0.1, 0.1, 0.1])) + "\n" for q in range(3)))
    b.write_text("".join(json.dumps(dict(qid=q, pred="A" if q else "B", p=[0.6, 0.1, 0.1, 0.1, 0.1] if q else [0.2, 0.5, 0.1, 0.1, 0.1]))
                         + "\n" for q in range(3)))
    n, nd, mx, first = diag18.compare_records(str(a), str(b))
    assert (n, nd) == (3, 1) and mx == pytest.approx(0.4) and first[0]["qid"] == 0


def test_the_diagnostic_jobs_read_results_only_and_hold_one_b200_each():
    cpu, gpu = job("hdepic18_diag_cpu.SBATCH"), job("hdepic18_diag_gpu.SBATCH")
    for text in (cpu, gpu):
        assert "--no-parity" not in text and "SBATCH_ACCOUNT" not in text and "--out-dir" not in text and "results/" not in text
        assert "HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}" in text and "diag18 $MODE" in text
    assert "gres" not in cpu and "#SBATCH --gres=gpu:b200:1" in gpu and "--expect $GPU" in gpu and "GPU=B200" in gpu
    assert "--requeue" in gpu and "WANT=hdepic18-diag-$MODE" in gpu and "diagnose18/$MODE" in gpu


def test_diagnostic_outputs_go_to_diagnose18_only(tmp_path, monkeypatch):
    monkeypatch.setenv("GV_ROOT", str(tmp_path / "gv"))
    monkeypatch.setenv("HDEPIC_ROOT", str(tmp_path / "hdepic"))
    assert diag18.out_dir("c4") == str(tmp_path / "gv" / "data" / "hdepic" / "diagnose18" / "c4")
    src = open(os.path.join(REPO, "accuracy_tables", "hpc", "hdepic", "diag18.py")).read()
    assert "open(" not in src.split("def results(")[1].split("def ")[0]          # results() only names a folder; nothing writes there


def test_the_runbook_note_lists_the_four_jobs():
    with open(os.path.join(REPO, "HPC_ACCURACY_RUNBOOK.md")) as f:
        section = f.read().split("\n## 18.")[1]
    for text in ("hdepic18-diag-c1", "hdepic18-diag-c2", "hdepic18-diag-c3", "hdepic18-diag-c4", "FRAMES IN SPAN",
                 "FOVEATED FRAMES = PLAN'S CANDIDATES", "FOVEATED IDENTITY", "VISION IDENTITY IN FP32", "AGREEMENT", "C4 DONE",
                 "rerun if C1 finds a frame bug", "gpu_ledger.jsonl", "diagnostic"):
        assert text in section, text

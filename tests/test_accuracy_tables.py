"""accuracy_tables/: config paths, frame selection, items and token budgets. CPU only, no data."""
import os

import pandas as pd
import pytest

from accuracy_tables import common
from accuracy_tables.summarize import budgets


def test_paths_resolve_keys_env_home_and_relative(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_SCRATCH", "/scratch/me")
    cfg = tmp_path / "cfg" / "config.yaml"
    cfg.parent.mkdir()
    cfg.write_text("paths:\n"
                   "  data_root: data\n"
                   "  folds_dir: ${data_root}/folds\n"
                   "  gate_ckpt_dir: ${MY_SCRATCH}/ckpt\n"
                   "  frames_cache_dir: ~/frames\n"
                   "  questions_json: null\n"
                   "run:\n  videos: all\n")
    c = common.load_config(str(cfg))
    p = c["paths"]
    assert p["folds_dir"] == os.path.join(str(cfg.parent), "data", "folds")
    assert p["gate_ckpt_dir"] == "/scratch/me/ckpt"
    assert p["frames_cache_dir"] == os.path.expanduser("~/frames")
    assert p["questions_json"] is None
    assert c["run"]["videos"] == "all" and c["run"]["K"] == 96 and c["run"]["max_frames"]["smolvlm"] == 64


def test_missing_paths_and_unknown_variables_are_reported(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("paths:\n  folds_dir: f\n")
    with pytest.raises(KeyError, match="gate_ckpt_dir"):
        common.load_config(str(cfg))
    cfg.write_text("paths:\n  folds_dir: ${NOT_A_KEY_OR_ENV_VAR_123}/f\n  gate_ckpt_dir: g\n  frames_cache_dir: h\n")
    with pytest.raises(KeyError, match="NOT_A_KEY_OR_ENV_VAR_123"):
        common.load_config(str(cfg))


def test_example_config_loads():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "accuracy_tables", "config.example.yaml")
    c = common.load_config(path)
    assert c["run"]["policy"] == "D+P" and len(c["ours"]["rings"]) == 4


@pytest.mark.parametrize("n,k", [(299, 96), (59, 96), (96, 96), (1, 96), (10, 3)])
def test_uniform_indices(n, k):
    idx = common.uniform_indices(n, k)
    assert len(idx) == min(n, k) and idx[0] == 0 and idx[-1] == n - 1
    assert idx == sorted(set(idx)) and all(0 <= i < n for i in idx)


def test_cap_frames_keeps_order_and_ends():
    frames = list(range(100))
    assert common.cap_frames(frames, None) == frames and common.cap_frames(frames, 200) == frames
    capped = common.cap_frames(frames, 64)
    assert len(capped) == 64 and capped[0] == 0 and capped[-1] == 99 and capped == sorted(capped)


def test_make_items_sorted_by_video_with_their_frames():
    qs = pd.DataFrame(dict(qid=[3, 1, 2], video=["b", "a", "b"], question=["q3", "q1", "q2"],
                           options=[list("vwxyz")] * 3, answer=["A", "B", "C"]))
    items = common.make_items(qs, {"a": ["fa"], "b": ["fb1", "fb2"]})
    assert [(it["video"], it["qid"]) for it in items] == [("a", 1), ("b", 2), ("b", 3)]
    assert items[1]["frames"] == ["fb1", "fb2"] and common.make_items(qs, {})[0]["frames"] == []


def test_budget_is_relative_to_uniform_on_common_questions():
    res = pd.DataFrame([
        dict(row="uniform", condition="U", qid=1, n_image_tokens=1000),
        dict(row="uniform", condition="U", qid=2, n_image_tokens=3000),
        dict(row="ours", condition="O", qid=1, n_image_tokens=100),
        dict(row="ours", condition="O", qid=2, n_image_tokens=100),
        dict(row="ours", condition="O", qid=9, n_image_tokens=10 ** 6),     # no uniform: ignored
        dict(row="blind", condition="B", qid=1, n_image_tokens=0)])
    b = budgets(res)
    assert b["U"] == pytest.approx(100) and b["O"] == pytest.approx(5) and b["B"] == 0

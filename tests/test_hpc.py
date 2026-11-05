"""accuracy_tables/hpc/: the Torch config and the result-file repair. CPU only, no data."""
import os

from accuracy_tables import common
from accuracy_tables.hpc.repair_results import repair

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "accuracy_tables", "hpc", "config.torch.yaml")


def test_config_resolves_under_the_two_roots_and_keeps_the_protocol(monkeypatch):
    monkeypatch.setenv("GV_ROOT", "/g")
    monkeypatch.setenv("EGO_ROOT", "/e")
    c = common.load_config(CONFIG)
    p = c["paths"]
    assert p["folds_dir"] == "/g/data/folds" and p["oof_cache"] == "/g/data/vlm/egoeverything/oof_preds_full.csv"
    assert p["questions_json"] == "/e/Egoeverything_VQA.json" and p["frames_local_dir"] == "/g/frames_1344"
    assert p["links_json"] is None
    assert c["run"]["K"] == 96 and c["run"]["policy"] == "D+P" and c["run"]["parity"] is True
    assert c["run"]["max_frames"] == {"smolvlm": 64} and c["ours"]["smolvlm_k"] == [1, 3, 5, 7]


def test_repair_drops_only_a_half_written_last_line(tmp_path):
    whole, cut, only = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    whole.write_text('{"a": 1}\n{"b": 2}\n')
    cut.write_text('{"a": 1}\n{"b"')
    only.write_text('{"a"')
    assert repair(str(tmp_path)) == [str(cut), str(only)]
    assert whole.read_text() == '{"a": 1}\n{"b": 2}\n' and cut.read_text() == '{"a": 1}\n' and only.read_text() == ""
    assert repair(str(tmp_path / "no_such_folder")) == []

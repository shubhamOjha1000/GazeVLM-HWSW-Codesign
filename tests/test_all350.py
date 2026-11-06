"""accuracy_tables/hpc/all350/ and the frame loading threads on Qwen2.5, Qwen3 and Phi. CPU only, no dataset, no model. The joined data
goes through the real load_data, labels, gaze and budget code."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from accuracy_tables import common, models, run
from accuracy_tables.hpc.all350 import gpu_summary, measure_tokens, prepare, report, seed_results
from accuracy_tables.hpc.newvideos import compare_records, paths
from eyehead import policy as pol
from vlmeval import budget_rule as br
from vlmeval import phi_fovea as ph
from vlmeval import qwen3_fovea as q3
from vlmeval import qwen_fovea as qf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = pol.S
SETTING = dict(R=70, a=160)


# ---------------------------------------------------------------------------- settings
def test_config_uses_the_joined_folder_and_keeps_the_protocol(monkeypatch):
    monkeypatch.setenv("GV_ROOT", "/g")
    monkeypatch.setenv("EGO_ROOT", "/e")
    c = common.load_config(prepare.CONFIG)
    p = c["paths"]
    assert p["folds_dir"] == "/g/newvideos/all350/folds" and p["oof_cache"] == "/g/newvideos/all350/preds.csv"
    assert p["frames_local_dir"] == "/g/newvideos/all350/frames" and p["questions_json"] == "/e/Egoeverything_VQA.json"
    assert c["run"]["videos"] == "all" and c["run"]["K"] == 96 and c["run"]["policy"] == "D+P" and c["run"]["parity"] is True
    assert c["run"]["max_frames"] == {"smolvlm": 64} and c["run"]["prefetch_workers"] == 0
    assert c["budget"]["fractions"] == [0.05, 0.1, 0.2] and c["budget"]["tokens_per_frame"]["smolvlm"] == 17
    assert c["ours"]["rings"][0] == SETTING


def test_the_groups_add_up_to_the_dataset():
    assert sum(q for q, _ in prepare.GROUPS.values()) == prepare.TOTAL[0] == 5985
    assert sum(v for _, v in prepare.GROUPS.values()) == prepare.TOTAL[1] == 350
    assert prepare.GROUPS["aea12"][0] + prepare.GROUPS["aea13"][0] == 2312 and prepare.GROUPS["nymeria"][0] == 3673
    assert sum(prepare.NYMERIA_SPLIT) == prepare.GROUPS["nymeria"][1]


# ---------------------------------------------------------------------------- the pilot
def test_pilot_has_both_kinds_of_nymeria_video_and_a_new_aea_video():
    many = [f"m{i:02d}" for i in range(21)]
    one = [f"o{i:03d}" for i in range(304)]
    aea = [f"loc{i:02d}" for i in range(13)]
    counts = dict({v: 160 for v in many}, **{v: 1 for v in one})
    pilot = prepare.pick_pilot(one + many, aea, counts)
    assert pilot == ["m05", "m15", "o038", "o114", "o190", "o266", "loc06"]
    assert prepare.pick_pilot(list(reversed(one + many)), list(reversed(aea)), counts) == pilot     # the order given doesn't matter


def test_pilot_stops_if_nymeria_isnt_split_as_expected():
    with pytest.raises(SystemExit, match="stop and tell the user"):
        prepare.pick_pilot(["a", "b"], ["loc"], dict(a=5, b=1))


# ---------------------------------------------------------------------------- a small world on disk
KNOWN_NAME = "/content/drive/x/{v}/feat_{k:05d}.npz"
NEW_NAME = "{v}/feat_{k:05d}.npz"
NOT_ASKED = "loc_not_asked"                     # a gate video with rows and predictions, and no questions


def fake_rows(video, n, name):
    """n frames' pairs with a gaze window that moves, and an extra column that the join must drop."""
    pairs = range(n - 1)
    return pd.DataFrame(dict(
        sequence=video, feat_frame_1=[name.format(v=video, k=k) for k in pairs], feat_frame_2=[name.format(v=video, k=k + 1) for k in pairs],
        quad=S, gaze_xy_window=[f"{0.3 + (k % 50) / 100:.4f},0.5000;{0.31 + (k % 50) / 100:.4f},0.5000" for k in pairs], n_oof_in_gap=0, extra=1))


def fake_preds(rows, seed):
    labels = np.random.default_rng(seed).integers(0, 4, len(rows))
    p = rows[["sequence", "feat_frame_1", "feat_frame_2", "quad"]].assign(source="test ensemble")
    for name in ("p_TRANSITION", "p_PURSUIT", "p_REFIXATION", "p_STABLE"):
        p[name] = 0.25
    return p.assign(pred=labels)


def write_frames(folder, video, n, fps):
    d = os.path.join(folder, video)
    os.makedirs(d)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(dict(native_fps=fps, step=fps, n=n, ts=[float(i) for i in range(n)], size=qf.IMG, orig_w=1408, orig_h=1408), f)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """3 groups of 1 or 2 videos, as the earlier sections left them -> {group: videos}."""
    g, e = tmp_path / "g", tmp_path / "e"
    monkeypatch.setenv("GV_ROOT", str(g))
    monkeypatch.setenv("EGO_ROOT", str(e))
    os.makedirs(e)
    lists = dict(aea12=["loc_known_a", "loc_known_b"], aea13=["loc_new"], nymeria=["2023_x", "2023_y"])
    questions = [dict(video_name=v, question=f"q{i}", options=list("abcde"), answer="A", keyframes=[dict(timestamp=1.0, object_name="o", bbox="")])
                 for i, v in enumerate(["loc_known_a", "2023_x", "loc_new", "loc_known_b", "2023_y", "2023_x", "loc_known_a"])]
    with open(e / "Egoeverything_VQA.json", "w") as f:
        json.dump(questions, f)

    folds = g / "data" / "folds"
    os.makedirs(folds)
    train = pd.concat([fake_rows("loc_known_a", 30, KNOWN_NAME), fake_rows(NOT_ASKED, 10, KNOWN_NAME)], ignore_index=True)
    test = fake_rows("loc_known_b", 12, KNOWN_NAME)
    train.to_csv(folds / "train_val_fold1.csv", index=False)
    test.to_csv(folds / "test.csv", index=False)
    os.makedirs(g / "data" / "vlm" / "egoeverything")
    fake_preds(pd.concat([train, test], ignore_index=True), 0).to_csv(paths.known_preds_csv(), index=False)
    for v, n in (("loc_known_a", 30), ("loc_known_b", 12)):
        write_frames(str(g / "frames_1344"), v, n, 20)

    for group, sizes, fps, seed in (("aea13", dict(loc_new=25), 20, 1), ("nymeria", {"2023_x": 40, "2023_y": 200}, 30, 2)):
        rows = pd.concat([fake_rows(v, n, NEW_NAME) for v, n in sizes.items()], ignore_index=True)
        os.makedirs(paths.folds_dir(group))
        rows.to_csv(os.path.join(paths.folds_dir(group), "test.csv"), index=False)
        rows.head(0).to_csv(os.path.join(paths.folds_dir(group), "train_val_fold1.csv"), index=False)
        fake_preds(rows, seed).to_csv(paths.preds_csv(group), index=False)
        for v, n in sizes.items():
            write_frames(paths.new_frames_root(), v, n, fps)
    for name, videos in lists.items():
        paths.write_list(name, videos)
    return lists


def joined_world(lists):
    prepare.join_data(lists)
    prepare.link_frames([v for vs in lists.values() for v in vs])
    return common.load_config(prepare.CONFIG)


# ---------------------------------------------------------------------------- joining
def test_join_keeps_each_groups_names_and_predictions(world):
    joined_world(world)
    rows = pd.read_csv(os.path.join(prepare.folds_dir(), "test.csv"))
    assert list(rows.columns) == prepare.COLUMNS
    assert sorted(rows["sequence"].unique()) == sorted(v for vs in world.values() for v in vs)            # the unasked gate video is gone
    assert rows[rows.sequence == "loc_known_b"]["feat_frame_2"].iloc[0] == KNOWN_NAME.format(v="loc_known_b", k=1)
    assert rows[rows.sequence == "2023_x"]["feat_frame_2"].iloc[0] == NEW_NAME.format(v="2023_x", k=1)
    assert len(pd.read_csv(os.path.join(prepare.folds_dir(), "train_val_fold1.csv"))) == 0
    preds, source = pd.read_csv(prepare.preds_csv()), pd.read_csv(paths.known_preds_csv())
    assert NOT_ASKED not in set(preds.sequence) and set(preds.feat_frame_2) == set(rows.feat_frame_2)
    for name in ("loc_known_a", "loc_known_b"):
        assert preds[preds.sequence == name]["pred"].tolist() == source[source.sequence == name]["pred"].tolist()
    assert preds[preds.sequence == "2023_y"]["pred"].tolist() == pd.read_csv(paths.preds_csv("nymeria")).query("sequence == '2023_y'")["pred"].tolist()


def test_join_problems_each_kind():
    rows = pd.DataFrame(dict(sequence=["a", "a"], feat_frame_2=["x", "y"]))
    preds = pd.DataFrame(dict(sequence=["a", "a"], feat_frame_2=["x", "y"]))
    assert prepare.join_problems(rows, preds, ["a"]) == []
    assert "cover 1 videos, expected 2" in prepare.join_problems(rows, preds, ["a", "b"])[0]
    assert "1 rows have no prediction" in prepare.join_problems(rows, preds.head(1), ["a"])[0]
    assert "duplicate frame names in the rows" in prepare.join_problems(rows.assign(feat_frame_2=["x", "x"]), preds.head(1), ["a"])[0]


def test_join_stops_on_a_group_without_predictions(world):
    p = pd.read_csv(paths.preds_csv("aea13"))
    p.head(3).to_csv(paths.preds_csv("aea13"), index=False)
    with pytest.raises(SystemExit, match="don't join"):
        prepare.join_data(world)


def test_frames_link_to_where_each_group_keeps_them(world):
    prepare.link_frames(["loc_known_a", "2023_x"])
    assert os.readlink(os.path.join(prepare.frames_root(), "loc_known_a")) == os.path.join(os.environ["GV_ROOT"], "frames_1344", "loc_known_a")
    assert os.readlink(os.path.join(prepare.frames_root(), "2023_x")) == os.path.join(paths.new_frames_root(), "2023_x")
    prepare.link_frames(["2023_x"])                                                # a rerun replaces the link
    with pytest.raises(SystemExit, match="no frames"):
        prepare.link_frames(["no_such_video"])


# ---------------------------------------------------------------------------- the joined data, through the real code
def test_joined_data_matches_each_group_read_from_where_it_was_built(world):
    cfg = joined_world(world)
    joined = common.load_data(cfg)
    assert len(joined["qs"]) == 7 and sorted(joined["videos"]) == sorted(v for vs in world.values() for v in vs)
    budget = prepare.smolvlm_budget_frames(cfg, joined)
    for group in world:
        group_cfg = prepare.group_config(group)
        data = common.load_data(group_cfg)
        assert sorted(data["videos"]) == sorted(world[group])
        assert prepare.compare_group(data, prepare.smolvlm_budget_frames(group_cfg, data), joined, budget) == []


def test_compare_group_finds_a_changed_label_and_a_changed_gaze(world):
    cfg = joined_world(world)
    joined = common.load_data(cfg)
    budget = prepare.smolvlm_budget_frames(cfg, joined)
    group_cfg = prepare.group_config("aea13")
    data = common.load_data(group_cfg)
    group_budget = prepare.smolvlm_budget_frames(group_cfg, data)
    data["labels"]["loc_new"] = data["labels"]["loc_new"].copy()
    data["labels"]["loc_new"][0] = (data["labels"]["loc_new"][0] + 1) % 4
    data["uniform"]["loc_new"][1] = dict(data["uniform"]["loc_new"][1], gaze=(0.0, 0.0))
    found = prepare.compare_group(data, group_budget, joined, budget)
    assert "loc_new: gate labels differ" in found and "loc_new: uniform frames or gaze differ" in found


def test_check_counts_stops_on_a_wrong_count(world):
    data = dict(qs=[1, 2], videos=["a"])
    prepare.check_counts("x", data, (2, 1))
    with pytest.raises(SystemExit, match="expected 3 questions"):
        prepare.check_counts("x", data, (3, 1))


def test_expected_frames_of_each_row_on_the_joined_data(world):
    cfg = joined_world(world)
    data = common.load_data(cfg)
    budget = prepare.smolvlm_budget_frames(cfg, data)
    (name, per), = prepare.expected_frames(cfg, data, budget, "phi", "uniform")[0].items()
    assert name == "uniform K=96 (stock input)" and per["loc_known_a"] == 30 and per["2023_y"] == 96
    assert prepare.expected_frames(cfg, data, budget, "phi", "uniform")[1] == 757
    (name, per), = prepare.expected_frames(cfg, data, budget, "smolvlm", "uniform")[0].items()
    assert name == "uniform K=64 (stock input)" and per["2023_y"] == 64
    (name, per), = prepare.expected_frames(cfg, data, budget, "qwen3", "blind")[0].items()
    assert name == run.BLIND and per["2023_y"] == 0
    per, tokens = prepare.expected_frames(cfg, data, budget, "smolvlm", "budget")
    assert tokens == 17 and set(per) == set(seed_results.condition_names("smolvlm", "budget"))
    assert [len(budget[f]["2023_y"]) for f in cfg["budget"]["fractions"]] == list(v["2023_y"] for v in per.values())


# ---------------------------------------------------------------------------- reuse
def record(cond, qid, video, **over):
    return dict(dict(condition=cond, qid=qid, video=video, answer="A", pred="A", correct=1, p=[1, 0, 0, 0, 0], n_frames=2, n_image_tokens=34,
                     error=None), **over)


def write_records(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in records))


VIDEO_OF = ["a", "b", "a", "c"]


def test_condition_names_are_what_a_run_writes():
    assert seed_results.condition_names("qwen2.5", "uniform") == ["uniform K=96 (stock input)"]
    assert seed_results.condition_names("smolvlm", "uniform") == ["uniform K=64 (stock input)"]
    assert seed_results.condition_names("phi", "blind") == [run.BLIND]
    assert seed_results.condition_names("smolvlm", "budget") == [f"budget rule {b}% (D+P gate, fovea k=1)" for b in (5, 10, 20)]


def test_record_problems_each_kind():
    good = [record("c", 0, "a"), record("c", 2, "a")]
    assert seed_results.record_problems(good, ["c"], ["a"], VIDEO_OF) == []
    check = lambda records: " ".join(seed_results.record_problems(records, ["c"], ["a"], VIDEO_OF))
    assert "1 x error record" in check([record("c", 0, "a"), record("c", 2, "a", error="oom")])
    assert "1 x duplicate" in check([record("c", 0, "a"), record("c", 0, "a")])
    assert "1 x other condition" in check(good + [record("d", 0, "a")])
    assert "1 x other video" in check(good + [record("c", 1, "b")])
    assert "1 x qid of another video" in check([record("c", 0, "a"), record("c", 1, "a")])
    assert "c: 1 records, expected 2" in check(good[:1])


def small_sources(tmp_path, model):
    return [("uniform", str(tmp_path / "src" / "u.jsonl"), "one"), ("uniform", str(tmp_path / "src" / "u2.jsonl"), "two"),
            ("blind", str(tmp_path / "src" / "b.jsonl"), "one")]


@pytest.fixture
def seed_world(tmp_path, monkeypatch):
    """Two lists of videos and their records -> where results go. The questions: a, b, a, c."""
    monkeypatch.setenv("GV_ROOT", str(tmp_path))
    monkeypatch.setenv("EGO_ROOT", str(tmp_path))
    with open(tmp_path / "Egoeverything_VQA.json", "w") as f:
        json.dump([dict(video_name=v) for v in VIDEO_OF], f)
    paths.write_list("one", ["a"])
    paths.write_list("two", ["b", "c"])
    monkeypatch.setattr(seed_results, "sources", lambda model: small_sources(tmp_path, model))
    name = seed_results.condition_names("phi", "uniform")[0]
    write_records(tmp_path / "src" / "u.jsonl", [record(name, 0, "a"), record(name, 2, "a")])
    write_records(tmp_path / "src" / "u2.jsonl", [record(name, 1, "b"), record(name, 3, "c")])
    write_records(tmp_path / "src" / "b.jsonl", [record(run.BLIND, 0, "a"), record(run.BLIND, 2, "a")])
    return tmp_path / "results" / "new"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def test_seed_copies_every_source_of_a_row_and_leaves_the_sources_alone(seed_world, tmp_path):
    before = (tmp_path / "src" / "u.jsonl").read_text()
    seed_results.seed_model("phi", str(seed_world))
    assert [r["qid"] for r in read_jsonl(seed_world / "phi" / "uniform.jsonl")] == [0, 2, 1, 3]
    assert [r["qid"] for r in read_jsonl(seed_world / "phi" / "blind.jsonl")] == [0, 2]
    assert (tmp_path / "src" / "u.jsonl").read_text() == before
    assert not list((seed_world / "phi").glob("*.tmp"))


def test_a_rerun_keeps_what_was_scored_since(seed_world):
    seed_results.seed_model("phi", str(seed_world))
    with open(seed_world / "phi" / "uniform.jsonl", "a") as f:
        f.write(json.dumps(dict(condition="new", qid=9)) + "\n")
    seed_results.seed_model("phi", str(seed_world))
    assert len(read_jsonl(seed_world / "phi" / "uniform.jsonl")) == 5


def test_seed_stops_on_an_error_record_and_writes_nothing_for_that_row(seed_world, tmp_path):
    name = seed_results.condition_names("phi", "uniform")[0]
    write_records(tmp_path / "src" / "u2.jsonl", [record(name, 1, "b", error="oom"), record(name, 3, "c")])
    with pytest.raises(SystemExit, match="can't be reused: 1 x error record"):
        seed_results.seed_model("phi", str(seed_world))
    assert not (seed_world / "phi" / "uniform.jsonl").exists()


def test_seed_stops_on_a_missing_source(seed_world, tmp_path):
    os.remove(tmp_path / "src" / "b.jsonl")
    with pytest.raises(SystemExit, match="no results to reuse"):
        seed_results.seed_model("phi", str(seed_world))


@pytest.mark.parametrize("folder", ["hpc_all", "hpc_budget", "nymeria_smolvlm"])
def test_seed_refuses_to_write_inside_a_read_only_results_folder(seed_world, tmp_path, folder):
    for out in (tmp_path / "results" / folder, tmp_path / "results" / folder / "inner"):
        with pytest.raises(SystemExit, match="only read"):
            seed_results.seed_model("phi", str(out))
    assert not (tmp_path / "results" / folder).exists()


def test_sources_never_include_no_gate_and_only_smolvlm_reuses_budget_rows(monkeypatch):
    monkeypatch.setenv("GV_ROOT", "/g")
    for model in ("qwen2.5", "qwen3", "phi"):
        assert [(row, group) for row, _, group in seed_results.sources(model)] == [("uniform", "aea12"), ("blind", "aea12")]
    got = {(os.path.dirname(p).split("/")[-2], row, group) for row, p, group in seed_results.sources("smolvlm")}
    assert got == {("hpc_all", "uniform", "aea12"), ("hpc_all", "blind", "aea12"), ("hpc_budget", "budget", "aea12"),
                   ("nymeria_smolvlm", "uniform", "nymeria"), ("nymeria_smolvlm", "blind", "nymeria"), ("nymeria_smolvlm", "budget", "nymeria")}


def test_reused_records_with_other_frames_are_found(world, monkeypatch, tmp_path):
    cfg = joined_world(world)
    data = common.load_data(cfg)
    budget = prepare.smolvlm_budget_frames(cfg, data)
    name = seed_results.condition_names("phi", "uniform")[0]
    good, wrong = str(tmp_path / "good.jsonl"), str(tmp_path / "wrong.jsonl")
    write_records(good, [record(name, 0, "2023_y", n_frames=96, n_image_tokens=96 * 757)])
    write_records(wrong, [record(name, 0, "2023_y", n_frames=95, n_image_tokens=95 * 757), record("other", 1, "2023_y"),
                           dict(condition=name, qid=2, video="2023_y", error="oom", n_frames=96)])
    monkeypatch.setattr(seed_results, "MODELS", ("phi",))
    monkeypatch.setattr(seed_results, "sources", lambda model: [("uniform", good, "x")])
    assert prepare.reused_differences(cfg, data, budget) == []
    monkeypatch.setattr(seed_results, "sources", lambda model: [("uniform", wrong, "x")])
    assert prepare.reused_differences(cfg, data, budget) == [f"{wrong}: 3 of 3 records"]


# ---------------------------------------------------------------------------- tokens per frame
def test_qwen3_has_qwen25s_token_count():
    for gaze in ((0.5, 0.5), (0.1, 0.9), (0.73, 0.21)):
        assert len(q3.foveated_cells(*gaze, a=SETTING["a"], R_deg=SETTING["R"])) == measure_tokens.qwen_tokens(gaze, SETTING)


def test_token_counts_at_the_image_centre_and_their_range():
    assert measure_tokens.qwen_tokens((0.5, 0.5), SETTING) == 40
    assert measure_tokens.phi_parts((0.5, 0.5), SETTING) == (84, 12)
    assert measure_tokens.phi_tokens((0.5, 0.5), SETTING) == 253                   # 84 crop + 12 breaks + separator + 156 global
    grid = [(u / 8, v / 8) for u in range(9) for v in range(9)]
    assert all(40 <= measure_tokens.qwen_tokens(g, SETTING) <= 64 for g in grid)
    assert all(253 <= measure_tokens.phi_tokens(g, SETTING) <= 320 for g in grid)


def test_a_frame_with_no_gaze_counts_at_the_centre():
    assert measure_tokens.tokens_of("qwen2.5", dict(gaze=None), SETTING) == 40
    assert measure_tokens.tokens_of("phi", dict(gaze=None), SETTING) == 253
    assert measure_tokens.tokens_of("qwen3", dict(gaze=(0.1, 0.9)), SETTING) == measure_tokens.qwen_tokens((0.1, 0.9), SETTING)


def test_the_first_ring_setting_is_the_smallest_on_every_gaze_point():
    for g in ((0.5, 0.5), (0.2, 0.3), (0.9, 0.9)):
        counts = [measure_tokens.qwen_tokens(g, s) for s in common.DEFAULTS["ours"]["rings"]]
        assert counts[0] == min(counts)
        counts = [measure_tokens.phi_tokens(g, s) for s in common.DEFAULTS["ours"]["rings"]]
        assert counts[0] == min(counts)


def test_the_count_is_checked_against_each_recorded_question():
    frames = dict(v=[dict(gaze=None), dict(gaze=(0.5, 0.5))])                      # both count at the centre
    data = dict(gate=frames)
    want = dict(qwen=80, phi=506)
    ok = lambda model: [dict(video="v", n_image_tokens=want["phi" if model == "phi" else "qwen"])] * 2
    assert measure_tokens.record_differences(data, SETTING, ok) == {"qwen2.5": (2, 0), "qwen3": (2, 0), "phi": (2, 0)}
    off = lambda model: [dict(video="v", n_image_tokens=81)]
    assert measure_tokens.record_differences(data, SETTING, off)["qwen3"] == (1, 1)


@pytest.mark.parametrize("mean,value", [(48.8, 49), (280.4, 281), (17.0, 17)])
def test_the_proposed_value_is_the_mean_rounded_up(mean, value):
    assert measure_tokens.proposed_value(mean) == value


def test_limits_table_at_the_values_seen_on_the_12_videos():
    cfg = common._merge(common.DEFAULTS, {})
    t = measure_tokens.limits_table({"qwen2.5": 48.8, "phi": 280.4, "smolvlm": 17.0}, cfg)
    limits = lambda m: t[t.model == m]["limit"].tolist()
    assert limits("qwen2.5") == [25, 50, 100] and limits("phi") == [12, 25, 51] and limits("smolvlm") == [15, 30, 60]
    assert t[t.model == "qwen2.5"]["expected_spend_pct"].iloc[0] == pytest.approx(100 * 25 * 48.8 / (96 * 256))


def test_frame_limits_on_a_short_video_stay_within_it():
    assert br.frame_limit(0.2, 59 * 256, 49, 59) == 59
    assert br.frame_limit(0.2, 59 * 757, 281, 59) == 31


# ---------------------------------------------------------------------------- the report
def test_groups_split_by_name_and_questions_add_up():
    g = report.video_groups(["loc_a", "2023_x", "loc_b"])
    assert g == dict(all={"loc_a", "2023_x", "loc_b"}, aea={"loc_a", "loc_b"}, nymeria={"2023_x"})
    assert report.expected_questions(dict(loc_a=3, loc_b=2, **{"2023_x": 1}), g) == dict(all=6, aea=5, nymeria=1)


def full_table(n=7):
    return pd.DataFrame([dict(group="all", model=m, condition=f"c{i}", n=n) for m in seed_results.MODELS for i in range(5)])


def test_n_problems_each_kind():
    assert report.n_problems(full_table(), dict(all=7, aea=0)) == []
    assert report.n_problems(full_table(6), dict(all=7))[0] == "all qwen2.5 c0: n = 6, expected 7"
    gone = full_table()
    assert report.n_problems(gone[gone.model != "phi"], dict(all=7)) == ["all phi: 0 conditions, expected 5"]
    assert report.n_problems(gone[gone.condition != "c4"], dict(all=7))[0] == "all qwen2.5: 4 conditions, expected 5"


def result_record(cond, qid, video, correct, tokens, frames):
    return dict(condition=cond, qid=qid, video=video, answer="A", pred="A", correct=correct, n_frames=frames, n_image_tokens=tokens,
                n_gaze_missing=0, sec=0.1, error=None)


def test_the_report_table_carries_the_note_and_each_group(tmp_path, monkeypatch):
    monkeypatch.setenv("GV_ROOT", "/g")
    monkeypatch.setenv("EGO_ROOT", "/e")
    u, b = "uniform K=96 (stock input)", "budget rule 5% (D+P gate, fovea R=70 a=160)"
    write_records(tmp_path / "phi" / "uniform.jsonl", [result_record(u, 0, "loc_a", 1, 757, 1), result_record(u, 1, "2023_x", 0, 757, 1)])
    write_records(tmp_path / "phi" / "budget.jsonl", [result_record(b, 0, "loc_a", 1, 281, 1), result_record(b, 1, "2023_x", 1, 281, 1)])
    cfg = common.load_config(prepare.CONFIG)
    cfg["paths"]["out_dir"] = str(tmp_path)
    t = report.build_table(cfg, report.video_groups(["loc_a", "2023_x"]))
    assert sorted(t.group.unique()) == ["aea", "all", "nymeria"] and (t.note == report.NOTE).all()
    assert "can't show whether the gate itself helps" in report.NOTE
    row = t[(t.group == "nymeria") & (t.condition == b)].iloc[0]
    assert row.n == 1 and row.accuracy == 100 and row.budget_pct == pytest.approx(100 * 281 / 757) and row.gaze_missing_pct == 0
    assert t[(t.group == "all") & (t.condition == u)].iloc[0].accuracy == 50


def test_a_group_with_no_results_is_left_out(tmp_path, monkeypatch):
    monkeypatch.setenv("GV_ROOT", "/g")
    monkeypatch.setenv("EGO_ROOT", "/e")
    write_records(tmp_path / "phi" / "uniform.jsonl", [result_record("u", 0, "loc_a", 1, 757, 1)])
    cfg = common.load_config(prepare.CONFIG)
    cfg["paths"]["out_dir"] = str(tmp_path)
    assert sorted(report.build_table(cfg, report.video_groups(["loc_a", "2023_x"])).group.unique()) == ["aea", "all"]


def test_error_records_are_counted(tmp_path):
    write_records(tmp_path / "phi" / "uniform.jsonl", [dict(error="oom"), dict(error=None)])
    assert report.error_records(str(tmp_path)) == {("phi", "uniform"): 1}


# ---------------------------------------------------------------------------- GPU use
def test_read_use_takes_the_percent_and_skips_what_isnt_a_number(tmp_path):
    p = tmp_path / "gpu.csv"
    p.write_text("2026/11/05 12:00:00.000, 45 %, 1234 MiB, 100.0 W\n2026/11/05 12:01:00.000, [N/A], 1 MiB, 1 W\nnothing\n"
                 "2026/11/05 12:02:00.000, 7 %, 1 MiB, 1 W\n")
    assert gpu_summary.read_use(str(p)) == [45, 7]


def test_lowest_mean_finds_the_idle_stretch():
    assert gpu_summary.lowest_mean([50] * 29) is None
    assert gpu_summary.lowest_mean([50] * 30) == 50
    assert gpu_summary.lowest_mean([90] * 40 + [0] * 30 + [90] * 40) == 0
    assert gpu_summary.lowest_mean([10, 0, 0, 10], window=2) == 0


def test_summary_line_says_when_a_reading_is_too_short(tmp_path):
    p = tmp_path / "gpu_x.csv"
    p.write_text("t, 40 %, 1 MiB, 1 W\nt, 60 %, 1 MiB, 1 W\n")
    assert gpu_summary.summary_line(str(p)) == "gpu_x.csv: 2 samples, mean 50.0%, under 30 samples, so no 30-minute reading"
    assert "no such log" in gpu_summary.summary_line(str(tmp_path / "missing.csv"))


# ---------------------------------------------------------------------------- comparing runs
def test_report_mode_fails_only_on_frames_or_tokens():
    changed = dict(records=30, missing=2, pred_differs=1, tokens_differ=0, max_prob_diff=0.01)
    assert compare_records.problems(changed) == 3 and compare_records.problems(changed, report=True) == 0
    assert compare_records.problems(dict(changed, tokens_differ=1), report=True) == 1
    assert compare_records.problems(dict(changed, records=0), report=True) == 1


# ---------------------------------------------------------------------------- frame loading threads
@pytest.fixture
def frames(tmp_path):
    rng, out = np.random.default_rng(0), []
    for i in range(5):
        path = tmp_path / f"{i}.png"
        Image.fromarray(rng.integers(0, 255, (56, 56, 3), dtype=np.uint8)).save(path)
        out.append(dict(key=("v", i), path=str(path), t=float(i), gaze=(0.5, 0.5)))
    return out


def test_qwen_threads_give_every_frame_exactly_what_loading_one_at_a_time_gives(frames):
    ready = {(frames[0]["key"], 448): "already loaded"}
    qf.load_resized([(fr, 448) for fr in frames], ready, workers=3)
    assert ready[(frames[0]["key"], 448)] == "already loaded" and len(ready) == 5
    for fr in frames[1:]:
        assert torch.equal(ready[(fr["key"], 448)], qf.resize01(qf.load_frame01(fr["path"], qf.IMG), 448))


def test_phi_threads_give_every_frame_exactly_what_loading_one_at_a_time_gives(frames):
    pytest.importorskip("torchvision")
    ready = {frames[0]["key"]: "already loaded"}
    ph.load_views(frames, ready, workers=3)
    assert ready[frames[0]["key"]] == "already loaded" and len(ready) == 5
    for fr in frames[1:]:
        hd, glb = ph.views(ph.load_pil(fr["path"]))
        assert torch.equal(ready[fr["key"]][0], hd) and torch.equal(ready[fr["key"]][1], glb)


def test_adapters_pass_the_workers_on(monkeypatch):
    got = []
    run_questions = lambda *a, **k: got.append(k["prefetch_workers"])
    monkeypatch.setattr(qf, "run_questions", run_questions)
    monkeypatch.setattr(q3, "run_questions", run_questions)
    monkeypatch.setattr(ph, "run_questions", run_questions)
    for cls, workers in ((models.Qwen25, 8), (models.Qwen3, 4), (models.Phi, 2)):
        ad = cls(cfg=dict(run=dict(prefetch_workers=workers)))
        ad.model = ad.proc = ad.letter_ids = ad.img = ad.llm = ad.tok = None
        ad.score([], {}, "out")
    assert got == [8, 4, 2]
    assert models.Qwen25(cfg={}).prefetch_workers() == 0


def test_a_model_without_threads_is_named_rather_than_ignored():
    assert run.models_without_threads(["internvl", "gemma", "phi", "qwen3", "qwen2.5", "smolvlm"], 8) == ["internvl", "gemma"]
    assert run.models_without_threads(["internvl", "gemma"], 0) == []


def items_on(frames):
    """2 questions on each of 2 videos, with frames keyed per video."""
    out = []
    for video in ("a", "b"):
        frs = [dict(fr, key=(video, fr["key"][1])) for fr in frames[:3]]
        out += [dict(qid=q, video=video, question="q", options=list("abcde"), answer="A", frames=frs) for q in (0, 1)]
    return [dict(it, qid=i) for i, it in enumerate(out)]


def probs_from(feats):
    v = float(torch.cat(list(feats)).float().mean())
    return np.array([v, 1 - v, 0.0, 0.0, 0.0])


def without_time(path):
    return [{k: x for k, x in r.items() if k != "sec"} for r in read_jsonl(path)]


def counted(monkeypatch, module, name):
    """Counts calls of a module's function, whichever thread makes them."""
    calls, real = [], getattr(module, name)

    def wrapper(*a, **k):
        calls.append(a[0])
        return real(*a, **k)

    monkeypatch.setattr(module, name, wrapper)
    return calls


PROC = SimpleNamespace(image_processor=SimpleNamespace(image_mean=[0.5] * 3, image_std=[0.5] * 3))


def fake_qwen25(monkeypatch):
    monkeypatch.setattr(qf, "visual_of", lambda model: None)
    monkeypatch.setattr(qf, "check_encoder", lambda vis: None)
    monkeypatch.setattr(qf, "encode_cells", lambda vis, x, cells, size, mean, std: torch.full((len(cells), 1), float(x.mean())))
    monkeypatch.setattr(qf, "build_inputs", lambda model, proc, msgs, feats, cells, sizes: (dict(feats=feats), torch.zeros((1, 9))))
    monkeypatch.setattr(qf, "letter_probs", lambda model, inputs, letter_ids: probs_from(inputs["feats"]))


def test_qwen25_scores_the_same_with_threads_and_loads_each_frame_once_per_video(frames, tmp_path, monkeypatch):
    fake_qwen25(monkeypatch)
    calls = counted(monkeypatch, qf, "load_frame01")
    conds = dict(stock=("uniform", 448), fovea=("fovea", dict(a=160, R=70, img=448, gaze="gaze")))
    build = lambda *a: []
    qf.run_questions(None, PROC, [0], items_on(frames), conds, str(tmp_path / "off.jsonl"), build)
    off = len(calls)
    calls.clear()
    qf.run_questions(None, PROC, [0], items_on(frames), conds, str(tmp_path / "on.jsonl"), build, prefetch_workers=3)
    assert off == 12 and len(calls) == 6                                           # 3 frames x 2 videos, x 2 conditions without threads
    assert without_time(tmp_path / "on.jsonl") == without_time(tmp_path / "off.jsonl") and len(read_jsonl(tmp_path / "on.jsonl")) == 8


def test_qwen3_scores_the_same_with_threads(frames, tmp_path, monkeypatch):
    monkeypatch.setattr(q3, "parts", lambda model: (None, None, None))
    monkeypatch.setattr(q3, "check_encoder", lambda vis: None)
    monkeypatch.setattr(q3, "encode_cells", lambda vis, x, cells, side, mean, std: (torch.full((len(cells), 1), float(x.mean())), []))
    monkeypatch.setattr(q3, "build_inputs", lambda model, proc, msgs, feats, deep, cells, sides: (dict(feats=feats), torch.zeros((1, 9))))
    monkeypatch.setattr(q3, "letter_probs", lambda model, inputs, letter_ids: probs_from(inputs["feats"]))
    conds = dict(stock=("uniform", q3.IMG), fovea=("fovea", dict(a=160, R=70, gaze="gaze")))
    build = lambda *a: []
    q3.run_questions(None, PROC, [0], items_on(frames), conds, str(tmp_path / "off.jsonl"), build)
    q3.run_questions(None, PROC, [0], items_on(frames), conds, str(tmp_path / "on.jsonl"), build, prefetch_workers=3)
    assert without_time(tmp_path / "on.jsonl") == without_time(tmp_path / "off.jsonl") and len(read_jsonl(tmp_path / "on.jsonl")) == 8


def test_phi_scores_the_same_with_threads(frames, tmp_path, monkeypatch):
    pytest.importorskip("torchvision")
    monkeypatch.setattr(ph, "global_rows", lambda img, glb: glb)

    def frame_tokens(img, conditions, c, fr, hd01, glb_seq):
        return torch.full((3, 1), float(hd01.mean()) + float(glb_seq.mean())), None, 3, None

    monkeypatch.setattr(ph, "frame_tokens", frame_tokens)
    monkeypatch.setattr(ph, "build_inputs", lambda llm, tok, msgs, feats: (dict(feats=feats), torch.zeros((1, 9))))
    monkeypatch.setattr(ph, "letter_probs", lambda llm, inputs, letter_ids: probs_from(inputs["feats"]))
    conds = dict(stock=("stock", None), fovea=("fovea", dict(a=160, R=70, gaze="gaze")))
    build = lambda *a: []
    ph.run_questions(None, None, None, [0], items_on(frames), conds, str(tmp_path / "off.jsonl"), build)
    ph.run_questions(None, None, None, [0], items_on(frames), conds, str(tmp_path / "on.jsonl"), build, prefetch_workers=3)
    assert without_time(tmp_path / "on.jsonl") == without_time(tmp_path / "off.jsonl") and len(read_jsonl(tmp_path / "on.jsonl")) == 8


# ---------------------------------------------------------------------------- the jobs
@pytest.mark.parametrize("name", ["all350_prepare.SBATCH", "all350_vlm.SBATCH", "all350_summary.SBATCH"])
def test_jobs_write_only_to_their_own_results(name):
    with open(os.path.join(REPO, "accuracy_tables", "hpc", name)) as f:
        text = f.read()
    for folder in ("hpc_all", "hpc_budget", "nymeria_smolvlm"):
        assert f"OUT=$GV_ROOT/results/{folder}" not in text and f"--out-dir $GV_ROOT/results/{folder}" not in text
    assert "--no-parity" not in text and "nogate" not in text and "compare_budget" not in text


def test_gpu_jobs_ask_for_one_gpu_type_and_can_resume():
    with open(os.path.join(REPO, "accuracy_tables", "hpc", "all350_vlm.SBATCH")) as f:
        text = f.read()
    assert "--gres=gpu:b200:1" in text and "--dependency=singleton" in text and "--requeue" in text and "ROWS=uniform,blind,budget" in text

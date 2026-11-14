"""Runbook section 18, steps 1 to 8 under the user's D1 to D5 (2026-11-14), on a fake HD-EPIC: the plan (single instants, segments in time
order, D2's query frame, D5), the gate per segment, the frames and their meta, the loader (stamps, segment starts, the query frame in every
row), the manifest's data check, the shard picks and merge, and the jobs. Also that the shared code changes nothing when section 18's keys
are absent. CPU only, fake data, no HD-EPIC. Structures that can hold NaN are never compared with ==."""
import copy
import json
import os
import pickle
import re

import numpy as np
import pandas as pd
import pytest

from accuracy_tables import common
from accuracy_tables.hpc.hdepic import checks18, data, frames18, gate18, groups, manifest18, plan18, s18
from vlmeval import budget_rule as br
from vlmeval import qwen_mcq

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG18 = os.path.join(REPO, "accuracy_tables", "hpc", "config.hdepic18.yaml")
DEV0 = 7_000_000_000
A, B = "P01-20240101-100000", "P01-20240101-120000"          # B was recorded 2 hours after A


def job(name):
    with open(os.path.join(REPO, "accuracy_tables", "hpc", name)) as f:
        return f.read()


# ============================================================================ the shared code, with and without section 18's keys
def test_segment_starts_make_each_segment_its_own_clip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n1, n2 = rng.integers(2, 12, size=2)
        lab1, lab2 = rng.integers(0, 4, n1 - 1).tolist(), rng.integers(0, 4, n2 - 1).tolist()
        for x in range(4):                                                    # the window before the 2nd segment: never read
            got, _ = br.send_reasons(lab1 + [x] + lab2, starts=[n1])
            assert got[:n1] == br.send_reasons(lab1)[0] and got[n1:] == br.send_reasons(lab2)[0]
        lab = rng.integers(0, 4, 20).tolist()
        assert br.keep_frames(lab, 5) == br.keep_frames(lab, 5, starts=()) and br.send_reasons(lab) == br.send_reasons(lab, ())


def test_a_moving_stretch_never_settles_across_a_gap():
    T, P, R, S = 0, 1, 2, 3
    assert br.send_reasons([T, T, T, S])[0][4] == br.SETTLE                    # one clip: the stretch settles at frame 4
    assert br.send_reasons([T, T, T, S], starts=[3])[0] == [br.FRAME0, None, None, br.FRAME0, None]   # frame 3 begins segment 2
    assert br.send_reasons([P, P, P, P])[0] == [br.FRAME0, br.PURSUIT, None, None, None]
    assert br.send_reasons([P, P, P, P], starts=[2])[0] == [br.FRAME0, br.PURSUIT, br.FRAME0, br.PURSUIT, None]   # a fresh run


def test_the_query_frame_takes_the_nearest_place():
    assert common.include_forced([0, 10, 20]) == [0, 10, 20] and common.include_forced([0, 10, 20], ()) == [0, 10, 20]
    assert common.include_forced([0, 10, 20], [12]) == [0, 12, 20]
    assert common.include_forced([0, 10, 20], [15]) == [0, 15, 20]            # a tie: the earlier one goes
    assert common.include_forced([3], [7]) == [7] and common.include_forced([0, 7, 9], [7]) == [0, 7, 9]


def test_a_stamp_is_a_float_that_carries_its_label():
    t = qwen_mcq.Stamp(62.4, s18.stamp(62.4, 1))
    assert qwen_mcq.fmt_time(62.4) == "1:02" and qwen_mcq.fmt_time(t) == "<TIME 00:01:02.400 video 1>"
    assert t + 1 == pytest.approx(63.4) and float(t) == pytest.approx(62.4) and json.loads(json.dumps(t)) == pytest.approx(62.4)
    for back in (copy.deepcopy(t), pickle.loads(pickle.dumps(t))):
        assert back.stamp == t.stamp and back == pytest.approx(62.4)
    msgs = qwen_mcq.build_messages("At <TIME 00:01:00.000 video 1> in <BBOX 1 2 3 4>?", list("abcde"), [("f.jpg", t), ("g.jpg", 70.0)], 0)
    texts = [c["text"] for c in msgs[0]["content"] if c["type"] == "text"]
    assert texts[1] == "Frame at <TIME 00:01:02.400 video 1>:" and texts[2] == "Frame at 1:10:"
    assert "At <TIME 00:01:00.000 video 1> in <BBOX 1 2 3 4>?" in texts[-1]                 # the question's tags stay verbatim


def test_every_row_shares_one_prompt_builder_and_one_label():
    from accuracy_tables import models
    assert models.build_messages is qwen_mcq.build_messages
    defs, labels = [], []
    for folder in ("vlmeval", "accuracy_tables"):
        for name in sorted(os.listdir(os.path.join(REPO, folder))):
            if name.endswith(".py"):
                text = open(os.path.join(REPO, folder, name)).read()
                defs += [name] * len(re.findall(r"def build_messages\(", text))
                labels += [name] * text.count("Frame at")
                for line in text.splitlines():
                    if "build_messages(" in line and "fr[" in line:
                        assert '(fr["path"], fr["t"])' in line, (name, line)    # every row passes each frame's own time
    assert defs == ["qwen_mcq.py"] and set(labels) == {"qwen_mcq.py"}


def test_stamps_numbers_and_clocks():
    assert s18.stamp(3723.0456, 2) == "<TIME 01:02:03.046 video 2>" and s18.STAMP.match(s18.stamp(0, 1))
    assert s18.input_number("video 2", 1) == 2 and s18.input_number("clip", 3) == 3
    assert s18.video_clock(A) == "20240101100000" and s18.video_clock("P01-x") is None


# ============================================================================ a fake HD-EPIC
def world_videos(base, videos):
    """{id: gaze Hz or None}: 60 s at 30 fps in device time, the gaze from 2 s in (CSV and projected npz), an IMU."""
    for v, hz in videos.items():
        d = base / "hdepic" / "videos" / v
        os.makedirs(d / "mps" / "eye_gaze")
        mp4 = (np.arange(1800) * 1e9 / 30).astype(np.int64)
        pd.DataFrame(dict(mp4_time_ns=mp4, relative_vrs_device_time_ns=mp4, vrs_device_time_ns=mp4 + DEV0)).to_csv(
            d / f"{v}_mp4_to_vrs_time_ns.csv", index=False)
        rng = np.random.default_rng(len(v) + int(hz or 0))
        t = np.arange(DEV0, DEV0 + 60 * 10 ** 9, 1_250_000)
        np.savez(d / "imu.npz", right_t_ns=t, right_accel=np.tile([0, -9.81, 0], (len(t), 1)) + rng.normal(0, 0.1, (len(t), 3)),
                 right_gyro=rng.normal(0, 0.05, (len(t), 3)))
        if hz:
            g = np.arange(DEV0 // 1000 + 2_000_000, DEV0 // 1000 + 59_000_000, int(1e6 / hz)).astype(np.int64)
            pd.DataFrame(dict(tracking_timestamp_us=g)).to_csv(d / "mps" / "eye_gaze" / "general_eye_gaze.csv", index=False)
            yaw, pit = rng.normal(0, 0.1, len(g)), rng.normal(0, 0.1, len(g))
            out = base / "gv" / "data" / "hdepic" / "gaze" / f"{v}.npz"
            os.makedirs(out.parent, exist_ok=True)
            np.savez(out, t_us=g, yaw=yaw, pitch=pit, x=np.clip(0.5 + yaw, 0, 1), y=np.clip(0.5 + pit, 0, 1), oof=np.zeros(len(g), bool))


def span(v, s, e):
    return dict(id=v, start_time=s, end_time=e)


def write_world_questions(folder):
    os.makedirs(folder)
    q = lambda inputs, question="q": dict(inputs=inputs, question=question, choices=list("abcde"), correct_idx=1)
    files = {
        "fine_grained_action_recognition.json": [q({"video 1": span(A, "00:00:10.000", "00:00:12.000")}),
                                                 q({"video 1": span(data.NO_GAZE_VIDEOS[0], "00:00:10.000", "00:00:12.000")})],
        "3d_perception_fixture_location.json": [q({"video 1": dict(id=A, time="00:00:30.000")})],
        "object_motion_object_movement_counting.json": [q({"video 1": dict(id=B), "video 2": dict(id=B, time="00:00:20.000")})],
        "object_motion_stationary_object_localization.json": [q({"video 1": dict(id=B)}, "Where at <TIME 00:00:40.500 video 1>?")],
        "object_motion_object_movement_itinerary.json": [q({"video 1": dict(id=B)})],
        "recipe_multi_step_localization.json": [q({"video 2": span(B, "00:00:40.000", "00:00:50.000"),
                                                   "video 1": span(A, "00:00:05.000", "00:00:15.000")})],
        "ingredient_ingredient_weight.json": [q({"video 1": span(A, "00:00:01.000", "00:00:03.000")},
                                                "How much at <TIME 00:00:02.000 video 1> in <BBOX 1 2 3 4>?")],
        "nutrition_nutrition_change.json": [q({"video 1": span(A, "00:00:20.000", "00:00:21.000"),
                                               "video 2": span(A, "00:00:21.500", "00:00:23.000")})],
    }
    for name, qs in files.items():
        with open(os.path.join(folder, name), "w") as f:
            json.dump({f"{name[:-5]}_{i}": x for i, x in enumerate(qs)}, f)


def world_plan():
    from accuracy_tables.hpc.hdepic import explore18
    loaded, _ = groups.load_groups()
    fg = [q for qs in loaded[groups.SAMPLE_GROUP].values() for q in qs]
    sample, quota = groups.stratified_sample(fg)
    groups.write_or_check_sample(sample, quota, len(fg))
    scope = explore18.in_scope(loaded, sample)
    cache, info = {}, {}
    for v in sorted({s["video"] for q in scope for s in q["segments"]}):
        mp4_ns, vrs_ns = data.read_time_csv(os.path.join(data.video_dir(v), f"{v}_mp4_to_vrs_time_ns.csv"))
        cache[v] = dict(mp4_ns=mp4_ns, vrs_ns=vrs_ns)
        info[v] = dict(fps=30.0, mp4_frames=1800, width=1408, height=1408, gaze_csv=data.gaze_csv(v))
        if info[v]["gaze_csv"]:
            cache[v]["gaze_us"] = pd.read_csv(info[v]["gaze_csv"])["tracking_timestamp_us"].to_numpy(np.float64)
    return plan18.make_plan(scope, cache, info), scope


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("HDEPIC_ROOT", str(tmp_path / "hdepic"))
    monkeypatch.setenv("GV_ROOT", str(tmp_path / "gv"))
    world_videos(tmp_path, {A: 10.0, B: 60.0, data.NO_GAZE_VIDEOS[0]: None})
    write_world_questions(str(tmp_path / "hdepic" / "annotations" / "vqa-benchmark"))
    p, scope = world_plan()
    os.makedirs(s18.root(), exist_ok=True)
    with open(s18.plan_json(), "w") as f:
        json.dump(p, f)
    return p


def by_type(p):
    return {q["type"]: q for q in p["questions"]}


def test_the_plan_follows_d1_d2_and_d5(world):
    p, q = world, by_type(world)
    assert [e["reason"] for e in p["excluded"]] == ["D5: on a video without gaze"] and p["excluded"][0]["group"] == "fine_grained"
    assert [x["pseudo"] for x in p["questions"]] == [s18.pseudo_name(i) for i in range(len(p["questions"]))]
    assert q["action_recognition"]["K"] == 61 and q["action_recognition"]["starts"] == []
    assert q["fixture_location"]["K"] == 61 and q["fixture_location"]["mp4_s"][0] == pytest.approx(29.0)     # 1 s either side
    assert q["object_movement_itinerary"]["K"] == 64 and not q["object_movement_itinerary"]["exception"]
    r = q["multi_step_localization"]
    assert r["videos"] == [A, B] and r["starts"] == [32] and r["K"] == 64                     # time order: A was recorded first
    assert r["video_n"] == [1] * 32 + [2] * 32 and r["video_of"] == [0] * 32 + [1] * 32
    n = q["nutrition_change"]
    assert n["starts"] == [26] and n["K"] == 64 and n["windows_into_previous_segment"] == 1          # a 0.5 s gap
    assert q["ingredient_weight"]["frames_outside_gaze"] == 30                                        # 1.0 to 1.97 s: before the gaze


@pytest.mark.parametrize("qtype,t", [("object_movement_counting", 20.0), ("stationary_object_localization", 40.5)])
def test_the_query_frame_is_a_candidate_and_nothing_repeats(world, qtype, t):
    q = by_type(world)[qtype]
    assert q["exception"] and q["K"] == 64 and q["starts"] == [] and len(q["forced"]) == 1
    assert q["mp4_s"][q["forced"][0]] == pytest.approx(t) and len(set(q["mp4_frames"])) == 64
    assert q["mp4_frames"] == sorted(q["mp4_frames"]) and q["mp4_frames"][0] == 0 and q["mp4_frames"][-1] == 1799


def test_d5s_count_is_checked_against_the_users():
    ok = dict(excluded=[dict(group=g, reason="D5: on a video without gaze") for g, n in s18.D5_EXPECTED.items() for _ in range(n)])
    assert plan18.d5_problems(ok) == [] and len(ok["excluded"]) == 67
    assert plan18.d5_problems(dict(excluded=ok["excluded"][1:])) != []


def fake_scores(monkeypatch):
    from accuracy_tables.hpc.newvideos import score_gate

    def scores(test, cks):
        df = pd.read_csv(test)
        return df.assign(pred=df["idx"] % 4)[["sequence", "feat_frame_1", "feat_frame_2", "quad", "pred"]]
    monkeypatch.setattr(score_gate, "load_models", lambda d: None)
    monkeypatch.setattr(score_gate, "score_ensemble", scores)


def test_the_gate_rows_chain_over_segments_and_videos(world, monkeypatch, capsys):
    gate18.rows_cmd(None)
    assert "ROWS WRITTEN: 8 questions" in capsys.readouterr().out
    for q in world["questions"]:
        rows = pd.read_csv(s18.rows_csv(q["pseudo"]))
        assert len(rows) == q["K"] - 1 and list(rows.feat_frame_1[1:]) == list(rows.feat_frame_2[:-1])
        assert rows.feat_frame_2.map(lambda p: int(re.search(r"feat_(\d+)", p).group(1))).tolist() == list(range(1, q["K"]))
    r = pd.read_csv(s18.rows_csv(by_type(world)["multi_step_localization"]["pseudo"]))
    assert (r.n_gaze_in_gap[:31] > 0).all() and (r.n_gaze_in_gap[31:] > 0).all() and (r.n_velocities >= 8).all()


@pytest.fixture
def loaded_world(world, monkeypatch):
    from PIL import Image
    gate18.rows_cmd(None)
    fake_scores(monkeypatch)
    gate18.score_cmd(None)
    for q in world["questions"]:
        os.makedirs(s18.frames_dir(q["pseudo"]))
        for j in range(q["K"]):
            Image.new("RGB", (8, 8), (j % 255, 0, 0)).save(frames18.frame_file(q["pseudo"], j))
        for k, v in enumerate(q["videos"]):
            mine = {str(j): t for j, (vi, t) in enumerate(zip(q["video_of"], q["mp4_s"])) if vi == k}
            with open(frames18.part_file(q["pseudo"], v), "w") as f:
                json.dump(mine, f)
    n, files, nbytes, unique, problems, worst = frames18.check(world)
    assert problems == [] and worst == 0.0 and files == sum(q["K"] for q in world["questions"])
    with open(s18.questions_json(), "w") as f:
        json.dump(manifest18.entries(world), f)
    cfg = common.load_config(CONFIG18)
    return world, cfg, common.load_data(cfg)


def test_the_loader_stamps_every_frame_and_forces_the_query_frame_into_every_row(loaded_world):
    from accuracy_tables import flops_calibrate as fc
    p, cfg, d = loaded_world
    found, missing = manifest18.data_problems(d, p, cfg)
    assert found == [] and sum(missing.values()) > 0                             # frames before the gaze starts: the centred fovea
    q = by_type(p)
    rc, om = q["multi_step_localization"]["pseudo"], q["object_movement_counting"]["pseudo"]
    assert d["segment_starts"][rc] == [32] and d["forced"] == {om: q["object_movement_counting"]["forced"],
                                                                q["stationary_object_localization"]["pseudo"]:
                                                                    q["stationary_object_localization"]["forced"]}
    f = q["object_movement_counting"]["forced"][0]
    for b in cfg["budget"]["fractions"]:
        for use_gate in (True, False):
            frames, why = common.budget_frames(cfg, d, "smolvlm", 81, b, use_gate)
            assert f in {fr["key"][1] for fr in frames[om]}
            plain = common.budget_frames(cfg, dict(d, forced={}), "smolvlm", 81, b, use_gate)[0]
            assert len(frames[om]) == len(plain[om])                              # the query frame takes a place: no row's count moves
    for k in (1, 3, 16):
        u = fc.with_uniform(d, k)["uniform"]
        assert f in {fr["key"][1] for fr in u[om]} and len(u[om]) == k
        assert [fr["key"] for fr in u[rc]] == [fr["key"] for fr in fc.with_uniform(dict(d, forced={}), k)["uniform"][rc]]
    fr = d["frame"](rc, 40)
    assert isinstance(fr["t"], qwen_mcq.Stamp) and s18.STAMP.match(fr["t"].stamp) and fr["t"].stamp.endswith(" video 2>")
    assert fr["t"] == pytest.approx(d["metas"][rc]["ts"][40]) and fr["t"].stamp == s18.stamp(d["metas"][rc]["ts"][40], 2)
    assert d["frame"](rc, 0)["t"].stamp == "<TIME 00:00:05.000 video 1>"


def test_the_gate_runs_per_segment_in_the_loader(loaded_world):
    p, cfg, d = loaded_world
    rc = by_type(p)["multi_step_localization"]["pseudo"]
    marks = {fr["key"][1] for fr in d["gate"][rc]}
    assert {0, 32} <= marks                                                       # each segment's first frame is a frame 0 of its own


def test_the_exception_comparison_runs(loaded_world, monkeypatch, capsys):
    capsys.readouterr()
    gate18.compare_cmd(None)
    out = capsys.readouterr().out
    assert "COMPARED: every row keeps its frame count" in out and out.count("budget ") == 3


def test_the_frames_check_finds_a_missing_part(loaded_world):
    p = loaded_world[0]
    q = by_type(p)["multi_step_localization"]
    os.remove(frames18.part_file(q["pseudo"], B))
    problems = frames18.check(p)[4]
    assert problems == [f"{q['pseudo']}: no part for {B}", f"{q['pseudo']}: 64 of 64 frames, decoder times for 32"]


# ============================================================================ picks, shards and the merge
def test_picks_take_the_groups_in_turn_and_shards_split_evenly():
    qs = [dict(video=f"hd18_{i:05d}", group=groups.GROUPS[i % 2]) for i in range(10)]
    assert checks18.pick_by_questions(qs, 4) == ["hd18_00000", "hd18_00001", "hd18_00002", "hd18_00003"]
    parts = [checks18.pick_shard(qs, s, 3) for s in range(3)]
    assert sorted(sum(parts, [])) == sorted(q["video"] for q in qs) and [len(x) for x in parts] == [4, 3, 3]
    with pytest.raises(SystemExit):
        checks18.pick_shard(qs, 3, 3)


def test_the_merge_carries_each_shards_gpu_ledger_and_overwrites_nothing(tmp_path):
    from accuracy_tables.hpc.all350 import flops_shards
    for s in range(3):
        d = tmp_path / f"shard{s}" / "smolvlm"
        os.makedirs(d)
        (d / "uniform.jsonl").write_text("".join(json.dumps(dict(condition="u", qid=10 * s + i)) + "\n" for i in range(s + 1)))
        (d / "gpu_types.txt").write_text("NVIDIA B200\n")
        (d / "gpu_ledger.jsonl").write_text(json.dumps(dict(row="uniform", condition="u", first_line=0, gpu="NVIDIA B200")) + "\n")
    got, state = checks18.merge(str(tmp_path), ["uniform", "blind"])
    assert got == {"uniform": 6, "blind": 0} and state["uniform.jsonl"] == "written" and state["gpu_ledger.jsonl"] == "written"
    led = [json.loads(x) for x in open(tmp_path / "smolvlm" / "gpu_ledger.jsonl")]
    assert [e["first_line"] for e in led] == [0, 1, 3] and [e["shard"] for e in led] == [0, 1, 2]     # moved by the earlier shards
    gpus, found = flops_shards.folder_gpus(str(tmp_path / "smolvlm"), ["uniform"], "run")
    assert dict(gpus) == {"NVIDIA B200": 6} and found == []                       # no "records before the first ledger line"
    got, state = checks18.merge(str(tmp_path), ["uniform", "blind"])             # a rerun: already there and identical
    assert set(state.values()) == {"unchanged"} and open(tmp_path / "smolvlm" / "gpu_types.txt").read() == "NVIDIA B200\n"
    with open(tmp_path / "smolvlm" / "uniform.jsonl", "a") as f:
        f.write(json.dumps(dict(condition="u", qid=99)) + "\n")
    with pytest.raises(SystemExit, match="nothing is overwritten"):
        checks18.merge(str(tmp_path), ["uniform"])
    (tmp_path / "shard2" / "smolvlm" / "uniform.jsonl").write_text(json.dumps(dict(condition="u", qid=0)) + "\n")
    with pytest.raises(SystemExit, match="in two shards"):
        checks18.merge(str(tmp_path), ["uniform"])


def test_the_records_wording_says_when_the_folder_is_merged(tmp_path, capsys):
    from accuracy_tables.hpc.all350 import smolvlm256_checks as sc
    d = tmp_path / "smolvlm"
    os.makedirs(d)
    (d / "uniform.jsonl").write_text(json.dumps(dict(condition="u", qid=0)) + "\n")
    (d / "gpu_ledger.jsonl").write_text(json.dumps(dict(row="uniform", condition="u", first_line=0, gpu="NVIDIA B200")) + "\n")
    (tmp_path / "plan.json").write_text(json.dumps(dict(lines=[dict(row="uniform", condition="u")])))
    args = ["records", "--folder", str(d), "--plan", str(tmp_path / "plan.json"), "--rows", "uniform", "--questions", "1"]
    sc.main(args + ["--merged-from", "3"])
    out = capsys.readouterr().out
    assert "(merged from 3 shards)" in out and "GPU per record: {'NVIDIA B200': 1}" in out
    sc.main(args)
    assert "(one job, so nothing to merge)" in capsys.readouterr().out


def test_the_tables_label_the_sample_and_the_separate_run():
    from accuracy_tables.hpc.hdepic import report18
    assert report18.label("fine_grained") == "fine_grained (2,000-question stratified sample of 21,000)" and report18.label("recipe") == "recipe"
    ref = pd.DataFrame([dict(group="both", row="uniform", budget_pct=100.0, accuracy=40.0, ci_low=37.0, ci_high=43.0, flops_pct=100.0)])
    ours = pd.DataFrame([dict(group=g, row="uniform", budget_pct=100.0, accuracy=30.0, ci_low=28.0, ci_high=32.0, flops_pct=100.0)
                         for g in ("all", "mean_of_groups")])
    vs = report18.vs_section17(ours, ref)
    assert len(vs) == 2 and (vs.note == report18.SEPARATE).all() and "separate run" in report18.SEPARATE
    assert set(s18.NOTES) == {"object_motion", "ingredient"}


# ============================================================================ the config and the jobs
def test_the_hdepic18_config_has_section_14s_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("GV_ROOT", str(tmp_path))
    monkeypatch.setenv("EGO_ROOT", str(tmp_path))
    hd, ego = common.load_config(CONFIG18), common.load_config(os.path.join(REPO, "accuracy_tables", "hpc", "config.all350.yaml"))
    assert hd["dataset"] == "hdepic18" and "dataset" not in ego
    for k in ("ours", "gazevlm", "streamingtom", "dycoke", "fastvid", "prunevid", "dtd", "avgpool", "gazevlm2", "run"):
        assert hd[k] == ego[k], k
    assert hd["budget"]["fractions"] == ego["budget"]["fractions"] and hd["budget"]["tokens_per_frame"]["smolvlm"] == 17
    assert all("/data/hdepic/s18/" in hd["paths"][k] for k in ("folds_dir", "oof_cache", "questions_json", "frames_local_dir"))


JOBS18 = ["hdepic18_prep.SBATCH", "hdepic18_frames.SBATCH", "hdepic18_checks.SBATCH", "hdepic18_flops.SBATCH", "hdepic18_summary.SBATCH"]
OURS = "$GV_ROOT/results/hdepic_all_smolvlm_flops"
READ_ONLY = re.compile(re.escape("--reference $GV_ROOT/results/hdepic_gaze_smolvlm_flops/summary_flops_by_group.csv") + r"(?=[\s\"']|$)")


def expand(value, assigned):
    for _ in range(10):
        new = re.sub(r"\$\{?([A-Z_][A-Z_0-9]*)\}?", lambda m: assigned.get(m.group(1), m.group(0)), value)
        if new == value:
            break
        value = new
    return value


@pytest.mark.parametrize("name", JOBS18)
def test_the_section_18_jobs_keep_the_rules(name):
    text = job(name)
    assert "--no-parity" not in text and "SBATCH_ACCOUNT" not in text and "data/folds" not in text and "HF_HOME=" not in text
    assigned = dict(re.findall(r"(?m)^\s*([A-Z_][A-Z_0-9]*)=(\S+)", text))
    ours = lambda p: p == OURS or p.startswith(OURS + "/")
    for line in text.splitlines():
        rest = READ_ONLY.sub("", line)
        for out in re.findall(r"--out-dir\s+(\S+)", rest):
            assert ours(expand(out.replace("\\$", "$"), assigned)), line
        if "--out-dir" in rest:
            for p in re.findall(r"\$GV_ROOT/results/[^\s\"']*", rest):
                assert ours(p), line
    for folder in ("qwen3_350", "qwen25_350", "smolvlm256_flops", "hpc_all", "nymeria_smolvlm"):
        assert folder not in text, folder
    if "gres" in text:
        assert "#SBATCH --gres=gpu:b200:1" in text and "GPU=B200" in text and "--expect $GPU" in text and "--requeue" in text
    else:
        assert name in ("hdepic18_prep.SBATCH", "hdepic18_frames.SBATCH", "hdepic18_summary.SBATCH")


def test_the_prep_and_flops_jobs_run_the_section_18_steps():
    prep = job("hdepic18_prep.SBATCH")
    for step in ("$H.plan18", 'run_in_aria "python -u -m $H.project --plan $PLAN"', "$H.sheets18", "$H.gate18 imu", "$H.gate resample-check",
                 "$H.gate18 rows", "$H.gate18 score", "$H.gate18 stats", "$H.gate18 compare", "$H.frames18 check", "$H.manifest18 --config"):
        assert step in prep, step
    fl = job("hdepic18_flops.SBATCH")
    assert "CONFIG=accuracy_tables/hpc/config.hdepic18.yaml" in fl and "pick --every 25" in fl and "WANT=hdepic18-flops-full-$SHARD" in fl
    assert "pick --shard" in fl and "DEST=$OUT/shard$SHARD" in fl and "chmod a-w" in fl
    assert "checks18 merge" in job("hdepic18_summary.SBATCH") and "report18" in job("hdepic18_summary.SBATCH")


def test_section_18_has_every_step_and_its_stop_lines():
    with open(os.path.join(REPO, "HPC_ACCURACY_RUNBOOK.md")) as f:
        section = f.read().split("\n## 18.")[1]
    for text in ("PLAN WRITTEN", "D5 COUNT DIFFERS", "SHEETS WRITTEN", "IMU MATCHES AEA", "REBUILD MATCHES", "VOR CHECK", "COMPARED",
                 "FRAMES MATCH PLAN", "STAMP TOKENS", "data check OK", "SAME ANSWERS: 0 of 100", "FOVEATED TOKENS 17", "PROJECTED full run",
                 "MERGED", "EVERY n IS COMPLETE", "<TIME hh:mm:ss.sss video n>", "deviation", "the user looks at them", "15%", "6 hours",
                 "summary_vs_section17.csv", "hdepic18_flops.SBATCH full 0"):
        assert text in section, text

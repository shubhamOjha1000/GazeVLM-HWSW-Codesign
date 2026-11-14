"""Runbook section 18: step 0's fovea diagnosis (the projection and motion checks' geometry, the answer analysis, the model's view of a
foveated frame) and the explore step for HD-EPIC's 6 non-gaze groups (the files, the inputs and tags, the seed-0 sample, K over segments,
the box hypotheses, a fake HD-EPIC end to end), and the jobs. CPU only, fake data, no HD-EPIC. Structures that can hold NaN are never
compared with ==."""
import collections
import json
import os

import numpy as np
import pandas as pd
import pytest

from accuracy_tables.hpc.hdepic import data, groups
from accuracy_tables.hpc.hdepic import diagnose as dg
from accuracy_tables.hpc.hdepic import diagnose_aria as da

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def job(name):
    with open(os.path.join(REPO, "accuracy_tables", "hpc", name)) as f:
        return f.read()


# ============================================================================ step 0a: geometry of the projection checks
def pinhole(f=0.5):
    """A camera of focal f (units of the image side), not turned: camera point -> (x, y) in [0, 1], x right, y down."""
    return lambda p: (0.5 + f * float(np.ravel(p)[0]) / float(np.ravel(p)[2]), 0.5 + f * float(np.ravel(p)[1]) / float(np.ravel(p)[2]))


def test_up_is_measured_from_the_images_up():
    proj = pinhole()
    assert da.up_angle_deg(proj, [0, -1, 0]) == pytest.approx(0, abs=1e-6)
    assert da.up_angle_deg(proj, [1, 0, 0]) == pytest.approx(90)
    assert abs(da.up_angle_deg(proj, [0, 1, 0])) == pytest.approx(180)
    turned = lambda p: (1 - proj(p)[1], proj(p)[0])                       # our 90-degree turn: the native left edge becomes the top
    assert da.up_angle_deg(turned, [-1, 0, 0]) == pytest.approx(0, abs=1e-6)
    assert abs(da.up_angle_deg(turned, [0, -1, 0])) == pytest.approx(90)


def test_the_motion_matrix_of_a_pinhole():
    M = da.motion_matrix(pinhole(0.5), np.eye(3))
    assert np.allclose(M, [[0, -0.5, 0], [0.5, 0, 0]], atol=1e-3)              # turning right moves the scene left
    R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])                          # an IMU turned 90 degrees about the optical axis
    assert np.allclose(da.motion_matrix(pinhole(0.5), R), M @ R, atol=1e-3)


def test_rotation_angles():
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    assert da.rotation_angle_deg(np.eye(3), np.eye(3)) == pytest.approx(0, abs=1e-6)
    assert da.rotation_angle_deg(np.eye(3), [[c, -s, 0], [s, c, 0], [0, 0, 1]]) == pytest.approx(90)


def fake_plan():
    qs = []
    for v, n in (("A", 3), ("B", 1), ("C", 2)):
        qs += [dict(qid=len(qs), video=v, type="gaze_estimation", pseudo=f"{v}{i}") for i in range(n)]
    qs.append(dict(qid=99, video="D", type="interaction_anticipation", pseudo="D0"))
    return dict(questions=qs, videos=["A", "B", "C", "D"])


def test_step_0a_spreads_its_questions_over_videos():
    p = fake_plan()
    assert [q["pseudo"] for q in da.choose_questions(p, 4)] == ["A0", "C0", "B0", "A1"]
    assert [q["pseudo"] for q in da.choose_questions(p, 20)] == ["A0", "C0", "B0", "A1", "C1", "A2"]
    assert da.choose_motion_videos(p, 2) == ["A", "C"]


def test_the_projection_findings_say_same_or_differs():
    q = dict(ours_native_px=[100.0, 200.0], meta_native_px=[100.2, 200.1], ours_upright_px=[50.0, 60.0], meta_upright_px=[50.0, 60.3],
             saved=[50 / 1408, 60 / 1408], size=[1408, 1408], saved_matches_sample=True)
    v = {"x": dict(up_deg=3.0, cam_vs_aea_deg=1.0, json_size=[2880, 2880], matched_size=[1408, 1408], model="Fisheye624")}
    lines = da.summarize([q], v)
    assert "SAME AS META'S" in lines[0] and "TURN OK" in lines[1] and "1.0 deg" in lines[2]
    lines = da.summarize([dict(q, meta_native_px=[105.0, 200.0])], {"x": dict(v["x"], up_deg=-88.0)})
    assert "DIFFERS FROM META'S" in lines[0] and "TURN WRONG ON 1" in lines[1]
    no_upright = {k: x for k, x in q.items() if k != "meta_upright_px"}
    assert "SAME AS META'S" in da.summarize([no_upright], v)[0] and "isn't in this projectaria-tools" in da.summarize([no_upright], v)[0]


# ============================================================================ step 0b: the motion test
def test_the_gyro_integrates_to_a_rotation():
    t = np.arange(0, 2e6, 1250.0)
    g = np.tile([0.1, -0.2, 0.3], (len(t), 1))
    th = dg.gyro_rotation(t, g, np.array([0.0, 5e5]), np.array([1e6, 1.5e6]))
    assert np.allclose(th, [[0.1, -0.2, 0.3], [0.1, -0.2, 0.3]], atol=1e-6)


@pytest.mark.parametrize("name", list(dg.HYPOTHESES))
def test_the_motion_fit_finds_the_orientation_and_the_scale(name):
    rng = np.random.default_rng(0)
    pred = rng.normal(0, 0.01, (600, 2))
    meas = 1.1 * pred @ np.asarray(dg.HYPOTHESES[name], float).T
    rs, best, slope, _, lag = dg.motion_fit(meas, pred, np.ones(600, bool))
    assert best == name and rs[name] == pytest.approx(1.0)
    if name == "as ours":
        assert slope == pytest.approx(1.1) and lag == 0 and dg.motion_ok(best, rs[best], slope, lag)
    else:
        assert not dg.motion_ok(best, rs["as ours"], slope, lag)


def test_the_lag_scan_finds_a_clock_offset_and_a_wrong_scale_fails():
    rng = np.random.default_rng(1)
    pred = rng.normal(0, 0.01, (600, 2))
    *_, lag = dg.motion_fit(np.roll(pred, -3, axis=0), pred, np.ones(600, bool))   # the video shows at f what the IMU predicts for f + 3
    assert lag == 3
    rs, best, slope, _, lag = dg.motion_fit(2.0 * pred, pred, np.ones(600, bool))
    assert best == "as ours" and slope == pytest.approx(2.0) and not dg.motion_ok(best, rs[best], slope, lag)


def test_phase_correlations_sign_is_found_on_a_known_shift():
    cv2 = pytest.importorskip("cv2")
    sx, sy = dg.shift_sign()
    rng = np.random.default_rng(2)
    a = cv2.GaussianBlur(rng.random((176, 176)).astype(np.float32), (0, 0), 3)
    (dx, dy), _ = dg.phase_shift(a, np.roll(a, (-4, 5), axis=(0, 1)))            # 5 right and 4 up
    assert sx * dx == pytest.approx(5, abs=0.5) and sy * dy == pytest.approx(-4, abs=0.5)


def test_the_other_orientations_of_a_gaze_point():
    alt = dg.alternatives((0.2, 0.3))
    assert alt["turned 180"] == pytest.approx((0.8, 0.7)) and alt["mirrored left-right"] == pytest.approx((0.8, 0.3))
    assert alt["turned 90 cw"] == pytest.approx((0.7, 0.2)) and "as ours" not in alt and len(set(dg.TAGS.values())) == 7


# ============================================================================ step 0b: files, answers and the model's view
def test_other_gaze_files_are_found_but_mps_csvs_are_not(tmp_path):
    assert not dg.gaze_like("videos/P01/mps/eye_gaze/general_eye_gaze.csv")
    assert dg.gaze_like("intermediate/P01/gaze_centre.csv") and dg.gaze_like("videos/P01/P01_gaze.npz") and not dg.gaze_like("videos/P01/imu.npz")
    p = tmp_path / "gaze_centre.csv"
    p.write_text("frame,x,y,label\n0,0.5,0.4,cup\n1,0.6,0.4,\n")
    got = dg.describe_file(str(p))
    assert got["columns"] == ["frame", "x", "y", "label"] and got["first_rows"][0] == ["0", "0.5", "0.4", "<text>"]


def write_results(base):
    """A fake section 17 results folder: 4 "where looking" clips, correct answer A; budget picks C on 3, nogate C or D."""
    model = base / "smolvlm"
    os.makedirs(model)
    lines = [dict(row="uniform", budget_pct=100.0, condition="uni"), dict(row="blind", budget_pct=0.0, condition="bl")]
    recs = collections.defaultdict(list)
    for b in (5.0, 10.0, 20.0):
        lines += [dict(row="budget", budget_pct=b, condition=f"b{b:g}"), dict(row="nogate", budget_pct=b, condition=f"n{b:g}")]
    for i in range(4):
        v = f"hd_ge_{i:05d}"
        recs["uniform"].append(dict(condition="uni", qid=i, video=v, answer="A", pred="A", correct=1, p=[.6, .1, .1, .1, .1]))
        recs["blind"].append(dict(condition="bl", qid=i, video=v, answer="A", pred="B", correct=0, p=[.1, .6, .1, .1, .1]))
        for b in (5.0, 10.0, 20.0):
            recs["budget"].append(dict(condition=f"b{b:g}", qid=i, video=v, answer="A", pred="C" if i < 3 else "A", correct=int(i == 3),
                                       p=[.2] * 5, n_frames=10, n_gaze_missing=0 if i < 2 else 1))
            recs["nogate"].append(dict(condition=f"n{b:g}", qid=i, video=v, answer="A", pred="C" if i < 2 else "D", correct=0, p=[.2] * 5,
                                       n_frames=10, n_gaze_missing=0))
    for row, rs in recs.items():
        (model / f"{row}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    (model / "flops_plan.json").write_text(json.dumps(dict(lines=lines)))
    return [dict(video=f"hd_ge_{i:05d}", type="gaze_estimation", answer="A", options=list("abcde"), real_video="P01-x",
                 hdepic_qid=f"gaze_gaze_estimation_{i}", question="q") for i in range(4)]


def test_the_answer_analysis_on_fake_records(tmp_path, monkeypatch, capsys):
    gv, results = tmp_path / "gv", tmp_path / "results"
    monkeypatch.setenv("GV_ROOT", str(gv))
    monkeypatch.setenv("HDEPIC_ROOT", str(tmp_path / "hdepic"))
    qs = write_results(results)
    os.makedirs(gv / "data" / "hdepic")
    (gv / "data" / "hdepic" / "questions.json").write_text(json.dumps(qs))
    dg.main(["answers", "--results", str(results)])
    out = capsys.readouterr().out
    assert "FINDING answers gaze_estimation budget 5%: accuracy 25.0%, picks C on 75% (correct answers are C on 0%)" in out
    with open(gv / "data" / "hdepic" / "diagnose" / "answers.json") as f:
        got = json.load(f)
    sw = {r["row"]: r for r in got["gaze_estimation"]["same_wrong"]}
    assert sw["budget"]["wrong_at_every_budget"] == 3 and sw["budget"]["same_option_at_every_budget"] == pytest.approx(100)
    assert sw["budget"]["same_as_nogate_5%"] == pytest.approx(200 / 3)
    split = {r["row"]: r for r in got["gaze_estimation"]["gaze_split"]}
    assert split["budget"]["with_gaze_acc"] == pytest.approx(0) and split["budget"]["missing_acc"] == pytest.approx(50)
    t = pd.DataFrame(got["table"]).set_index(["type", "label"])
    assert t.loc[("gaze_estimation", "budget 5%"), "frames_without_gaze"] == pytest.approx(5.0)
    assert t.loc[("gaze_estimation", "budget 5%"), "same_as_uniform"] == pytest.approx(25.0)
    assert t.loc[("gaze_estimation", "correct answers"), "pick_A"] == pytest.approx(100.0)
    # the results folder is only read: nothing was added to it
    assert sorted(os.listdir(results / "smolvlm")) == ["blind.jsonl", "budget.jsonl", "flops_plan.json", "nogate.jsonl", "uniform.jsonl"]


def test_the_model_view_is_what_the_encoder_gets():
    torch = pytest.importorskip("torch")
    from vlmeval import smolvlm_fovea as sm
    x01 = torch.rand(3, sm.IMG, sm.IMG)
    cells = sm.foveated_cells(0.5, 0.5, 1)
    view = dg.model_view(sm, x01, cells, [0.5] * 3, [0.5] * 3)
    assert len(cells) == 17 and dg.fine_cells_match(sm, x01, cells, view) < 1e-5
    y, x, s = next(c for c in np.asarray(cells).tolist() if c[2] == sm.TOP)
    block = view[:, y:y + 3, x:x + 3]
    assert float((block - block[:, :1, :1]).abs().max()) == 0.0              # a 126 px cell is its 42 px version, shown 3 times larger


# ============================================================================ the explore step: files, inputs, tags
def test_files_map_to_their_group_and_type():
    assert groups.file_group("fine_grained_action_recognition.json") == ("fine_grained", "action_recognition")
    assert groups.file_group("3d_perception_fixture_location.json") == ("3d_perception", "fixture_location")
    assert groups.file_group("ingredients_ingredient_weight.json") == ("ingredient", "ingredient_weight")
    assert groups.file_group("ingredient_ingredient_retrieval.json") == ("ingredient", "ingredient_retrieval")
    assert groups.file_group("gaze_gaze_estimation.json") == ("gaze", "gaze_estimation")
    assert groups.file_group("readme.json") == (None, None)


def test_inputs_take_clips_whole_videos_and_boxes():
    segs, boxes, other = groups.parse_inputs({
        "video 1": dict(id="P01-a", start_time="00:00:10.000", end_time="00:00:14.500"),
        "video 2": dict(id="P02-b", time="00:01:00.000", bbox=[1, 2, 3, 4]),
        "video 3": dict(id="P03-c"),
        "bounding box": [10, 20, 30, 40],
        "odd": "text"})
    assert [s["form"] for s in segs] == ["span", "time", "whole"] and [s["video"] for s in segs] == ["P01-a", "P02-b", "P03-c"]
    assert segs[0]["start_s"] == 10.0 and segs[1]["start_s"] == segs[1]["end_s"] == 60.0 and segs[2]["start_s"] is None
    assert [b["values"] for b in boxes] == [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]] and [b["segment"] for b in boxes] == [1, 2]
    assert other == [dict(name="odd", shape="str")]
    assert groups.as_box(dict(xmin=1, ymin=2, xmax=3, ymax=4)) == [1.0, 2.0, 3.0, 4.0] and groups.as_box([True, 1, 2, 3]) is None


def test_tags_show_their_shape_and_numbers_not_their_text():
    t = groups.tags("Where is it at <TIME 00:01:02.500 video 1> in <BBOX 693.1 847.2 775.0 979.8>?")
    assert [k for k, _, _ in t] == ["TIME", "BBOX"]
    assert t[0][1] == "<TIME 99:99:99.999 aaaaa 9>" and t[1][2] == [693.1, 847.2, 775.0, 979.8]
    assert groups.tag_time_s("x <TIME 00:01:02.500 video 1>") == pytest.approx(62.5) and groups.tag_time_s("none") is None
    r = groups.question_row("ingredient", "weight", "ingredient_weight_3", dict(
        inputs={"video 1": dict(id="v", time="00:00:05.000")}, question="How much at <TIME 00:00:05.000 video 1>?",
        choices=["a <BBOX 1 2 3 4>", "b", "c", "d", "e"], correct_idx=1))
    assert (r["q_time_tags"], r["q_bbox_tags"], r["c_time_tags"], r["c_bbox_tags"]) == (1, 0, 0, 1)
    assert r["bbox_tag_numbers"] == [[1.0, 2.0, 3.0, 4.0]] and r["q_tag_time_s"] == pytest.approx(5.0)
    assert "question" not in r and "choices" not in r


# ============================================================================ the explore step: the sample
def fake_fg(sizes=(600, 300, 100), videos=7):
    return [dict(qid=f"{t}_{i}", type=t, videos=[f"v{i % videos}"]) for t, n in zip("abc", sizes) for i in range(n)]


def test_the_sample_is_proportional_spread_and_seeded():
    rows = fake_fg()
    s, quota = groups.stratified_sample(rows, n=200, seed=0)
    assert quota == {"a": 120, "b": 60, "c": 20} and {t: len(v) for t, v in s.items()} == quota
    assert all(len(set(v)) == len(v) for v in s.values())
    for ids in s.values():
        per = collections.Counter(int(i.split("_")[1]) % 7 for i in ids)
        assert len(per) == 7 and max(per.values()) - min(per.values()) <= 1        # as evenly over the videos as the type allows
    assert groups.stratified_sample(rows, n=200, seed=0)[0] == s and groups.stratified_sample(rows, n=200, seed=1)[0] != s
    assert groups.apportion({"x": 1, "y": 1, "z": 1}, 2) == {"x": 1, "y": 1, "z": 0}


def test_a_type_smaller_than_its_share_gives_all_it_has():
    s, quota = groups.stratified_sample([dict(qid=f"a_{i}", type="a", videos=["v"]) for i in range(5)], n=10)
    assert quota == {"a": 10} and sorted(s["a"]) == sorted(f"a_{i}" for i in range(5))


def test_the_sample_is_written_once_and_never_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("GV_ROOT", str(tmp_path / "gv"))
    monkeypatch.setenv("HDEPIC_ROOT", str(tmp_path / "hdepic"))
    s, quota = groups.stratified_sample(fake_fg(), n=50)
    assert groups.write_or_check_sample(s, quota, 1000) == "written"
    assert groups.write_or_check_sample(s, quota, 1000) == "unchanged"
    with open(groups.sample_json()) as f:
        body = json.load(f)
    assert body["qids"] == s and body["seed"] == 0 and body["n"] == 50 and body["of"] == 1000
    assert not any("question" in k for k in body)
    with pytest.raises(SystemExit, match="SAMPLE DIFFERS"):
        groups.write_or_check_sample({k: v[::-1] for k, v in s.items()}, quota, 1000)


# ============================================================================ the explore step: K over segments, boxes
def test_k_splits_over_segments_by_duration_with_one_at_least_each():
    assert groups.split_k([10, 30], [300, 900]) == [17, 47]
    ks = groups.split_k([1, 1, 100], [30, 30, 3000])
    assert sum(ks) == 64 and min(ks) >= 1 and ks[2] > ks[0]
    assert groups.split_k([0], [60]) == [60]
    assert groups.split_k([5, 5], [3, 100]) == [3, 61]                          # capped at a segment's own frames, the rest to the other
    assert groups.split_k([2, 2], [10, 10]) == [10, 10]                         # under 64 frames in all: every frame


def test_segment_bounds():
    mp4 = (np.arange(1800) * 1e9 / 30).astype(np.int64)
    assert groups.segment_bounds(dict(form="whole"), mp4) == (0.0, pytest.approx(59.9667, abs=1e-3))
    assert groups.segment_bounds(dict(form="time", start_s=0.5, end_s=0.5), mp4, 1.0) == (0.0, 1.5)
    assert groups.segment_bounds(dict(form="span", start_s=2.0, end_s=4.0), mp4, 1.0) == (2.0, 4.0)   # only a single instant widens


def test_a_box_turns_with_the_image_and_each_resolution_is_drawn():
    b = [10, 20, 30, 60]
    assert groups.rotate_box(b, 0, 100) == [10, 20, 30, 60]
    assert groups.rotate_box(b, 1, 100) == [40, 10, 80, 30]                     # (x, y) -> (S - y, x), as our 90-degree turn
    assert groups.rotate_box(b, 2, 100) == [70, 40, 90, 80]
    assert groups.rotate_box(groups.rotate_box(b, 1, 100), 3, 100) == b
    assert list(groups.box_scales([0.1, 0.2, 0.3, 0.4])) == ["normalized"]
    assert list(groups.box_scales([693.1, 847.2, 775.0, 979.8])) == ["MP4 px", "2880 sensor", "0-1000 grid"]
    assert list(groups.box_scales([100, 200, 1300, 1400])) == ["MP4 px", "2880 sensor"]


# ============================================================================ the explore step end to end, on a fake HD-EPIC
def fake_videos(base, videos):
    """{id: gaze Hz or None}: 60 s at 30 fps, a time CSV, gaze, an IMU file and a calibration; no MP4."""
    for v, hz in videos.items():
        d = base / "videos" / v
        os.makedirs(d / "mps" / "eye_gaze")
        mp4 = (np.arange(1800) * 1e9 / 30).astype(np.int64)
        pd.DataFrame(dict(mp4_time_ns=mp4, relative_vrs_device_time_ns=mp4, vrs_device_time_ns=mp4 + 7_000_000_000)).to_csv(
            d / f"{v}_mp4_to_vrs_time_ns.csv", index=False)
        if hz:
            g_us = np.arange(7_000_000, 7_000_000 + 59_000_000, int(1e6 / hz))
            pd.DataFrame(dict(tracking_timestamp_us=g_us, yaw_rads_cpf=0.0, pitch_rads_cpf=0.0)).to_csv(
                d / "mps" / "eye_gaze" / "general_eye_gaze.csv", index=False)
        t = np.arange(0, 60 * 10 ** 9, 1_250_000)
        np.savez(d / "imu.npz", right_t_ns=t, right_accel=np.tile([0, 0, 9.81], (len(t), 1)), right_gyro=np.zeros((len(t), 3)))
        (d / "calibration.json").write_text(json.dumps(dict(CameraCalibrations=[dict(Label="camera-rgb")])))


def span(v, s, e):
    return dict(id=v, start_time=s, end_time=e)


def write_groups(folder):
    os.makedirs(folder)
    q = lambda inputs, question="q": dict(inputs=inputs, question=question, choices=list("abcde"), correct_idx=0)
    files = {
        "fine_grained_action_recognition.json": [q({"video 1": span("P01-x" if i % 2 else "P04-y", "00:00:10.000", "00:00:12.000")})
                                                 for i in range(30)],
        "fine_grained_why_recognition.json": [q({"video 1": span("P01-x", "00:00:20.000", "00:00:25.000")}) for _ in range(10)],
        "3d_perception_fixture_location.json": [q({"video 1": dict(id="P01-x", time="00:00:30.000"), "bbox": [700, 800, 760, 900]})] * 3,
        "object_motion_object_movement_counting.json": [q({"video 1": dict(id="P04-y")})] * 2,
        "recipe_multi_step_localization.json": [q({"video 1": span("P01-x", "00:00:05.000", "00:00:15.000"),
                                                   "video 2": span("P01-x", "00:00:40.000", "00:00:50.000")}),
                                                q({"video 1": span("P01-x", "00:00:05.000", "00:00:15.000"),
                                                   "video 2": span("P04-y", "00:00:40.000", "00:00:45.000")})],
        "ingredient_ingredient_weight.json": [q({"video 1": span("P04-y", "00:00:10.000", "00:00:20.000")},
                                                "How much at <TIME 00:00:12.000 video 1> in <BBOX 0.1 0.2 0.3 0.4>?")],
        "nutrition_nutrition_change.json": [q({"video 1": span(data.NO_GAZE_VIDEOS[0], "00:00:05.000", "00:00:08.000")})],
    }
    for name, qs in files.items():
        with open(os.path.join(folder, name), "w") as f:
            json.dump({f"{name[:-5]}_{i}": x for i, x in enumerate(qs)}, f)
    with open(os.path.join(folder, "gaze_gaze_estimation.json"), "w") as f:
        f.write("not json: the gaze group isn't in this run, so its files are never opened")


def test_the_explore_step_on_a_fake_hdepic(tmp_path, monkeypatch, capsys):
    pytest.importorskip("cv2")
    from accuracy_tables.hpc.hdepic import explore18
    root, gv = tmp_path / "hdepic", tmp_path / "gv"
    fake_videos(root, {"P01-x": 10.0, "P04-y": 60.0, data.NO_GAZE_VIDEOS[0]: None})
    write_groups(str(root / "annotations" / "vqa-benchmark"))
    monkeypatch.setenv("HDEPIC_ROOT", str(root))
    monkeypatch.setenv("GV_ROOT", str(gv))
    explore18.main(["--boxes", "3"])
    out = capsys.readouterr().out
    assert "fine_grained sample: 40 of 40 (seed 0), written" in out and "STOP: D1 to D5 are the user's" in out
    assert "other files, not opened: ['gaze_gaze_estimation.json']" in out
    assert "D4: every touched video's gaze has a whole step to 10 Hz" in out
    pq = pd.read_csv(gv / "data" / "hdepic" / "explore18" / "per_question.csv")
    assert len(pq) == 49 and not {"question", "choices"} & set(pq.columns)
    by = pq.groupby("group")
    assert by.size().to_dict() == {"3d_perception": 3, "fine_grained": 40, "ingredient": 1, "nutrition": 1, "object_motion": 2, "recipe": 2}
    p3 = pq[pq.group == "3d_perception"].iloc[0]
    assert p3.K_rule == 1 and p3.K_proposed == 61 and p3.n_boxes == 1 and p3.single_instant == 1      # 1 s either side at 30 fps
    om = pq[pq.group == "object_motion"].iloc[0]
    assert om.whole_video == 1 and om.K_rule == 64 and om.K_proposed == 64
    rc = pq[pq.group == "recipe"].reset_index(drop=True)
    assert list(rc.n_segments) == [2, 2] and list(rc.n_videos) == [1, 2] and list(rc.K_proposed) == [64, 64]
    assert int(pq.no_gaze_video.sum()) == 1 and pq[pq.group == "ingredient"].iloc[0].q_bbox_tags == 1
    with open(gv / "data" / "hdepic" / "sample_fine_grained_2000.json") as f:
        assert sum(len(v) for v in json.load(f)["qids"].values()) == 40
    explore18.main(["--boxes", "0"])
    assert "(seed 0), unchanged" in capsys.readouterr().out


# ============================================================================ the jobs and the runbook
@pytest.mark.parametrize("name", ["hdepic18_diagnose.SBATCH", "hdepic18_explore.SBATCH"])
def test_the_section_18_jobs_are_cpu_only_and_write_only_derived_data(name):
    text = job(name)
    assert "gres" not in text and "SBATCH_ACCOUNT" not in text and "--no-parity" not in text and "HF_HOME=" not in text
    assert "HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}" in text and "--out-dir" not in text
    assert "results/" not in text.replace("results/hdepic_gaze_smolvlm_flops/", "")


def test_step_0_runs_both_environments_and_stops():
    text = job("hdepic18_diagnose.SBATCH")
    assert 'run_in_aria "python -u -m $H.diagnose_aria"' in text and "$H.diagnose render" in text and "$H.diagnose motion" in text
    assert text.index("$H.diagnose_aria") < text.index("$H.diagnose motion") and "STOP: step 0's findings go to the user" in text
    assert "myquota" in job("hdepic18_explore.SBATCH") and "accuracy_tables.hpc.hdepic.explore18" in job("hdepic18_explore.SBATCH")


def test_section_18_puts_step_0_first_and_stops_for_the_decisions():
    with open(os.path.join(REPO, "HPC_ACCURACY_RUNBOOK.md")) as f:
        section = f.read().split("\n## 18.")[1]
    for text in ("Step 0", "STOP", "D1", "D2", "D3", "D4", "D5", "sample_fine_grained_2000.json", "seed 0", "results/hdepic_all_smolvlm_flops",
                 "hdepic18_diagnose.SBATCH", "hdepic18_explore.SBATCH", "read-only", "Never commit", "5,550", "B200", "FINDING",
                 "2,000-question stratified sample of 21,000", "separate run", "482adb537c021c86670beed01cd58990d01e72e4"):
        assert text in section, text
    assert section.index("hdepic18_diagnose.SBATCH") < section.index("hdepic18_explore.SBATCH") and "no winner" in section.lower()

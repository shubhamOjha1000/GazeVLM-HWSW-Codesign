"""HD-EPIC's gaze group, step 0 (runbook section 17): the question files, the time CSV, gaze rates, the IMU and calibration readers and the
exploration report, on a small fake HD-EPIC folder. CPU only, no video decoding, no real HD-EPIC data."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from accuracy_tables.hpc.hdepic import data, explore

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------- questions
def test_times_parse_with_and_without_hours():
    assert data.parse_time("00:01:02.500") == pytest.approx(62.5) and data.parse_time("01:00:00.000") == 3600
    assert data.parse_time("02:03.250") == pytest.approx(123.25)
    for bad in ("12", "1:2:3:4", "-00:00:01.0", "aa:bb"):
        with pytest.raises(ValueError):
            data.parse_time(bad)


def test_inputs_take_one_time_or_a_span_and_nothing_else():
    got = data.parse_inputs({"video 1": dict(id="P01-a", start_time="00:00:10.000", end_time="00:00:14.500"),
                             "video 2": dict(id="P02-b", time="00:01:00.000")})
    assert got == [dict(name="video 1", video="P01-a", form="span", start_s=10.0, end_s=14.5),
                   dict(name="video 2", video="P02-b", form="time", start_s=60.0, end_s=60.0)]
    with pytest.raises(ValueError, match="neither"):
        data.parse_inputs({"video 1": dict(id="P01-a", at="00:00:01.000")})


def test_tags_are_counted_in_questions_and_choices():
    q = dict(inputs={"video 1": dict(id="v", start_time="00:00:00.000", end_time="00:00:02.000")},
             question="What at <TIME 00:00:01.000 video 1> in <BBOX 1 2 3 4>?", choices=["a <BBOX 5 6 7 8>", "b", "c", "d", "e"],
             correct_idx=2)
    r = data.question_row("gaze_estimation", "gaze_gaze_estimation_0", q)
    assert (r["q_time_tags"], r["q_bbox_tags"], r["choice_time_tags"], r["choice_bbox_tags"]) == (1, 1, 0, 1)
    assert r["n_choices"] == 5 and r["correct_idx"] == 2 and r["videos"] == ["v"] and "question" not in r


def write_questions(folder, per_type):
    os.makedirs(folder, exist_ok=True)
    for qtype, name in data.GAZE_TYPES.items():
        with open(os.path.join(folder, name), "w") as f:
            json.dump({f"{name[:-5]}_{i}": q for i, q in enumerate(per_type[qtype])}, f)
    with open(os.path.join(folder, "fine_grained_action_recognition.json"), "w") as f:
        f.write("not json: only the gaze group's 2 files may be opened")


def span_q(video, s, e, n_choices=5):
    return dict(inputs={"video 1": dict(id=video, start_time=s, end_time=e)}, question="q", choices=list("abcde")[:n_choices],
                correct_idx=0)


def test_only_the_gaze_groups_2_files_are_read(tmp_path):
    write_questions(tmp_path, dict(gaze_estimation=[span_q("v1", "00:00:01.000", "00:00:03.000")],
                                   interaction_anticipation=[span_q("v1", "00:00:05.000", "00:00:09.000")] * 2))
    got = data.load_gaze_questions(str(tmp_path))
    assert {k: len(v) for k, v in got.items()} == {"gaze_estimation": 1, "interaction_anticipation": 2}
    assert got["interaction_anticipation"][1]["qid"] == "gaze_interaction_anticipation_1"


# ---------------------------------------------------------------------------- the time CSV
def test_mp4_frames_map_through_the_time_csv():
    mp4 = np.array([0, 33, 66, 66, 100, 133]) * 1_000_000                        # one repeated row
    assert data.frames_between(mp4, 0.030, 0.070).tolist() == [1, 2, 3]
    assert data.frames_between(mp4, 0.090, 0.090).tolist() == [4]                  # one time: the nearest frame
    assert data.frames_between(mp4, 0.040, 0.040).tolist() == [1]
    vrs = mp4 + 5_000_000_000
    assert data.device_ns(vrs, mp4, 0.1) == 5_100_000_000
    st = data.time_csv_stats(mp4, 30.0)
    assert st["rows"] == 6 and st["repeated"] == 1 and st["backwards"] == 0 and st["max_frame_offset"] == 1
    assert data.time_csv_stats(np.array([0, 10, 5]), 30.0)["backwards"] == 1


@pytest.mark.parametrize("hz,step", [(10.0, 1), (30.0, 3), (60.0, 6), (20.0, 2), (10.4, 1), (15.0, None), (45.0, None), (0, None)])
def test_a_gaze_rate_reaches_10_hz_by_a_whole_step_or_not_at_all(hz, step):
    assert data.gate_step(hz) == step


def test_rates_and_gaps_of_timestamps():
    t = np.arange(0, 2_000_000, 100_000)                                       # 10 Hz in us
    assert data.rate_hz(t) == pytest.approx(10) and data.longest_gap_s(np.r_[t, 3_000_000]) == pytest.approx(1.1)
    assert data.rate_hz(np.arange(0, 10 ** 9, 10 ** 6), 1e9) == pytest.approx(1000)


def test_imu_and_calibration_summaries(tmp_path):
    t = np.arange(0, 10 ** 9, 1_250_000)                                       # 800 Hz
    np.savez(tmp_path / "imu.npz", right_t_ns=t, right_accel=np.tile([0, 0, 9.81], (len(t), 1)), right_gyro=np.zeros((len(t), 3)),
             left_t_ns=np.arange(0, 10 ** 9, 10 ** 6), left_accel=np.zeros((1000, 3)), left_gyro=np.zeros((1000, 3)))
    s = data.imu_summary(str(tmp_path / "imu.npz"))
    assert s["right"]["hz"] == pytest.approx(800) and s["right"]["accel_mag"] == pytest.approx(9.81) and s["left"]["hz"] == pytest.approx(1000)
    assert s["arrays"]["right_accel"] == [len(t), 3]
    (tmp_path / "c.json").write_text(json.dumps(dict(CameraCalibrations=[dict(Label="camera-rgb", Projection=dict(Name="Fisheye624"),
                                                                              T_Device_Camera=dict(Translation=[0, 0, 0]))], ImuCalibrations=[])))
    c = data.calibration_fields(str(tmp_path / "c.json"))
    assert c["rgb"]["Label"] == "camera-rgb" and c["rgb"]["Projection"] == ["Name"] and c["top"]["ImuCalibrations"] == "list of 0"


def test_nothing_is_written_inside_hdepic(tmp_path, monkeypatch):
    monkeypatch.setenv("HDEPIC_ROOT", str(tmp_path / "hdepic"))
    os.makedirs(tmp_path / "hdepic")
    with pytest.raises(SystemExit, match="only read"):
        data.refuse_inside_raw(str(tmp_path / "hdepic" / "x"))
    data.refuse_inside_raw(str(tmp_path / "derived"))


# ---------------------------------------------------------------------------- the report
def test_section_14s_limits_and_1fps_counts():
    assert [explore.limit(f, 64) for f in explore.FRACTIONS] == [15, 30, 60]
    assert [explore.limit(f, 10) for f in explore.FRACTIONS] == [2, 4, 9] and explore.limit(0.05, 3) == 1
    assert explore.frames_1fps(1.2, 4.0) == 3 and explore.frames_1fps(2.0, 2.0) == 1 and explore.frames_1fps(2.2, 2.8) == 0


def test_option_frames_count_once_per_video_at_1_fps():
    pq = pd.DataFrame(dict(video=["a", "a", "b"], start_s=[0.0, 1.5, 0.0], end_s=[3.0, 4.0, 70.0], span_s=[3.0, 2.5, 70.0]))
    assert explore.option_frames(pq) == dict(a=3 * 64, b=len({0, 1, 2, 3, 4}) + 71, c=64)


def fake_hdepic(base, videos):
    """videos: {id: gaze Hz or None}; 60 s at 30 fps, a time CSV with one repeated row, an IMU file, a calibration."""
    for v, hz in videos.items():
        d = base / "videos" / v
        os.makedirs(d / "mps" / "eye_gaze")
        mp4 = (np.arange(1800) * 1e9 / 30).astype(np.int64)
        mp4[100] = mp4[99]
        dev = mp4 + 7_000_000_000
        pd.DataFrame(dict(mp4_time_ns=mp4, relative_vrs_device_time_ns=mp4, vrs_device_time_ns=dev)).to_csv(
            d / f"{v}_mp4_to_vrs_time_ns.csv", index=False)
        if hz:
            g_us = np.arange(7_000_000, 7_000_000 + 58_000_000, int(1e6 / hz))
            pd.DataFrame(dict(tracking_timestamp_us=g_us, yaw_rads_cpf=0.0, pitch_rads_cpf=0.0)).to_csv(
                d / "mps" / "eye_gaze" / "general_eye_gaze.csv", index=False)
        t = np.arange(0, 60 * 10 ** 9, 1_250_000)
        np.savez(d / "imu.npz", right_t_ns=t, right_accel=np.tile([0, 0, 9.81], (len(t), 1)), right_gyro=np.zeros((len(t), 3)))
        (d / "calibration.json").write_text(json.dumps(dict(CameraCalibrations=[dict(Label="camera-rgb")])))


def test_the_report_on_a_fake_hdepic(tmp_path, monkeypatch, capsys):
    root, gv = tmp_path / "hdepic", tmp_path / "gv"
    fake_hdepic(root, {"P01-x": 10.0, "P04-y": 60.0, data.NO_GAZE_VIDEOS[0]: None, "P05-unused": 10.0})
    write_questions(root / "annotations" / "vqa-benchmark", dict(
        gaze_estimation=[span_q("P01-x", "00:00:10.000", "00:00:13.000"), span_q("P04-y", "00:00:20.000", "00:00:20.000"),
                         span_q(data.NO_GAZE_VIDEOS[0], "00:00:05.000", "00:00:08.000")],
        interaction_anticipation=[span_q("P01-x", "00:00:00.500", "00:00:59.000"), span_q("P04-y", "00:00:30.000", "00:00:35.000")]))
    monkeypatch.setenv("HDEPIC_ROOT", str(root))
    monkeypatch.setenv("GV_ROOT", str(gv))
    explore.main(["--sample-frames", "1"])
    out = capsys.readouterr().out
    assert "the gaze questions touch 3 of 4 videos (fewer than about 20" in out and "STOP: D1 to D5 are the user's" in out
    pv = pd.read_csv(gv / "data" / "hdepic" / "explore" / "per_video.csv").set_index("video")
    assert pv.loc["P04-y", "gaze_hz"] == pytest.approx(60) and pv.loc["P04-y", "gaze_gate_step"] == 6
    assert pv.loc["P01-x", "timecsv_repeated"] == 1 and pd.isna(pv.loc[data.NO_GAZE_VIDEOS[0], "gaze_file"])
    assert pv.loc["P01-x", "gaze_ends_before_video_s"] == pytest.approx(59.967 - 58 + 0.1, abs=0.05)
    pq = pd.read_csv(gv / "data" / "hdepic" / "explore" / "per_question.csv")
    assert len(pq) == 5 and not {"question", "choices"} & set(pq.columns)
    r = pq.set_index("qid").loc["gaze_gaze_estimation_0"]
    assert r.frames_native == 91 and r.frames_1fps == 4 and r.gaze_in_span == 31 and r.option_a_gap_s == pytest.approx(3 / 63)
    assert pq.set_index("qid").loc["gaze_gaze_estimation_1", "frames_native"] == 1                 # one time: one frame
    assert int(pq.no_gaze_video.sum()) == 1 and "D5: gaze questions on the 2 videos without gaze" in out


def test_the_examples_and_question_text_stay_out_of_the_written_files(tmp_path, monkeypatch):
    root, gv = tmp_path / "hdepic", tmp_path / "gv"
    fake_hdepic(root, {"P01-x": 10.0})
    secret = "SECRET-QUESTION-TEXT"
    write_questions(root / "annotations" / "vqa-benchmark", dict(
        gaze_estimation=[dict(span_q("P01-x", "00:00:10.000", "00:00:13.000"), question=secret)],
        interaction_anticipation=[span_q("P01-x", "00:00:20.000", "00:00:30.000")]))
    monkeypatch.setenv("HDEPIC_ROOT", str(root))
    monkeypatch.setenv("GV_ROOT", str(gv))
    explore.main(["--sample-frames", "1"])
    for name in os.listdir(gv / "data" / "hdepic" / "explore"):
        assert secret not in (gv / "data" / "hdepic" / "explore" / name).read_text()


# ---------------------------------------------------------------------------- the job and the runbook
def test_the_explore_job_is_cpu_only_and_reads_hdepic_from_scratch():
    with open(os.path.join(REPO, "accuracy_tables", "hpc", "hdepic_explore.SBATCH")) as f:
        text = f.read()
    assert "gres" not in text and "HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}" in text and "myquota" in text
    assert "accuracy_tables.hpc.hdepic.explore" in text and "SBATCH_ACCOUNT" not in text


def test_section_17_stops_for_the_decisions():
    with open(os.path.join(REPO, "HPC_ACCURACY_RUNBOOK.md")) as f:
        section = f.read().split("\n## 17.")[1]
    for text in ("gaze", "D1", "D2", "D3", "D4", "D5", "STOP", "read-only", "Never commit", "gaze-driven rows see the wearer's gaze",
                 "482adb537c021c86670beed01cd58990d01e72e4", "hdepic_gaze_smolvlm_flops", "git pull --ff-only origin eyehead-gate"):
        assert text in section, text

"""HD-EPIC's gaze group, step 0 (runbook section 17): the question files, the time CSV, gaze rates, the IMU and calibration readers and the
exploration report, on a small fake HD-EPIC folder. CPU only, no video decoding, no real HD-EPIC data."""
import json
import os
import re

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


# ============================================================================ steps 1 to 8 (the user's D1 to D5, 2026-11-13)
from accuracy_tables import common  # noqa: E402
from accuracy_tables.hpc.hdepic import checks as hchecks  # noqa: E402
from accuracy_tables.hpc.hdepic import frames as hframes  # noqa: E402
from accuracy_tables.hpc.hdepic import gate, manifest, plan  # noqa: E402
from accuracy_tables.hpc.hdepic import load as hload  # noqa: E402
from accuracy_tables.hpc.hdepic import report as hreport  # noqa: E402
from accuracy_tables.hpc.newvideos import build_rows, paths as npaths  # noqa: E402

CONFIG_HD = os.path.join(REPO, "accuracy_tables", "hpc", "config.hdepic.yaml")
DEV0_NS = 7_000_000_000                       # device time of MP4 frame 0 in the fake world


# ---------------------------------------------------------------------------- D1: the candidates
def test_candidates_are_distinct_inside_the_span_and_capped_at_64():
    mp4 = (np.arange(1800) * 1e9 / 30).astype(np.int64)
    mp4[301] = mp4[300]                                                       # a repeated time-CSV row inside the span
    idx, n = plan.candidates(mp4, 10.0, 20.0)
    assert n == 300 and len(idx) == 64 and len(set(idx.tolist())) == 64 and 301 not in idx.tolist()
    assert idx[0] == 300 and idx[-1] == 600 and (mp4[idx] <= 20e9).all() and (mp4[idx] >= 10e9).all()   # never after the span's end
    idx, n = plan.candidates(mp4, 10.0, 11.0)
    assert n == 30 and idx.tolist() == [300] + list(range(302, 331))                       # K = all of them, frame 301 counted once
    assert plan.pseudo_name("gaze_estimation", "gaze_gaze_estimation_12") == "hd_ge_00012"
    assert plan.pseudo_name("interaction_anticipation", "gaze_interaction_anticipation_7") == "hd_ia_00007"


def test_section_14s_rule_for_fewer_than_64_frames():
    """Uniform K is all of a short clip's frames; the budget rule's limits scale with them; the baselines' K' is one number."""
    from vlmeval import budget_rule as br
    assert common.uniform_indices(35, 96) == list(range(35))
    lim = lambda k: [br.frame_limit(f, min(k, 64) * 81, 17, k, 64) for f in (0.05, 0.1, 0.2)]
    assert lim(64) == [15, 30, 60] and lim(35) == [8, 16, 33] and lim(15) == [3, 7, 14]
    from accuracy_tables import flops
    assert flops.encoded([15, 35, 64], 16).tolist() == [15, 16, 16]          # a clip with fewer frames than K' gives all it has


# ---------------------------------------------------------------------------- D4: one resampler to 10 Hz
@pytest.mark.parametrize("hz,step", [(10.0, 1), (30.0, 3), (60.04, 6), (59.9, 6), (20.0, 2)])
def test_the_resampler_takes_every_nth_sample(hz, step):
    t = np.arange(0, 5e6, 1e6 / hz)
    assert gate.gate_step_any(t)[1] == step


@pytest.mark.parametrize("hz", [15.0, 45.0, 5.0])
def test_a_rate_with_no_whole_step_stops(hz):
    with pytest.raises(ValueError, match="no whole step"):
        gate.gate_step_any(np.arange(0, 5e6, 1e6 / hz))


@pytest.mark.parametrize("hz", [10.0, 30.0])
def test_the_resampler_is_section_9s_at_10_and_30_hz(hz):
    t = np.arange(0, 5e6, 1e6 / hz)
    assert gate.gate_step_any(t)[1] == build_rows.gate_gaze_step(t)[1]


def synthetic_streams(hz=10.0, seconds=6, seed=0):
    rng = np.random.default_rng(seed)
    g_ts = np.arange(0, seconds * 1e6, 1e6 / hz)
    yaw, pit = rng.normal(0, 0.1, len(g_ts)), rng.normal(0, 0.1, len(g_ts))
    x, y, oof = 0.5 + yaw, 0.5 + pit, np.zeros(len(g_ts), bool)
    imu_ts = np.arange(0, seconds * 1e6, 1250.0)
    imu = np.concatenate([np.tile([0, -9.81, 0], (len(imu_ts), 1)), rng.normal(0, 0.05, (len(imu_ts), 3))], 1).astype(np.float32)
    return g_ts, yaw, pit, build_rows.lookup_projector(yaw, pit, x, y, oof), imu_ts, imu


def nan_aware_differences(old, new):
    """Where two lists of row dicts differ: the same keys, NaN in exactly the same places, every other value exactly equal (same
    type and ==). A plain == can't compare rows holding NaN, since nan != nan."""
    isnan = lambda v: isinstance(v, (float, np.floating)) and bool(np.isnan(v))
    if len(old) != len(new):
        return [f"{len(old)} rows against {len(new)}"]
    out = []
    for i, (a, b) in enumerate(zip(old, new)):
        if a.keys() != b.keys():
            out.append(f"row {i}: keys {sorted(set(a) ^ set(b))}")
            continue
        for k in a:
            if isnan(a[k]) or isnan(b[k]):
                if not (isnan(a[k]) and isnan(b[k])):
                    out.append(f"row {i} {k}: NaN in only one ({a[k]!r} against {b[k]!r})")
            elif type(a[k]) is not type(b[k]) or a[k] != b[k]:
                out.append(f"row {i} {k}: {a[k]!r} against {b[k]!r}")
    return out


def test_a_1s_window_at_1_fps_is_section_9s_pair_window():
    g_ts, yaw, pit, project, imu_ts, imu = synthetic_streams()
    f_ts = np.array([1e6, 2e6, 3e6, 4e6])
    old = build_rows.pair_rows("v", f_ts, g_ts, yaw, pit, project, imu_ts, imu)
    new = build_rows.pair_rows("v", f_ts, g_ts, yaw, pit, project, imu_ts, imu, window_us=1e6)
    assert len(old) == 3
    assert nan_aware_differences(old, new) == []
    bent = [dict(r) for r in new]                                            # the comparison itself catches each kind of difference
    bent[0]["n_velocities"] += 1
    bent[1]["frame_similarity"] = 0.0
    bent[2]["imu_accel_mag_mean"] = float("nan")
    assert len(nan_aware_differences(old, bent)) == 3


def test_close_candidates_still_get_the_gates_1s_window():
    g_ts, yaw, pit, project, imu_ts, imu = synthetic_streams()
    f_ts = 2e6 + np.arange(10) * 33_333.0                                    # 0.033 s apart, as "where looking"
    rows = build_rows.pair_rows("v", f_ts, g_ts, yaw, pit, project, imu_ts, imu, window_us=1e6)
    assert all(r["n_velocities"] >= 8 and len(r["imu_window"].split(";")) == 10 for r in rows)


# ---------------------------------------------------------------------------- D4's proof on section 9's 30 Hz path
def test_the_resampler_rebuilds_a_30_hz_video_byte_for_byte(tmp_path, monkeypatch):
    monkeypatch.setenv("GV_ROOT", str(tmp_path / "gv"))
    monkeypatch.setenv("EGO_ROOT", str(tmp_path / "ego"))
    for v, hz in (("2023_thirty", 30.0), ("2023_ten", 10.0)):
        g_ts, yaw, pit, _, imu_ts, imu = synthetic_streams(hz=hz, seconds=8, seed=int(hz))
        os.makedirs(os.path.dirname(npaths.gaze_npz(v)), exist_ok=True)
        np.savez(npaths.gaze_npz(v), t_us=g_ts, yaw=yaw, pitch=pit, x=0.5 + yaw, y=0.5 + pit, oof=np.zeros(len(g_ts), bool))
        os.makedirs(npaths.frames_dir(v))
        with open(os.path.join(npaths.frames_dir(v), "meta.json"), "w") as f:
            json.dump(dict(ts=[float(i) for i in range(7)], n=7), f)
        os.makedirs(npaths.video_dir(v))
        np.savez(os.path.join(npaths.video_dir(v), "imu.npz"), right_t_ns=(imu_ts * 1e3).astype(np.int64), right_accel=imu[:, :3],
                 right_gyro=imu[:, 3:])
        build_rows.build_video(v, force=True)
    n, same, differ = gate.resample_check(["2023_thirty", "2023_ten"], str(tmp_path / "check"))
    assert (n, same, differ) == (1, 1, [])                                    # only the 30 Hz video is compared


# ---------------------------------------------------------------------------- the IMU against AEA's
def imu_npz(path, gravity=(0, -9.81, 0), scale=1.0, hz=800):
    t = np.arange(0, 2 * 10 ** 9, int(1e9 / hz))
    rng = np.random.default_rng(1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, right_t_ns=t, right_accel=scale * (np.tile(gravity, (len(t), 1)) + rng.normal(0, 0.1, (len(t), 3))),
             right_gyro=rng.normal(0, 0.05, (len(t), 3)))


@pytest.mark.parametrize("case,says", [("same", None), ("axes", "other axes"), ("units", "not m/s^2"), ("rate", "Hz")])
def test_the_imu_is_checked_against_aea(case, says, tmp_path, monkeypatch):
    monkeypatch.setattr(data, "video_dir", lambda v: str(tmp_path / "hd" / v))
    monkeypatch.setattr(npaths, "video_dir", lambda v: str(tmp_path / "aea" / v))
    imu_npz(str(tmp_path / "aea" / "loc_a" / "imu.npz"))
    kw = dict(axes=dict(gravity=(-9.81, 0, 0)), units=dict(scale=1000 / 9.81), rate=dict(hz=1000), same={})[case]
    imu_npz(str(tmp_path / "hd" / "P01-x" / "imu.npz"), **kw)
    _, found = gate.imu_check(["P01-x"], ["loc_a"])
    assert (found == []) if says is None else any(says in f for f in found)


# ---------------------------------------------------------------------------- a whole small HD-EPIC world, steps 1 to 4
def device_world(base, videos):
    """Videos with their time CSV, gaze CSV, projected gaze (as project.py writes it), and an IMU in device time."""
    for v, (hz, oof_from) in videos.items():
        d = base / "hdepic" / "videos" / v
        os.makedirs(d / "mps" / "eye_gaze")
        mp4 = (np.arange(1800) * 1e9 / 30).astype(np.int64)
        pd.DataFrame(dict(mp4_time_ns=mp4, relative_vrs_device_time_ns=mp4, vrs_device_time_ns=mp4 + DEV0_NS)).to_csv(
            d / f"{v}_mp4_to_vrs_time_ns.csv", index=False)
        g = np.arange(DEV0_NS // 1000, DEV0_NS // 1000 + 59_000_000, int(1e6 / hz)).astype(np.int64)
        pd.DataFrame(dict(tracking_timestamp_us=g)).to_csv(d / "mps" / "eye_gaze" / "general_eye_gaze.csv", index=False)
        rng = np.random.default_rng(len(v))
        yaw, pit = rng.normal(0, 0.1, len(g)), rng.normal(0, 0.1, len(g))
        oof = np.zeros(len(g), bool)
        if oof_from is not None:
            oof[(g > DEV0_NS // 1000 + oof_from * 1e6) & (g < DEV0_NS // 1000 + (oof_from + 0.3) * 1e6)] = True
        out = base / "gv" / "data" / "hdepic" / "gaze" / f"{v}.npz"
        os.makedirs(out.parent, exist_ok=True)
        np.savez(out, t_us=g, yaw=yaw, pitch=pit, x=np.clip(0.5 + yaw, 0, 1), y=np.clip(0.5 + pit, 0, 1), oof=oof)
        t = np.arange(DEV0_NS, DEV0_NS + 60 * 10 ** 9, 1_250_000)
        np.savez(d / "imu.npz", right_t_ns=t, right_accel=np.tile([0, -9.81, 0], (len(t), 1)) + rng.normal(0, 0.1, (len(t), 3)),
                 right_gyro=rng.normal(0, 0.05, (len(t), 3)))


@pytest.fixture
def hd_world(tmp_path, monkeypatch):
    monkeypatch.setenv("HDEPIC_ROOT", str(tmp_path / "hdepic"))
    monkeypatch.setenv("GV_ROOT", str(tmp_path / "gv"))
    device_world(tmp_path, {"P01-x": (10.0, 10.5), "P04-y": (60.0, None)})
    write_questions(tmp_path / "hdepic" / "annotations" / "vqa-benchmark", dict(
        gaze_estimation=[span_q("P01-x", "00:00:10.000", "00:00:11.500"), span_q("P04-y", "00:00:40.000", "00:00:41.000")],
        interaction_anticipation=[span_q("P04-y", "00:00:20.000", "00:00:30.000"), span_q("P01-x", "00:00:30.000", "00:00:40.000")]))
    qs = data.load_gaze_questions()
    info, times, gz = {}, {}, {}
    for v in ("P01-x", "P04-y"):
        info[v] = dict(fps=30.0, mp4_frames=1800, width=1408, height=1408, gaze_csv=data.gaze_csv(v))
        times[v] = data.read_time_csv(os.path.join(data.video_dir(v), f"{v}_mp4_to_vrs_time_ns.csv"))
        gz[v] = pd.read_csv(info[v]["gaze_csv"])["tracking_timestamp_us"].to_numpy(np.int64)
        info[v]["gaze_hz"] = data.rate_hz(gz[v])
    p = plan.make_plan(qs, info, times, gz)
    os.makedirs(os.path.dirname(data.plan_json()), exist_ok=True)
    with open(data.plan_json(), "w") as f:
        json.dump(p, f)
    gate.rows_cmd(None)
    from accuracy_tables.hpc.newvideos import score_gate

    def fake_scores(test, cks):
        df = pd.read_csv(test)
        return df.assign(pred=df["idx"] % 4)[["sequence", "feat_frame_1", "feat_frame_2", "quad", "pred"]]
    monkeypatch.setattr(score_gate, "load_models", lambda d: None)
    monkeypatch.setattr(score_gate, "score_ensemble", fake_scores)
    gate.score_cmd(None)
    from PIL import Image
    for q in p["questions"]:
        d = data.frames_dir(q["pseudo"])
        os.makedirs(d)
        for j in range(q["K"]):
            Image.new("RGB", (8, 8), (j % 255, 0, 0)).save(os.path.join(d, f"frame_{j:05d}.jpg"))
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(dict(n=q["K"], dir=d, ts=q["mp4_s"], device_us=q["device_us"], mp4_frames=q["mp4_frames"], decoder_s=q["mp4_s"],
                           native_fps=30.0, size=1344, orig_w=1408, orig_h=1408, video=q["video"]), f)
    return p


def test_the_plan_numbers_questions_and_flags_frames_past_the_gaze(hd_world):
    p = hd_world
    assert [q["pseudo"] for q in p["questions"]] == ["hd_ge_00000", "hd_ge_00001", "hd_ia_00000", "hd_ia_00001"]
    assert [q["qid"] for q in p["questions"]] == [0, 1, 2, 3] and p["excluded"] == []
    k = {q["pseudo"]: q["K"] for q in p["questions"]}
    assert k == {"hd_ge_00000": 46, "hd_ge_00001": 31, "hd_ia_00000": 64, "hd_ia_00001": 64}
    late = {q["pseudo"]: q["frames_after_gaze_end"] for q in p["questions"]}
    assert late["hd_ia_00001"] == 0 and all(v == 0 for v in late.values())   # the fake gaze covers 59 s


def test_each_candidate_gets_a_1s_gate_window(hd_world):
    s = pd.read_csv(os.path.join(data.derived_root(), "rows_stats.csv")).set_index("pseudo")
    assert s.loc["hd_ge_00000", "rows"] == 45 and s.loc["hd_ia_00000", "rows"] == 63
    assert (s.velocities_min >= 8).all() and s.loc["hd_ia_00000", "step"] == 6 and s.loc["hd_ge_00000", "step"] == 1
    assert s.loc["hd_ge_00000", "distinct_gaze_inputs"] < s.loc["hd_ge_00000", "rows"]           # 0.033 s apart: windows repeat


def test_the_frames_check_matches_the_plan_and_finds_what_differs(hd_world):
    n, frames, nbytes, problems, worst = hframes.check(hd_world)
    assert (n, frames, problems, worst) == (4, 46 + 31 + 64 + 64, [], 0.0) and nbytes > 0
    q = hd_world["questions"][0]
    os.remove(os.path.join(data.frames_dir(q["pseudo"]), "frame_00003.jpg"))
    meta = os.path.join(data.frames_dir(hd_world["questions"][1]["pseudo"]), "meta.json")
    m = json.load(open(meta))
    m["decoder_s"] = [t + 0.2 for t in m["decoder_s"]]
    json.dump(m, open(meta, "w"))
    _, _, _, problems, worst = hframes.check(hd_world)
    assert len(problems) == 2 and worst == pytest.approx(0.2)


def test_the_manifest_and_the_loader_give_section_14s_data(hd_world, capsys):
    manifest.main(["--config", CONFIG_HD])
    out = capsys.readouterr().out
    assert "data check OK: 4 questions on 2 videos" in out and "excluded: 0 (none)" in out
    qs = json.load(open(data.questions_json()))
    assert [q["answer"] for q in qs] == ["A"] * 4 and qs[0]["video"] == "hd_ge_00000" and qs[0]["real_video"] == "P01-x"
    cfg = common.load_config(CONFIG_HD)
    d = common.load_data(cfg)
    assert sorted(d["videos"]) == sorted(q["video"] for q in qs) and d["test_videos"] == set(d["videos"])
    v = "hd_ge_00000"
    assert d["metas"][v]["n"] == 46 and len(d["uniform"][v]) == 46 and len(d["labels"][v]) == 45
    fr = d["frame"](v, 0)
    assert fr["t"] == pytest.approx(10.0) and fr["key"] == (v, 0) and fr["path"].endswith("frame_00000.jpg")
    gaze = [d["frame"](v, f)["gaze"] for f in range(46)]
    times = d["metas"][v]["ts"]
    assert all(g is None for g, t in zip(gaze, times) if 10.65 < t <= 11.3) and gaze[0] is not None  # out-of-view samples at 10.6 and 10.7 s
    from accuracy_tables.models import ADAPTERS
    frames, _ = common.budget_frames(cfg, d, "smolvlm", ADAPTERS["smolvlm"].stock_tokens, 0.2, use_gate=True)
    assert len(frames[v]) <= 43 and len(frames["hd_ia_00000"]) <= 60       # the limits of 46 and 64 candidates at 20%


def test_the_dispatch_leaves_every_other_config_alone(monkeypatch):
    monkeypatch.setattr(hload, "load_data", lambda cfg: "hdepic")
    assert common.load_data(dict(dataset="hdepic")) == "hdepic"


def test_the_hdepic_config_has_section_14s_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("GV_ROOT", str(tmp_path))
    monkeypatch.setenv("EGO_ROOT", str(tmp_path))
    hd, ego = common.load_config(CONFIG_HD), common.load_config(os.path.join(REPO, "accuracy_tables", "hpc", "config.all350.yaml"))
    assert hd["dataset"] == "hdepic" and "dataset" not in ego
    for k in ("ours", "gazevlm", "streamingtom", "dycoke", "fastvid", "prunevid", "dtd", "avgpool", "gazevlm2"):
        assert hd[k] == ego[k], k
    assert hd["run"] == ego["run"] and hd["budget"]["fractions"] == ego["budget"]["fractions"]
    assert hd["budget"]["tokens_per_frame"]["smolvlm"] == ego["budget"]["tokens_per_frame"]["smolvlm"] == 17
    assert hd["run"]["K"] == 96 and hd["run"]["max_frames"]["smolvlm"] == 64


# ---------------------------------------------------------------------------- the checks and the summary
def test_picks_alternate_the_types_and_fovea_records_need_17_a_frame(tmp_path):
    qs = [dict(video=f"hd_ge_{i}", type="gaze_estimation") for i in range(3)] + [dict(video=f"hd_ia_{i}", type="interaction_anticipation")
                                                                               for i in range(3)]
    assert hchecks.pick_by_questions(qs, 3) == ["hd_ge_0", "hd_ia_0", "hd_ge_1"] and hchecks.pick_every(qs, 4) == ["hd_ge_0", "hd_ia_1"]
    with open(tmp_path / "budget.jsonl", "w") as f:
        f.write(json.dumps(dict(n_frames=3, n_image_tokens=51, error=None)) + "\n")
    with open(tmp_path / "nogate.jsonl", "w") as f:
        f.write(json.dumps(dict(n_frames=2, n_image_tokens=34, error=None)) + "\n" + json.dumps(dict(n_frames=1, error="oom")) + "\n")
    assert hchecks.fovea_records(str(tmp_path)) == (2, 0)
    with open(tmp_path / "budget.jsonl", "a") as f:
        f.write(json.dumps(dict(n_frames=3, n_image_tokens=48, error=None)) + "\n")
    assert hchecks.fovea_records(str(tmp_path)) == (3, 1)


def summary_res():
    rows = []
    for t, real in (("gaze_estimation", ["P01", "P02", "P03"]), ("interaction_anticipation", ["P04", "P05", "P06"])):
        for i, v in enumerate(real * 4):
            for cond, ok in (("uni", i % 2 == 0), ("blind", i % 4 == 0)):
                rows.append(dict(condition=cond, correct=int(ok), type=t, video=v))
    return pd.DataFrame(rows)


def test_chance_uniform_minus_blind_and_the_side_table():
    res = summary_res()
    plan_ = dict(lines=[dict(row="uniform", condition="uni"), dict(row="blind", condition="blind")])
    t = hreport.chance_and_uniform_minus_blind(res, plan_, {"both": 5, "gaze_estimation": 5, "interaction_anticipation": 5}).set_index("group")
    assert t.loc["gaze_estimation", "chance_pct"] == 20 and t.loc["both", "uniform_minus_blind"] == pytest.approx(25.0)
    assert hreport.LEAK in t.loc["gaze_estimation", "note"] and t.loc["interaction_anticipation", "note"] == ""
    ours = pd.DataFrame([dict(group=g, row="uniform", budget_pct=100.0, accuracy=40.0, ci_low=37.0, ci_high=43.0, flops_pct=100.0)
                         for g in ("both", "gaze_estimation", "interaction_anticipation", "mean_of_types")])
    ref = pd.DataFrame([dict(group="all", row="uniform", budget_pct=100.0, accuracy=42.0, ci_low=40.0, ci_high=44.0, flops_pct=100.0),
                        dict(group="aea", row="uniform", budget_pct=100.0, accuracy=1.0, ci_low=0.0, ci_high=2.0, flops_pct=100.0)])
    vs = hreport.vs_egoeverything(ours, ref)
    assert len(vs) == 4 and (vs.accuracy_egoeverything == 42.0).all() and set(vs.hdepic_group) == {"both", "gaze_estimation",
                                                                                                       "interaction_anticipation",
                                                                                                       "mean_of_types"}
    assert vs.note.str.contains("no winner").all()


def test_the_mean_of_types_is_the_mean_of_their_accuracies():
    res = summary_res()
    cols = ["group", "model", "row", "budget_pct", "target_pct", "planned_pct", "K", "knob", "condition", "token_share_pct", "n", "accuracy",
            "ci_low", "ci_high", "flops_pct", "token_pct", "gflops_per_q", "frames_per_q", "frames_encoded_per_q", "prompt_tokens_per_q",
            "videos", "miss_pts", "nearest_only", "note"]
    tables = {}
    for g, acc in (("both", 50.0), ("gaze_estimation", 50.0), ("interaction_anticipation", 50.0)):
        tables[g] = pd.DataFrame([{**{c: 0.0 for c in cols}, "group": g, "model": "smolvlm", "row": "uniform", "budget_pct": 100.0,
                                   "condition": "uni", "n": 12, "accuracy": acc, "nearest_only": False,
                                   "note": "x; " + hreport.LEAK if g == "gaze_estimation" else "x"}])[cols]
    m = hreport.mean_of_types(tables, res, None, "smolvlm")
    assert list(m.columns) == cols and m.group.iloc[0] == "mean_of_types" and m.n.iloc[0] == 24
    assert m.accuracy.iloc[0] == pytest.approx(50.0) and m.ci_low.iloc[0] <= 50.0 <= m.ci_high.iloc[0] and hreport.LEAK not in m.note.iloc[0]


# ---------------------------------------------------------------------------- the step 1 to 8 jobs
def hjob(name):
    with open(os.path.join(REPO, "accuracy_tables", "hpc", name)) as f:
        return f.read()


HD_JOBS = ["hdepic_prep.SBATCH", "hdepic_frames.SBATCH", "hdepic_tests.SBATCH", "hdepic_checks.SBATCH", "hdepic_flops.SBATCH",
           "hdepic_summary.SBATCH"]
OUR_RESULTS = "$GV_ROOT/results/hdepic_gaze_smolvlm_flops"
# The one results/ path outside ours the rules allow on a writing line: section 14's table, which the summary job's report only reads.
READ_ONLY_REFERENCE = re.compile(re.escape("--reference $GV_ROOT/results/smolvlm_flops/summary_flops_by_group.csv") + r"(?=[\s\"']|$)")


def expand(value, assigned):
    """$NAME and ${NAME} replaced by the job's own NAME=... lines, until nothing the job assigns is left."""
    for _ in range(10):
        new = re.sub(r"\$\{?([A-Z_][A-Z_0-9]*)\}?", lambda m: assigned.get(m.group(1), m.group(0)), value)
        if new == value:
            break
        value = new
    return value


def ours(path):
    return path == OUR_RESULTS or path.startswith(OUR_RESULTS + "/")


@pytest.mark.parametrize("name", HD_JOBS)
def test_the_hdepic_jobs_keep_the_rules(name):
    text = hjob(name)
    assert "--no-parity" not in text and "SBATCH_ACCOUNT" not in text and "data/folds" not in text and "HF_HOME=" not in text
    assigned = dict(re.findall(r"(?m)^\s*([A-Z_][A-Z_0-9]*)=(\S+)", text))
    for line in text.splitlines():
        rest = READ_ONLY_REFERENCE.sub("", line)
        for out in re.findall(r"--out-dir\s+(\S+)", rest):                  # every write, resolved through the job's variables
            assert ours(expand(out.replace("\\$", "$"), assigned)), line
        if "--out-dir" in rest:                                              # and no other results/ path on a writing line
            for path in re.findall(r"\$GV_ROOT/results/[^\s\"']*", rest):
                assert ours(path), line
    for folder in ("qwen3_350", "qwen25_350", "smolvlm256_flops", "hpc_all", "nymeria_smolvlm"):
        assert folder not in text, folder
    if "HDEPIC_ROOT" in text:
        assert "HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}" in text


def test_only_the_gpu_jobs_hold_one_b200():
    for name in ("hdepic_checks.SBATCH", "hdepic_flops.SBATCH"):
        text = hjob(name)
        assert "#SBATCH --gres=gpu:b200:1" in text and "GPU=B200" in text and "--expect $GPU" in text and "--requeue" in text
    for name in ("hdepic_prep.SBATCH", "hdepic_frames.SBATCH", "hdepic_tests.SBATCH", "hdepic_summary.SBATCH"):
        assert "gres" not in hjob(name), name
    flops_job = hjob("hdepic_flops.SBATCH")
    assert "OUT=$GV_ROOT/results/hdepic_gaze_smolvlm_flops" in flops_job and "M=smolvlm" in flops_job
    assert "CONFIG=accuracy_tables/hpc/config.hdepic.yaml" in flops_job and "estimate --timing $TIMING/$M" in flops_job
    checks_job = hjob("hdepic_checks.SBATCH")
    assert "--config $CONFIG14 --model $M --rows uniform,blind,budget" in checks_job and "same --new $CHK/regression/$M --ref $REF" in checks_job
    assert "$H fovea --folder $CHK/on/$M" in checks_job and "--prefetch-workers 0 --out-dir $CHK/off" in checks_job
    prep = hjob("hdepic_prep.SBATCH")
    assert 'run_in_aria "python -u -m $H.project"' in prep and "$H.gate resample-check" in prep and "$H.manifest --config $CONFIG" in prep


def test_section_17_has_every_step_and_its_stop_lines():
    with open(os.path.join(REPO, "HPC_ACCURACY_RUNBOOK.md")) as f:
        section = f.read().split("\n## 17.")[1]
    for text in ("REBUILD MATCHES", "IMU MATCHES AEA", "SHEETS WRITTEN", "VOR CHECK FAILED", "FRAMES MATCH PLAN", "data check OK",
                 "SAME ANSWERS: 0 of 100", "SAME AS UNIFORM K", "FOVEATED TOKENS 17 ON EVERY KEPT FRAME", "PLAN WRITTEN",
                 "PROJECTED full run", "15%", "RUN DONE", "EVERY n IS COMPLETE", "mean_of_types", "summary_vs_egoeverything.csv"):
        assert text in section, text

"""accuracy_tables/hpc/newvideos/: gate data for new videos, and the frame loading threads. CPU only, no dataset, no
projectaria-tools. The rows go through the real label, gaze and gate code, with the repo's real gate weights."""
import csv
import json
import os
import sys
import types
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from accuracy_tables import common, summarize
from accuracy_tables.hpc.newvideos import (build_rows, check_gate, compare_records, compare_rows, draw_gaze, paths, plan,
                                           project_gaze, score_gate)
from eyehead import features
from eyehead import policy as pol
from eyehead.infer import load_ckpt
from src.common import gaze_geometry as gg
from src.loss1.dataset import parse_rates
from vlmeval import budget_rule as br
from vlmeval import egoev
from vlmeval import qwen_fovea as qf
from vlmeval import smolvlm_fovea as sm
from vlmeval import spatial as sp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(REPO, "accuracy_tables", "hpc", "eyehead_weights")
T, P, R, S = pol.T, pol.P, pol.R, pol.S


# ---------------------------------------------------------------------------- settings
def test_frame_size_and_projection_match_the_existing_pipeline():
    assert paths.FRAME_SIZE == qf.IMG
    with open(os.path.join(REPO, "configs", "temporal_analysis.yaml")) as f:
        cfg = yaml.safe_load(f)
    assert project_gaze.DEPTH_M == cfg["gaze_depth_m"] and project_gaze.ROTATE_CW90 == cfg["rotate_cw90"]


def test_config_uses_only_the_new_folders_and_keeps_the_protocol(monkeypatch):
    monkeypatch.setenv("GV_ROOT", "/g")
    monkeypatch.setenv("EGO_ROOT", "/e")
    c = common.load_config(os.path.join(REPO, "accuracy_tables", "hpc", "config.newvideos.yaml"))
    p = c["paths"]
    assert p["folds_dir"] == "/g/newvideos/nymeria/folds" and p["oof_cache"] == "/g/newvideos/nymeria/preds.csv"
    assert p["frames_local_dir"] == "/g/newvideos/frames" and p["gate_ckpt_dir"] == "/g/data/runs/eyehead"
    assert c["run"]["K"] == 96 and c["run"]["policy"] == "D+P" and c["run"]["parity"] is True
    assert c["run"]["max_frames"] == {"smolvlm": 64} and c["run"]["prefetch_workers"] == 0
    assert c["budget"]["fractions"] == [0.05, 0.1, 0.2] and c["budget"]["tokens_per_frame"]["smolvlm"] == 17


# ---------------------------------------------------------------------------- the lists
def test_groups_split_by_name_and_gate_data():
    g = plan.split_groups(["loc_b", "2023_x", "loc_a", "loc_c", "2023_x", "2022_y"], {"loc_a", "loc_never_asked"})
    assert g == dict(aea12=["loc_a"], aea13=["loc_b", "loc_c"], nymeria=["2022_y", "2023_x"])


def test_pilot_is_spread_through_the_sorted_list():
    names = [f"v{i:03d}" for i in range(325)]
    assert plan.pilot_videos(names) == ["v032", "v097", "v162", "v227", "v292"]


# ---------------------------------------------------------------------------- reading gaze
def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return str(path)


def test_read_gaze_with_one_yaw(tmp_path):
    p = write_csv(tmp_path / "g.csv", ["tracking_timestamp_us", "yaw_rads_cpf", "pitch_rads_cpf", "left_yaw_rads_cpf"],
                  [[100, 0.1, 0.2, 0], [200, "", 0.3, 0]])
    g = project_gaze.read_gaze_csv(p)
    assert g["t_us"].tolist() == [100, 200] and g["yaw"][0] == 0.1 and np.isnan(g["yaw"][1]) and g["left_yaw"] is None


@pytest.mark.parametrize("extra", [[], [0.0]])
def test_read_gaze_with_left_and_right_yaw(tmp_path, extra):
    header = ["tracking_timestamp_us", "left_yaw_rads_cpf", "right_yaw_rads_cpf", "pitch_rads_cpf"]
    header += ["yaw_rads_cpf"] if extra else []                                    # a column of zeros counts as absent
    g = project_gaze.read_gaze_csv(write_csv(tmp_path / "g.csv", header, [[100, 0.1, 0.3, 0.2, *extra]]))
    assert g["yaw"] is None and g["left_yaw"][0] == 0.1 and g["right_yaw"][0] == 0.3


def test_align_yaw_matches_by_time_and_marks_missing_rows():
    gazes = [SimpleNamespace(tracking_timestamp=timedelta(microseconds=t), yaw=y) for t, y in ((200, 0.2), (100, 0.1))]
    yaw = project_gaze.align_yaw(np.array([100.0, 200.0, 300.0]), gazes)
    assert yaw[:2].tolist() == [0.1, 0.2] and np.isnan(yaw[2])


def test_project_all_skips_samples_without_angles():
    x, y, oof = project_gaze.project_all(np.array([0.1, np.nan]), np.array([0.2, 0.2]), lambda a, b: (a + 1, b + 1, True))
    assert x[0] == 1.1 and y[0] == 1.2 and oof.tolist() == [True, False] and np.isnan(x[1]) and np.isnan(y[1])


# ---------------------------------------------------------------------------- matching the camera to the video
class RecordingCamera:
    def __init__(self, size):
        self.size, self.calls = size, []

    def get_image_size(self):
        return np.array(self.size)

    def rescale(self, new_resolution, scale, origin_offset):
        self.calls.append((list(new_resolution), scale, list(origin_offset)))
        return "rescaled"


def test_a_sensor_calibration_gets_arias_crop_rule_for_a_1408_video():
    cam = RecordingCamera((2880, 2880))
    assert project_gaze.video_calibration(cam, 1408) == "rescaled"
    assert cam.calls == [([1408, 1408], 0.5, [32.0, 32.0])]                        # a scale alone would not match the VRS


def test_a_calibration_already_at_the_video_size_is_left_alone():
    cam = RecordingCamera((1408, 1408))
    assert project_gaze.video_calibration(cam, 1408) is cam and cam.calls == []


@pytest.mark.parametrize("size,video", [((2880, 2880), 704), ((1000, 1000), 1408), ((2880, 1408), 1408)])
def test_any_other_size_pair_stops(size, video):
    with pytest.raises(ValueError, match="no crop rule"):
        project_gaze.video_calibration(RecordingCamera(size), video)


def test_video_size_reads_a_square_video_and_stops_otherwise():
    assert project_gaze.video_size(dict(orig_w=1408, orig_h=1408)) == 1408
    for meta in (dict(orig_w=1408, orig_h=704), dict(orig_w=None, orig_h=None), {}):
        with pytest.raises(ValueError, match="square"):
            project_gaze.video_size(meta)


class Identity:
    """Stands in for a transform: its inverse is itself, and applying it changes nothing."""

    def inverse(self):
        return self

    def __matmul__(self, other):
        return other


def test_the_projector_uses_the_camera_it_is_given_and_its_size(monkeypatch):
    mps = types.ModuleType("projectaria_tools.core.mps")
    mps.get_eyegaze_point_at_depth = lambda yaw, pitch, depth: np.zeros(3)
    monkeypatch.setitem(sys.modules, "projectaria_tools", types.ModuleType("projectaria_tools"))
    monkeypatch.setitem(sys.modules, "projectaria_tools.core", types.ModuleType("projectaria_tools.core"))
    monkeypatch.setitem(sys.modules, "projectaria_tools.core.mps", mps)
    cam = SimpleNamespace(get_transform_device_camera=Identity, get_image_size=lambda: np.array([1408, 1408]),
                          project=lambda point: np.array([352.0, 704.0]))
    device = SimpleNamespace(get_camera_calib=lambda label: pytest.fail("used the device's camera, not the one given"),
                             get_transform_device_cpf=Identity)
    project = gg.calibration_projector(device, cam_calib=cam)
    assert project(0.0, 0.0) == (0.5, 0.25, False)                                 # (352, 704) / 1408, then upright


def test_lookup_returns_what_was_saved_for_numpy_scalars():
    yaw, pitch = np.array([0.1, np.nan, 0.3]), np.array([0.2, 0.2, 0.4])
    project = build_rows.lookup_projector(yaw, pitch, np.array([0.5, 0.0, 0.7]), np.array([0.6, 0.0, 0.8]),
                                          np.array([False, False, True]))
    assert project(yaw[0], pitch[0]) == (0.5, 0.6, False) and project(yaw[2], pitch[2]) == (0.7, 0.8, True)


def test_load_imu_is_rebased_with_accel_before_gyro(tmp_path):
    p = tmp_path / "imu.npz"
    accel, gyro = np.ones((3, 3), dtype=np.float32), 2 * np.ones((3, 3), dtype=np.float32)
    np.savez(p, right_t_ns=np.array([5_000_000, 6_000_000, 7_000_000], dtype=np.int64), right_accel=accel, right_gyro=gyro)
    ts, arr = build_rows.load_imu(str(p))
    assert ts.tolist() == [0.0, 1000.0, 2000.0] and arr.shape == (3, 6) and arr.dtype == np.float32
    assert arr[0].tolist() == [1, 1, 1, 2, 2, 2]


# ---------------------------------------------------------------------------- the rows, through the real code
def synthetic_rows(video):
    """3 pairs at 1 fps, 10 Hz gaze, 1 kHz IMU."""
    f_ts, g_ts = np.arange(4) * 1e6, np.arange(40) * 1e5
    g_yaw, g_pit = np.linspace(-0.2, 0.2, 40), np.linspace(0.1, -0.1, 40)
    imu_ts = np.arange(4000) * 1e3
    imu_arr = np.random.default_rng(0).normal(size=(4000, 6)).astype(np.float32)
    return build_rows.pair_rows(video, f_ts, g_ts, g_yaw, g_pit, lambda y, p: (0.5 + y, 0.5 + p, False), imu_ts, imu_arr)


def test_pair_rows_counts_and_formats():
    rows = synthetic_rows("v")
    assert len(rows) == 3 and rows[0]["feat_frame_1"] == "v/feat_00000.npz" and rows[2]["feat_frame_2"] == "v/feat_00003.npz"
    r = rows[0]
    assert r["n_velocities"] == 9 and r["n_gaze_in_gap"] == 10 and r["n_oof_in_gap"] == 0 and r["n_imu_in_gap"] == 1000
    assert parse_rates(r["gaze_rates_window"]).shape == (9, 3) and parse_rates(r["imu_window"]).shape == (10, 6)
    assert np.isfinite(r["imu_gyro_mag_mean"]) and r["quad"] == build_rows.PLACEHOLDER_QUAD


@pytest.fixture
def merged(tmp_path, monkeypatch):
    """Two videos' rows merged into a group's fold files -> the test.csv path."""
    monkeypatch.setenv("GV_ROOT", str(tmp_path))
    monkeypatch.setenv("EGO_ROOT", str(tmp_path))
    for v in ("a", "b"):
        os.makedirs(os.path.dirname(paths.rows_csv(v)), exist_ok=True)
        pd.DataFrame(synthetic_rows(v)).to_csv(paths.rows_csv(v), index=False)
    return score_gate.merge_rows("g", ["a", "b"])


def test_merged_rows_read_by_the_gate_label_and_gaze_code(merged):
    folds = paths.folds_dir("g")
    clips = pol.make_clips(pol.load_labels(folds)[0])
    assert sorted(c["sequence"] for c in clips) == ["a", "b"] and all(len(c["frames"]) == 4 for c in clips)
    assert all(c["labels"].tolist() == [S, S, S] for c in clips)
    E, G, M, imu_ok = features.raw_signals(features.load_rows(merged))
    assert imu_ok.all() and M.all()                                               # all 6 IMU columns: head speed is real
    tracks, dropped = sp.gaze_tracks(folds, ["a", "b"], lambda v, k: float(k))
    assert dropped == 0 and len(tracks["a"][0]) == 30
    assert sp.gaze_fixation(tracks["a"], 0.0) is None and sp.gaze_fixation(tracks["a"], 1.0) is not None   # frame 0 has none


def test_ensemble_predictions_join_the_clips(merged):
    cks = score_gate.load_models(WEIGHTS)
    preds = score_gate.score_ensemble(merged, cks)
    assert len(preds) == 6 and preds["pred"].between(0, 3).all() and (preds["source"] == "test ensemble").all()
    clips, _ = egoev.gate_clips(paths.folds_dir("g"), test_only=False)
    labels = egoev.video_labels(clips, preds)
    assert sorted(labels) == ["a", "b"] and len(labels["a"]) == 3
    stats = check_gate.input_stats(pd.read_csv(merged), load_ckpt(os.path.join(WEIGHTS, "fold1.pt"), "cpu"))
    assert stats["eye_speed_p90"] >= stats["eye_speed_p50"] > 0 and stats["head_speed_p90"] >= stats["head_speed_p50"] > 0


def test_merge_stops_on_a_video_without_rows(merged):
    with pytest.raises(SystemExit, match="no rows"):
        score_gate.merge_rows("g", ["a", "missing"])


# ---------------------------------------------------------------------------- check 1
def test_compare_column_cases():
    assert compare_rows.compare_column(["1.00000,2.00000"], ["1.00000,2.00000"]) == dict(same_text=1, same_shape=1, max_abs_diff=0.0)
    small = compare_rows.compare_column(["1.00001"], ["1.00000"])
    assert small["same_text"] == 0 and small["same_shape"] == 1 and small["max_abs_diff"] == pytest.approx(1e-5)
    assert compare_rows.compare_column(["1,2;3,4"], ["1,2"])["same_shape"] == 0
    assert compare_rows.compare_column(["nan,1"], ["0,1"])["max_abs_diff"] == float("inf")
    assert compare_rows.compare_column(["nan,1"], ["nan,1"])["max_abs_diff"] == 0.0
    assert compare_rows.compare_column([float("nan")], [""])["same_text"] == 1     # an empty cell, read back by pandas


def known_row(frame, **over):
    base = dict(frame=frame, n_velocities=9, n_gaze_in_gap=10, n_oof_in_gap=0, n_imu_in_gap=1000, gaze_rates_window="0.1,0.2,0.3",
                imu_window="1,2,3,4,5,6", gaze_xy_window="0.5,0.5")
    return dict(base, **over)


def test_compare_video_needs_every_column_to_match():
    old = pd.DataFrame([known_row(1), known_row(2)])
    assert compare_rows.compare_video("v", old.copy(), old)[1] is True
    assert compare_rows.compare_video("v", old.head(1), old)[1] is False
    assert compare_rows.compare_video("v", pd.DataFrame([known_row(1), known_row(3)]), old)[1] is False
    assert compare_rows.compare_video("v", pd.DataFrame([known_row(1), known_row(2, n_oof_in_gap=1)]), old)[1] is False
    assert compare_rows.compare_video("v", pd.DataFrame([known_row(1), known_row(2, imu_window="1,2,3,4,5,7")]), old)[1] is False
    near = pd.DataFrame([known_row(1), known_row(2, gaze_xy_window="0.5001,0.5")])
    assert compare_rows.compare_video("v", near, old)[1] is True                   # within the last printed digit


def test_each_video_is_compared_with_the_model_behind_its_existing_predictions():
    assert compare_rows.models_for("fold2", ["m1", "m2", "m3"]) == ["m2"]
    assert compare_rows.models_for("test ensemble", ["m1", "m2", "m3"]) == ["m1", "m2", "m3"]


# ---------------------------------------------------------------------------- checks 2 and 3
def test_video_summary_counts_marks_gaze_and_limits():
    r = check_gate.video_summary("v", np.array([S, T, P, S, R]), [10, 0, 10, 10, 10], [0, 0, 1, 0, 0], [0, 1000, 1000, 1000, 1000])
    assert r["frames"] == 6 and r["marks"] == 4 and r["marks_per_min"] == 40
    assert (r[br.FRAME0], r[br.PURSUIT], r[br.SETTLE], r[br.REFIX]) == (1, 1, 1, 1)
    assert r["no_gaze_frames"] == 3 and r["windows_out_of_view"] == 1 and r["windows_no_gaze"] == 1 and r["windows_no_imu"] == 1
    assert (r["limit_5"], r["limit_10"], r["limit_20"]) == (1, 2, 5) and r["thinned_5"] and not r["thinned_20"]


def test_group_summary_adds_up():
    rows = [check_gate.video_summary("a", np.array([S, T, P, S, R]), [10] * 5, [0] * 5, [1000] * 5),
            check_gate.video_summary("b", np.array([S, S, S, S, S]), [10] * 5, [0] * 5, [1000] * 5)]
    g = check_gate.group_summary(pd.DataFrame(rows), np.array([S, T, P, S, R, S, S, S, S, S]))
    assert g["videos"] == 2 and g["frames"] == 12 and g["marks_per_min"] == pytest.approx(25)
    assert g["class_STABLE_pct"] == pytest.approx(70) and g["frames_no_gaze_pct"] == pytest.approx(100 * 2 / 12)


def test_pick_frames_skips_frame_0_and_stays_in_the_video():
    picks = draw_gaze.pick_frames(259)
    assert len(picks) == 12 and picks[0] == 1 and picks[-1] == 258
    assert draw_gaze.pick_frames(5) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------- reports
def test_compare_records_reports_each_kind_of_difference():
    a = dict(pred="A", n_image_tokens=10, n_frames=2, p=[0.6, 0.4])
    new = {("c", 1): a, ("c", 2): dict(a, pred="B"), ("c", 3): dict(a, n_image_tokens=11), ("c", 4): a}
    ref = {("c", 1): a, ("c", 2): a, ("c", 3): dict(a, p=[0.5, 0.5])}
    r = compare_records.compare_records(new, ref)
    assert (r["records"], r["missing"], r["pred_differs"], r["tokens_differ"]) == (4, 1, 1, 1)
    assert r["max_prob_diff"] == pytest.approx(0.1)


def test_gaze_missing_share():
    assert summarize.gaze_missing_pct(pd.DataFrame(dict(n_gaze_missing=[1, 1], n_frames=[10, 10]))) == 10
    assert np.isnan(summarize.gaze_missing_pct(pd.DataFrame(dict(n_frames=[10]))))
    assert np.isnan(summarize.gaze_missing_pct(pd.DataFrame(dict(n_gaze_missing=[np.nan], n_frames=[10]))))
    assert np.isnan(summarize.gaze_missing_pct(pd.DataFrame(dict(n_gaze_missing=[0], n_frames=[0]))))


# ---------------------------------------------------------------------------- frame loading threads
class FakeProcessor:
    image_mean, image_std = [0.5] * 3, [0.5] * 3

    def __call__(self, images, do_image_splitting, return_tensors):
        a = np.asarray(images[0][0].resize((sm.IMG, sm.IMG))).astype(np.float32) / 255
        return dict(pixel_values=torch.from_numpy((a.transpose(2, 0, 1) - 0.5) / 0.5)[None, None])


def test_threads_give_every_frame_exactly_what_loading_one_at_a_time_gives(tmp_path):
    rng, ip, frames = np.random.default_rng(0), FakeProcessor(), []
    for i in range(6):
        path = tmp_path / f"{i}.png"
        Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(path)
        frames.append(dict(key=("v", i), path=str(path)))
    recs = {("v", 0): "already loaded"}
    sm.load_stock_inputs(ip, frames, recs, workers=3)
    assert recs[("v", 0)] == "already loaded" and sorted(k[1] for k in recs) == list(range(6))
    for fr in frames[1:]:
        one = sm.stock_input(ip, Image.open(fr["path"]).convert("RGB"))
        assert torch.equal(recs[fr["key"]]["pv"], one["pv"]) and torch.equal(recs[fr["key"]]["x01"], one["x01"])

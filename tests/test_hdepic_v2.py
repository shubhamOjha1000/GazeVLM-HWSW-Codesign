"""Ours v2, design and Stage 1 (diagnostic dev work): the layouts' tokens and encoder GFLOPs, the rings' boxes, each ring's input against
the source pixels of its crop, the prompt's extra ring images, the dev question choice, and the jobs. CPU only, fake data."""
import os

import numpy as np
import pytest

from accuracy_tables.hpc.hdepic import v2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def job(name):
    with open(os.path.join(REPO, "accuracy_tables", "hpc", name)) as f:
        return f.read()


def test_tokens_and_encoder_gflops_of_each_layout():
    assert v2.image_gflops(378) == pytest.approx(669.9, abs=0.2)                 # section 14's stock frame, about 670
    assert v2.image_gflops(252) == pytest.approx(281.4, abs=0.2) and v2.image_gflops(84) == pytest.approx(30.0, abs=0.1)
    assert {k: v2.layout_tokens(k) for k in v2.LAYOUTS} == {"L1": 44, "L2": 61, "L3": 90, "P64": 64}
    assert v2.layout_gflops("L2") == pytest.approx(471.1, abs=0.3) and v2.layout_gflops("L3") == pytest.approx(703.9, abs=0.3)
    assert all(s % 42 == 0 and s <= 384 for rings in v2.LAYOUTS.values() for _, s, _ in rings)
    assert all(s >= 252 for rings in v2.LAYOUTS.values() for r, s, _ in rings if r == "periphery")


def test_the_rings_are_centred_on_the_gaze_and_stay_inside_the_frame():
    b = {r: box for r, _, box in v2.ring_boxes((0.5, 0.5), "L2")}
    assert b["periphery"] == (0, 0, 1344, 1344) and b["mid"] == (452, 452, 893, 893) and b["fovea"] == (525, 525, 819, 819)
    e = {r: box for r, _, box in v2.ring_boxes((0.0, 1.0), "L2")}
    assert e["mid"] == (0, 903, 441, 1344) and e["fovea"] == (0, 1050, 294, 1344)      # slid inside at the corner, never padded
    assert v2.ring_boxes(None, "L2") == v2.ring_boxes((0.5, 0.5), "L2")              # no gaze: v1's centred fallback
    sizes = {r: box[2] - box[0] for r, _, box in v2.ring_boxes((0.3, 0.6), "L3")}
    assert sizes == {"periphery": 1344, "mid": 588, "fovea": 368}                     # 1x and 2x stock scale


def test_each_rings_input_is_the_source_pixels_of_its_crop():
    torch = pytest.importorskip("torch")
    x = torch.rand(3, 200, 200)
    same = v2.ring_images(x, [("mid", 64, (10, 20, 74, 84))])[0]
    assert torch.equal(same, x[:, 20:84, 10:74])                                     # a box already that size: the pixels themselves
    small = v2.ring_images(x, [("fovea", 42, (0, 0, 84, 84))])[0]
    assert tuple(small.shape) == (3, 42, 42) and float(small.min()) >= 0 and float(small.max()) <= 1
    flat = torch.full((3, 200, 200), 0.25)
    assert torch.allclose(v2.ring_images(flat, [("p", 126, (0, 0, 200, 200))])[0], torch.full((3, 126, 126), 0.25), atol=1e-5)


def test_each_frame_keeps_one_label_and_gets_its_rings_as_images():
    from vlmeval.qwen_mcq import build_messages
    it = dict(question="q?", options=list("abcde"), frames=[dict(path="a.jpg", t=1.0), dict(path="b.jpg", t=2.0)])
    content = v2.ring_messages(build_messages, it, 3, 384 * 384)[0]["content"]
    kinds = [c["type"] for c in content]
    assert kinds.count("image") == 6 and [c["text"] for c in content if c["type"] == "text" and c["text"].startswith("Frame at")] == \
        ["Frame at 0:01:", "Frame at 0:02:"]
    assert kinds[1:5] == ["text", "image", "image", "image"] and [c["image"] for c in content if c["type"] == "image"] == ["a.jpg"] * 3 + ["b.jpg"] * 3


def test_the_dev_questions_are_outside_the_sample_on_videos_with_gaze_and_seeded(monkeypatch):
    from accuracy_tables.hpc.hdepic import data, groups
    monkeypatch.setattr(data, "gaze_csv", lambda v: None if v == "no_gaze_csv" else f"{v}.csv")
    mk = lambda i, v: dict(qid=f"fine_grained_action_recognition_{i}", parse_error=None, segments=[dict(video=v)], videos=[v])
    rows = [mk(i, f"v{i % 5}") for i in range(40)] + [mk(100, data.NO_GAZE_VIDEOS[0]), mk(101, "no_gaze_csv")]
    loaded = {groups.SAMPLE_GROUP: {"action_recognition": rows, "why_recognition": [mk(200, "v0")]}}
    sample = {"action_recognition": [f"fine_grained_action_recognition_{i}" for i in range(10)]}
    pool = v2.dev_pool(loaded, sample)
    assert len(pool) == 30 and not {q["qid"] for q in pool} & set(sample["action_recognition"])
    ids = v2.choose_dev(pool, n=10, seed=1)
    assert len(set(ids)) == 10 and ids == v2.choose_dev(pool, n=10, seed=1) and ids != v2.choose_dev(pool, n=10, seed=2)
    per = {}
    for q in pool:
        if q["qid"] in ids:
            per[q["videos"][0]] = per.get(q["videos"][0], 0) + 1
    assert sorted(per.values()) == [2, 2, 2, 2, 2]                                    # spread over the 5 videos


def test_the_v2_jobs_write_only_data_hdepic_v2_and_hold_one_b200():
    cpu, frames, gpu = job("hdepic_v2_cpu.SBATCH"), job("hdepic_v2_frames.SBATCH"), job("hdepic_v2_gpu.SBATCH")
    for text in (cpu, frames, gpu):
        assert "--no-parity" not in text and "SBATCH_ACCOUNT" not in text and "--out-dir" not in text and "results/" not in text
        assert "HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}" in text
    assert "gres" not in cpu and "gres" not in frames and "#SBATCH --gres=gpu:b200:1" in gpu and "--expect $GPU" in gpu
    assert "GPU=B200" in gpu and "--requeue" in gpu and "data/hdepic/v2/stage1" in gpu and "v2 stage1" in gpu
    assert 'run_in_aria "python -u -m $H.project --plan $GV_ROOT/data/hdepic/v2/dev_plan.json"' in cpu


def test_the_runbook_note_has_the_design_and_stage_1():
    with open(os.path.join(REPO, "HPC_ACCURACY_RUNBOOK.md")) as f:
        section = f.read().split("\n## 18.")[1]
    for text in ("Ours v2", "LAYOUTS WRITTEN", "DEV QUESTIONS", "DEV FRAMES MATCH", "STAGE 1", "the user picks one layout",
                 "hdepic_v2_cpu.SBATCH layouts", "hdepic_v2_gpu.SBATCH", "diagnostic"):
        assert text in section, text

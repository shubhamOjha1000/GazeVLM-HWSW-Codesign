"""The 10% FLOPs sweep on 4 models (diagnostic dev work): the rho rule and every model's sizes, the conditions and their rho levels, the
frames at a frame limit against common.budget_frames, the tile, the two-view prompt, the FLOPs arithmetic, the summary's FLOPs, paired
intervals and best-rho choice, the Blind-source facts, and the jobs. CPU only, fake data."""
import inspect
import os

import numpy as np
import pytest
import yaml

from accuracy_tables import common
from accuracy_tables.hpc.hdepic import sweep10 as sw
from accuracy_tables.hpc.hdepic import sweep10_views as views
from vlmeval import budget_rule as br

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLIND = "blind (question and options only)"


def job(name):
    with open(os.path.join(REPO, "accuracy_tables", "hpc", name)) as f:
        return f.read()


def test_every_models_sizes_by_the_rho_rule():
    want = {"smolvlm": [(7, 294, 49), (8, 336, 64), (9, 384, 81)], "smolvlm256": [(6, 384, 36), (7, 448, 49), (8, 512, 64)],
            "qwen2.5": [(12, 336, 144), (14, 392, 196), (16, 448, 256)], "qwen3": [(12, 384, 144), (14, 448, 196), (16, 512, 256)]}
    for m, rows in want.items():
        assert [sw.res_grid(m, r) for r in sw.RHOS] == rows, m
        g = sw.MODELS[m]
        for (k, side, t), rho in zip(rows, sw.RHOS):
            assert t >= sw.FLOOR * g["stock_tokens"] and (side % g["base"] == 0 or side == g["stock_side"])
            assert sw.avgpool_side(m, rho) == k                                   # pooling lands on the same grids
    assert {m: [sw.gazevlm2_tokens(m, r) for r in (0.5, 0.75)] for m in sw.MODELS} == \
        {"smolvlm": [42, 60], "smolvlm256": [32, 48], "qwen2.5": [128, 192], "qwen3": [128, 192]}
    assert all(sw.twoview(m)["tile_src"] == 672 and sw.twoview(m)["tokens"] == 2 * sw.MODELS[m]["stock_tokens"] for m in sw.MODELS)
    assert sw.twoview("qwen2.5")["detail"] == 2.0
    assert sw.sizes_lines("smolvlm")[0].startswith("SIZES smolvlm (SmolVLM2-2.2B, B200): stock 384 px = 81 tokens a frame")


def test_the_rho_rule_keeps_the_floor_and_takes_the_larger_on_a_tie():
    cands = [(36, "6x6"), (49, "7x7"), (64, "8x8")]
    assert sw.pick(cands, 0.5, 81) == (49, "7x7")                                 # 36 is nearer 40.5 but under the 0.50 floor
    assert sw.pick([(40, "a"), (42, "b")], 0.5, 82, floor=0.0) == (42, "b")       # 41 is a tie: the larger
    assert sw.pick([(40, "a"), (42, "b")], 0.5, 82) == (42, "b")                  # 40 is under the floor of 41
    assert sw.pick([(0.49, 1), (0.52, 2), (0.80, 3)], 0.5, 1.0) == (0.52, 2)
    with pytest.raises(ValueError):
        sw.pick([(0.2, 1)], 0.5, 1.0)


def test_the_conditions_and_their_rho_levels():
    names = [c[0] for c in sw.conditions()]
    assert len(names) == len(set(names)) == 28
    assert {"uniform", "blind", "uniform_flops", "ours_res_0.50", "ours_res_1.00", "ours_twoview", "ours_twoview_centre",
            "nogate_res_0.75", "nogate_twoview", "gazevlm_0.50", "gazevlm2_0.75", "avgpool_0.50"} <= set(names)
    assert not any(n.endswith("_1.00") for n, _, _, kind in sw.conditions() if kind == "baseline")   # at 1.00 they are uniform_flops
    ms = sw.method_settings()
    assert all(ms[row] == {0.5: f"{row}_0.50", 0.75: f"{row}_0.75", 1.0: "uniform_flops"} for row in sw.BASELINE_ROWS)
    assert set(ms["ours_res"]) == set(sw.RHOS) and ms["ours_twoview"] == {2.0: "ours_twoview"}
    assert {c for s in ms.values() for c in s.values()} == set(names) - {"uniform", "blind"}
    pytest.importorskip("torch")
    from accuracy_tables import run
    assert sw.BASELINE_ROWS == run.BASELINE_ROWS


def fake_data(starts=None, forced=None):
    rng = np.random.default_rng(3)
    n = 40
    return dict(clips=[dict(sequence="a", frame_no=np.arange(n))], metas={"a": dict(n=n)}, labels={"a": rng.integers(0, 4, n - 1)},
                uniform={"a": list(range(n))}, frame=lambda v, f: dict(key=(v, int(f))), segment_starts={"a": starts or []},
                forced={"a": forced} if forced else {})


@pytest.mark.parametrize("starts,forced", [(None, None), ([17], None), (None, [5])])
def test_frames_at_a_limit_are_the_budget_rules(starts, forced):
    d = fake_data(starts, forced)
    cfg = dict(budget=dict(tokens_per_frame=dict(smolvlm=17)), run=dict(max_frames=dict(smolvlm=64)))
    for fraction in (0.05, 0.1, 0.2):
        L = br.frame_limit(fraction, 40 * 81, 17, 40, 64)
        for use_gate in (True, False):
            want, _ = common.budget_frames(cfg, d, "smolvlm", 81, fraction, use_gate)
            assert sw.kept_positions(d, L, use_gate)["a"] == [fr["key"][1] for fr in want["a"]]
            assert [fr["key"] for fr in sw.gate_frames(d, L, use_gate)["a"]] == [fr["key"] for fr in want["a"]]
    assert sw.kept_positions(d, 99, False)["a"] == list(range(40)) and len(sw.kept_positions(d, 3, False)["a"]) == 3
    if forced:
        assert 5 in sw.kept_positions(d, 4, True)["a"]


def test_the_tile_is_half_the_source_around_the_gaze_and_inside_the_frame():
    assert views.tile_box((0.5, 0.5), 1344, 1344) == (336, 336, 1008, 1008) == views.tile_box(None, 1344, 1344)
    assert views.tile_box((0.0, 1.0), 1344, 1344) == (0, 672, 672, 1344)
    assert views.tile_box((0.9, 0.2), 1344, 1344) == (672, 0, 1344, 672)
    assert views.tile_gaze(dict(gaze=None), dict(gaze="gaze")) == ((0.5, 0.5), True)
    assert views.tile_gaze(dict(gaze=(0.2, 0.3)), dict(gaze="centre")) == ((0.5, 0.5), False)
    assert views.tile_gaze(dict(gaze=(0.2, 0.3)), dict(gaze="gaze")) == ((0.2, 0.3), False)
    assert views.n_views(("res", 336)) == 1 and views.n_views(("twoview", dict(gaze="gaze", tile=True))) == 2
    assert views.n_views(("twoview", dict(gaze="gaze", tile=False))) == 1


def test_two_views_a_frame_under_one_label():
    pytest.importorskip("torch")
    from vlmeval.qwen_mcq import build_messages
    it = dict(question="q?", options=list("abcde"), frames=[dict(path="a.jpg", t=1.0), dict(path="b.jpg", t=2.0)])
    content = views.view_messages(build_messages, it, 2, 384 * 384)[0]["content"]
    kinds = [c["type"] for c in content]
    assert kinds.count("image") == 4 and [c["text"] for c in content if c["type"] == "text" and c["text"].startswith("Frame at")] == \
        ["Frame at 0:01:", "Frame at 0:02:"]
    assert [c["image"] for c in content if c["type"] == "image"] == ["a.jpg", "a.jpg", "b.jpg", "b.jpg"]
    assert views.view_messages(build_messages, it, 1, 384 * 384) == build_messages("q?", list("abcde"), [("a.jpg", 1.0), ("b.jpg", 2.0)],
                                                                                  384 * 384)


def test_the_flops_arithmetic():
    pytest.importorskip("torch")
    units = dict(prefill=[1.0, 2.0, 0.0], gate_per_decision=10.0)
    got = sw.question_flops(units, [2], [8], 100, [10], dict(c0=5.0, o=1.0), decisions=[3])
    assert float(got[0]) == pytest.approx(5 * 10 * 3 + 200 + 1 + 2 * 25)            # gate + vision + prefill on 10 + 5 + 2 + 8 tokens
    assert sw.best_k(lambda k: 0.04 * k) == (2, pytest.approx(0.08))                  # 0.08 and 0.12 tie: the smaller
    assert sw.nearest_only(10.6) and not sw.nearest_only(10.4)


def test_the_summarys_flops_per_record():
    pytest.importorskip("torch")
    u = dict(v_stock=100.0, gate_per_decision=1.0, prefill=[0.0, 1.0, 0.0])
    assert sw.record_flops(dict(n_frames=3, n_tokens=50), dict(kind="baseline", views=2, K=8), u, 5) == pytest.approx(2 * 5 * 100 + 50)
    assert sw.record_flops(dict(n_frames=4, n_tokens=40), dict(kind="ours", v_frame=30.0), u, 10) == pytest.approx(5 * 9 + 120 + 40)
    assert sw.record_flops(dict(n_frames=0, n_tokens=12), dict(kind="blind"), u, 10) == pytest.approx(12)
    assert sw.record_flops(dict(n_frames=10, n_tokens=900), dict(kind="uniform", v_frame=100.0), u, 10) == pytest.approx(1900)


def test_a_model_table_on_fake_answers():
    pytest.importorskip("torch")
    plan = dict(units=dict(v_stock=100.0, gate_per_decision=0.0, prefill=[0.0, 1.0, 0.0]), K_candidates={"0": 4, "1": 4},
                lines=[dict(name="uniform", method="uniform", kind="uniform", rho=1.0, rho_real=1.0, K=64, tokens_per_frame=81, views=1,
                            v_frame=100.0, planned_pct=100.0),
                       dict(name="ours_res_0.50", method="ours_res", kind="ours", rho=0.5, rho_real=0.6, K=2, tokens_per_frame=49, views=1,
                            v_frame=50.0, planned_pct=24.0)])
    rec = lambda q, c, f, t: dict(qid=q, correct=c, n_frames=f, n_tokens=t, n_image_tokens=f * 10, error=None)
    answers = {"uniform": [rec(0, 1, 4, 400), rec(1, 0, 4, 400)], "ours_res_0.50": [rec(0, 1, 2, 100), rec(1, 1, 2, 100)]}
    t = {r["condition"]: r for r in sw.model_table("smolvlm", plan, answers)}
    assert t["uniform"]["flops_pct"] == pytest.approx(100) and t["ours_res_0.50"]["flops_pct"] == pytest.approx(25)
    assert t["ours_res_0.50"]["accuracy"] == pytest.approx(100) and t["ours_res_0.50"]["d_uniform"] == pytest.approx(50)
    assert t["ours_res_0.50"]["nearest_only"] and not t["uniform"]["nearest_only"]
    best = {b["method"]: b for b in sw.best_rows("smolvlm", list(t.values()))}
    assert best["ours_res"]["best_rho"] == 0.5 and best["ours_res"]["settings"] == 1


def test_paired_intervals_and_the_best_rho():
    same = {0: 1, 1: 0, 2: 1}
    assert sw.paired(same, same) == (0.0, 0.0, 0.0, 3)
    d, lo, hi, n = sw.paired({q: 1 for q in range(10)}, {q: 0 for q in range(10)})
    assert (d, lo, hi, n) == (100.0, 100.0, 100.0, 10)
    rows = [dict(rho=0.5, accuracy=40.0, flops_pct=10.4), dict(rho=0.75, accuracy=40.0, flops_pct=9.8), dict(rho=1.0, accuracy=39.0, flops_pct=10.0)]
    assert sw.best_of(rows)["rho"] == 0.75                                         # a tie: the FLOPs nearest 10%
    assert sw.best_of([dict(rho=0.5, accuracy=40.0, flops_pct=10.3), dict(rho=1.0, accuracy=40.0, flops_pct=10.3)])["rho"] == 1.0


def blind_inputs():
    dev = dict(questions=[dict(qid=i, hdepic_qid=f"fg_{i}", pseudo=f"hd18dev_{i:05d}") for i in range(4)])
    v2 = [dict(condition=BLIND, qid=i, video=f"hd18dev_{i:05d}", answer="A", correct=i % 2, p=[0.1 * (i + 1), 0.2, 0.2, 0.2, 0.1])
          for i in range(4)]
    s18q = [dict(qid=0, type="action_recognition", hdepic_qid="other_0"), dict(qid=1, type="why_recognition", hdepic_qid="other_1")]
    s18b = [dict(qid=0, correct=1, p=[0.9, 0.025, 0.025, 0.025, 0.025]), dict(qid=1, correct=0, p=[0.5, 0.5, 0, 0, 0])]
    return v2, dev, {i: "A" for i in range(4)}, s18b, s18q, {"other_0"}


def test_the_blind_source_facts():
    facts, problems = sw.blind_source(*blind_inputs(), BLIND)
    assert not problems and facts["records"] == 4 and facts["correct"] == 2 and facts["s18_records"] == 1 and facts["same_p"] == 0
    v2, dev, ans, s18b, s18q, sample = blind_inputs()
    assert sw.blind_source(v2, dev, ans, s18b, s18q, {"fg_2"}, BLIND)[1]                      # a dev question in the sample
    copied = [dict(r, p=s18b[0]["p"]) for r in v2]
    assert sw.blind_source(copied, dev, ans, s18b, s18q, sample, BLIND)[1]                    # section 18's probabilities
    assert sw.blind_source([dict(r, video="hd18_00000") for r in v2], dev, ans, s18b, s18q, sample, BLIND)[1]
    assert sw.blind_source(v2[:3], dev, ans, s18b, s18q, sample, BLIND)[1]


def test_records_compared_by_question():
    a = [dict(qid=0, pred="A", p=[0.6, 0.1, 0.1, 0.1, 0.1], n_frames=2, n_image_tokens=98, n_tokens=300, error=None)]
    assert sw.same_by_qid(a, a) == (1, 0, 0.0)
    b = [dict(a[0], p=[0.5, 0.2, 0.1, 0.1, 0.1])]
    assert sw.same_by_qid(a, b)[1:] == (1, pytest.approx(0.1))
    assert sw.same_by_qid(a, [])[1] == 1


def test_the_gpu_code_names_and_checks():
    pytest.importorskip("torch")
    from accuracy_tables.hpc.hdepic import sweep10_gpu as g
    name, spec = g.view_condition(dict(kind="ours", K=7, method="ours_res", side=294, tokens_per_frame=49))
    assert (name, spec) == ("ours gate L=7 + whole frame 294 px (49 tokens)", ("res", 294))
    name, spec = g.view_condition(dict(kind="nogate", K=3, method="nogate_twoview", side=384, tile_src=672, tokens_per_frame=162))
    assert name.startswith("no gate L=3 (slot middles) + two views") and "gaze tile 672 of 1344 px at 384 px (162 tokens)" in name
    assert spec == ("twoview", dict(gaze="gaze", tile=True))
    assert g.view_condition(dict(kind="ours", K=3, method="ours_twoview_centre", side=384, tile_src=672, tokens_per_frame=162))[1] == \
        ("twoview", dict(gaze="centre", tile=True))
    u = dict(prefill=[0.0, 1.0, 0.0])
    got = dict(ve=200.0, pf=10.0, frames=2, n_tokens=10, n_image_tokens=98)
    assert g.check_counted(got, 100.0, u, 49, False)[2] and not g.check_counted(got, 100.0, u, 50, False)[2]
    assert g.check_counted(dict(got, ve=203.0), 100.0, u, None, True)[2] and not g.check_counted(dict(got, ve=205.0), 100.0, u, None, True)[2]


def test_v2s_extractor_takes_the_sweeps_folder():
    from accuracy_tables.hpc.hdepic import v2
    assert inspect.signature(v2.extract_video).parameters["folder"].default is None


def test_the_config_reads_section_18s_candidates_into_sweep10():
    with open(os.path.join(REPO, sw.CONFIG)) as f:
        cfg = yaml.safe_load(f)
    assert cfg["dataset"] == "hdepic18" and cfg["run"]["K"] == 64 and cfg["run"]["max_frames"] == dict(smolvlm=64, smolvlm256=64)
    assert all("data/hdepic/sweep10" in v for k, v in cfg["paths"].items() if v and k not in ("gate_ckpt_dir", "raw_download_dir"))


def test_the_jobs_hold_their_gpu_type_and_write_only_sweep10():
    b200, a100, cpu, frames = (job(f"hdepic_sweep10_{x}.SBATCH") for x in ("b200", "a100", "cpu", "frames"))
    for text in (b200, a100, cpu, frames):
        assert "--no-parity" not in text and "SBATCH_ACCOUNT" not in text and "HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}" in text
        assert all("results/" not in line for line in text.splitlines() if not line.lstrip().startswith(("#", "smolvlm", "qwen")))
    assert "#SBATCH --gres=gpu:b200:1" in b200 and "GPU=B200" in b200 and "smolvlm256)" in b200 and "qwen" not in b200.split("set -euo")[1]
    assert "#SBATCH --gres=gpu:a100:1" in a100 and "GPU=A100" in a100 and "qwen3)" in a100 and "smolvlm)" not in a100
    assert "gres" not in cpu and "gres" not in frames
    for text in (b200, a100):
        assert '--comment="preemption=yes;requeue=true"' in text and "#SBATCH --requeue" in text and "#SBATCH --signal=B:USR1@900" in text
        assert "--expect $GPU" in text and "--same-as $D/gpu_types.txt" in text and "WANT=sweep10-$TAG-$MODE" in text
        assert "same --new $D/checks/regression/$M --ref $REF --rows uniform,blind,budget" in text and "D=$GV_ROOT/data/hdepic/sweep10/$M" in text
        assert "GPU MEAN: last %d min" in text and "while sleep 1800" in text and 'chmod a-w "$D/plan.json"' in text
        assert "repair_results $D/answers" in text and "$G checks --model $M" in text and "$G run --model $M" in text
    assert "results/smolvlm_flops/smolvlm; RQ=20" in b200 and "results/smolvlm256_flops/smolvlm256; RQ=10" in b200
    assert "results/qwen25_flops/qwen2.5; RQ=10" in a100 and "results/qwen3_flops/qwen3; RQ=10" in a100
    assert 'run_in_aria "python -u -m $H.project --plan $GV_ROOT/data/hdepic/sweep10/plan.json"' in cpu


def test_the_runbook_note():
    with open(os.path.join(REPO, "HPC_ACCURACY_RUNBOOK.md")) as f:
        section = f.read().split("\n## 18.")[1]
    for text in ("10% FLOPs sweep", "BLIND SOURCE", "VOR CHECK OK", "DEV FRAMES MATCH", "data check OK: 940 questions", "CHECKS DONE",
                 "PLAN WRITTEN", "RUN DONE", "SUMMARY DONE", "hdepic_sweep10_b200.SBATCH", "hdepic_sweep10_a100.SBATCH", "GPU LOW",
                 "diagnostic", "no winner"):
        assert text in section, text

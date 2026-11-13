"""SmolVLM on all 350 videos with every row matched on total FLOPs (runbook section 14): the FLOPs arithmetic, the plan, the run's jobs,
the report and the job files. CPU only, no dataset, no model: the encoder, the LLM and the FLOP counts of real questions are faked, and
everything around them is the real code."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

import test_smolvlm350 as shared
from test_smolvlm350 import fake_smolvlm  # noqa: F401  (a fixture)
from accuracy_tables import calibrate as cal
from accuracy_tables import common, flops, run
from accuracy_tables import flops_calibrate as fc
from accuracy_tables import flops_run as fr
from accuracy_tables.hpc.all350 import flops_report as rep
from accuracy_tables.hpc.all350 import prepare
from accuracy_tables.models import build_messages
from eyehead.model import EyeHeadGate

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------- the arithmetic
def test_count_flops_counts_a_matrix_product():
    lin = torch.nn.Linear(4, 3)
    n, out = flops.count_flops(lambda: lin(torch.zeros(2, 4)))
    assert n == 2 * 2 * 4 * 3 and out.shape == (2, 3)


def test_one_gate_decision_is_one_window_through_one_model():
    gate = EyeHeadGate(("eye", "head", "cos"), d=4, h=8, k=3, dropout=0.0).eval()
    one = flops.measure_gate(gate, 20)
    with torch.no_grad():
        two, _ = flops.count_flops(lambda: gate(torch.zeros(2, 3, 20), torch.ones(2, 20, dtype=torch.bool)))
    assert one > 0 and two == 2 * one


C = [2e8, 3.2e9, 2e5]                       # a fake LLM: FLOPs = c0 + c1 n + c2 n^2


def test_the_prefill_quadratic_is_exact_and_checked():
    pts = [(n, float(flops.prefill(C, n))) for n in fc.PREFILL_LENGTHS]
    got = flops.fit_prefill(pts)
    assert all(abs(float(flops.prefill(got, n)) - f) <= 1e-9 * f for n, f in pts)
    cubic = [(n, f + float(n) ** 3) for n, f in pts]                              # about 0.1% off at 1024 and 4096 tokens
    with pytest.raises(SystemExit, match="quadratic"):
        flops.fit_prefill(cubic)
    with pytest.raises(ValueError):
        flops.fit_prefill(pts[:3])


def test_the_prompt_is_the_blind_prompt_plus_an_intro_and_a_label_per_frame():
    p = flops.fit_prompt([(f, 30 + 9 * f) for f in (1, 4, 16, 64)])
    assert p["c0"] == pytest.approx(30) and p["o"] == pytest.approx(9) and p["max_residual"] < 1e-9
    n = flops.prompt_tokens([50, 50], [0, 2], [0, 34], p)
    assert n.tolist() == pytest.approx([50, 50 + 30 + 18 + 34])                    # no frame: no intro either


def test_question_flops_add_the_gate_the_views_and_the_prefill():
    units = dict(v_stock=10.0, v_fovea=2.0, prefill=[1.0, 0.0, 0.0], gate_per_decision=3.0)
    assert flops.question_flops(units, [4], "v_stock", [7], views=2).tolist() == [2 * 4 * 10 + 1]
    assert flops.question_flops(units, [4], "v_fovea", [7], decisions=[5]).tolist() == [flops.GATE_MODELS * 3 * 5 + 4 * 2 + 1]


def test_uniform_k_never_encodes_more_frames_than_a_video_has():
    assert flops.encoded([3, 100, 0], 10).tolist() == [3, 10, 0]
    units = dict(v_stock=1.0, v_fovea=1.0, prefill=[0.0, 0.0, 0.0], gate_per_decision=0.0)
    q = dict(blind=np.array([0, 0]), video_frames=np.array([3, 100]))
    p = dict(c0=0.0, o=0.0)
    assert flops.uniform_share(q, units, p, 100.0, 10, 0) == pytest.approx(13 / 100)
    assert flops.uniform_share(q, units, p, 100.0, 10, 0, views=2) == pytest.approx(26 / 100)


def test_the_closest_frame_count_wins_and_a_tie_takes_the_smaller():
    assert flops.choose_k(lambda k: k / 100, 0.025, 64) == (2, 0.02)
    assert flops.choose_k(lambda k: k / 100, 0.031, 64) == (3, 0.03)
    assert flops.choose_k(lambda k: k / 100, 0.9, 64) == (64, 0.64)


# ---------------------------------------------------------------------------- one row's inputs at another frame count
def protocol():
    return common._merge(common.DEFAULTS, dict(paths=dict(out_dir="x")))


@pytest.mark.parametrize("row,section", [("streamingtom", "streamingtom"), ("dycoke", "dycoke"), ("dtd", "dtd"), ("avgpool", "avgpool"),
                                         ("gazevlm2", "gazevlm2"), ("gazevlm", "gazevlm"), ("prunevid", "prunevid"),
                                         ("fastvid", "fastvid"), ("uniform_flops", None)])
def test_a_planned_row_changes_only_its_frame_count_and_its_budget_parameter(row, section):
    cfg = protocol()
    c = fc.with_knob(cfg, row, 0.3, 7)
    assert c["run"] == dict(cfg["run"], K=7) and cfg["run"]["K"] == 96
    for k in cfg:
        if k not in ("run", section):
            assert c[k] == cfg[k], k
    if section == "dycoke":
        assert c["dycoke"]["settings"] == [dict(window=32, prune=0.3)]
    if section == "prunevid":
        assert c["prunevid"]["settings"] == [dict(tau=0.8, beta=0.3)] and c["prunevid"]["gamma"] == cfg["prunevid"]["gamma"]
    if section == "dtd":
        assert c["dtd"]["settings"] == [dict(threshold=0.3, absolute=False)]


def test_with_uniform_spreads_k_frames_over_each_video():
    frames = {"v": [dict(key=("v", i)) for i in range(50)], "w": [dict(key=("w", i)) for i in range(3)]}
    data = dict(videos=["v", "w"], metas={"v": dict(n=50), "w": dict(n=3)}, frame=lambda v, f: frames[v][f], uniform=None, qs=None)
    d = fc.with_uniform(data, 5)
    assert [f["key"][1] for f in d["uniform"]["v"]] == common.uniform_indices(50, 5) and len(d["uniform"]["w"]) == 3
    assert data["uniform"] is None and d["qs"] is None                            # the original is untouched


def test_the_check_questions_take_turns_between_nymeria_and_aea():
    qs = pd.DataFrame(dict(video=["loc_b", "loc_a", "2023_x", "2023_x", "2023_y"], qid=[5, 4, 9, 2, 7]))
    assert fc.verify_qids(qs, 3) == [2, 4, 7] and fc.verify_qids(qs, 9) == [2, 4, 7, 5]


UNITS = dict(v_stock=6.7e11, v_fovea=1.3e11, prefill=C, gate_per_decision=1000.0)


@pytest.mark.parametrize("row,vision,prefill,ok", [
    ("dtd", 1.01, 1.0, True), ("dtd", 1.03, 1.0, False), ("dtd", 0.99, 1.0, False), ("dtd", 1.0, 1.001, False),
    ("uniform", 1.001, 1.0, False), ("uniform_flops", 1.0, 1.0, True), ("budget", 1.00001, 1.0, True), ("gazevlm2", 1.0, 1.0, True)])
def test_the_flops_check_allows_only_the_baselines_own_work(row, vision, prefill, ok):
    unit = UNITS["v_fovea"] if row == "budget" else UNITS["v_stock"]
    views = 2 if row == "gazevlm2" else 1
    got = dict(ve=vision * views * 4 * unit, pf=prefill * float(flops.prefill(C, 500)), frames=4, n_tokens=500)
    r_ve, r_pf, good = fc.check(row, got, UNITS)
    assert good is ok and r_ve == pytest.approx(vision) and r_pf == pytest.approx(prefill)


# ---------------------------------------------------------------------------- the whole plan on 2 small videos
N = 100
VIDEOS = {"2023_m": 3, "loc_a": 2}
TRUE = dict(v_stock=6.7e11, v_fovea=1.3e11)


class SmolVLM(shared.SmolVLM):
    """The test adapter with a blind prompt of known length. Adapters find their backends by class name, so it keeps this one."""
    def blind_ids(self, messages):
        return torch.zeros((1, blind_len(messages)))


def blind_len(messages):
    return 20 + len(messages[0]["content"][-1]["text"]) % 7


def fake_prompt_len(ad, question, options, frames, image_tokens):
    return blind_len(build_messages(question, options, [], 0)) + 30 + 9 * len(frames) + sum(image_tokens)


def fake_counted(extra=None):
    """counted() without a model: what flops.py says, plus the baselines' own work (0.5%, or `extra` for one row)."""
    extra = extra or {}

    def counted(job, qid, decoder, tmp):
        item = next(it for it in job["items"] if int(it["qid"]) == qid)
        row, frames = job["row"], len(item["frames"])
        unit = TRUE["v_fovea"] if row in flops.FOVEA_ROWS else TRUE["v_stock"]
        own = extra.get(row, 1.005 if row in run.BASELINE_ROWS else 1.0)
        n = 40 + 3 * frames
        return dict(ve=own * (2 if row in flops.TWO_VIEWS else 1) * frames * unit, pf=float(flops.prefill(C, n)), frames=frames,
                    n_tokens=n)
    return counted


@pytest.fixture
def world(tmp_path, monkeypatch, fake_smolvlm):
    monkeypatch.setenv("GV_ROOT", str(tmp_path / "g"))
    monkeypatch.setenv("EGO_ROOT", str(tmp_path / "e"))
    frames = {v: shared.write_frames(str(tmp_path), v, N) for v in VIDEOS}
    rng = np.random.default_rng(0)
    qid, rows = 0, []
    for v, n in VIDEOS.items():
        for i in range(n):
            rows.append(dict(qid=qid, video=v, question=f"q{qid} " * (qid + 1), options=list("abcde"), answer="A"))
            qid += 1
    qs = pd.DataFrame(rows)
    data = dict(qs=qs, videos=list(VIDEOS), test_videos=set(), metas={v: dict(n=N) for v in VIDEOS}, gate={},
                uniform={v: [frames[v][f] for f in common.uniform_indices(N, 96)] for v in VIDEOS},
                clips=[dict(sequence=v, frame_no=np.arange(N)) for v in VIDEOS], labels={v: rng.integers(0, 4, N - 1) for v in VIDEOS},
                frame=lambda v, f: frames[v][f])
    cfg = common.load_config(prepare.CONFIG)
    cfg["paths"]["out_dir"] = str(tmp_path / "out")
    monkeypatch.setitem(flops.MEASURE, "smolvlm", dict(prefill=lambda ad, n: float(flops.prefill(C, n)), prompt_len=fake_prompt_len))
    monkeypatch.setattr(flops, "measure_gate_ckpt", lambda path: 1000)
    monkeypatch.setattr(run, "check_parity", lambda *a: {})
    from system_perf import probe
    monkeypatch.setattr(probe, "find_decoder", lambda model: None)
    monkeypatch.setattr(fc, "counted", fake_counted())
    out_model = tmp_path / "out" / "smolvlm"
    os.makedirs(out_model)
    return dict(cfg=cfg, data=data, ad=SmolVLM(cfg=cfg), out_model=str(out_model), monkeypatch=monkeypatch)


@pytest.fixture
def plan(world):
    return fc.make_plan(world["ad"], world["cfg"], world["data"], world["out_model"], log=lambda *a: None)


def test_the_plan_has_every_row_at_every_budget(plan):
    t = fc.targets_table(plan)
    assert len(t) == 2 + 3 * (2 + 1 + 8)
    assert set(t.row) == {"uniform", "blind", "budget", "nogate", "uniform_flops"} | set(run.BASELINE_ROWS)
    for row in ["budget", "nogate", "uniform_flops", *run.BASELINE_ROWS]:
        assert sorted(t[t.row == row].budget_pct) == [5, 10, 20], row
    assert t[t.row == "uniform"].predicted_pct.tolist() == [100] and t[t.row == "blind"].target_pct.tolist() == [0]


def test_the_units_and_the_prompt_come_from_the_counts(plan):
    u = plan["units"]
    assert u["v_stock"] == pytest.approx(TRUE["v_stock"]) and u["v_fovea"] == pytest.approx(TRUE["v_fovea"])
    assert u["gate_per_decision"] == 1000 and u["prefill"] == pytest.approx(C, rel=1e-4)
    assert plan["prompt"]["c0"] == pytest.approx(30) and plan["prompt"]["o"] == pytest.approx(9)
    assert plan["share"] == pytest.approx(17 / 81) and plan["K_reference"] == 64
    assert plan["videos"] == {v: dict(frames=N, decisions=N - 1) for v in VIDEOS}


def test_ours_sets_the_targets_and_every_other_row_aims_at_them(plan):
    t = fc.targets_table(plan).set_index(["row", "budget_pct"])
    for b in (5.0, 10.0, 20.0):
        target = plan["targets"][f"{b:g}"] * 100
        assert t.loc[("budget", b), "target_pct"] == pytest.approx(target)
        assert t.loc[("budget", b), "predicted_pct"] == pytest.approx(target)
        for row in ["uniform_flops", *run.BASELINE_ROWS]:
            assert t.loc[(row, b), "target_pct"] == pytest.approx(target), row
        assert np.isnan(t.loc[("nogate", b), "target_pct"])
    assert t.loc[("budget", 5.0), "target_pct"] < t.loc[("budget", 10.0), "target_pct"] < t.loc[("budget", 20.0), "target_pct"]


def test_k_prime_is_the_frame_count_closest_to_the_target(plan, world):
    """Recomputed here from the plan's own numbers: one more or one fewer frame is never closer."""
    t = fc.targets_table(plan)
    q = world["data"]["qs"].sort_values(["video", "qid"])
    blind = np.array([blind_len(build_messages(r.question, list(r.options), [], 0)) for r in q.itertuples()])
    qq = dict(blind=blind, video_frames=np.full(len(q), N))
    units = dict(plan["units"])
    fu = np.full(len(q), 64)
    total = flops.question_flops(units, fu, "v_stock", flops.prompt_tokens(blind, fu, 81 * fu, plan["prompt"])).sum()
    for r in t[t.row == "uniform_flops"].itertuples():
        share = lambda k: flops.uniform_share(qq, units, plan["prompt"], total, k, 81)
        assert abs(share(r.K) - r.target_pct / 100) <= min(abs(share(k) - r.target_pct / 100) for k in (max(1, r.K - 1), min(64, r.K + 1)))


def test_the_baselines_keep_ours_tokens_per_frame_and_share_one_k_per_budget(plan):
    t = fc.targets_table(plan)
    for b in (5, 10, 20):
        at = t[(t.budget_pct == b) & t.row.isin(fc.SINGLE_VIEW)]
        assert at.K.nunique() == 1, b
        assert at.set_index("row").loc["streamingtom", "token_share_pct"] == pytest.approx(100 * 17 / 81)
        assert at.set_index("row").loc["avgpool", "token_share_pct"] == pytest.approx(100 * 16 / 81)              # 4 x 4
    g = t[t.row == "gazevlm2"]                          # an even count, 2 views of the same size: 16 and 18 are as near 17
    assert g.token_share_pct.nunique() == 1 and round(g.token_share_pct.iloc[0] * 81 / 100) in (16, 18)
    for b in (5, 10, 20):
        two, one = t[(t.row == "gazevlm2") & (t.budget_pct == b)].K.iloc[0], t[(t.row == "dtd") & (t.budget_pct == b)].K.iloc[0]
        assert two <= one


def test_every_planned_condition_names_its_frame_count_and_was_checked(plan):
    for ln in plan["lines"]:
        if ln["row"] in ("uniform_flops", *run.BASELINE_ROWS):
            assert f"K={ln['K']}" in ln["condition"], ln
        if ln["row"] != "blind":
            assert ln["verified"] and all(v["ok"] for v in ln["verified"]), ln
    assert [ln["condition"] for ln in plan["lines"] if ln["row"] == "uniform"] == [run.uniform_name(64)]
    assert json.loads(json.dumps(plan)) == plan


def test_the_counts_are_kept_per_frame_count(plan, world):
    ks = {ln["K"] for ln in plan["lines"] if ln["row"] in fc.SINGLE_VIEW}
    assert sorted(os.listdir(world["out_model"])) == sorted(f"flops_counts_K{k}.jsonl" for k in ks)


def test_a_baseline_doing_more_than_its_encoder_and_llm_stops_the_plan(world):
    world["monkeypatch"].setattr(fc, "counted", fake_counted(dict(prunevid=1.05)))
    with pytest.raises(SystemExit, match="FLOPS DIFFER"):
        fc.make_plan(world["ad"], world["cfg"], world["data"], world["out_model"], log=lambda *a: None)


def test_the_run_scores_exactly_the_planned_conditions(plan, world):
    ad, cfg, data = world["ad"], world["cfg"], world["data"]
    ref, planned = fr.reference_jobs(ad, cfg, data, plan), fr.plan_jobs(ad, cfg, data, plan)
    scored = [(j["row"], c) for j in ref + planned for c in j["conds"]]
    assert set(scored) == {(ln["row"], ln["condition"]) for ln in plan["lines"]} and len(scored) == len(set(scored))
    k_of = {(ln["row"], ln["condition"]): ln["K"] for ln in plan["lines"]}
    for j in planned:
        (cond,) = j["conds"]
        assert len(j["items"]) == sum(VIDEOS.values())
        assert all(len(it["frames"]) == min(k_of[j["row"], cond], N) for it in j["items"]), j["row"]
    u = [j for j in planned if j["row"] == "uniform_flops"]
    assert u and all(next(iter(j["conds"])).startswith("uniform K=") for j in u)


def test_the_run_refuses_a_plan_whose_names_differ(plan, world):
    bad = json.loads(json.dumps(plan))
    next(ln for ln in bad["lines"] if ln["row"] == "dtd")["condition"] = "something else"
    with pytest.raises(SystemExit, match="Stop and tell the user"):
        fr.plan_jobs(world["ad"], world["cfg"], world["data"], bad)


def test_the_run_refuses_a_changed_config(tmp_path):
    cfg_path, plan_path = tmp_path / "c.yaml", tmp_path / "p.json"
    cfg_path.write_text("run: {}\n")
    plan_path.write_text(json.dumps(dict(model="smolvlm", config_sha256="not this one")))
    with pytest.raises(SystemExit, match="changed since the plan"):
        fr.main(["--config", str(cfg_path), "--plan", str(plan_path), "--out-dir", str(tmp_path)])


def test_flops_are_only_planned_for_measured_models(tmp_path):
    """Section 15 measures qwen2.5 and qwen3 too, so the refusal is checked on internvl, which still has no measuring functions."""
    with pytest.raises(SystemExit, match="only measured"):
        fc.main(["--config", "x", "--model", "internvl", "--out-dir", str(tmp_path)])


# ---------------------------------------------------------------------------- the report
def records(plan, data, drop=0):
    """One record per planned condition and question: frames and tokens as the rows would give them, every answer 'A' but one."""
    out = []
    for ln in {(ln["row"], ln["condition"]): ln for ln in plan["lines"]}.values():
        frames = {"uniform": 64, "blind": 0, "budget": 15, "nogate": 15}.get(ln["row"], ln["K"])
        sent = max(0, frames - drop) if ln["row"] in run.BASELINE_ROWS else frames
        tokens = {"uniform": 81, "uniform_flops": 81, "blind": 0}.get(ln["row"], 17) * sent
        for q in data["qs"].itertuples():
            out.append(dict(row=ln["row"], condition=ln["condition"], qid=int(q.qid), video=q.video, answer="A",
                            pred="B" if q.qid == 0 else "A", correct=int(q.qid != 0), n_frames=sent, n_image_tokens=tokens,
                            n_tokens=60 + 9 * sent + tokens, error=None))
    return pd.DataFrame(out)


def test_the_report_charges_every_encoded_frame_and_compares_with_uniform_k(plan, world):
    res = rep.add_flops(records(plan, world["data"], drop=2), plan)
    base = res[res.row == "dtd"].iloc[0]
    k = next(ln["K"] for ln in plan["lines"] if ln["row"] == "dtd" and ln["condition"] == base.condition)
    assert base.frames_encoded == min(k, N) and base.n_frames == max(0, k - 2)              # dropped frames were still encoded
    two = res[res.row == "gazevlm2"].iloc[0]
    u = plan["units"]
    assert two.flops == pytest.approx(2 * two.frames_encoded * u["v_stock"] + float(flops.prefill(u["prefill"], two.n_tokens)))
    b = res[res.row == "budget"].iloc[0]
    assert b.flops == pytest.approx(flops.GATE_MODELS * 1000 * (N - 1) + 15 * u["v_fovea"] + float(flops.prefill(u["prefill"], b.n_tokens)))
    ng = res[res.row == "nogate"].iloc[0]
    assert ng.flops == pytest.approx(15 * u["v_fovea"] + float(flops.prefill(u["prefill"], ng.n_tokens)))
    stats = rep.condition_stats(res, run.uniform_name(64)).set_index(["row", "condition"])
    assert stats.loc[("uniform", run.uniform_name(64)), "flops_pct"] == pytest.approx(100)
    blind = stats.loc[("blind", run.BLIND)]
    assert blind.flops_pct < 1 and blind.token_pct == 0 and blind.n == sum(VIDEOS.values())


def test_the_report_lists_each_row_and_budget_in_a_fixed_order_and_finds_gaps(plan, world):
    res = rep.add_flops(records(plan, world["data"]), plan)
    stats = rep.condition_stats(res, run.uniform_name(64))
    table = rep.table_for(stats, plan, "all", "smolvlm")
    assert table.row.tolist()[:5] == ["uniform", "blind", "budget", "budget", "budget"] and table.row.iloc[-1] == "gazevlm2"
    assert list(table.budget_pct[table.row == "budget"]) == [5, 10, 20]
    assert rep.problems(table, plan, dict(all=5, aea=0), dict(all=stats)) == []
    short = rep.condition_stats(res[~((res.row == "dtd") & (res.qid == 0))], run.uniform_name(64))
    found = rep.problems(rep.table_for(short, plan, "all", "smolvlm"), plan, dict(all=5), dict(all=short))
    assert found and all("dtd" in f and "n = 4" in f for f in found)
    extra = pd.concat([stats, stats.head(1).assign(row="dtd", condition="stale")], ignore_index=True)
    assert rep.problems(table, plan, dict(all=5), dict(all=extra)) == ["all: results for a condition not in the plan: dtd stale"]


def test_the_report_refuses_videos_the_plan_doesnt_have(plan, world):
    res = records(plan, world["data"]).assign(video="other")
    with pytest.raises(SystemExit, match="doesn't have"):
        rep.add_flops(res, plan)


# ---------------------------------------------------------------------------- the jobs
def job(name):
    with open(os.path.join(REPO, "accuracy_tables", "hpc", name)) as f:
        return f.read()


@pytest.mark.parametrize("name", ["smolvlm_flops.SBATCH", "smolvlm_flops_summary.SBATCH", "smolvlm_flops_tests.SBATCH"])
def test_jobs_write_only_to_their_own_results_and_keep_the_rules(name):
    text = job(name)
    for folder in ("hpc_all", "hpc_budget", "nymeria_smolvlm", "hpc350_check", "hpc350_pilot", "smolvlm_350", "qwen3_350", "qwen25_350"):
        assert f"results/{folder}" not in text, folder
    assert "--no-parity" not in text and "seed_results" not in text and "data/folds" not in text


def test_the_gpu_job_stays_on_one_gpu_type_and_can_resume():
    text = job("smolvlm_flops.SBATCH")
    assert "--gres=gpu:b200:1" in text and "GPU=B200" in text and text.count("--expect $GPU") == 2
    assert "--same-as $OUT/$M/gpu_types.txt" in text and "--dependency=singleton" in text and "--requeue" in text
    assert "OUT=$GV_ROOT/results/smolvlm_flops" in text and "accuracy_tables.flops_calibrate" in text
    assert "accuracy_tables.flops_run" in text and "--keep-going" in text and "repair_results" in text


def test_the_summary_and_the_tests_jobs():
    assert "flops_report --out-dir $OUT" in job("smolvlm_flops_summary.SBATCH")
    assert "gpu_smolvlm-flops-full_" in job("smolvlm_flops_summary.SBATCH")
    assert "tests/test_smolvlm_flops.py" in job("smolvlm_flops_tests.SBATCH")

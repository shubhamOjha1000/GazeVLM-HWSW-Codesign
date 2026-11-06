"""SmolVLM on all 350 videos with every row at 5, 10 and 20%: frame loading threads on the baseline rows, the knobs, the identity check, the
GPU check and the report. CPU only, no dataset, no model: the encoder and the LLM are faked, and everything around them is the real code."""
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from accuracy_tables import calibrate as cal
from accuracy_tables import common, models, run
from accuracy_tables.hpc import check_gpu
from accuracy_tables.hpc.all350 import check_identity, prepare, report
from vlmeval import avgpool as ap
from vlmeval import dtd, dycoke as dy, fastvid as fv
from vlmeval import gazevlm as gz
from vlmeval import gazevlm2 as gv2
from vlmeval import prunevid as pv
from vlmeval import smolvlm_fovea as sm
from vlmeval import streamingtom as stm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS = ["uniform", "blind", "budget", "avgpool", "gazevlm", "gazevlm2", "streamingtom", "dycoke", "fastvid", "prunevid", "dtd"]


# ---------------------------------------------------------------------------- a fake SmolVLM
class FakeProcessor:
    """Resizes to SmolVLM's 384 px and scales to [-1, 1], as its image processor's output looks."""
    image_mean, image_std = [0.5] * 3, [0.5] * 3

    def __call__(self, images, do_image_splitting, return_tensors):
        pil = images[0][0].resize((sm.IMG, sm.IMG))
        x = torch.as_tensor(np.array(pil), dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1
        return dict(pixel_values=x[None, None])


def fake_tokens(rec):
    """81 tokens that depend on the image, as encode_uniform's do."""
    return torch.nn.functional.adaptive_avg_pool2d(rec["pv"], 9).flatten(1).T


@contextmanager
def fake_capture(target, score=None):
    """Attention for the saliency: 729 patches for StreamingTOM, 81 tokens for FastVID, as the backends read them."""
    n = 729 if score is stm.received_attention else 81
    yield [torch.arange(n, dtype=torch.float32)[None]]


@pytest.fixture
def fake_smolvlm(monkeypatch):
    vision = SimpleNamespace(encoder=SimpleNamespace(layers=[SimpleNamespace(self_attn=None)]))
    monkeypatch.setattr(sm, "parts", lambda model: (vision, None))
    monkeypatch.setattr(sm, "encode_uniform", lambda model, rec, side, mean, std: fake_tokens(rec))
    monkeypatch.setattr(sm, "build_inputs", lambda model, proc, msgs, feats: (dict(feats=feats), torch.zeros((1, 9))))
    monkeypatch.setattr(sm, "letter_probs", lambda model, inputs, letter_ids: probs_from(inputs["feats"]))
    monkeypatch.setattr(stm, "capture_saliency", fake_capture)


def probs_from(feats):
    v = float(torch.cat(list(feats)).float().mean())
    return np.array([v, 1 - v, 0.0, 0.0, 0.0])


PROC = SimpleNamespace(image_processor=FakeProcessor())


def write_frames(folder, video, n, side=56):
    rng, out = np.random.default_rng(sum(map(ord, video))), []
    for i in range(n):
        path = os.path.join(folder, f"{video}_{i}.png")
        Image.fromarray(rng.integers(0, 255, (side, side, 3), dtype=np.uint8)).save(path)
        out.append(dict(key=(video, i), path=path, t=float(i), gaze=None if i == 1 else (0.3, 0.6), gaze_trace=[]))
    return out


def items_on(folder):
    """2 questions on each of 2 videos."""
    out = []
    for video in ("a", "b"):
        frames = write_frames(str(folder), video, 4)
        out += [dict(video=video, question="q", options=list("abcde"), answer="A", frames=frames) for _ in range(2)]
    return [dict(it, qid=i) for i, it in enumerate(out)]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def without_time(path):
    return [{k: x for k, x in r.items() if k != "sec"} for r in read_jsonl(path)]


# ---------------------------------------------------------------------------- threads on each baseline row
BASELINES = {
    "avgpool": (ap.run_questions, ap.SmolVLM, dict(x=dict(side=2), y=dict(side=3)), 1),
    "gazevlm": (gz.run_questions, gz.SmolVLM, dict(x=dict(rho=0.3), y=dict(rho=0.6)), 1),
    "gazevlm2": (gv2.run_questions, gv2.SmolVLM, dict(x=dict(tokens=4), y=dict(tokens=8)), 2),
    "streamingtom": (stm.run_questions, stm.SmolVLM, dict(x=dict(ratio=0.05), y=dict(ratio=0.2)), 1),
    "dycoke": (dy.run_questions, dy.SmolVLM, dict(x=dict(window=2, prune=0.9), y=dict(window=2, prune=0.5)), 1),
    "fastvid": (fv.run_questions, fv.SmolVLM, dict(x=dict(r=0.05), y=dict(r=0.2)), 1),
    "prunevid": (pv.run_questions, pv.SmolVLM, dict(x=dict(tau=0.8, beta=0.05), y=dict(tau=0.8, beta=0.2)), 1),
    "dtd": (dtd.run_questions, dtd.SmolVLM, dict(x=dict(threshold=0.9, absolute=False), y=dict(threshold=0.5, absolute=False)), 1),
}


def counted_loads(monkeypatch):
    """Counts each frame's input being made, whichever thread makes it."""
    calls, real = [], sm.stock_input

    def wrapper(ip, pil):
        calls.append(1)
        return real(ip, pil)

    monkeypatch.setattr(sm, "stock_input", wrapper)
    return calls


@pytest.mark.parametrize("row", sorted(BASELINES))
def test_threads_change_nothing_and_load_each_view_once(row, tmp_path, fake_smolvlm, monkeypatch):
    run_questions, backend, conds, views = BASELINES[row]
    calls = counted_loads(monkeypatch)
    items = items_on(tmp_path)
    build = lambda *a: []
    be = backend(None, PROC, [0])
    run_questions(be, items, conds, str(tmp_path / "off.jsonl"), build)
    assert len(calls) == 2 * 4 * views                                             # 2 videos, 4 frames, once for all 2 conditions
    calls.clear()
    be = backend(None, PROC, [0])
    run_questions(be, items, conds, str(tmp_path / "on.jsonl"), build, prefetch_workers=3)
    assert len(calls) == 2 * 4 * views
    assert be.ready == {}                                                          # every loaded frame was used
    assert without_time(tmp_path / "on.jsonl") == without_time(tmp_path / "off.jsonl")
    assert len(read_jsonl(tmp_path / "on.jsonl")) == 4 * len(conds)


def test_two_view_threads_keep_each_views_own_image(tmp_path, fake_smolvlm):
    """The two views of a frame share its key, so a lookup by key would give the crop the whole frame."""
    be = gv2.SmolVLM(None, PROC, [0])
    frame = write_frames(str(tmp_path), "a", 1)[0]
    gv2.prefetch_views(be, [frame], [0.5], str(tmp_path), workers=2)
    crop = gv2.crop_file(str(tmp_path), frame, 0.5)
    assert sorted(be.ready) == sorted([frame["path"], crop])
    assert not torch.equal(be.ready[frame["path"]]["pv"], be.ready[crop]["pv"])
    with Image.open(crop) as im:
        assert im.size == (28, 28)


def test_a_two_view_rerun_resumes_with_threads(tmp_path, fake_smolvlm):
    run_questions, backend, conds, _ = BASELINES["gazevlm2"]
    out = str(tmp_path / "on.jsonl")
    items = items_on(tmp_path)
    run_questions(backend(None, PROC, [0]), items[:1], conds, out, lambda *a: [], prefetch_workers=2)
    run_questions(backend(None, PROC, [0]), items, conds, out, lambda *a: [], prefetch_workers=2)
    assert len(read_jsonl(out)) == 4 * len(conds) and len({(r["condition"], r["qid"]) for r in read_jsonl(out)}) == 4 * len(conds)


def test_only_smolvlm_baselines_take_threads():
    assert run.baselines_without_threads(["smolvlm"], ROWS, 8) == []
    assert run.baselines_without_threads(["phi", "qwen2.5", "smolvlm"], ["dtd"], 8) == ["phi", "qwen2.5"]
    assert run.baselines_without_threads(["phi"], ["uniform", "budget"], 8) == []
    assert run.baselines_without_threads(["phi"], ["dtd"], 0) == []
    with pytest.raises(NotImplementedError, match="threads"):
        gz.Backend().prefetch([], 2)


def test_the_adapter_hands_its_workers_to_a_baseline_row(monkeypatch):
    got = []
    monkeypatch.setattr(dtd, "run_questions", lambda *a, **k: got.append(k["prefetch_workers"]))
    jobs = run.row_jobs(SmolVLM(cfg=dict(run=dict(prefetch_workers=8))), protocol(), small_data(), ["dtd"])
    jobs[0]["score"]([], {}, "out")
    assert got == [8]


# ---------------------------------------------------------------------------- the knobs
class SmolVLM(models.SmolVLM):
    """The real adapter without a model. Adapters find their backends by class name, so it keeps this one."""
    def __init__(self, model_id=None, cfg=None):
        super().__init__(model_id, cfg)
        self.model, self.proc, self.letter_ids = None, PROC, [0]

    def load(self):
        return self

    def free(self):
        pass


def protocol():
    return common._merge(common.DEFAULTS, dict(paths=dict(out_dir="x")))


def small_data(frames=None):
    frames = frames or {v: [dict(key=(v, i), path="", t=float(i), gaze=(0.3, 0.6)) for i in range(70)] for v in ("a", "b")}
    qs = pd.DataFrame(dict(qid=[0, 1, 2], video=["a", "a", "b"], question="q", options=[list("abcde")] * 3, answer="A"))
    return dict(qs=qs, uniform=frames)


@pytest.fixture
def calibrated(tmp_path, monkeypatch, fake_smolvlm):
    """calibrate.py --fractions through its real main, on 2 small videos -> dict(cfg, written, targets, record, data)."""
    monkeypatch.setenv("GV_ROOT", str(tmp_path / "g"))
    monkeypatch.setenv("EGO_ROOT", str(tmp_path / "e"))
    frames = {v: write_frames(str(tmp_path), v, 6) for v in ("a", "b")}
    data = small_data(frames)
    monkeypatch.setattr(common, "load_data", lambda cfg: data)
    monkeypatch.setattr(cal, "ADAPTERS", dict(smolvlm=SmolVLM))
    out, written = tmp_path / "out", tmp_path / "out" / "config_baselines.yaml"
    os.makedirs(out)
    cal.main(["--config", prepare.CONFIG, "--model", "smolvlm", "--fractions", "--out-dir", str(out), "--write", str(written),
              "--prefetch-workers", "2"])
    with open(out / "smolvlm" / "calibration.json") as f:
        record = json.load(f)
    return dict(cfg=common.load_config(str(written)), written=written, targets=pd.read_csv(out / "smolvlm" / "targets.csv"), record=record, data=data)


def test_fractions_target_the_protocols_budgets_with_nothing_scored(calibrated):
    assert calibrated["record"]["targets"] == [0.05, 0.1, 0.2] and calibrated["record"]["ours"] is None
    t = calibrated["targets"]
    assert sorted(t[t.row == "avgpool"].target_pct) == [5, 10, 20]
    assert t[t.row == "uniform"].target_pct.tolist() == [100] and t[t.row == "blind"].target_pct.tolist() == [0]
    assert sorted(t[t.row == "budget"].condition) == sorted(f"budget rule {b}% (D+P gate, fovea k=1)" for b in (5, 10, 20))
    assert len(t) == 2 + 3 + 8 * 3 and set(t.row) == set(ROWS)


def test_the_written_config_changes_only_the_budget_parameters(calibrated):
    c, d = calibrated["cfg"], common.DEFAULTS
    assert c["run"]["K"] == d["run"]["K"] and c["run"]["policy"] == d["run"]["policy"] and c["budget"]["fractions"] == d["budget"]["fractions"]
    assert c["ours"] == d["ours"] and c["budget"]["tokens_per_frame"]["smolvlm"] == 17
    for section, keys in (("streamingtom", ("tau", "knn", "beta", "released_dpc")), ("fastvid", ("c", "tau", "ignore", "d", "p", "beta")),
                          ("prunevid", ("gamma", "knn", "min_tokens")), ("gazevlm", ("sigma", "r_min", "thumb", "gaze")), ("gazevlm2", ("alpha", "beta"))):
        assert all(c[section][k] == d[section][k] for k in keys)
    assert {s["window"] for s in c["dycoke"]["settings"]} == {32} and {s["tau"] for s in c["prunevid"]["settings"]} == {0.8}
    assert {s["absolute"] for s in c["dtd"]["settings"]} == {False}
    assert c["run"]["prefetch_workers"] == 0                                       # the job passes its workers each time


def test_each_condition_in_the_targets_is_what_a_run_scores(calibrated):
    """The report finds a row's results by this name."""
    t = calibrated["targets"]
    jobs = run.row_jobs(SmolVLM(cfg=calibrated["cfg"]), calibrated["cfg"], calibrated["data"], [r for r in ROWS if r != "budget"])
    scored = {job["row"]: set(job["conds"]) for job in jobs}
    for row, names in scored.items():
        assert names == set(t[t.row == row].condition), row


def test_exact_rows_land_where_the_token_rule_says(calibrated):
    t = calibrated["targets"].set_index(["row", "target_pct"]).predicted_pct
    assert t["avgpool", 5] == pytest.approx(100 * 4 / 81) and t["avgpool", 10] == pytest.approx(100 * 9 / 81)
    assert t["avgpool", 20] == pytest.approx(100 * 16 / 81)
    for f, tokens in ((5, 4), (10, 8), (20, 16)):
        assert t["gazevlm2", f] == pytest.approx(100 * tokens / 81) and t["streamingtom", f] == pytest.approx(100 * tokens / 81)


def test_budgets_below_a_rows_floor_share_one_condition(calibrated):
    """The thumbnail and the smallest gaze region alone pass 5% and 10% on SmolVLM's 9 x 9 grid."""
    t = calibrated["targets"]
    g = t[t.row == "gazevlm"].set_index("target_pct").condition
    assert g[5] == g[10] and g[10] != g[20]
    assert calibrated["targets"][calibrated["targets"].row == "gazevlm"].predicted_pct.min() > 8


def test_a_prediction_matches_what_a_scored_row_measures(calibrated, tmp_path):
    """DTD on the small videos, scored by the real loop, must spend what calibrate.py predicted."""
    t = calibrated["targets"]
    line = t[(t.row == "dtd") & (t.target_pct == 10)].iloc[0]
    items = common.make_items(calibrated["data"]["qs"], calibrated["data"]["uniform"])
    out = str(tmp_path / "dtd.jsonl")
    dtd.run_questions(dtd.SmolVLM(None, PROC, [0]), items, {"c": dict(threshold=float(line.knob), absolute=False)}, out, lambda *a: [])
    assert np.mean([r["n_image_tokens"] for r in read_jsonl(out)]) / (6 * 81) == pytest.approx(line.predicted_pct / 100)


def test_fractions_stop_on_phi_until_its_row_breaks_are_counted():
    with pytest.raises(SystemExit, match="row breaks"):
        cal.main(["--config", "x", "--model", "phi", "--fractions", "--write", "y"])


def test_threads_are_refused_on_a_model_without_them(capsys):
    with pytest.raises(SystemExit):
        cal.main(["--config", "x", "--model", "qwen2.5", "--points", "1", "--prefetch-workers", "2", "--write", "y"])
    assert "threads aren't built" in capsys.readouterr().err


# ---------------------------------------------------------------------------- the identity check
def test_the_identity_config_keeps_every_token(tmp_path, monkeypatch):
    monkeypatch.setenv("GV_ROOT", "/g")
    monkeypatch.setenv("EGO_ROOT", "/e")
    path = tmp_path / "x" / "identity.yaml"
    check_identity.write_config(prepare.CONFIG, str(path))
    c = common.load_config(str(path))
    assert ap.pool_side(c["avgpool"]["budget"][0], 81, 9) == 9 and c["run"]["K"] == 96
    tokens = [torch.randn(81, 4, generator=torch.Generator().manual_seed(i)) for i in range(5)]
    assert dtd.dtd_plans(tokens, c["dtd"]["settings"][0]["threshold"], False) == [list(range(81))] * 5
    s = c["dycoke"]["settings"][0]
    assert dy.ttm_plans(tokens, s["window"], s["prune"]) == [list(range(81))] * 5
    plan, _ = stm.ctr_plan(tokens[1], tokens[0], torch.ones(81), stm.budget(c["streamingtom"]["ratio"][0], 81))
    assert [p for p, _ in plan] == list(range(81)) and not any(m for _, m in plan)


def test_identity_compare_each_kind():
    ref = dict(qid=0, pred="A", n_image_tokens=81, n_frames=1, p=[0.5, 0.5])
    assert check_identity.problems(check_identity.compare([ref], [ref])) == 0
    r = check_identity.compare([ref], [dict(ref, pred="B"), dict(ref, qid=1), dict(ref, n_image_tokens=80), dict(ref, n_frames=2)])
    assert (r["answers"], r["missing"], r["tokens"], r["frames"]) == (1, 1, 1, 1) and check_identity.problems(r) == 4
    assert check_identity.problems(check_identity.compare([ref], [])) == 1
    assert check_identity.compare([ref], [dict(ref, p=[0.4, 0.6])])["max_prob_diff"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------- the GPU type
def test_a_job_stops_on_another_gpu_type_and_records_its_own(tmp_path):
    assert not check_gpu.wrong_type("NVIDIA B200", "B200") and not check_gpu.wrong_type("NVIDIA b200 x", "B200")
    assert check_gpu.wrong_type("NVIDIA L40S", "B200") and not check_gpu.wrong_type("NVIDIA L40S", None)
    path = tmp_path / "a" / "gpu_types.txt"
    check_gpu.record(str(path), "NVIDIA B200")
    check_gpu.record(str(path), "NVIDIA B200")
    assert report.gpu_types(str(tmp_path), ["a", "missing"]) == {"a": ["NVIDIA B200"]}


# ---------------------------------------------------------------------------- the report
def table_of(rows):
    return pd.DataFrame([dict(group="all", model="smolvlm", row=r, condition=c, n=7, budget_pct=b) for r, c, b in rows])


TARGETS = pd.DataFrame([dict(model="smolvlm", row="uniform", target_pct=100.0, knob=np.nan, predicted_pct=np.nan, condition="u"),
                        dict(model="smolvlm", row="gazevlm", target_pct=5.0, knob=0.01, predicted_pct=10.6, condition="floor"),
                        dict(model="smolvlm", row="gazevlm", target_pct=10.0, knob=0.01, predicted_pct=10.6, condition="floor"),
                        dict(model="smolvlm", row="gazevlm", target_pct=20.0, knob=0.6, predicted_pct=20.0, condition="g20")])


def test_a_condition_two_budgets_share_is_listed_under_both():
    t = report.add_targets(table_of([("gazevlm", "floor", 10.6), ("gazevlm", "g20", 19.9), ("uniform", "u", 100.0)]), TARGETS)
    assert t.row.tolist() == ["uniform", "gazevlm", "gazevlm", "gazevlm"]                 # the fixed order, not by accuracy
    assert t.target_pct.tolist() == [100, 5, 10, 20]
    assert t.miss_pts.round(1).tolist() == [0, 5.6, 0.6, -0.1]
    assert t.nearest_only.tolist() == [False, True, True, False]


def test_completeness_counts_conditions_not_lines():
    table = table_of([("gazevlm", "floor", 10.6), ("gazevlm", "g20", 19.9), ("uniform", "u", 100.0)])
    want = TARGETS.condition.nunique()
    assert want == 3 and report.n_problems(table, dict(all=7), ["smolvlm"], want) == []
    assert report.n_problems(table.head(2), dict(all=7), ["smolvlm"], want) == ["all smolvlm: 2 conditions, expected 3"]
    assert report.n_problems(table, dict(all=8), ["smolvlm"], want)[0] == "all smolvlm floor: n = 7, expected 8"


def test_conditions_are_checked_against_the_targets_both_ways():
    table = table_of([("gazevlm", "floor", 10.6), ("gazevlm", "stale", 19.9), ("uniform", "u", 100.0)])
    found = report.condition_problems(table, TARGETS, dict(all=7, nymeria=0))
    assert found == ["all smolvlm: no results for g20", "all smolvlm: results for a condition not in the targets: stale"]


def test_the_row_order_covers_every_row_of_this_run():
    assert set(report.ROW_ORDER) == set(ROWS) and report.ROW_ORDER[:3] == report.ROW_FILES
    assert "can't show whether the gate itself helps" in report.NOTE


# ---------------------------------------------------------------------------- the jobs
def job(name):
    with open(os.path.join(REPO, "accuracy_tables", "hpc", name)) as f:
        return f.read()


@pytest.mark.parametrize("name", ["smolvlm350.SBATCH", "smolvlm350_tests.SBATCH", "smolvlm350_summary.SBATCH"])
def test_jobs_write_only_to_their_own_results_and_keep_the_rules(name):
    text = job(name)
    for folder in ("hpc_all", "hpc_budget", "nymeria_smolvlm", "hpc_350"):
        assert f"OUT=$GV_ROOT/results/{folder}" not in text and f"--out-dir $GV_ROOT/results/{folder}" not in text
    assert "--no-parity" not in text and "nogate" not in text and "compare_budget" not in text and "seed_results" not in text


def test_the_gpu_job_stays_on_one_gpu_type_scores_every_row_and_can_resume():
    text = job("smolvlm350.SBATCH")
    assert "--gres=gpu:b200:1" in text and "GPU=B200" in text and text.count("--expect $GPU") == 4
    assert "--dependency=singleton" in text and "--requeue" in text and "ROWS=uniform,blind,budget,$BASELINES" in text
    assert "--fractions" in text and "--prefetch-workers $W" in text
    for row in ROWS[3:]:
        assert row in text.split("BASELINES=")[1].splitlines()[0]
    assert "--ref $GV_ROOT/results/nymeria_smolvlm/$M" in text              # only read, to compare


def test_the_knobs_stay_out_of_the_shared_config():
    with open(prepare.CONFIG) as f:
        assert set(yaml.safe_load(f)) == {"paths", "run", "budget"}

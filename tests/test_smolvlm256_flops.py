"""SmolVLM2-256M with every row matched on total FLOPs (runbook section 16). CPU only, no dataset, no model: the encoder, the LLM and the
FLOP counts of real questions are faked, everything around them is the real code. Covers the 256M's geometry and fovea, its backends
(threads on and off give the same records, gaze lands on its own 8 x 8 grid), a whole FLOPs plan, the checks module, the job files,
and that SmolVLM2-2.2B's path is as section 14 ran it."""
import contextlib
import json
import os
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import test_qwen3350 as q3t
import test_smolvlm350 as shared
from accuracy_tables import common, flops, models, run
from accuracy_tables import flops_calibrate as fc
from accuracy_tables import flops_run as fr
from accuracy_tables.hpc.all350 import flops_report as rep
from accuracy_tables.hpc.all350 import prepare, shards
from accuracy_tables.hpc.all350 import smolvlm256_checks as chk
from accuracy_tables.models import build_messages
from vlmeval import avgpool as ap
from vlmeval import dtd, dycoke as dy, fastvid as fv
from vlmeval import gazevlm as gz
from vlmeval import gazevlm2 as gv2
from vlmeval import prunevid as pv
from vlmeval import smolvlm256_fovea as s2
from vlmeval import smolvlm_fovea as sm
from vlmeval import streamingtom as stm

REPO = shared.REPO
CONFIG = os.path.join(REPO, "accuracy_tables", "hpc", "config.smolvlm256.yaml")
M = "smolvlm256"


# ---------------------------------------------------------------------------- the 256M's geometry and fovea
def test_the_geometry_is_the_pinned_configs():
    assert (s2.IMG, s2.PATCH, s2.SHUFFLE, s2.BASE, s2.GRID, s2.TOP, s2.CELLS, s2.STOCK_TOKENS) == (512, 16, 4, 64, 8, 256, 2, 64)
    assert s2.MODEL_ID == "HuggingFaceTB/SmolVLM2-256M-Video-Instruct" and len(s2.REVISION) == 40
    cfg = SimpleNamespace(vision_config=SimpleNamespace(image_size=512, patch_size=16), scale_factor=4)
    ip = SimpleNamespace(max_image_size={"longest_edge": 512})
    assert s2.geometry_problems(cfg, ip, SimpleNamespace(image_seq_len=64)) == []
    bad = SimpleNamespace(vision_config=SimpleNamespace(image_size=384, patch_size=14), scale_factor=3)
    found = s2.geometry_problems(bad, SimpleNamespace(max_image_size={"longest_edge": 384}), SimpleNamespace(image_seq_len=81))
    assert len(found) == 5
    with pytest.raises(RuntimeError, match="Stop and tell the user"):
        s2.check_geometry(SimpleNamespace(config=bad), SimpleNamespace(image_processor=ip, image_seq_len=64))


@pytest.mark.parametrize("gaze", [(0.5, 0.5), (0.1, 0.1), (0.9, 0.2), (0.3, 0.95), (0.0, 1.0)])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_the_fovea_tiles_the_frame_with_the_same_tokens_for_every_gaze(gaze, k):
    cells = s2.foveated_cells(*gaze, k)
    assert len(cells) == s2.tokens_at(k) == {1: 19, 2: 34, 3: 49}[k]
    assert set(cells[:, 2].tolist()) <= {64, 256} and int((cells[:, 2] ** 2).sum()) == 512 * 512
    cover = np.zeros((512, 512), int)
    for y, x, s in cells.tolist():
        cover[y:y + s, x:x + s] += 1
    assert (cover == 1).all()                                                     # no gap, no overlap


def test_the_full_resolution_cell_is_the_one_nearest_the_gaze():
    for gaze, corner in (((0.1, 0.1), (0, 0)), ((0.9, 0.1), (0, 256)), ((0.1, 0.9), (256, 0)), ((0.8, 0.7), (256, 256))):
        fine = s2.foveated_cells(*gaze, 1)
        fine = fine[fine[:, 2] == 64]
        assert (fine[:, 0].min(), fine[:, 1].min()) == corner and len(fine) == 16


def test_patches_and_their_positions_follow_the_model_grid():
    cells = np.array([[0, 0, 64], [256, 256, 256]])
    img = torch.rand(3, 512, 512)
    p = s2.cells_to_patches(img, cells, [0.5] * 3, [0.5] * 3)
    assert p.shape == (32, 3, 16, 16)
    assert torch.allclose(p[1], (img[:, 0:16, 16:32] - 0.5) / 0.5)              # the 64 px cell's second patch, untouched
    h, w = s2.patch_positions(cells)
    assert h[:16].tolist() == np.repeat(np.arange(4), 4).tolist() and w[:16].tolist() == np.tile(np.arange(4), 4).tolist()
    assert h[16:].tolist() == (16 + np.repeat(np.arange(4), 4) * 4 + 1).tolist()  # 4 x 4 blocks: offset (4 - 1) // 2 = 1
    assert len(s2.uniform_cells()) == 64 and s2.same_tokens_size(19) == 256


def test_the_layout_the_plan_reads_is_the_one_scored():
    conds = dict(f=("fovea", dict(k=1, gaze="gaze")), u=("uniform", 512))
    cells, side, used = s2.frame_layout(conds, "f", dict(gaze=(0.2, 0.7)))
    assert side == 512 and used == (0.2, 0.7) and np.array_equal(cells, s2.foveated_cells(0.2, 0.7, 1))
    assert s2.frame_layout(conds, "f", dict(gaze=None))[2] is None                 # no gaze: the centre, counted
    assert len(s2.frame_layout(conds, "u", {})[0]) == 64


def test_the_two_point_two_billion_geometry_is_unchanged():
    assert (sm.IMG, sm.PATCH, sm.BASE, sm.TOP, sm.GRID, sm.SHUFFLE) == (384, 14, 42, 126, 9, 3)
    assert len(sm.foveated_cells(0.3, 0.6, 1)) == 17
    assert common.DEFAULTS["run"]["max_frames"] == {"smolvlm": 64} and common.DEFAULTS["budget"]["tokens_per_frame"] == {"smolvlm": 17}


SECTION_14_FILES = {
    "accuracy_tables/hpc/config.all350.yaml": "7fdfe63a5dcf5930917eea673e610d839f098e0ba1d1ac0f3187fd2585fb1014",
    "accuracy_tables/hpc/smolvlm_flops.SBATCH": "472169c2d9229e1b13b909025f9fbe54cf6c034d6f16a92942a148070a46716d",
    "accuracy_tables/hpc/smolvlm_flops_summary.SBATCH": "3c20a172c5f279b67ae829b6d097d4aaea11770f21fb2ee7d3045921513eaa46",
    "accuracy_tables/hpc/smolvlm_flops_tests.SBATCH": "50ee0e45a025f451bbedc306bcf3d7b664d9b576ea8bd968a0c9d19876ef5486",
}


@pytest.mark.parametrize("path", sorted(SECTION_14_FILES))
def test_sections_14_and_15s_config_and_jobs_are_byte_for_byte_what_ran(path):
    assert shards.file_sha256(os.path.join(REPO, path)) == SECTION_14_FILES[path]


# ---------------------------------------------------------------------------- the config and the adapter
def test_the_256ms_config_adds_only_its_own_entries(monkeypatch):
    monkeypatch.setenv("GV_ROOT", "/g")
    monkeypatch.setenv("EGO_ROOT", "/e")
    mine, theirs = common.load_config(CONFIG), common.load_config(prepare.CONFIG)
    assert mine["run"]["max_frames"] == {"smolvlm": 64, "smolvlm256": 64} and mine["ours"]["smolvlm256_k"] == [1, 2, 3]
    assert mine["budget"]["tokens_per_frame"] == dict(theirs["budget"]["tokens_per_frame"], smolvlm256=19)
    for k in ("paths", "gazevlm", "streamingtom", "dycoke", "fastvid", "prunevid", "dtd", "avgpool", "gazevlm2"):
        assert mine[k] == theirs[k], k
    assert mine["run"]["K"] == theirs["run"]["K"] and mine["ours"]["rings"] == theirs["ours"]["rings"]
    assert dict(mine["budget"], tokens_per_frame=None) == dict(theirs["budget"], tokens_per_frame=None)


def test_the_adapter_is_the_256ms_and_finds_its_own_backends():
    a = models.ADAPTERS[M]
    assert a is models.SmolVLM256 and a.stock_tokens == 64 and a.default_ids == (s2.MODEL_ID,)
    assert a.loads_on_threads and a.baselines_load_on_threads and models.ADAPTERS["smolvlm"].stock_tokens == 81
    assert a.download_kwargs() == dict(revision=s2.REVISION, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    assert models.SmolVLM.download_kwargs() == {}
    ad = a(cfg={})
    assert ad.fovea_settings(dict(smolvlm256_k=[1, 2, 3])) == [dict(k=1), dict(k=2), dict(k=3)] and ad.setting_label(dict(k=1)) == "k=1"
    assert ad.stock_condition() == ("uniform", 512) and ad.fovea_condition(dict(k=1)) == ("fovea", dict(k=1, gaze="gaze"))
    for mod in (gz, stm, dy, fv, pv, dtd, ap, gv2):
        assert hasattr(mod, "SmolVLM256"), mod.__name__


def test_the_frame_limits_are_the_proposers():
    cfg = common._merge(common.DEFAULTS, dict(run=dict(max_frames=dict(smolvlm256=64)), budget=dict(tokens_per_frame=dict(smolvlm256=19))))
    from accuracy_tables.hpc.all350 import measure_tokens
    t = measure_tokens.limits_table({"smolvlm": 17.0, M: 19.0}, cfg)
    assert t[t.model == M].limit.tolist() == [10, 21, 43] and t[t.model == "smolvlm"].limit.tolist() == [15, 30, 60]


# ---------------------------------------------------------------------------- a fake 256M
class FakeProcessor:
    """Resizes to the 256M's 512 px and scales to [-1, 1], as its image processor's output looks."""
    image_mean, image_std = [0.5] * 3, [0.5] * 3
    max_image_size = {"longest_edge": 512}

    def __call__(self, images, do_image_splitting, return_tensors):
        pil = images[0][0].resize((s2.IMG, s2.IMG))
        x = torch.as_tensor(np.array(pil), dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1
        return dict(pixel_values=x[None, None])


PROC = SimpleNamespace(image_processor=FakeProcessor())


def fake_tokens(rec):
    """64 tokens that depend on the image, as encode_uniform's do."""
    return torch.nn.functional.adaptive_avg_pool2d(rec["pv"], s2.GRID).flatten(1).T


@contextlib.contextmanager
def fake_capture(target, score=None):
    """Attention for the saliency: 1,024 patches for StreamingTOM, 64 tokens for FastVID, as the 256M's backends read them."""
    n = 1024 if score is stm.received_attention else 64
    yield [torch.arange(n, dtype=torch.float32)[None]]


@pytest.fixture
def fake_256(monkeypatch):
    vision = SimpleNamespace(encoder=SimpleNamespace(layers=[SimpleNamespace(self_attn=None)]))
    monkeypatch.setattr(s2, "parts", lambda model: (vision, None))
    monkeypatch.setattr(s2, "encode_uniform", lambda model, rec, side, mean, std: fake_tokens(rec))
    monkeypatch.setattr(s2, "build_inputs", lambda model, proc, msgs, feats: (dict(feats=feats), torch.zeros((1, 9))))
    monkeypatch.setattr(s2, "letter_probs", lambda model, inputs, letter_ids: shared.probs_from(inputs["feats"]))
    monkeypatch.setattr(stm, "capture_saliency", fake_capture)


BASELINES = {
    "avgpool": (ap.run_questions, ap.SmolVLM256, dict(x=dict(side=2), y=dict(side=4)), 1),
    "gazevlm": (gz.run_questions, gz.SmolVLM256, dict(x=dict(rho=0.3), y=dict(rho=0.6)), 1),
    "gazevlm2": (gv2.run_questions, gv2.SmolVLM256, dict(x=dict(tokens=4), y=dict(tokens=8)), 2),
    "streamingtom": (stm.run_questions, stm.SmolVLM256, dict(x=dict(ratio=0.05), y=dict(ratio=0.3)), 1),
    "dycoke": (dy.run_questions, dy.SmolVLM256, dict(x=dict(window=2, prune=0.9), y=dict(window=2, prune=0.5)), 1),
    "fastvid": (fv.run_questions, fv.SmolVLM256, dict(x=dict(r=0.05), y=dict(r=0.3)), 1),
    "prunevid": (pv.run_questions, pv.SmolVLM256, dict(x=dict(tau=0.8, beta=0.05), y=dict(tau=0.8, beta=0.3)), 1),
    "dtd": (dtd.run_questions, dtd.SmolVLM256, dict(x=dict(threshold=0.9, absolute=False), y=dict(threshold=0.5, absolute=False)), 1),
}


@pytest.mark.parametrize("row", sorted(BASELINES))
def test_each_backend_runs_on_the_256m_module_and_threads_change_nothing(row, tmp_path, fake_256, monkeypatch):
    run_questions, backend, conds, views = BASELINES[row]
    calls, real = [], s2.stock_input
    monkeypatch.setattr(s2, "stock_input", lambda ip, pil: calls.append(1) or real(ip, pil))
    be = backend(None, PROC, [0])
    assert be.sm is s2 and be.side == 8 and be.max_pixels == 512 * 512
    items = shared.items_on(tmp_path)
    run_questions(be, items, conds, str(tmp_path / "off.jsonl"), lambda *a: [])
    assert len(calls) == 2 * 4 * views
    calls.clear()
    be = backend(None, PROC, [0])
    run_questions(be, items, conds, str(tmp_path / "on.jsonl"), lambda *a: [], prefetch_workers=3)
    assert len(calls) == 2 * 4 * views and be.ready == {}
    assert shared.without_time(tmp_path / "on.jsonl") == shared.without_time(tmp_path / "off.jsonl")
    assert len(shared.read_jsonl(tmp_path / "on.jsonl")) == 4 * len(conds)


def test_streamingtoms_saliency_pools_the_256ms_4_x_4_patches(fake_256, tmp_path):
    fr_ = shared.write_frames(str(tmp_path), "a", 1)[0]
    st = stm.SmolVLM256(None, PROC, [0]).frame_state(fr_)
    assert st["tokens"].shape[0] == 64 and st["sal"].shape == (64,)
    assert float(st["sal"][0]) == pytest.approx(np.mean([r * 32 + c for r in range(4) for c in range(4)]))
    st_cls = fv.SmolVLM256(None, PROC, [0]).frame_state(fr_)
    assert st_cls["sal"].shape == (64,)


def test_full_budget_avgpool_keeps_the_256ms_own_grid():
    be = ap.SmolVLM256(None, PROC, [0])
    grid = torch.arange(8 * 8 * 3, dtype=torch.float32).view(8, 8, 3)
    assert torch.equal(be.pooled(grid, 8)[0], grid.reshape(64, 3))
    assert ap.pool_side(19 / 64, 64, 8, False) == 4                                 # ours' 19 of 64 -> 4 x 4, its nearest square


def test_gaze_lands_on_the_256ms_8_x_8_grid():
    assert gz.box_on_grid(gz.roi_box([(0.5, 0.5)], 0.05), 8) == (3, 5, 3, 5)          # the 0.125 minimum: 1 token of 64 px a side
    assert gz.box_on_grid(gz.roi_box([(0.5, 0.5)], 0.05), 9) == (3, 6, 3, 6)          # the 2.2B's 9 x 9, as before
    assert gv2.gaze_token((0.6, 0.3), 8) == (4, 2) and gv2.gaze_token((0.999, 0.0), 8) == (7, 0)
    idx = gv2.select(8, (4, 2), 18)
    assert len(idx) == len(set(idx.tolist())) == 18 and 0 <= idx.min() and idx.max() < 64
    rows = gz.layout(8, (3, 5, 3, 5))
    assert len(rows) == 4 + 4 and rows[:4, 2].tolist() == [4, 4, 4, 4]                # 2 x 2 thumbnail of 4 x 4 tokens, then the ROI


# ---------------------------------------------------------------------------- a whole FLOPs plan on the fake 256M
REAL = dict(flops.MEASURE[M])
TRUE = [2.1e8, 3.0e4]                       # FLOPs per patch and per patch squared
C = [5e7, 2.1e8, 4 * 30 * 9 * 64.0]
SHAPE = dict(layers=30, heads=9, head_dim=64, non_embedding_params=106_000_000)
N = 100
VIDEOS = {"2023_m": 3, "loc_a": 2}


def true_vision(cells, side):
    return float(np.dot(REAL["features"](cells, side), TRUE))


def blind_len(messages):
    return 20 + len(messages[0]["content"][-1]["text"]) % 7


def fake_prompt_len(ad, question, options, frames, image_tokens):
    return blind_len(build_messages(question, options, [], 0)) + 30 + 9 * len(frames) + sum(image_tokens)


def fake_measure(**change):
    out = dict(REAL, prefill=lambda ad, n: float(flops.prefill(C, n)), prompt_len=fake_prompt_len, llm_shape=lambda ad: SHAPE,
               vision=lambda ad, cells, side: true_vision(cells, side))
    out.update(change)
    return out


def fake_counted(job, qid, decoder, tmp):
    item = next(it for it in job["items"] if int(it["qid"]) == qid)
    row, frames = job["row"], len(item["frames"])
    if row in flops.FOVEA_ROWS:
        lay = [REAL["layout"](job["conds"], next(iter(job["conds"])), f) for f in item["frames"]]
        ve, tok = sum(true_vision(c, s) for c, s in lay), sum(len(c) for c, _ in lay)
    else:
        own = 1.005 if row in run.BASELINE_ROWS else 1.0
        ve, tok = own * (2 if row in flops.TWO_VIEWS else 1) * frames * true_vision(*REAL["stock"]()), None
    n = 40 + 3 * frames
    return dict(ve=ve, pf=float(flops.prefill(C, n)), frames=frames, n_tokens=n, n_image_tokens=tok, video=item["video"])


class SmolVLM256(models.SmolVLM256):
    """The real adapter without a model. Adapters find their backends by class name, so it keeps this one."""
    def __init__(self, model_id=None, cfg=None):
        super().__init__(model_id, cfg)
        self.model, self.proc, self.letter_ids, self.model_id = None, PROC, [0], "fake"

    def load(self):
        return self

    def free(self):
        pass

    def blind_ids(self, messages):
        return torch.zeros((1, blind_len(messages)))


@pytest.fixture
def world(tmp_path, monkeypatch, fake_256):
    from system_perf import probe
    monkeypatch.setenv("GV_ROOT", str(tmp_path / "g"))
    monkeypatch.setenv("EGO_ROOT", str(tmp_path / "e"))
    frames = {v: shared.write_frames(str(tmp_path), v, N) for v in VIDEOS}
    rng = np.random.default_rng(0)
    rows, qid = [], 0
    for v, n in VIDEOS.items():
        for _ in range(n):
            rows.append(dict(qid=qid, video=v, question=f"q{qid} " * (qid + 1), options=list("abcde"), answer="A"))
            qid += 1
    data = dict(qs=pd.DataFrame(rows), videos=list(VIDEOS), test_videos=set(), metas={v: dict(n=N) for v in VIDEOS},
                gate={v: [] for v in VIDEOS}, uniform={v: [frames[v][f] for f in common.uniform_indices(N, 96)] for v in VIDEOS},
                clips=[dict(sequence=v, frame_no=np.arange(N)) for v in VIDEOS], labels={v: rng.integers(0, 4, N - 1) for v in VIDEOS},
                frame=lambda v, f: frames[v][f])
    cfg = common.load_config(CONFIG)
    cfg["paths"]["out_dir"] = str(tmp_path / "out")
    monkeypatch.setitem(flops.MEASURE, M, fake_measure())
    monkeypatch.setattr(flops, "measure_gate_ckpt", lambda path: 1000)
    monkeypatch.setattr(run, "check_parity", lambda *a: {})
    monkeypatch.setattr(probe, "find_decoder", lambda model: None)
    monkeypatch.setattr(fc, "counted", fake_counted)
    out_model = tmp_path / "out" / M
    os.makedirs(out_model)
    return dict(cfg=cfg, data=data, ad=SmolVLM256(cfg=cfg), out_model=str(out_model), monkeypatch=monkeypatch)


def plan_of(world, logs=None):
    return fc.make_plan(world["ad"], world["cfg"], world["data"], world["out_model"], log=(logs.append if logs is not None else lambda *a: None))


def test_the_plan_has_every_row_and_the_256ms_numbers(world):
    plan = plan_of(world)
    t = fc.targets_table(plan)
    assert len(t) == 35 and set(t.row) == {"uniform", "blind", "budget", "nogate", "uniform_flops"} | set(run.BASELINE_ROWS)
    assert plan["K_reference"] == 64 and plan["tokens_per_frame"] == 19 and plan["stock_tokens"] == 64
    assert plan["share"] == pytest.approx(19 / 64) and [ln["condition"] for ln in plan["lines"] if ln["row"] == "uniform"] == [run.uniform_name(64)]
    assert plan["units"]["vision_coef"] == pytest.approx(TRUE, rel=1e-6) and plan["llm_check"]["want_c"] == 69_120
    assert len(plan["vision_layouts"]) == 16 and {x["kind"] for x in plan["vision_layouts"]} == {"stock", "uniform", "fovea", "other fovea"}
    for cond, per_video in plan["fovea"].items():
        assert all(tokens == 19 * frames for _, tokens, frames in per_video.values()), cond
    budget = {ln["budget_pct"]: ln["condition"] for ln in plan["lines"] if ln["row"] == "budget"}
    cfg, data = world["cfg"], world["data"]
    assert [chk.video_limits(cfg, data, M, 64, b / 100)["2023_m"] for b in (5, 10, 20)] == [10, 21, 43]   # on a 100-frame video
    for b in (5, 10, 20):
        kept = common.budget_frames(cfg, data, M, 64, b / 100, use_gate=True)[0]["2023_m"]
        assert plan["fovea"][budget[b]]["2023_m"][2] == len(kept) <= {5: 10, 10: 21, 20: 43}[b]
    for ln in plan["lines"]:
        if ln["row"] != "blind":
            assert ln["verified"] and all(v["ok"] for v in ln["verified"]), ln
    assert json.loads(json.dumps(plan)) == plan


def test_k_prime_and_the_knobs_are_chosen_as_on_the_two_point_two_billion(world):
    plan = plan_of(world)
    t, u = fc.targets_table(plan), plan["units"]
    q = world["data"]["qs"].sort_values(["video", "qid"])
    blind = np.array([blind_len(build_messages(r.question, list(r.options), [], 0)) for r in q.itertuples()])
    qq = dict(blind=blind, video_frames=np.full(len(q), N))
    fu = np.full(len(q), 64)
    total = flops.question_flops(u, fu, "v_stock", flops.prompt_tokens(blind, fu, 64 * fu, plan["prompt"])).sum()
    for rows, tpf in ((["uniform_flops"], 64), (fc.SINGLE_VIEW, 19)):
        for r in t[t.row.isin(rows)].itertuples():
            share = lambda k: flops.uniform_share(qq, u, plan["prompt"], total, k, tpf)
            assert r.K == flops.choose_k(share, r.target_pct / 100, 64)[0], (r.row, r.budget_pct)
    for b in (5, 10, 20):
        at = t[(t.budget_pct == b) & t.row.isin(fc.SINGLE_VIEW)].set_index("row")
        assert at.K.nunique() == 1
        assert at.loc["avgpool", "token_share_pct"] == pytest.approx(100 * 16 / 64)                # 4 x 4
        assert abs(at.loc["streamingtom", "token_share_pct"] - 100 * 19 / 64) < 1
    g = t[t.row == "gazevlm2"]
    assert round(g.token_share_pct.iloc[0] * 64 / 100) % 2 == 0


def test_the_run_scores_exactly_the_planned_conditions(world):
    plan = plan_of(world)
    ad, cfg, data = world["ad"], world["cfg"], world["data"]
    scored = [(j["row"], c) for j in fr.reference_jobs(ad, cfg, data, plan) + fr.plan_jobs(ad, cfg, data, plan) for c in j["conds"]]
    assert set(scored) == {(ln["row"], ln["condition"]) for ln in plan["lines"]} and len(scored) == len(set(scored))


def test_the_report_charges_ours_its_foveated_frames(world):
    plan = plan_of(world)
    ln = next(x for x in plan["lines"] if x["row"] == "budget" and x["budget_pct"] == 10)
    _, tokens, frames = plan["fovea"][ln["condition"]]["2023_m"]
    res = pd.DataFrame([dict(row="budget", condition=ln["condition"], qid=0, video="2023_m", n_frames=frames, n_image_tokens=tokens,
                             n_tokens=600)])
    got = rep.add_flops(res, plan).iloc[0]
    u = plan["units"]
    assert got.flops == pytest.approx(flops.GATE_MODELS * 1000 * (N - 1) + plan["fovea"][ln["condition"]]["2023_m"][0]
                                      + float(flops.prefill(u["prefill"], 600)))
    assert rep.fovea_problems(res, plan) == [] and rep.fovea_problems(res.assign(n_image_tokens=17 * frames), plan)


def test_an_n2_term_off_the_config_stops_with_attention_not_counted(world):
    world["monkeypatch"].setitem(flops.MEASURE, M, fake_measure(llm_shape=lambda ad: dict(SHAPE, heads=8)))
    logs = []
    with pytest.raises(SystemExit, match="c DIFFERS FROM CONFIG"):
        plan_of(world, logs)
    assert any("ATTENTION NOT COUNTED" in x for x in logs)
    assert flops.MEASURE["smolvlm256"]["llm_shape_tolerance"] == flops.EXACT_TOLERANCE  # checked exactly, not to 1%


def test_a_vision_cost_that_isnt_linear_stops_the_plan(world):
    world["monkeypatch"].setitem(flops.MEASURE, M, fake_measure(vision=lambda ad, cells, side: true_vision(cells, side) + 10 * (16.0 * len(cells)) ** 3))
    with pytest.raises(SystemExit, match="aren't linear"):
        plan_of(world)


def test_frames_paired_or_extra_work_on_a_baseline_stop_the_plan(world):
    def off(job, qid, decoder, tmp):
        got = fake_counted(job, qid, decoder, tmp)
        return dict(got, ve=got["ve"] * 1.05) if job["row"] == "dtd" else got
    world["monkeypatch"].setattr(fc, "counted", off)
    with pytest.raises(SystemExit, match="FLOPS DIFFER"):
        plan_of(world)


def test_the_fit_has_16_layouts_with_enough_patch_counts():
    ad = models.SmolVLM256(cfg={})
    lay = fc.vision_fit_layouts(ad, dict(ours=dict(smolvlm256_k=[1, 2, 3])), REAL)
    assert [k for k, *_ in lay] == ["stock"] + ["uniform"] * 4 + ["fovea"] * 7 + ["other fovea"] * 4
    patches = {16 * len(c) for _, _, c, _ in lay}
    assert len(patches) > 4 and max(patches) == 1024
    pts = [(REAL["features"](c, s), true_vision(c, s)) for _, _, c, s in lay]
    assert flops.fit_vision(pts) == pytest.approx(TRUE, rel=1e-6)
    with pytest.raises(SystemExit, match="aren't linear"):
        flops.fit_vision([(f, v + 10 * f[0] ** 3) for f, v in pts])


# ---------------------------------------------------------------------------- the checks module
def test_videos_are_picked_the_same_way_every_time():
    counts = q3t.like_the_dataset()
    got = chk.pick_by_questions(counts, 100)
    assert got[0].startswith("2023_") and got[1].startswith("loc_") and sum(counts[v] for v in got) >= 100
    assert chk.pick_by_questions(dict(reversed(list(counts.items()))), 100) == got
    assert len(chk.pick_every(counts, 25)) == 14 and chk.pick_every(counts, 25)[0] == sorted(counts)[0]


def rec(cond, qid, p=(0.6, 0.4, 0, 0, 0), **kw):
    return dict(dict(condition=cond, qid=qid, video="v", pred="A", p=list(p), n_frames=3, n_image_tokens=57, n_tokens=99, error=None), **kw)


def test_same_answers_means_every_field_and_every_probability():
    a = [rec("c", 0), rec("c", 1)]
    assert chk.differences(a, a) == (2, 0, 0.0, 0)
    assert chk.differences(a, [rec("c", 0), rec("c", 1, p=(0.6, 0.39, 0, 0, 0))])[1:3] == (1, pytest.approx(0.01))
    assert chk.differences(a[:1], a, need_all=True)[3] == 1 and chk.differences(a[:1], a)[3] == 0
    assert chk.differences([rec("c", 0, error="oom")], a)[1] == 1


def write_rows(folder, rows):
    os.makedirs(folder, exist_ok=True)
    for name, recs in rows.items():
        with open(os.path.join(folder, f"{name}.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)


def test_same_and_identity_print_or_stop(tmp_path, capsys):
    write_rows(tmp_path / "a", dict(uniform=[rec("u", 0)], dtd=[rec("d", 0)]))
    write_rows(tmp_path / "b", dict(uniform=[rec("u", 0)], dtd=[rec("d", 0, p=(0.61, 0.39, 0, 0, 0))]))
    chk.main(["same", "--new", str(tmp_path / "a"), "--ref", str(tmp_path / "a"), "--rows", "uniform,dtd", "--all"])
    assert "SAME ANSWERS: 0 of 2 records differ, max_prob_diff 0.0" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="ANSWERS DIFFER"):
        chk.main(["same", "--new", str(tmp_path / "b"), "--ref", str(tmp_path / "a"), "--rows", "uniform,dtd"])
    same = {r: [rec(r, 0)] for r in ("avgpool", "dtd", "dycoke", "streamingtom")}
    write_rows(tmp_path / "i", dict(same, uniform=[rec("u", 0)]))
    chk.main(["identity", "--folder", str(tmp_path / "i")])
    assert "SAME AS UNIFORM K at max_prob_diff 0.0" in capsys.readouterr().out
    write_rows(tmp_path / "j", dict(same, uniform=[rec("u", 0)], dtd=[rec("dtd", 0, p=(0.61, 0.39, 0, 0, 0))]))
    with pytest.raises(SystemExit, match="NOT THE SAME"):                            # same answer, other probabilities
        chk.main(["identity", "--folder", str(tmp_path / "j")])


def test_the_projection_scales_the_timing_run(tmp_path, capsys):
    write_rows(tmp_path, dict(uniform=[dict(rec("u", q), video=f"v{q % 2}") for q in range(10)]))
    with open(tmp_path / "run_info_flops.json", "w") as f:
        json.dump(dict(status={"uniform": "ok (10 records, 20 s)", "dtd 5%": "ok (30 records, 2 with errors (oom / nan), 40 s)"}), f)
    e = chk.estimate(str(tmp_path))
    assert (e["seconds"], e["questions"], e["videos"]) == (60, 10, 2)
    assert e["by_questions_h"] == pytest.approx(60 * 5985 / 10 / 3600) and e["by_videos_h"] == pytest.approx(60 * 350 / 2 / 3600)
    chk.main(["estimate", "--timing", str(tmp_path)])
    assert "PROJECTED full run" in capsys.readouterr().out
    with open(tmp_path / "run_info_flops.json", "w") as f:
        json.dump(dict(status={"uniform": "FAILED: RuntimeError: x"}), f)
    with pytest.raises(SystemExit, match="didn't finish"):
        chk.estimate(str(tmp_path))


def test_one_jobs_records_are_checked_for_errors_gpu_and_completeness(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(prepare, "TOTAL", (3, 2))
    plan = dict(lines=[dict(row="uniform", condition="u"), dict(row="dtd", condition="d1"), dict(row="dtd", condition="d2")])
    with open(tmp_path / "plan.json", "w") as f:
        json.dump(plan, f)
    folder = tmp_path / M
    write_rows(folder, dict(uniform=[rec("u", q) for q in range(3)], dtd=[rec(c, q) for c in ("d1", "d2") for q in range(3)]))
    with open(folder / "gpu_ledger.jsonl", "w") as f:
        f.write(json.dumps(dict(row="uniform", condition="u", first_line=0, gpu="NVIDIA B200")) + "\n")
        f.write(json.dumps(dict(row="dtd", condition="d1", first_line=0, gpu="NVIDIA B200")) + "\n")
    args = ["records", "--folder", str(folder), "--plan", str(tmp_path / "plan.json"), "--rows", "uniform,dtd"]
    chk.main(args)
    out = capsys.readouterr().out
    assert "oom / nan records: 0" in out and "GPU per record: {'NVIDIA B200': 9}" in out and "RECORDS: 9 of 9 planned answers" in out
    write_rows(folder, dict(dtd=[rec(c, q) for c in ("d1", "d2") for q in range(3)][:-1]))
    with pytest.raises(SystemExit, match="INCOMPLETE"):
        chk.main(args)


def test_the_side_table_puts_both_models_next_to_each_other():
    row = lambda acc: dict(group="all", row="dtd", budget_pct=5.0, accuracy=acc, ci_low=acc - 1, ci_high=acc + 1, flops_pct=5.1, n=9)
    t = chk.vs_table(pd.DataFrame([row(40.0)]), pd.DataFrame([row(45.0)]))
    assert list(t.columns) == ["group", "row", "budget_pct", "accuracy_2p2b", "ci_low_2p2b", "ci_high_2p2b", "flops_pct_2p2b",
                               "accuracy_256m", "ci_low_256m", "ci_high_256m", "flops_pct_256m", "note"]
    assert t.accuracy_256m.iloc[0] == 40.0 and t.accuracy_2p2b.iloc[0] == 45.0 and "no winner" in t.note.iloc[0]


def test_the_2p2b_frames_are_compared_only_where_both_limits_are_equal(tmp_path, monkeypatch):
    monkeypatch.setenv("GV_ROOT", str(tmp_path / "g"))
    monkeypatch.setenv("EGO_ROOT", str(tmp_path / "e"))
    cfg = common.load_config(CONFIG)
    lengths = dict(long=100, short=3)
    frames = {v: [dict(key=(v, i), path="", t=float(i), gaze=(0.3, 0.6)) for i in range(n)] for v, n in lengths.items()}
    rng = np.random.default_rng(1)
    data = dict(videos=list(lengths), metas={v: dict(n=n) for v, n in lengths.items()}, frame=lambda v, f: frames[v][f],
                uniform={v: [frames[v][i] for i in common.uniform_indices(n, 96)] for v, n in lengths.items()},
                clips=[dict(sequence=v, frame_no=np.arange(n)) for v, n in lengths.items()],
                labels={v: rng.integers(0, 4, n - 1) for v, n in lengths.items()})
    label = "k=1"
    recs = []
    for b in cfg["budget"]["fractions"]:
        got, _ = common.budget_frames(cfg, data, "smolvlm", 81, float(b), use_gate=True)
        recs += [dict(condition=run.budget_name(b, cfg["run"]["policy"], label), qid=i, video=v, n_frames=len(got[v])) for i, v in enumerate(lengths)]
    write_rows(tmp_path / "ref", dict(budget=recs))
    applies, differ, na, wrong, n = chk.same_frames(cfg, data, str(tmp_path / "ref"))
    assert (differ, wrong, n) == (0, 0, 6) and applies >= 1 and na >= 3                 # the long video's limits differ at every budget
    assert chk.video_limits(cfg, data, "smolvlm", 81, 0.1)["long"] == 30 and chk.video_limits(cfg, data, M, 64, 0.1)["long"] == 21
    write_rows(tmp_path / "ref", dict(budget=[dict(r, n_frames=r["n_frames"] + 1) for r in recs]))
    assert chk.same_frames(cfg, data, str(tmp_path / "ref"))[3] == 6


# ---------------------------------------------------------------------------- the jobs and the runbook
def job(name):
    with open(os.path.join(REPO, "accuracy_tables", "hpc", name)) as f:
        return f.read()


JOBS = ["smolvlm256_download.SBATCH", "smolvlm256_flops_tests.SBATCH", "smolvlm256_flops_checks.SBATCH", "smolvlm256_flops.SBATCH",
        "smolvlm256_flops_summary.SBATCH"]


@pytest.mark.parametrize("name", JOBS)
def test_jobs_keep_the_rules(name):
    text = job(name)
    for folder in ("hpc_all", "hpc_budget", "nymeria_smolvlm", "hpc350", "smolvlm_350", "qwen3_350", "qwen25_350", "qwen25_flops",
                   "qwen3_flops", "data/folds"):
        assert folder not in text, folder
    assert "--no-parity" not in text and "SBATCH_ACCOUNT" not in text and "seed_results" not in text
    assert "HF_HOME=" not in text                                                      # the user's own HF_HOME is kept
    for line in text.splitlines():
        if "results/smolvlm_flops" in line:                                            # the 2.2B's folder is only read
            assert "--ref" in line or "REF=" in line or "--other" in line or "only read" in line or line.startswith("#"), line


def test_gpu_jobs_ask_for_one_b200_and_cpu_jobs_for_none():
    for name in ("smolvlm256_flops.SBATCH", "smolvlm256_flops_checks.SBATCH"):
        text = job(name)
        assert "#SBATCH --gres=gpu:b200:1" in text and "GPU=B200" in text and "--cpus-per-task=16" in text and "W=16" in text
        assert '#SBATCH --comment="preemption=yes;requeue=true"' in text and "--requeue" in text and "--dependency=singleton" in text
        assert "gpu_summary $GV_ROOT/logs/gpu_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.csv" in text
    for name in ("smolvlm256_download.SBATCH", "smolvlm256_flops_tests.SBATCH", "smolvlm256_flops_summary.SBATCH"):
        assert "gres" not in job(name)


def test_the_flops_job_calibrates_once_times_and_resumes():
    text = job("smolvlm256_flops.SBATCH")
    assert "OUT=$GV_ROOT/results/smolvlm256_flops" in text and "CONFIG=accuracy_tables/hpc/config.smolvlm256.yaml" in text
    assert "WANT=smolvlm256-flops-$MODE" in text and 'if [ -f "$PLAN" ]' in text and "--same-as $OUT/$M/gpu_types.txt" in text
    assert "flops_calibrate --config $CONFIG --model $M --videos all" in text and "pick-videos --every 25" in text
    assert "estimate --timing $TIMING/$M" in text and "flops_run --config $CONFIG --model $M --plan $PLAN --videos all" in text
    assert text.count("repair_results $OUT/$M") == 2 and "--keep-going" in text


def test_the_checks_job_runs_the_four_checks_on_small_sets():
    text = job("smolvlm256_flops_checks.SBATCH")
    assert "--model smolvlm --rows uniform,blind,budget" in text and "--max-questions 20" in text and "--prefetch-workers 8" in text
    assert "same --new $CHK/regression/smolvlm --ref $REF --rows uniform,blind,budget" in text
    assert "check_identity --write $CHK/identity.yaml --config $CONFIG" in text and "identity --folder $CHK/identity/$M" in text
    assert "--max-questions 50" in text and "--prefetch-workers 0 --out-dir $CHK/off" in text and "--prefetch-workers $W --out-dir $CHK/on" in text
    assert "same --new $CHK/on/$M --ref $CHK/off/$M --rows $ROWS --all" in text
    assert "ROWS=uniform,blind,budget,nogate,gazevlm,streamingtom,dycoke,fastvid,prunevid,dtd,avgpool,gazevlm2" in text


def test_the_tests_summary_and_download_jobs():
    assert "python -m pytest tests -q" in job("smolvlm256_flops_tests.SBATCH") and "smolvlm256_checks tokens" in job("smolvlm256_flops_tests.SBATCH")
    text = job("smolvlm256_flops_summary.SBATCH")
    assert "smolvlm256_checks records --folder $OUT/$M" in text and "flops_report --out-dir $OUT --model $M" in text
    assert "vs2p2b" in text and "summary_vs_2p2b.csv" in text and "TOKENS MATCH: not applicable, different model" in text
    assert "download_models smolvlm256" in job("smolvlm256_download.SBATCH") and "HF_HUB_OFFLINE=0" in job("smolvlm256_download.SBATCH")


def test_section_16_guards_the_pull_and_stops_for_the_limits():
    with open(os.path.join(REPO, "HPC_ACCURACY_RUNBOOK.md")) as f:
        section = f.read().split("\n## 16.")[1]
    for text in ("git pull --ff-only origin eyehead-gate", "STOP", "PROPOSED", "SAME FRAMES AS 2.2B", "SAME AS UNIFORM K", "SAME ANSWERS",
                 "parity OK", "PLAN WRITTEN", "15%", "smolvlm256_flops_tests.SBATCH", "smolvlm256_flops_checks.SBATCH", s2.REVISION):
        assert text in section, text
    rest = re.sub(r"don't claim a winner|no winner", "", section.lower())
    assert "winner" not in rest                                                        # only ever said not to claim one

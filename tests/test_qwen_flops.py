"""Qwen2.5-VL and Qwen3-VL on all 350 videos with every row matched on total FLOPs, in 3 shards per model (runbook section 15). CPU only, no
dataset, no model: the encoder and the LLM are faked as in tests/test_qwen3350.py and test_qwen25350.py, and so are the FLOP counts of real
questions and the encoder's count per layout. Everything around them is the real code: the frames' layouts and their attention windows,
flops_calibrate.py's main, the split, the sealed plan, flops_run.py's main per shard and whole, the merge, flops_report.py's main."""
import contextlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
import torch

import test_qwen25350 as s25
import test_qwen3350 as shared
from accuracy_tables import common, flops, run
from accuracy_tables import flops_calibrate as fc
from accuracy_tables import flops_run as fr
from accuracy_tables.hpc import check_gpu
from accuracy_tables.hpc.all350 import flops_progress, flops_report as rep, flops_shards as fs, merge_shards, prepare, shards
from accuracy_tables.hpc.repair_results import repair
from vlmeval import egoev
from vlmeval import qwen_fovea as qf

REPO = shared.REPO
GPU = shared.GPU
ROWS = ["uniform", "blind", "budget", "nogate", "uniform_flops", "gazevlm", "streamingtom", "dycoke", "fastvid", "prunevid", "dtd",
        "avgpool", "gazevlm2"]
SECTION_14_COLUMNS = ["group", "model", "row", "budget_pct", "target_pct", "planned_pct", "K", "knob", "condition", "token_share_pct", "n",
                      "accuracy", "ci_low", "ci_high", "flops_pct", "token_pct", "gflops_per_q", "frames_per_q", "frames_encoded_per_q",
                      "prompt_tokens_per_q", "videos", "miss_pts", "nearest_only", "note"]

# ---------------------------------------------------------------------------- a fake Qwen, per model
REAL = {m: dict(flops.MEASURE[m]) for m in ("qwen2.5", "qwen3")}
TRUE = {"qwen2.5": [1.25e9, 1.2e5, 2.0e4], "qwen3": [7.0e8, 7.3e4]}           # FLOPs per term of flops.py's features
C = {"qwen2.5": [6e8, 5.5e9, 4 * 36 * 16 * 128.0], "qwen3": [6e8, 7.3e9, 4 * 36 * 32 * 128.0]}
SHAPE = {"qwen2.5": dict(layers=36, heads=16, head_dim=128, non_embedding_params=2_770_000_000),
         "qwen3": dict(layers=36, heads=32, head_dim=128, non_embedding_params=3_630_000_000)}
ADAPTER = {"qwen2.5": s25.Qwen25, "qwen3": shared.Qwen3}
FAKE = {"qwen2.5": s25.fake_model, "qwen3": shared.fake_model}
TAG = {"qwen2.5": "qwen25", "qwen3": "qwen3"}
VIDEOS = {"loc_a": 3, "loc_b": 2, "2023_m": 3, "2023_o1": 1}
ORDER = ["2023_m", "loc_a", "2023_o1", "loc_b", "loc_a", "2023_m", "loc_b", "2023_m", "loc_a"]      # a qid is a position
N = 40
BLIND = 9                                                                                         # the test adapters' blind prompt


def true_vision(m, cells, side):
    return float(np.dot(REAL[m]["features"](cells, side), TRUE[m]))


def stock_vision(m):
    return true_vision(m, *REAL[m]["stock"]())


def fake_prompt_len(ad, question, options, frames, image_tokens):
    return BLIND + 30 + 9 * len(frames) + sum(image_tokens)


def fake_measure(m, **change):
    out = dict(REAL[m], prefill=lambda ad, n: float(flops.prefill(C[m], n)), prompt_len=fake_prompt_len,
               llm_shape=lambda ad: SHAPE[m], vision=lambda ad, cells, side: true_vision(m, cells, side))
    out.update(change)
    return out


def fake_counted(m, extra=None, pair=False):
    """counted() without a model: vision from each frame's real layout (foveated rows) or frames x a stock frame, the baselines' own
    work (0.5%, or extra for one row), and with pair=True an encoder that pairs frames into temporal patches."""
    extra = extra or {}

    def counted(job, qid, decoder, tmp):
        item = next(it for it in job["items"] if int(it["qid"]) == qid)
        row, frames = job["row"], len(item["frames"])
        cond = next(iter(job["conds"]))
        if row in flops.FOVEA_ROWS:
            lay = [REAL[m]["layout"](job["conds"], cond, fr_) for fr_ in item["frames"]]
            ve, tok = sum(true_vision(m, c, s) for c, s in lay), sum(len(c) for c, _ in lay)
        else:
            paid = 2 * -(-frames // 2) if pair else frames
            ve = extra.get(row, 1.005 if row in run.BASELINE_ROWS else 1.0) * (2 if row in flops.TWO_VIEWS else 1) * paid * stock_vision(m)
            tok = None
        n = 40 + 3 * frames
        return dict(ve=ve, pf=float(flops.prefill(C[m], n)), frames=frames, n_tokens=n, n_image_tokens=tok, video=item["video"])
    return counted


def write_frames(base, video):
    """test_qwen3350's frames, with every gaze inside the frame: its gaze (0.2 + 0.15 i, 0.6) is made for 5 frames, and on 40 it runs
    6 frame widths off the frame, where GazeVLM's heatmap underflows to 0 and divides 0 by 0. Frame 1 keeps no gaze."""
    frames = shared.write_frames(base, video, N)
    for i, fr in enumerate(frames):
        if fr["gaze"] is not None:
            fr["gaze"] = (0.05 + 0.9 * ((7 * i) % N) / (N - 1), 0.3 + 0.2 * (i % 3))
    return frames


def data_of(w, videos):
    """What common.load_data gives. gate: no gate frames per video (no row here reads them), so describe() has something to average."""
    qs = w.qs[w.qs.video.isin(set(videos))].reset_index(drop=True)
    return dict(qs=qs, videos=list(videos), test_videos=set(), metas={v: dict(n=N) for v in videos}, gate={v: [] for v in videos},
                uniform={v: [w.frames[v][f] for f in common.uniform_indices(N, 96)] for v in videos},
                clips=[dict(sequence=v, frame_no=np.arange(N)) for v in videos], labels={v: w.labels[v] for v in videos},
                frame=lambda v, f: w.frames[v][f])


def videos_of(cfg):
    sel = cfg["run"]["videos"]
    return list(VIDEOS) if sel in ("all", "test") else [v for v in sel]


def build(base, mp, m):
    from system_perf import probe
    g, e = base / "g", base / "e"
    os.makedirs(g)
    os.makedirs(e)
    mp.setenv("GV_ROOT", str(g))
    mp.setenv("EGO_ROOT", str(e))
    FAKE[m](mp)
    real_bootstrap = egoev.video_bootstrap
    mp.setattr(egoev, "video_bootstrap", lambda df, stat, B=1000, seed=0: real_bootstrap(df, stat, 50, seed))
    with open(e / "Egoeverything_VQA.json", "w") as f:
        json.dump([dict(video_name=v) for v in ORDER], f)
    rng = np.random.default_rng(1)
    w = type("World", (), {})()
    w.model, w.base = m, base
    w.frames = {v: write_frames(base, v) for v in VIDEOS}
    w.labels = {v: rng.integers(0, 4, N - 1) for v in VIDEOS}
    w.qs = pd.DataFrame(dict(qid=range(len(ORDER)), video=ORDER, question=[f"q{i}" for i in range(len(ORDER))],
                             options=[list("abcde")] * len(ORDER), answer=["A", "B"] * 4 + ["A"]))
    mp.setattr(common, "load_data", lambda cfg: data_of(w, videos_of(cfg)))
    mp.setattr(fc, "ADAPTERS", {m: ADAPTER[m]})
    mp.setattr(fr, "ADAPTERS", {m: ADAPTER[m]})
    mp.setitem(flops.MEASURE, m, fake_measure(m))
    mp.setattr(flops, "measure_gate_ckpt", lambda path: 1000)
    mp.setattr(run, "check_parity", lambda *a: {})
    mp.setattr(probe, "find_decoder", lambda model: None)
    mp.setattr(fc, "counted", fake_counted(m))
    mp.setattr(fr, "gpu_name", lambda: GPU)
    mp.setattr(prepare, "TOTAL", (len(ORDER), len(VIDEOS)))
    w.root = g / "results" / f"{TAG[m]}_flops"
    w.source = g / "results" / f"{TAG[m]}_350"
    w.repo = shared.fake_repo(str(base / "repo"))
    w.files = fs.plan_files(str(w.root), m)
    # sections 12 and 13's split, which this section only reads
    shards.write_manifest(str(w.source), shards.manifest_of(dict(VIDEOS), 3, "all"))
    # calibrate, as the job runs it
    fs.split(str(w.source), str(w.root))
    os.makedirs(os.path.dirname(w.files["plan"]))
    check_gpu.record(w.files["gpu"], GPU)
    fc.main(["--config", prepare.CONFIG, "--model", m, "--videos", "all", "--out-dir", str(w.root / "calibrate")])
    fs.seal(str(w.root), m)
    with open(w.files["plan"]) as f:
        w.plan = json.load(f)
    # the whole run, then 3 shards, each as its job runs it
    w.whole = base / "whole"
    run_flops(w, w.whole, "all")
    for s in range(3):
        run_shard(w, s)
    w.merge_info = fs.merge(str(w.root), m, prepare.CONFIG, ROWS)
    with pytest.MonkeyPatch.context() as argv:                         # the merge job's report
        argv.setattr(sys, "argv", ["flops_report", "--out-dir", str(w.root), "--model", m, "--plan", w.files["plan"]])
        rep.main()
    return w


def run_flops(w, out_dir, videos):
    fr.main(["--config", prepare.CONFIG, "--model", w.model, "--plan", w.files["plan"], "--videos", videos, "--out-dir", str(out_dir),
             "--prefetch-workers", "2", "--keep-going"])


def run_shard(w, s, root=None):
    root = root or w.root
    folder = shards.shard_folder(str(root), s, w.model)
    os.makedirs(folder, exist_ok=True)
    check_gpu.record(os.path.join(folder, "gpu_types.txt"), GPU)
    repair(folder)
    fs.start(str(root), s, w.model, prepare.CONFIG, ROWS, 2, GPU, shared.VERSIONS, repo=w.repo)
    run_flops(w, root / "shards" / str(s), ",".join(shards.read_manifest(str(root))["shards"][s]["videos"]))


@pytest.fixture(scope="module", params=["qwen2.5", "qwen3"])
def world(request, tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        yield build(tmp_path_factory.mktemp(TAG[request.param]), mp, request.param)
    finally:
        mp.undo()


def without_time(path):
    with open(path) as f:
        return [{k: x for k, x in json.loads(line).items() if k != "sec"} for line in f if line.strip()]


def table(w):
    return fc.targets_table(w.plan)


# ---------------------------------------------------------------------------- the arithmetic that is new on Qwen
def test_qwen25s_layout_terms_follow_its_attention_windows():
    cells, side = REAL["qwen2.5"]["stock"]()
    assert side == 448 and flops.features_qwen25(cells, side) == [1024.0, 16 * 64.0 ** 2, 1024.0 ** 2]
    conds = dict(f=("fovea", dict(a=160, R=70, img=448, gaze="gaze")))
    a, b = (qf.frame_layout(conds, "f", dict(gaze=g))[0] for g in ((0.5, 0.5), (0.03, 0.97)))
    fa, fb = flops.features_qwen25(a, 448), flops.features_qwen25(b, 448)
    assert fa[0] == 4 * len(a) and fa[2] == fa[0] ** 2 and fa[1] <= fa[2]
    assert fa != fb                                                     # the gaze moves the cost


def test_qwen3s_layout_terms_are_patches_and_their_square():
    cells, side = REAL["qwen3"]["stock"]()
    assert side == 512 and flops.features_qwen3(cells, side) == [1024.0, 1024.0 ** 2]


def fit_layouts(m):
    """The layouts calibrate counts the encoder on, as (kind, cells, side)."""
    return [(kind, c, s) for kind, _, c, s in fc.vision_fit_layouts(ADAPTER[m](), dict(ours=common.DEFAULTS["ours"]), REAL[m])]


@pytest.mark.parametrize("m", ["qwen2.5", "qwen3"])
def test_calibrate_counts_more_layouts_than_terms_and_enough_patch_counts(m):
    lay = fit_layouts(m)
    n_terms = len(TRUE[m])
    assert [k for k, _, _ in lay] == ["stock"] + ["uniform"] * 2 + ["fovea"] * 7 + ["other fovea"] * 6 and len(lay) == 16
    assert lay[0][2] == {"qwen2.5": 448, "qwen3": 512}[m] and [s for k, _, s in lay if k == "uniform"] == list(REAL[m]["fit_sides"])
    patches = sorted({4 * len(c) for _, c, _ in lay})
    assert len(patches) > 2 * n_terms and patches[-1] == 1024           # spread from the run's fovea up to the stock frame
    fovea = {len(c) for k, c, _ in lay if k == "fovea"}
    assert len(fovea) > 1 and max(fovea) < 256                          # the run's fovea: its cell count follows the gaze


@pytest.mark.parametrize("m", ["qwen2.5", "qwen3"])
def test_the_vision_fit_is_exact_and_stops_on_a_cost_that_isnt_linear(m):
    """The section 15 run's failure: + P^3 (about 0.1% of a stock frame) is non-linear, but least squares absorbs most of it into P
    and P^2, leaving under the 0.01% the check allows. 10 P^3 (about 1% of a stock frame) leaves over 10 times that on calibrate's
    16 layouts."""
    feats = REAL[m]["features"]
    pts = [(feats(c, s), true_vision(m, c, s)) for _, c, s in fit_layouts(m)]
    assert flops.fit_vision(pts) == pytest.approx(TRUE[m], rel=1e-6)
    with pytest.raises(SystemExit, match="aren't linear"):              # cubic in the patches: not in P and P^2
        flops.fit_vision([(f, v + 10 * f[0] ** 3) for f, v in pts])
    with pytest.raises(SystemExit, match="aren't linear"):              # a cost that follows where the cells are
        flops.fit_vision([(feats(c, s), true_vision(m, c, s) * (1 + float(np.asarray(c)[:, 0].mean()) / 1e4))
                          for _, c, s in fit_layouts(m)])


def test_the_vision_fit_refuses_layouts_that_cant_check_it():
    m = "qwen2.5"
    lay = fit_layouts(m)
    two_counts = [([200.0, w, 200.0 ** 2], 1e11 + w) for w in (3000.0, 5000.0, 7000.0, 9000.0)] + [([1024.0, 65536.0, 1024.0 ** 2], 1e12)]
    with pytest.raises(SystemExit, match="can't be checked"):           # 5 layouts but 2 patch counts: any cost in P would fit
        flops.fit_vision(two_counts)
    exact = [(flops.features_qwen25(c, s), true_vision(m, c, s)) for _, c, s in lay[:1] + lay[3:5]]
    with pytest.raises(SystemExit, match="can't be checked"):           # as many layouts as terms
        flops.fit_vision(exact)
    uniform = [(flops.features_qwen25(qf.uniform_cells(s), s), 1.0e12 * s) for s in (224, 336, 448)]
    with pytest.raises(SystemExit, match="don't tell"):                 # uniform layouts alone: windows grow with patches
        flops.fit_vision(uniform)


@pytest.mark.parametrize("m", ["qwen2.5", "qwen3"])
def test_the_n2_term_is_checked_against_the_config(m):
    got = flops.llm_shape_check(C[m], SHAPE[m])
    assert got["want_c"] == {"qwen2.5": 294_912, "qwen3": 589_824}[m] and got["c_off"] == 0
    off = flops.llm_shape_check([C[m][0], C[m][1], C[m][2] * 1.02], SHAPE[m])
    assert off["c_off"] == pytest.approx(0.02) and abs(off["c_off"]) > flops.LLM_SHAPE_TOLERANCE


def test_frames_paired_into_temporal_patches_fail_the_check():
    units = dict(v_stock=1e12, v_fovea=2e11, prefill=C["qwen2.5"])
    n = 500
    ok = dict(ve=5e12, pf=float(flops.prefill(C["qwen2.5"], n)), frames=5, n_tokens=n)
    paired = dict(ok, ve=6e12)                                          # 5 frames paid as 3 pairs
    assert fc.check("uniform_flops", ok, units)[2] and not fc.check("uniform_flops", paired, units)[2]
    assert not fc.check("dtd", paired, units)[2]


def test_grouped_query_attention_is_counted_over_the_querys_heads():
    """Qwen's LLMs pass SDPA 16 (or 32) query heads against 2 (or 8) key/value heads. Inside gqa_attention that counts as the repeated
    keys and values would, 4 b h s_q s_k d; equal heads keep torch's own formula; outside, the registry is as it was."""
    from torch.utils.flop_counter import flop_registry
    op = torch.ops.aten._scaled_dot_product_flash_attention
    before = flop_registry[op]
    q, k_same, k_gqa = torch.zeros(1, 16, 10, 128), torch.zeros(1, 16, 7, 128), torch.zeros(1, 2, 7, 128)
    with flops.gqa_attention():
        f = flop_registry[op]
        assert f(q, k_gqa, k_gqa, out_val=None) == 4 * 1 * 16 * 10 * 7 * 128
        assert f(q, k_same, k_same, out_val=None) == before(q, k_same, k_same, out_val=None) == 4 * 1 * 16 * 10 * 7 * 128
    assert flop_registry[op] is before


def test_the_watch_sees_attention_with_no_flop_formula():
    with flops.AttentionWatch() as w:
        torch.zeros(2, 3) @ torch.zeros(3, 4)
    assert w.missed == set()
    lib = torch.library.Library("gvtest", "DEF")                       # an attention op the counter has no formula for
    lib.define("fake_attention(Tensor q) -> Tensor")
    lib.impl("fake_attention", lambda q: q.clone(), "CompositeExplicitAutograd")
    with flops.AttentionWatch() as w:
        torch.ops.gvtest.fake_attention(torch.zeros(2))
    assert w.missed == {"gvtest.fake_attention"}


# ---------------------------------------------------------------------------- the plan, on each fake Qwen
def test_the_plan_has_every_row_at_every_budget_and_the_settled_values(world):
    t = table(world)
    assert len(t) == 2 + 3 * (2 + 1 + 8)
    assert world.plan["K_reference"] == 96 and world.plan["tokens_per_frame"] == 51 and world.plan["stock_tokens"] == 256
    assert world.plan["share"] == pytest.approx(51 / 256) and world.plan["fractions"] == [0.05, 0.1, 0.2]
    assert [ln["condition"] for ln in world.plan["lines"] if ln["row"] == "uniform"] == [run.uniform_name(96)]
    assert all("R=70 a=160" in ln["condition"] for ln in world.plan["lines"] if ln["row"] in flops.FOVEA_ROWS)


def test_the_units_come_from_the_counts(world):
    m, u = world.model, world.plan["units"]
    assert u["v_stock"] == pytest.approx(stock_vision(m)) and u["vision_coef"] == pytest.approx(TRUE[m], rel=1e-6)
    assert u["prefill"] == pytest.approx(C[m], rel=1e-4) and u["gate_per_decision"] == 1000
    assert world.plan["llm_check"]["want_c"] == C[m][2] and world.plan["attention"] == dict(implementation="the model's own", missed=[])
    assert world.plan["prompt"]["c0"] == pytest.approx(30) and world.plan["prompt"]["o"] == pytest.approx(9)


def test_ours_vision_and_tokens_are_summed_over_each_frames_own_layout(world):
    m, ad = world.model, ADAPTER[world.model]()
    cfg = common.load_config(prepare.CONFIG)
    for job in run.row_jobs(ad, cfg, data_of(world, list(VIDEOS)), ["budget", "nogate"]):
        cond = next(iter(job["conds"]))
        for it in job["items"]:
            lay = [REAL[m]["layout"](job["conds"], cond, f) for f in it["frames"]]
            vision, tokens, frames = world.plan["fovea"][cond][it["video"]]
            assert vision == pytest.approx(sum(true_vision(m, c, s) for c, s in lay), rel=1e-9)
            assert tokens == sum(len(c) for c, _ in lay) and frames == len(it["frames"])


def test_ours_sets_the_targets_from_its_own_frames(world):
    """Each target recomputed from the plan: ours' gate, vision and prefill over Uniform K's, on every question."""
    p, u = world.plan, world.plan["units"]
    q = world.qs
    fu = np.full(len(q), min(96, N))
    total = flops.question_flops(u, fu, "v_stock", flops.prompt_tokens(np.full(len(q), BLIND), fu, 256 * fu, p["prompt"])).sum()
    for ln in p["lines"]:
        if ln["row"] != "budget":
            continue
        per = [p["fovea"][ln["condition"]][v] for v in q.video]
        vision, tokens, frames = (np.array([x[i] for x in per], dtype=float) for i in range(3))
        n = flops.prompt_tokens(np.full(len(q), BLIND), frames, tokens, p["prompt"])
        mine = flops.question_flops_vision(u, vision, n, decisions=np.full(len(q), N - 1)).sum() / total
        assert ln["target_pct"] == pytest.approx(100 * mine, rel=1e-9)
    assert p["targets"]["5"] < p["targets"]["10"] < p["targets"]["20"]


def test_k_prime_and_the_knobs_are_chosen_as_on_smolvlm(world):
    t, p = table(world), world.plan
    u = p["units"]
    qq = dict(blind=np.full(len(world.qs), BLIND), video_frames=np.full(len(world.qs), N))
    fu = np.full(len(world.qs), min(96, N))
    total = flops.question_flops(u, fu, "v_stock", flops.prompt_tokens(qq["blind"], fu, 256 * fu, p["prompt"])).sum()
    for fam, rows, tpf, views in (("uniform_flops", ["uniform_flops"], 256, 1), ("single", fc.SINGLE_VIEW, 51, 1)):
        for r in t[t.row.isin(rows)].itertuples():
            share = lambda k: flops.uniform_share(qq, u, p["prompt"], total, k, tpf, views)
            want = flops.choose_k(share, r.target_pct / 100, 96)
            assert r.K == want[0], (fam, r.row, r.budget_pct)
    for b in (5, 10, 20):
        at = t[(t.budget_pct == b) & t.row.isin(fc.SINGLE_VIEW)].set_index("row")
        assert at.K.nunique() == 1
        assert abs(at.loc["streamingtom", "token_share_pct"] - 100 * 51 / 256) < 0.5
        assert at.loc["avgpool", "token_share_pct"] == pytest.approx(100 * 49 / 256)          # 7 x 7
        assert t[(t.row == "gazevlm2") & (t.budget_pct == b)].K.iloc[0] <= at.K.iloc[0]
    for ln in p["lines"]:
        if ln["row"] != "blind":
            assert ln["verified"] and all(v["ok"] for v in ln["verified"]), ln
        if ln["row"] in ("uniform_flops", *run.BASELINE_ROWS):
            assert f"K={ln['K']}" in ln["condition"]


def test_the_summary_columns_are_section_14s(world):
    with open(world.root / "summary_flops_by_group.csv") as f:
        assert f.readline().strip().split(",") == SECTION_14_COLUMNS


def again(world, tmp_path, **measure):
    """make_plan once more on the world, with changed measuring functions; the counts files are copied, so nothing recounts there."""
    out = tmp_path / "again"
    shutil.copytree(os.path.dirname(world.files["plan"]), out)
    mp = pytest.MonkeyPatch()
    try:
        mp.setitem(flops.MEASURE, world.model, fake_measure(world.model, **measure))
        logs = []
        ad = ADAPTER[world.model](cfg=common.load_config(prepare.CONFIG)).load()
        plan = fc.make_plan(ad, ad.cfg, data_of(world, list(VIDEOS)), str(out), log=logs.append)
        return plan, logs
    finally:
        mp.undo()


def test_a_wrong_n2_term_stops_the_plan(world, tmp_path):
    wrong = dict(SHAPE[world.model], heads=SHAPE[world.model]["heads"] - 1)
    with pytest.raises(SystemExit, match="c DIFFERS FROM CONFIG"):
        again(world, tmp_path, llm_shape=lambda ad: wrong)


def test_a_vision_count_that_isnt_linear_in_the_layout_stops_the_plan(world, tmp_path):
    m = world.model
    with pytest.raises(SystemExit, match="aren't linear"):
        again(world, tmp_path, vision=lambda ad, cells, side: true_vision(m, cells, side) * (1 + float(np.asarray(cells)[:, 0].mean()) / 1e4))


def test_unseen_attention_redoes_the_counts_with_eager_attention(world, tmp_path, monkeypatch):
    state, always = dict(impl=None), dict(on=False)

    @contextlib.contextmanager
    def impl_(model, impl):
        state["impl"] = impl
        yield
        state["impl"] = None

    class Watch:
        def __enter__(self):
            self.missed = {"aten.fake_attention"} if state["impl"] is None or always["on"] else set()
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(flops, "attention_implementation", impl_)
    monkeypatch.setattr(flops, "AttentionWatch", Watch)
    plan, logs = again(world, tmp_path)
    assert plan["attention"] == dict(implementation="eager", missed=[])
    assert any("redone with eager attention" in x for x in logs) and not any("ATTENTION NOT COUNTED" in x for x in logs)
    always["on"] = True
    plan, logs = again(world, tmp_path / "2")
    assert plan["attention"]["missed"] == ["aten.fake_attention"] and any("ATTENTION NOT COUNTED" in x for x in logs)


def test_a_question_whose_vision_isnt_its_layouts_stops_the_plan(world, tmp_path, monkeypatch):
    real = fake_counted(world.model)

    def off(job, qid, decoder, tmp):
        got = real(job, qid, decoder, tmp)
        return dict(got, ve=got["ve"] * 1.001) if job["row"] == "nogate" else got

    monkeypatch.setattr(fc, "counted", off)
    with pytest.raises(SystemExit, match="FLOPS DIFFER"):
        again(world, tmp_path)


# ---------------------------------------------------------------------------- the split, the sealed plan, the shards
def test_the_split_is_sections_12_and_13s_byte_for_byte(world):
    for name in ("manifest.json", "shard_0.txt", "shard_1.txt", "shard_2.txt", "all_videos.txt"):
        with open(world.source / "manifest" / name, "rb") as a, open(world.root / "manifest" / name, "rb") as b:
            assert a.read() == b.read(), name
    assert fs.split(str(world.source), str(world.root))["n_questions"] == len(ORDER)          # again: unchanged


def test_the_split_is_refused_if_wrong_or_missing(world, tmp_path):
    with pytest.raises(SystemExit, match="no "):
        fs.split(str(tmp_path / "nothing"), str(tmp_path / "out"))
    bad = tmp_path / "bad"
    m = shards.manifest_of(dict(VIDEOS), 3, "all")
    m["shards"][0]["videos"] = m["shards"][0]["videos"][1:]
    shards.write_manifest(str(bad), m)
    with pytest.raises(SystemExit, match="is wrong"):
        fs.split(str(bad), str(tmp_path / "out"))


@pytest.mark.parametrize("folder", ["qwen25_350", "qwen3_350", "smolvlm_flops", "hpc350_pilot"])
def test_nothing_is_written_inside_a_folder_this_section_only_reads(world, folder):
    with pytest.raises(SystemExit, match="only read"):
        fs.split(str(world.source), str(world.root.parent / folder / "x"))


def test_a_shard_refuses_a_plan_that_isnt_the_sealed_one(world, tmp_path):
    root = tmp_path / "results" / "copy"
    shutil.copytree(world.root, root)
    plan = fs.plan_files(str(root), world.model)["plan"]
    os.chmod(plan, 0o644)
    with open(plan, "a") as f:
        f.write(" ")
    with pytest.raises(SystemExit, match="aren't the ones calibrate sealed"):
        fs.start(str(root), 0, world.model, prepare.CONFIG, ROWS, 2, GPU, shared.VERSIONS, repo=world.repo)
    os.remove(fs.plan_files(str(root), world.model)["seal"])
    with pytest.raises(SystemExit, match="run the calibrate job first"):
        fs.start(str(root), 0, world.model, prepare.CONFIG, ROWS, 2, GPU, shared.VERSIONS, repo=world.repo)


def test_the_jobs_refuse_a_changed_config(world, tmp_path):
    other = tmp_path / "c.yaml"
    shutil.copy(prepare.CONFIG, other)
    with open(other, "a") as f:
        f.write("\n# changed\n")
    with pytest.raises(SystemExit, match="changed since the plan"):
        fs.start(str(world.root), 0, world.model, str(other), ROWS, 2, GPU, shared.VERSIONS, repo=world.repo)


def test_a_rerun_skips_every_scored_record_and_a_cut_shard_resumes(world, tmp_path):
    root = tmp_path / "results" / "copy"
    shutil.copytree(world.root, root)
    folder = shards.shard_folder(str(root), 2, world.model)
    before = {r: open(os.path.join(folder, f"{r}.jsonl"), "rb").read() for r in ROWS}
    run_shard(world, 2, root)
    assert {r: open(os.path.join(folder, f"{r}.jsonl"), "rb").read() for r in ROWS} == before         # nothing scored again
    path = os.path.join(folder, "dtd.jsonl")
    lines = open(path).read().splitlines(True)
    with open(path, "w") as f:
        f.writelines(lines[:-2] + [lines[-2][:9]])                                                    # killed mid-write
    run_shard(world, 2, root)
    assert without_time(path) == without_time(os.path.join(shards.shard_folder(str(world.root), 2, world.model), "dtd.jsonl"))


# ---------------------------------------------------------------------------- the merge and the report
@pytest.mark.parametrize("row", ROWS)
def test_merged_files_match_a_whole_run_line_by_line(world, row):
    assert without_time(world.root / world.model / f"{row}.jsonl") == without_time(world.whole / world.model / f"{row}.jsonl")


def test_every_planned_answer_is_merged_once_on_one_gpu(world):
    conditions = merge_shards.read_targets(world.files["targets"])
    want = sum(len(c) for c in conditions.values()) * len(ORDER)
    assert sum(world.merge_info["records"].values()) == want
    assert world.merge_info["gpu_per_record"] == {GPU: want} and world.merge_info["gpu"] == GPU


def test_the_merge_stops_and_writes_nothing_on_a_missing_record(world, tmp_path):
    root = tmp_path / "results" / "copy"
    shutil.copytree(world.root, root)
    path = os.path.join(shards.shard_folder(str(root), 1, world.model), "nogate.jsonl")
    lines = open(path).read().splitlines(True)
    with open(path, "w") as f:
        f.writelines(lines[:-1])
    merged_before = sorted(os.listdir(root / world.model))
    with pytest.raises(SystemExit, match="INCOMPLETE"):
        fs.merge(str(root), world.model, prepare.CONFIG, ROWS)
    assert sorted(os.listdir(root / world.model)) == merged_before


def test_the_merge_stops_on_a_record_off_the_gpu_ledger(world, tmp_path):
    root = tmp_path / "results" / "copy"
    shutil.copytree(world.root, root)
    path = os.path.join(shards.shard_folder(str(root), 0, world.model), fs.LEDGER)
    with open(path, "a") as f:
        f.write(json.dumps(dict(row="dtd", condition="x", first_line=0, gpu="NVIDIA L40S", job=None, node=None)) + "\n")
    with pytest.raises(SystemExit, match="INCOMPLETE"):
        fs.merge(str(root), world.model, prepare.CONFIG, ROWS)


def report(root, plan, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["flops_report", "--out-dir", str(root), "--model", plan["model"], "--plan",
                                      fs.plan_files(str(root), plan["model"])["plan"]])
    rep.main()
    return capsys.readouterr().out


def test_the_report_is_complete_and_charges_ours_its_layouts(world, monkeypatch, capsys):
    out = report(world.root, world.plan, monkeypatch, capsys)
    assert "EVERY n IS COMPLETE" in out and f"GPU types recorded: {{'{world.model}': ['{GPU}']}}" in out
    res, _ = rep.load(str(world.root), world.model)
    res = rep.add_flops(res, world.plan)
    b = res[res.row == "budget"].iloc[0]
    vision = world.plan["fovea"][b.condition][b.video][0]
    u = world.plan["units"]
    assert b.flops == pytest.approx(flops.GATE_MODELS * 1000 * (N - 1) + vision + float(flops.prefill(u["prefill"], b.n_tokens)))
    assert rep.fovea_problems(res, world.plan) == []
    assert rep.fovea_problems(res.assign(n_image_tokens=res.n_image_tokens + 1), world.plan)


def test_the_report_flags_a_missing_record_as_incomplete(world, tmp_path, monkeypatch, capsys):
    root = tmp_path / "results" / "copy"
    shutil.copytree(world.root, root)
    path = root / world.model / "dtd.jsonl"
    lines = open(path).read().splitlines(True)
    with open(path, "w") as f:
        f.writelines(lines[1:])
    with pytest.raises(SystemExit, match="INCOMPLETE"):
        report(root, world.plan, monkeypatch, capsys)


def test_progress_counts_answers_over_the_plans_rows(world, capsys):
    got = flops_progress.progress(str(world.root), world.model)
    conditions = merge_shards.read_targets(world.files["targets"])
    assert [r for r, _, _ in got[0]["rows"]] == list(conditions)
    assert all(s["done"] == s["total"] == s["questions"] * sum(len(c) for c in conditions.values()) for s in got)
    flops_progress.main(["--root", str(world.root), "--model", world.model])
    assert "all shards:" in capsys.readouterr().out


# ---------------------------------------------------------------------------- the jobs and the runbook
def job(name):
    with open(os.path.join(REPO, "accuracy_tables", "hpc", name)) as f:
        return f.read()


@pytest.mark.parametrize("name", ["qwen_flops.SBATCH", "qwen_flops_merge.SBATCH", "qwen_flops_tests.SBATCH"])
def test_jobs_keep_the_rules(name):
    text = job(name)
    for folder in ("hpc_all", "hpc_budget", "nymeria_smolvlm", "hpc350_check", "hpc350_pilot", "smolvlm_350", "smolvlm_flops",
                   "qwen3_350", "qwen25_350", "data/folds"):
        assert folder not in text.replace("SECTION=qwen25_350", "").replace("SECTION=qwen3_350", ""), folder
    assert "--no-parity" not in text and "seed_results" not in text and "SBATCH_ACCOUNT" not in text


def test_the_gpu_job_is_one_a100_per_job_and_reads_the_earlier_split_only():
    text = job("qwen_flops.SBATCH")
    assert "#SBATCH --gres=gpu:a100:1" in text and "GPU=A100" in text and text.count("--expect $GPU") == 2
    assert '#SBATCH --comment="preemption=yes;requeue=true"' in text and "--requeue" in text and "--dependency=singleton" in text
    assert "--same-as $CAL_GPU --record $S/$M/gpu_types.txt" in text and "flops_shards split --source $GV_ROOT/results/$SECTION --root $OUT" in text
    assert "OUT=$GV_ROOT/results/${TAG}_flops" in text and "flops_shards start" in text and "flops_shards seal" in text
    assert 'echo "SHARD DONE: full $SHARD"' in text and "--keep-going" in text and "repair_results" in text
    assert "WANT=$TAG-flops-full-$SHARD" in text and "WANT=$TAG-flops-calibrate" in text
    assert "ROWS=" + ",".join(ROWS) in text


def test_the_merge_job_merges_reports_and_only_compares_with_sections_12_and_13():
    text = job("qwen_flops_merge.SBATCH")
    assert "gres" not in text and "--time=08:00:00" in text
    assert "flops_shards merge --root $OUT" in text and "flops_report --out-dir $OUT --model $M --plan $PLAN" in text
    assert "compare_records --new $OUT/$M --ref $GV_ROOT/results/$SECTION/$M --rows uniform,blind,budget --report" in text
    assert "gpu_${TAG}-flops-full-*_*.csv" in text


def test_the_tests_job_runs_every_test():
    assert 'python -m pytest tests -q' in job("qwen_flops_tests.SBATCH") and "gres" not in job("qwen_flops_tests.SBATCH")


def test_section_15_guards_the_pull():
    with open(os.path.join(REPO, "HPC_ACCURACY_RUNBOOK.md")) as f:
        section = f.read().split("\n## 15.")[1]
    for text in ("smolvlm-flops", "qwen3350", "qwen25350", "summary_flops_by_group.csv", "git checkout -- accuracy_tables/flops.py",
                 "git pull --ff-only origin eyehead-gate", "66285546d2b821cf421d4f5eb2576359d3770cd3", "STOP"):
        assert text in section, text

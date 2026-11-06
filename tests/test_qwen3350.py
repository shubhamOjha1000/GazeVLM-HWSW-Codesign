"""Qwen3 on all 350 videos, split across 3 one-GPU jobs (runbook section 12): the assignment, threads on Qwen3's baseline rows, the resumable
calibration, resuming a shard, the merge and everything it refuses, and that a merged run equals an unsharded one. CPU only, no dataset, no model:
the encoder and the LLM are faked, and everything around them is the real code, from calibrate.py's main through run_model to report.py's main."""
import json
import os
import shutil
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from accuracy_tables import calibrate as cal
from accuracy_tables import common, models, run, summarize
from accuracy_tables.hpc import check_gpu
from accuracy_tables.hpc.all350 import check_identity, estimate_hours, merge_shards, prepare, report, shards
from accuracy_tables.hpc.newvideos import compare_records, paths
from accuracy_tables.hpc.repair_results import repair
from vlmeval import avgpool as ap
from vlmeval import budget_rule as br
from vlmeval import dtd, dycoke as dy, egoev, fastvid as fv
from vlmeval import gazevlm as gz
from vlmeval import gazevlm2 as gv2
from vlmeval import prunevid as pv
from vlmeval import qwen3_fovea as q3
from vlmeval import qwen_fovea as qf
from vlmeval import streamingtom as stm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS = ["uniform", "blind", "budget", "avgpool", "gazevlm", "gazevlm2", "streamingtom", "dycoke", "fastvid", "prunevid", "dtd"]
GPU = "NVIDIA A100-SXM4-80GB"
VERSIONS = dict(python="3.12", torch="2.15", transformers="5.20")
M = "qwen3"


# ---------------------------------------------------------------------------- a fake Qwen3
PROC = SimpleNamespace(image_processor=SimpleNamespace(image_mean=[0.5] * 3, image_std=[0.5] * 3))
VISION = SimpleNamespace(blocks=[SimpleNamespace(attn=None)])


def fake_encode(vis, x, cells, side, mean, std):
    """One token per cell from the pixels it covers, and 3 DeepStack features from the same."""
    tok = torch.stack([x[:, y:y + s, c:c + s].mean((1, 2)) for y, c, s in np.asarray(cells, dtype=int).tolist()]) - 0.5
    return tok, [tok * (k + 1) for k in range(3)]


@contextmanager
def fake_capture(target, score=None):
    """Attention for the saliency: 1,024 patches for StreamingTOM, 256 tokens for FastVID, as the backends read them."""
    n = 1024 if score is stm.received_attention else 256
    yield [torch.arange(n, dtype=torch.float32)[None]]


def fake_probs(model, inputs, letter_ids):
    """Probabilities that depend on every token and DeepStack feature in the prompt."""
    deep = torch.cat([d for frame in inputs["deep"] for d in frame]).float().mean()
    v = (float(torch.cat(list(inputs["feats"])).float().mean() + deep) * 37) % 1
    return np.array([v, 1 - v, 0.0, 0.0, 0.0])


def fake_model(mp):
    mp.setattr(q3, "parts", lambda model: (VISION, None, None))
    mp.setattr(q3, "check_encoder", lambda vis: None)
    mp.setattr(q3, "encode_cells", fake_encode)
    mp.setattr(q3, "build_inputs", lambda model, proc, msgs, feats, deep, cells, sides: (dict(feats=feats, deep=deep), torch.zeros((1, 9), dtype=torch.long)))
    mp.setattr(q3, "letter_probs", fake_probs)
    mp.setattr(stm, "capture_saliency", fake_capture)


class Qwen3(models.Qwen3):
    """The real adapter without a model. Adapters find their backends by class name, so it keeps this one."""
    def load(self):
        self.model, self.proc, self.letter_ids, self.model_id = None, PROC, list(range(5)), "fake"
        return self

    def free(self):
        pass

    def blind_ids(self, messages):
        return torch.zeros((1, 9), dtype=torch.long)

    def blind_logits(self, ids):
        return torch.tensor([2.0, 1.0, 0.0, 0.0, 0.0])


def write_frames(folder, video, n):
    rng, out = np.random.default_rng(sum(map(ord, video))), []
    for i in range(n):
        path = os.path.join(str(folder), f"{video}_{i}.png")
        Image.fromarray(rng.integers(0, 255, (56, 56, 3), dtype=np.uint8)).save(path)
        out.append(dict(key=(video, i), path=path, t=float(i), gaze=None if i == 1 else (0.2 + 0.15 * i, 0.6), gaze_trace=[]))
    return out


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def without_time(path):
    return [{k: x for k, x in r.items() if k != "sec"} for r in read_jsonl(path)]


# ---------------------------------------------------------------------------- the assignment
def like_the_dataset():
    """25 AEA videos, 21 Nymeria videos with many questions and 304 with one."""
    counts = {f"loc_{i:02d}": 30 + 7 * i for i in range(25)}
    counts.update({f"2023_m{i:02d}": 60 + 15 * i for i in range(21)})
    counts.update({f"2023_o{i:03d}": 1 for i in range(304)})
    return counts


def test_every_video_lands_in_one_shard_and_every_question_appears_once():
    counts = like_the_dataset()
    manifest = shards.manifest_of(counts, 3, "all")
    videos = [v for s in manifest["shards"] for v in s["videos"]]
    assert sorted(videos) == sorted(counts) and len(set(videos)) == len(videos) == manifest["n_videos"] == 350
    assert sum(s["questions"] for s in manifest["shards"]) == sum(counts.values()) == manifest["n_questions"]
    for s in manifest["shards"]:
        assert s["questions"] == sum(counts[v] for v in s["videos"])
    assert shards.manifest_problems(manifest, counts) == []


def test_shards_are_balanced_by_questions_not_videos():
    loads = [s["questions"] for s in shards.manifest_of(like_the_dataset(), 3, "all")["shards"]]
    assert max(loads) - min(loads) <= 1
    small = dict(big=100, a=1, b=1, c=1, d=1)
    assert sorted(sum(small[v] for v in m) for m in shards.assign(small, 2)) == [4, 100]        # 1 video against 4, by questions


def test_the_assignment_ties_are_fixed_and_the_order_given_doesnt_matter():
    counts = like_the_dataset()
    backwards = dict(reversed(list(counts.items())))
    assert shards.assign(counts, 3) == shards.assign(backwards, 3) == shards.assign(counts, 3)
    assert shards.assign(dict(m=4, b=3, c=2, x=1, y=1), 3) == [["m"], ["b", "y"], ["c", "x"]]


def test_the_manifest_is_written_once_and_a_different_one_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("GV_ROOT", str(tmp_path))
    root = tmp_path / "results" / "new"
    manifest = shards.manifest_of(dict(a=5, b=4, c=3, d=1), 3, "all")
    assert shards.write_manifest(str(root), manifest) is True
    before = {name: (root / "manifest" / name).read_text() for name in os.listdir(root / "manifest")}
    assert sorted(before) == ["all_videos.txt", "manifest.json", "shard_0.txt", "shard_1.txt", "shard_2.txt"]
    assert before["shard_0.txt"] == "a\n" and before["shard_1.txt"] == "b\n" and before["shard_2.txt"] == "c\nd\n"
    assert shards.write_manifest(str(root), json.loads(json.dumps(manifest))) is False                  # as a rerun reads it back
    assert {name: (root / "manifest" / name).read_text() for name in before} == before
    with pytest.raises(SystemExit, match="different assignment"):
        shards.write_manifest(str(root), shards.manifest_of(dict(a=5, b=4, c=3, d=2), 3, "all"))
    assert {name: (root / "manifest" / name).read_text() for name in before} == before


def test_manifest_problems_each_kind():
    counts = dict(a=2, b=1, c=1)
    good = dict(scope="all", shards=[dict(shard=0, questions=2, videos=["a"]), dict(shard=1, questions=2, videos=["b", "c"])])
    assert shards.manifest_problems(good, counts) == []
    twice = dict(scope="all", shards=[dict(shard=0, questions=2, videos=["a"]), dict(shard=1, questions=4, videos=["a", "b", "c"])])
    assert "a: in more than one shard" in shards.manifest_problems(twice, counts)
    gone = dict(scope="all", shards=[dict(shard=0, questions=2, videos=["a"]), dict(shard=1, questions=1, videos=["b"])])
    assert shards.manifest_problems(gone, counts) == ["c: in no shard"]
    assert shards.manifest_problems(dict(gone, scope="pilot"), counts) == []                            # a pilot covers only its videos
    wrong = dict(scope="all", shards=[dict(shard=0, questions=9, videos=["a", "b", "c"])])
    assert "9 questions in the manifest, 4 in the questions file" in shards.manifest_problems(wrong, counts)[0]
    assert "x: has no questions" in shards.manifest_problems(dict(scope="p", shards=[dict(shard=0, questions=0, videos=["x"])]), counts)
    assert "shard 0: no videos" in shards.manifest_problems(dict(scope="p", shards=[dict(shard=0, questions=0, videos=[])]), counts)


def test_the_small_set_comes_from_the_questions_file_alone():
    counts = dict(loc_big=50, loc_small=20, loc_tie=20, m_big=90, m_small=4, o3=1, o1=1, o2=1, o4=1)
    assert shards.smallest_videos(counts) == ["loc_small", "m_small", "o1", "o2", "o3"]


def test_write_command_for_each_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("GV_ROOT", str(tmp_path))
    monkeypatch.setenv("EGO_ROOT", str(tmp_path))
    with open(tmp_path / "Egoeverything_VQA.json", "w") as f:
        json.dump([dict(video_name=v) for v in ["loc_a"] * 3 + ["m"] * 2 + ["o1", "o2", "o3"]], f)
    paths.write_list("all350_pilot", ["loc_a", "o1", "o2"])
    arg = lambda name, **k: SimpleNamespace(**dict(dict(root=str(tmp_path / "results" / name), shards=3, all=False, list=None, smallest=False), **k))
    shards.write_cmd(arg("pilot", list="all350_pilot"))
    pilot = shards.read_manifest(tmp_path / "results" / "pilot")
    assert pilot["scope"] == "all350_pilot" and pilot["n_videos"] == 3 and pilot["n_questions"] == 5
    shards.write_cmd(arg("check", smallest=True))
    assert shards.read_manifest(tmp_path / "results" / "check")["n_videos"] == 5
    with pytest.raises(SystemExit, match="expected"):                                                   # --all must be the real dataset
        shards.write_cmd(arg("full", all=True))
    assert not (tmp_path / "results" / "full").exists()


@pytest.mark.parametrize("folder", shards.PROTECTED)
def test_nothing_is_written_inside_a_read_only_results_folder(folder, tmp_path, monkeypatch):
    monkeypatch.setenv("GV_ROOT", str(tmp_path))
    with pytest.raises(SystemExit, match="only read"):
        shards.write_manifest(str(tmp_path / "results" / folder / "x"), shards.manifest_of(dict(a=1), 1, "p"))
    with pytest.raises(SystemExit, match="only read"):
        merge_shards.merge(str(tmp_path / "results" / folder), M, "t", "c", ROWS)


# ---------------------------------------------------------------------------- threads on Qwen3's baseline rows
BASELINES = {
    "avgpool": (ap.run_questions, ap.Qwen3, dict(x=dict(side=2), y=dict(side=4)), 1),
    "gazevlm": (gz.run_questions, gz.Qwen3, dict(x=dict(rho=0.3), y=dict(rho=0.6)), 1),
    "gazevlm2": (gv2.run_questions, gv2.Qwen3, dict(x=dict(tokens=8), y=dict(tokens=16)), 2),
    "streamingtom": (stm.run_questions, stm.Qwen3, dict(x=dict(ratio=0.05), y=dict(ratio=0.2)), 1),
    "dycoke": (dy.run_questions, dy.Qwen3, dict(x=dict(window=2, prune=0.9), y=dict(window=2, prune=0.5)), 1),
    "fastvid": (fv.run_questions, fv.Qwen3, dict(x=dict(r=0.05), y=dict(r=0.2)), 1),
    "prunevid": (pv.run_questions, pv.Qwen3, dict(x=dict(tau=0.8, beta=0.05), y=dict(tau=0.8, beta=0.2)), 1),
    "dtd": (dtd.run_questions, dtd.Qwen3, dict(x=dict(threshold=0.9, absolute=False), y=dict(threshold=0.5, absolute=False)), 1),
}


def small_items(folder):
    """2 questions on each of 2 videos."""
    out = []
    for video in ("a", "b"):
        frames = write_frames(folder, video, 3)
        out += [dict(video=video, question="q", options=list("abcde"), answer="A", frames=frames) for _ in range(2)]
    return [dict(it, qid=i) for i, it in enumerate(out)]


def counted_loads(monkeypatch):
    calls, real = [], qf.load_frame01

    def wrapper(path, size=qf.IMG):
        calls.append(path)
        return real(path, size)

    monkeypatch.setattr(qf, "load_frame01", wrapper)
    return calls


@pytest.mark.parametrize("row", sorted(BASELINES))
def test_threads_change_nothing_and_load_each_view_once(row, tmp_path, monkeypatch):
    fake_model(monkeypatch)
    run_questions, backend, conds, views = BASELINES[row]
    calls = counted_loads(monkeypatch)
    items = small_items(tmp_path)
    build = lambda *a: []
    run_questions(backend(None, PROC, [0]), items, conds, str(tmp_path / "off.jsonl"), build)
    assert len(calls) == 2 * 3 * views                                                     # 2 videos, 3 frames, once for both conditions
    calls.clear()
    be = backend(None, PROC, [0])
    run_questions(be, items, conds, str(tmp_path / "on.jsonl"), build, prefetch_workers=3)
    assert len(calls) == 2 * 3 * views
    assert be.ready == {}                                                                  # every loaded frame was used
    assert without_time(tmp_path / "on.jsonl") == without_time(tmp_path / "off.jsonl")
    assert len(read_jsonl(tmp_path / "on.jsonl")) == 4 * len(conds)


def test_two_view_threads_keep_each_views_own_image(tmp_path, monkeypatch):
    """The two views of a frame share its key, so a lookup by key would give the crop the whole frame."""
    fake_model(monkeypatch)
    be = gv2.Qwen3(None, PROC, [0])
    frame = write_frames(tmp_path, "a", 1)[0]
    gv2.prefetch_views(be, [frame], [0.5], str(tmp_path), workers=2)
    crop = gv2.crop_file(str(tmp_path), frame, 0.5)
    assert sorted(be.ready) == sorted([(frame["path"], q3.IMG), (crop, q3.IMG)])
    assert not torch.equal(be.ready[(frame["path"], q3.IMG)], be.ready[(crop, q3.IMG)])


def test_threads_give_exactly_what_loading_one_at_a_time_gives(tmp_path, monkeypatch):
    fake_model(monkeypatch)
    frames = write_frames(tmp_path, "a", 3)
    one, ahead = stm.Qwen3(None, PROC, [0]), stm.Qwen3(None, PROC, [0])
    ahead.prefetch(frames, 2)
    for fr in frames:
        assert torch.equal(one.frame_input(fr), ahead.frame_input(fr))
    assert ahead.ready == {}


def test_qwen3_is_the_only_model_this_adds_baseline_threads_to(monkeypatch):
    monkeypatch.setattr(run, "ADAPTERS", models.ADAPTERS)
    assert run.baselines_without_threads(["qwen3", "smolvlm"], ROWS, 8) == []
    assert run.baselines_without_threads(["qwen2.5", "phi", "internvl", "gemma"], ["dtd"], 8) == ["qwen2.5", "phi", "internvl", "gemma"]


# ---------------------------------------------------------------------------- the token counts
def scenes(seed, n_frames, n_tokens, same=False):
    """Frames in runs of near-duplicates with changes of scene between them, so segments, static tokens and skipped frames all occur."""
    g = torch.Generator().manual_seed(seed)
    base, out = torch.randn(n_tokens, 8, generator=g), []
    for f in range(n_frames):
        if not same and f and f % 5 == 0:
            base = torch.randn(n_tokens, 8, generator=g)
        noise = 0.0 if same else 0.02 * (f % 3)
        out.append(dict(tokens=base + noise * torch.randn(n_tokens, 8, generator=g), sal=torch.rand(n_tokens, generator=g)))
    return out


@pytest.mark.parametrize("seed,n_frames,n_tokens,same", [(0, 1, 36, False), (1, 2, 16, True), (2, 12, 64, False), (3, 30, 81, False), (4, 6, 256, False),
                                                         (5, 9, 64, True), (6, 3, 16, False)])
def test_the_token_arithmetic_equals_the_methods_own_plans(seed, n_frames, n_tokens, same):
    states = scenes(seed, n_frames, n_tokens, same)
    s_fv, s_pv = common.DEFAULTS["fastvid"], common.DEFAULTS["prunevid"]
    assert cal.fastvid_counts(states, cal.GRID_R, s_fv) == [cal.fastvid_kept(states, r, s_fv) for r in cal.GRID_R]
    assert cal.prunevid_counts(states, cal.GRID_BETA, s_pv) == [cal.prunevid_kept(states, b, s_pv) for b in cal.GRID_BETA]


def test_deepstack_features_dont_change_the_counts():
    states = scenes(7, 12, 64)
    deep = [dict(st, deep=[st["tokens"] * 2, st["tokens"] * 3, st["tokens"] * 4]) for st in states]
    s_fv, s_pv = common.DEFAULTS["fastvid"], common.DEFAULTS["prunevid"]
    assert [cal.fastvid_kept(deep, r, s_fv) for r in (0.05, 0.2)] == [cal.fastvid_kept(states, r, s_fv) for r in (0.05, 0.2)]
    assert [cal.prunevid_kept(deep, b, s_pv) for b in (0.05, 0.2)] == [cal.prunevid_kept(states, b, s_pv) for b in (0.05, 0.2)]


# ---------------------------------------------------------------------------- a small dataset, calibrated, scored whole and in 3 shards, merged
VIDEOS = {"loc_a": 3, "loc_b": 2, "2023_m": 4, "2023_o1": 1, "2023_o2": 1}
QUESTION_ORDER = ["2023_m", "loc_a", "2023_o1", "loc_b", "loc_a", "2023_m", "2023_m", "loc_b", "2023_o2", "loc_a", "2023_m"]     # a qid is a position
N_FRAMES = 5


def data_of(w, videos):
    """What common.load_data gives for these videos."""
    qs = w.qs[w.qs.video.isin(set(videos))].reset_index(drop=True)
    return dict(qs=qs, videos=list(videos), test_videos=set(), metas={v: dict(n=N_FRAMES) for v in videos}, gate={},
                uniform={v: w.frames[v] for v in videos}, clips=[dict(sequence=v, frame_no=np.arange(N_FRAMES)) for v in videos],
                labels={v: w.labels[v] for v in videos}, frame=lambda v, f: w.frames[v][f])


def score(w, out_dir, videos, workers):
    """A run of every row, as run.py does it. Only here does parity stay off: the fake has no model to compare with."""
    cfg = common.load_config(w.base)
    cfg["paths"]["out_dir"] = str(out_dir)
    cfg["run"]["parity"] = False
    cfg["run"]["prefetch_workers"] = workers
    run.run_model(M, cfg, data_of(w, videos), ROWS)


def run_shard(w, root, shard, workers=2, repo=None, gpu=GPU):
    """What the job does: record the GPU, repair, check and record the job, score."""
    folder = shards.shard_folder(str(root), shard, M)
    os.makedirs(folder, exist_ok=True)
    check_gpu.record(os.path.join(folder, "gpu_types.txt"), gpu)
    repair(folder)
    shards.start(str(root), shard, M, w.base, w.targets, ROWS, workers, gpu, VERSIONS, repo=repo or w.repo)
    score(w, os.path.join(str(root), "shards", str(shard)), shards.read_manifest(str(root))["shards"][shard]["videos"], workers)


def fake_repo(folder, commit="abc123", code="x = 1\n"):
    os.makedirs(os.path.join(folder, ".git", "refs", "heads"))
    os.makedirs(os.path.join(folder, "accuracy_tables"))
    with open(os.path.join(folder, ".git", "HEAD"), "w") as f:
        f.write("ref: refs/heads/eyehead-gate\n")
    with open(os.path.join(folder, ".git", "refs", "heads", "eyehead-gate"), "w") as f:
        f.write(commit + "\n")
    with open(os.path.join(folder, "accuracy_tables", "x.py"), "w") as f:
        f.write(code)
    return folder


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    base = tmp_path_factory.mktemp("world")
    g, e = base / "g", base / "e"
    os.makedirs(g)
    os.makedirs(e)
    mp.setenv("GV_ROOT", str(g))
    mp.setenv("EGO_ROOT", str(e))
    fake_model(mp)
    real_bootstrap = egoev.video_bootstrap
    mp.setattr(egoev, "video_bootstrap", lambda df, stat, B=1000, seed=0: real_bootstrap(df, stat, 50, seed))     # the same seed, fewer draws
    with open(e / "Egoeverything_VQA.json", "w") as f:
        json.dump([dict(video_name=v) for v in QUESTION_ORDER], f)
    rng = np.random.default_rng(0)
    w = SimpleNamespace(frames={v: write_frames(base, v, N_FRAMES) for v in VIDEOS}, labels={v: rng.integers(0, 4, N_FRAMES - 1) for v in VIDEOS},
                        qs=pd.DataFrame(dict(qid=range(len(QUESTION_ORDER)), video=QUESTION_ORDER, question="q", options=[list("abcde")] * len(QUESTION_ORDER),
                                             answer=["A", "B"] * 5 + ["A"])),
                        repo=fake_repo(str(base / "repo")), root=g / "results" / "qwen3_350", whole=g / "results" / "whole")
    # calibrate, through its real main
    mp.setattr(common, "load_data", lambda cfg: data_of(w, list(VIDEOS)))
    mp.setattr(cal, "ADAPTERS", dict(qwen3=Qwen3))
    mp.setattr(run, "ADAPTERS", dict(qwen3=Qwen3))
    w.cal = w.root / "calibrate"
    w.base = str(w.cal / "config_baselines.yaml")
    w.targets = str(w.cal / M / "targets.csv")
    w.counts = str(w.cal / M / "kept_counts.jsonl")
    w.cal_gpu = str(w.cal / M / "gpu_types.txt")
    os.makedirs(w.cal / M)
    check_gpu.record(w.cal_gpu, GPU)
    cal.main(["--config", prepare.CONFIG, "--model", M, "--fractions", "--prefetch-workers", "2", "--counts", w.counts, "--out-dir", str(w.cal), "--write", w.base])
    # scored whole with no threads, then in 3 shards with threads
    score(w, w.whole, list(VIDEOS), 0)
    shards.write_manifest(str(w.root), shards.manifest_of(dict(VIDEOS), 3, "all"))
    for shard in range(3):
        run_shard(w, w.root, shard)
    w.merge_info = merge_shards.merge(str(w.root), M, w.targets, w.base, ROWS, w.cal_gpu)
    yield w
    mp.undo()


@pytest.fixture
def copy(world, tmp_path):
    """The shards and the merged results, free to damage."""
    out = tmp_path / "copy"
    shutil.copytree(world.root, out)
    return out


# ---------------------------------------------------------------------------- calibration
def test_targets_use_the_protocols_budgets_and_the_exact_rows_land_on_the_token_rule(world):
    t = pd.read_csv(world.targets)
    assert sorted(t[t.row == "avgpool"].target_pct) == [5, 10, 20] and set(t.row) == set(ROWS) and len(t) == 2 + 3 + 8 * 3
    p = t.set_index(["row", "target_pct"]).predicted_pct
    for row, tokens in (("avgpool", (16, 25, 49)), ("gazevlm2", (12, 26, 52)), ("streamingtom", (13, 26, 51))):
        for f, n in zip((5, 10, 20), tokens):
            assert p[row, f] == pytest.approx(100 * n / 256), row
    assert sorted(t[t.row == "budget"].condition) == sorted(f"budget rule {b}% (D+P gate, fovea R=70 a=160)" for b in (5, 10, 20))


def test_each_rows_predicted_budget_is_what_scoring_it_measures(world):
    """Token counts only, and no gap: every baseline's prediction equals its measured budget."""
    res = summarize.load_model_results(str(world.whole), M)
    measured = summarize.budgets(res)
    t = pd.read_csv(world.targets)
    lines = t[t.predicted_pct.notna()]
    assert len(lines) == 8 * 3
    for line in lines.itertuples():
        assert measured[line.condition] == pytest.approx(line.predicted_pct, abs=1e-9), (line.row, line.target_pct)


def test_the_counts_and_the_whole_plans_choose_the_same_knobs(world, tmp_path):
    out, written = tmp_path / "slow", tmp_path / "slow" / "config.yaml"
    os.makedirs(out)
    cal.main(["--config", prepare.CONFIG, "--model", M, "--fractions", "--out-dir", str(out), "--write", str(written)])
    assert (out / M / "targets.csv").read_text() == open(world.targets).read()
    assert not (out / M / "kept_counts.jsonl").exists()


def test_the_knobs_change_only_the_budget_parameters(world):
    c, d = common.load_config(world.base), common.DEFAULTS
    assert c["run"]["K"] == 96 and c["run"]["policy"] == "D+P" and c["run"]["parity"] is True and c["ours"] == d["ours"]
    assert c["budget"]["fractions"] == [0.05, 0.1, 0.2] and c["budget"]["tokens_per_frame"]["qwen3"] == 51
    for section, keys in (("streamingtom", ("tau", "knn", "beta", "released_dpc")), ("fastvid", ("c", "tau", "ignore", "d", "p", "beta")),
                          ("prunevid", ("gamma", "knn", "min_tokens")), ("gazevlm", ("sigma", "r_min", "thumb", "gaze")), ("gazevlm2", ("alpha", "beta"))):
        assert all(c[section][k] == d[section][k] for k in keys)
    assert {s["window"] for s in c["dycoke"]["settings"]} == {32} and {s["tau"] for s in c["prunevid"]["settings"]} == {0.8}


def small_counts(world):
    frames = {v: world.frames[v] for v in VIDEOS}
    be = fv.Qwen3(None, PROC, [0])
    real, seen = be.frame_state, []

    def frame_state(fr):
        seen.append(fr["key"])
        return real(fr)

    be.frame_state = frame_state
    return frames, be, seen


def test_calibrate_continues_from_its_counts_after_a_kill(world, tmp_path):
    cfg = common.load_config(prepare.CONFIG)
    frames, be, seen = small_counts(world)
    path = tmp_path / "counts" / "kept_counts.jsonl"
    quiet = lambda *a: None
    first = cal.resumable_counts(str(path), frames, be, cfg, 16, quiet)
    assert len(seen) == len(VIDEOS) * N_FRAMES
    seen.clear()
    assert cal.resumable_counts(str(path), frames, be, cfg, 16, quiet) == first and seen == []       # all counted: nothing encoded
    text = path.read_text()
    path.write_text(text[:-20])                                                                      # killed writing the last video's line
    assert repair(str(path.parent)) == [str(path)]
    last = json.loads(text.splitlines()[-1])["video"]
    assert cal.resumable_counts(str(path), frames, be, cfg, 16, quiet) == first
    assert seen == [(last, f) for f in range(N_FRAMES)]                                              # only that video again


def test_counts_from_other_settings_are_refused(world, tmp_path):
    cfg = common.load_config(prepare.CONFIG)
    frames, be, _ = small_counts(world)
    path = tmp_path / "kept_counts.jsonl"
    cal.resumable_counts(str(path), {"loc_b": frames["loc_b"]}, be, cfg, 16, lambda *a: None)
    other = common.load_config(prepare.CONFIG)
    other["fastvid"]["tau"] = 0.5
    with pytest.raises(SystemExit, match="other settings"):
        cal.resumable_counts(str(path), frames, be, other, 16, lambda *a: None)


def test_a_job_stops_if_the_arithmetic_differs_from_the_plans_on_real_frames(world, tmp_path, monkeypatch):
    cfg = common.load_config(prepare.CONFIG)
    frames, be, _ = small_counts(world)
    real = cal.fastvid_counts
    monkeypatch.setattr(cal, "fastvid_counts", lambda *a: [n + 1 for n in real(*a)])
    with pytest.raises(SystemExit, match="token arithmetic"):
        cal.resumable_counts(str(tmp_path / "kept_counts.jsonl"), frames, be, cfg, 16, lambda *a: None)
    assert (tmp_path / "kept_counts.jsonl").read_text() == ""                                        # nothing wrong was kept


# ---------------------------------------------------------------------------- full budget against Uniform K
def test_full_budget_rows_give_uniform_ks_answers_on_qwen3(world, tmp_path):
    path = tmp_path / "identity.yaml"
    check_identity.write_config(prepare.CONFIG, str(path))
    cfg = common.load_config(str(path))
    cfg["paths"]["out_dir"] = str(tmp_path / "out")
    cfg["run"]["parity"] = False
    cfg["run"]["prefetch_workers"] = 2
    run.run_model(M, cfg, data_of(world, ["loc_b", "2023_o1"]), ["uniform", *check_identity.ROWS])
    folder = tmp_path / "out" / M
    uniform = check_identity.read_records(folder / "uniform.jsonl")
    for row in check_identity.ROWS:
        r = check_identity.compare(uniform, check_identity.read_records(folder / f"{row}.jsonl"))
        assert r["records"] == 3 and check_identity.problems(r) == 0 and r["max_prob_diff"] == 0, row


# ---------------------------------------------------------------------------- merged against unsharded
@pytest.mark.parametrize("row", ROWS)
def test_merged_files_match_an_unsharded_run_line_by_line(world, row):
    merged, whole = world.root / M / f"{row}.jsonl", world.whole / M / f"{row}.jsonl"
    assert without_time(merged) == without_time(whole)
    assert len(read_jsonl(merged)) > 0


def test_merged_lines_are_each_shards_own_lines_unchanged(world):
    for row in ROWS:
        lines = set()
        for shard in range(3):
            with open(os.path.join(shards.shard_folder(str(world.root), shard, M), f"{row}.jsonl")) as f:
                lines |= set(f.readlines())
        with open(world.root / M / f"{row}.jsonl") as f:
            assert set(f.readlines()) == lines


def test_every_question_is_merged_exactly_once(world):
    conditions = merge_shards.read_targets(world.targets)
    for row in ROWS:
        got = [(r["condition"], r["qid"]) for r in read_jsonl(world.root / M / f"{row}.jsonl")]
        assert len(got) == len(set(got)) == len(conditions[row]) * len(QUESTION_ORDER)
        assert {q for _, q in got} == set(range(len(QUESTION_ORDER)))


def summary_of(folder):
    cfg = common.load_config(prepare.CONFIG)
    cfg["paths"]["out_dir"] = str(folder)
    return cfg


def test_the_summary_gives_identical_numbers(world):
    a, b = summarize.summarize(summary_of(world.whole)), summarize.summarize(summary_of(world.root))
    assert len(a) == pd.read_csv(world.targets).condition.nunique()
    pd.testing.assert_frame_equal(a, b)
    groups = report.video_groups(list(VIDEOS))
    pd.testing.assert_frame_equal(report.build_table(summary_of(world.whole), groups), report.build_table(summary_of(world.root), groups))


def test_the_existing_report_reads_the_merged_results(world, tmp_path, monkeypatch, capsys):
    root = tmp_path / "copy"
    shutil.copytree(world.root, root)
    monkeypatch.setattr(sys, "argv", ["report", "--out-dir", str(root), "--models", M, "--targets", world.targets])
    report.main()
    out = capsys.readouterr().out
    assert "EVERY n IS COMPLETE" in out and report.NOTE in out and GPU in out
    table = pd.read_csv(root / "summary_by_group.csv")
    assert set(table.group) == {"all", "aea", "nymeria"} and (table.note == report.NOTE).all()
    assert set(table[table.group == "aea"].n) == {5} and set(table[table.group == "nymeria"].n) == {6} and set(table[table.group == "all"].n) == {11}


def test_the_shards_are_what_the_assignment_says(world):
    assert world.merge_info["gpu"] == GPU and world.merge_info["commit"] == "abc123" and world.merge_info["n_questions"] == 11
    assert [(s["questions"], s["videos"]) for s in world.merge_info["shards"]] == [(4, 1), (4, 2), (3, 2)]
    for s in shards.read_manifest(str(world.root))["shards"]:
        videos = {r["video"] for r in read_jsonl(os.path.join(shards.shard_folder(str(world.root), s["shard"], M), "uniform.jsonl"))}
        assert videos == set(s["videos"])                                                     # whole videos, and only its own
    with open(world.root / M / "gpu_types.txt") as f:
        assert f.read() == GPU + "\n"


# ---------------------------------------------------------------------------- what the merge refuses
def shard_file(copy, shard, name):
    return copy / "shards" / str(shard) / M / name


def edit_jobs(copy, shard, **changes):
    path = shard_file(copy, shard, "shard_info.jsonl")
    lines = [dict(line, **changes) for line in read_jsonl(path)]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def drop_a_question(copy):
    path = shard_file(copy, 0, "dtd.jsonl")
    path.write_text("".join(path.read_text().splitlines(True)[:-1]))


def drop_a_row(copy):
    shard_file(copy, 1, "gazevlm.jsonl").unlink()


def duplicate_a_record(copy):
    path = shard_file(copy, 2, "uniform.jsonl")
    path.write_text(path.read_text() + path.read_text().splitlines(True)[0])


def record_in_the_wrong_shard(copy):
    other = shard_file(copy, 0, "uniform.jsonl").read_text().splitlines(True)[0]
    path = shard_file(copy, 1, "uniform.jsonl")
    path.write_text(path.read_text() + other)


def half_written_line(copy):
    path = shard_file(copy, 0, "blind.jsonl")
    path.write_text(path.read_text()[:-20])


def garbled_line(copy):
    path = shard_file(copy, 0, "blind.jsonl")
    path.write_text("not json\n" + path.read_text())


def error_record(copy):
    path = shard_file(copy, 1, "budget.jsonl")
    lines = path.read_text().splitlines(True)
    path.write_text("".join(lines[:-1]) + json.dumps(dict(json.loads(lines[-1]), error="oom")) + "\n")


def stray_condition(copy):
    path = shard_file(copy, 2, "dtd.jsonl")
    lines = path.read_text().splitlines(True)
    path.write_text("".join(lines) + json.dumps(dict(json.loads(lines[0]), condition="DTD from an older run")) + "\n")


def wrong_qid(copy):
    path = shard_file(copy, 2, "uniform.jsonl")
    line = json.loads(path.read_text().splitlines()[0])
    other = QUESTION_ORDER.index("2023_m")
    path.write_text(path.read_text() + json.dumps(dict(line, qid=other)) + "\n")


def other_gpu_file(copy):
    shard_file(copy, 2, "gpu_types.txt").write_text("NVIDIA A100 80GB PCIe\n")


def two_gpus_in_one_shard(copy):
    path = shard_file(copy, 1, "gpu_types.txt")
    path.write_text(path.read_text() + "NVIDIA A100 80GB PCIe\n")


def no_gpu_file(copy):
    shard_file(copy, 0, "gpu_types.txt").unlink()


def video_in_two_shards(copy):
    path = copy / "manifest" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["shards"][1]["videos"].append(manifest["shards"][0]["videos"][0])
    path.write_text(json.dumps(manifest))


def never_started(copy):
    shard_file(copy, 1, "shard_info.jsonl").unlink()


DAMAGE = [
    (drop_a_question, "questions missing"),
    (drop_a_row, "no results file"),
    (duplicate_a_record, "duplicate"),
    (record_in_the_wrong_shard, "isn't this shard's"),
    (half_written_line, "half-written"),
    (garbled_line, "unreadable"),
    (error_record, "error record (oom)"),
    (stray_condition, "condition not in the targets"),
    (wrong_qid, "qid of another video"),
    (other_gpu_file, "more than one GPU name"),
    (two_gpus_in_one_shard, "not one GPU name"),
    (no_gpu_file, "not one GPU name"),
    (video_in_two_shards, "in more than one shard"),
    (never_started, "no job has started it"),
    (lambda c: edit_jobs(c, 1, gpu="NVIDIA A100 80GB PCIe"), "gpu"),
    (lambda c: edit_jobs(c, 2, commit="def456"), "commit"),
    (lambda c: edit_jobs(c, 2, code_sha256="local edit"), "code_sha256"),
    (lambda c: edit_jobs(c, 0, targets_sha256="other"), "targets_sha256"),
    (lambda c: edit_jobs(c, 1, config_sha256="other"), "config_sha256"),
    (lambda c: edit_jobs(c, 2, manifest_sha256="other"), "manifest_sha256"),
    (lambda c: edit_jobs(c, 1, rows=ROWS[:-1]), "rows"),
    (lambda c: edit_jobs(c, 1, workers=4), "workers"),
    (lambda c: edit_jobs(c, 2, versions=dict(VERSIONS, torch="2.0")), "versions"),
]


def contents(folder):
    out = {}
    for base, _, files in os.walk(folder):
        for name in files:
            with open(os.path.join(base, name)) as f:
                out[os.path.relpath(os.path.join(base, name), folder)] = f.read()
    return out


@pytest.mark.parametrize("damage,says", DAMAGE, ids=[str(i) for i in range(len(DAMAGE))])
def test_the_merge_stops_and_writes_nothing(damage, says, world, copy, capsys):
    before = contents(copy / M)
    damage(copy)
    with pytest.raises(SystemExit, match="MERGE STOPPED"):
        merge_shards.merge(str(copy), M, world.targets, world.base, ROWS, world.cal_gpu)
    assert says in capsys.readouterr().out
    assert contents(copy / M) == before                                                  # the earlier results are untouched
    assert not (copy / f"{M}.merging").exists()


def test_a_refused_merge_leaves_no_final_results(world, copy):
    shutil.rmtree(copy / M)
    drop_a_question(copy)
    with pytest.raises(SystemExit, match="MERGE STOPPED"):
        merge_shards.merge(str(copy), M, world.targets, world.base, ROWS, world.cal_gpu)
    assert not (copy / M).exists()


def test_the_merge_refuses_a_targets_or_settings_file_that_isnt_what_the_shards_ran_with(world, copy, tmp_path):
    changed = tmp_path / "targets.csv"
    changed.write_text(open(world.targets).read() + "\n")
    assert any("targets file isn't" in m for m in merge_shards.plan_merge(str(copy), M, str(changed), world.base, ROWS, world.cal_gpu)[2])
    settings = tmp_path / "config.yaml"
    settings.write_text(open(world.base).read() + "# edited\n")
    assert any("settings file isn't" in m for m in merge_shards.plan_merge(str(copy), M, world.targets, str(settings), ROWS, world.cal_gpu)[2])


def test_the_merge_refuses_a_calibration_on_another_gpu(world, copy, tmp_path):
    other = tmp_path / "gpu_types.txt"
    other.write_text("NVIDIA A100 80GB PCIe\n")
    assert any("more than one GPU name" in m for m in merge_shards.plan_merge(str(copy), M, world.targets, world.base, ROWS, str(other))[2])
    other.write_text(GPU + "\nNVIDIA A100 80GB PCIe\n")
    assert any("calibration" in m for m in merge_shards.plan_merge(str(copy), M, world.targets, world.base, ROWS, str(other))[2])


def test_the_merge_refuses_rows_other_than_the_targets(world, copy):
    found = merge_shards.plan_merge(str(copy), M, world.targets, world.base, ROWS[:-1], world.cal_gpu)[2]
    assert any("aren't the rows of the targets" in m for m in found)


def test_an_undamaged_copy_merges_again_to_the_same_files(world, copy):
    before = contents(copy / M)
    merge_shards.merge(str(copy), M, world.targets, world.base, ROWS, world.cal_gpu)
    assert contents(copy / M) == before
    assert not (copy / f"{M}.old").exists() and not (copy / f"{M}.merging").exists()


# ---------------------------------------------------------------------------- resuming a shard
def test_a_shard_continues_after_a_kill_left_a_half_written_last_line(world, copy):
    others = {s: contents(copy / "shards" / str(s)) for s in (1, 2)}
    path = shard_file(copy, 0, "dtd.jsonl")
    complete = path.read_text()
    lines = complete.splitlines(True)
    path.write_text("".join(lines[:-1]) + lines[-1][:30])                                # killed in the middle of writing its last record
    with pytest.raises(SystemExit, match="MERGE STOPPED"):
        merge_shards.merge(str(copy), M, world.targets, world.base, ROWS, world.cal_gpu)
    run_shard(world, copy, 0)                                                            # one shard alone
    assert without_time(path) == [{k: x for k, x in json.loads(line).items() if k != "sec"} for line in lines]
    for s, before in others.items():
        assert contents(copy / "shards" / str(s)) == before                              # the other shards weren't touched
    assert len(read_jsonl(shard_file(copy, 0, "shard_info.jsonl"))) == 2                 # both of its jobs are on record
    merge_shards.merge(str(copy), M, world.targets, world.base, ROWS, world.cal_gpu)
    for row in ROWS:
        assert without_time(copy / M / f"{row}.jsonl") == without_time(world.whole / M / f"{row}.jsonl")


def test_a_shard_that_ran_only_part_way_finishes_the_rest(world, copy):
    path = shard_file(copy, 2, "gazevlm2.jsonl")
    lines = path.read_text().splitlines(True)
    path.write_text("".join(lines[:2]))
    shard_file(copy, 2, "budget.jsonl").unlink()
    run_shard(world, copy, 2)
    merge_shards.merge(str(copy), M, world.targets, world.base, ROWS, world.cal_gpu)
    for row in ROWS:
        assert without_time(copy / M / f"{row}.jsonl") == without_time(world.whole / M / f"{row}.jsonl")


@pytest.mark.parametrize("change,says", [(dict(commit="def456"), "commit"), (dict(code="x = 2\n"), "code_sha256")])
def test_a_rerun_under_other_code_stops_before_scoring(change, says, world, copy, tmp_path):
    other = fake_repo(str(tmp_path / "other"), **change)
    before = shard_file(copy, 0, "shard_info.jsonl").read_text()
    with pytest.raises(SystemExit, match=says):
        shards.start(str(copy), 0, M, world.base, world.targets, ROWS, 2, GPU, VERSIONS, repo=other)
    assert shard_file(copy, 0, "shard_info.jsonl").read_text() == before


def test_a_job_on_another_gpu_or_targets_stops_before_scoring(world, copy, tmp_path):
    with pytest.raises(SystemExit, match="gpu"):
        shards.start(str(copy), 1, M, world.base, world.targets, ROWS, 2, "NVIDIA A100 80GB PCIe", VERSIONS, repo=world.repo)
    other = tmp_path / "targets.csv"
    other.write_text("changed\n")
    with pytest.raises(SystemExit, match="targets_sha256"):
        shards.start(str(copy), 1, M, world.base, str(other), ROWS, 2, GPU, VERSIONS, repo=world.repo)
    with pytest.raises(SystemExit, match="shards 0 to 2"):
        shards.start(str(copy), 3, M, world.base, world.targets, ROWS, 2, GPU, VERSIONS, repo=world.repo)


# ---------------------------------------------------------------------------- what a job records
def test_the_commit_is_read_without_git(tmp_path):
    repo = fake_repo(str(tmp_path / "r"))
    assert shards.git_commit(repo) == "abc123"
    os.remove(os.path.join(repo, ".git", "refs", "heads", "eyehead-gate"))
    with open(os.path.join(repo, ".git", "packed-refs"), "w") as f:
        f.write("# pack-refs with: peeled\nfff000 refs/heads/main\nabc999 refs/heads/eyehead-gate\n")
    assert shards.git_commit(repo) == "abc999"
    with open(os.path.join(repo, ".git", "HEAD"), "w") as f:
        f.write("deadbeef\n")
    assert shards.git_commit(repo) == "deadbeef"


def test_the_code_hash_sees_a_local_edit_and_ignores_what_isnt_code(tmp_path):
    repo = fake_repo(str(tmp_path / "r"))
    first = shards.code_sha256(repo)
    with open(os.path.join(repo, "accuracy_tables", "notes.txt"), "w") as f:
        f.write("not code")
    assert shards.code_sha256(repo) == first
    with open(os.path.join(repo, "accuracy_tables", "x.py"), "w") as f:
        f.write("x = 2\n")
    assert shards.code_sha256(repo) != first


def test_a_job_stops_on_any_other_gpu_name(tmp_path):
    path = tmp_path / "a" / "gpu_types.txt"
    assert check_gpu.not_the_same(GPU, str(path))                                        # no calibration on record
    check_gpu.record(str(path), GPU)
    assert not check_gpu.not_the_same(GPU, str(path))
    assert check_gpu.not_the_same("NVIDIA A100 80GB PCIe", str(path))
    assert not check_gpu.wrong_type(GPU, "A100") and check_gpu.wrong_type("NVIDIA L40S", "A100")
    check_gpu.record(str(path), "NVIDIA A100 80GB PCIe")
    assert check_gpu.not_the_same(GPU, str(path))                                        # a file with two names matches nothing


def test_exact_comparison_fails_on_any_probability_and_on_records_only_the_reference_has():
    a = dict(pred="A", n_image_tokens=10, n_frames=2, p=[0.6, 0.4])
    same = compare_records.compare_records({("c", 1): a}, {("c", 1): a})
    assert same["ref_only"] == 0 and compare_records.problems(same, exact=True) == 0
    moved = compare_records.compare_records({("c", 1): a}, {("c", 1): dict(a, p=[0.6001, 0.3999])})
    assert compare_records.problems(moved) == 0 and compare_records.problems(moved, exact=True) == 1
    short = compare_records.compare_records({("c", 1): a}, {("c", 1): a, ("c", 2): a})
    assert short["ref_only"] == 1 and compare_records.problems(short) == 0 and compare_records.problems(short, exact=True) == 1


def test_the_estimate_scales_answering_by_questions_and_encoding_by_videos():
    assert estimate_hours.split_seconds([1.0, 1.0, 1.0, 10.0]) == (4.0, 9.0)
    records = [dict(condition="c", sec=s, error=None) for s in (1.0, 1.0, 1.0, 10.0)] + [dict(condition="c", sec=99.0, error="oom")]
    assert estimate_hours.estimate(records, (4, 1), (40, 100)) == (13.0, 4.0 * 10 + 9.0 * 100)


def test_the_estimate_runs_on_merged_results(world, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["estimate", "--root", str(world.root), "--model", M])
    estimate_hours.main()
    out = capsys.readouterr().out
    assert "per shard" in out and "dtd:" in out


# ---------------------------------------------------------------------------- settings
def test_the_repos_qwen3_is_qwen3_vl_4b():
    assert models.Qwen3.default_ids == ("Qwen/Qwen3-VL-4B-Instruct",) and q3.MODEL_CANDIDATES == ("Qwen/Qwen3-VL-4B-Instruct",)
    with open(prepare.CONFIG) as f:
        assert "models" not in yaml.safe_load(f)                                         # no config points it elsewhere


def test_the_rule_keeps_its_approved_settings_on_qwen3():
    with open(prepare.CONFIG) as f:
        assert yaml.safe_load(f)["budget"]["tokens_per_frame"]["qwen3"] == 51
    assert [br.frame_limit(b, 96 * 256, 51, 10 ** 6) for b in (0.05, 0.1, 0.2)] == [24, 48, 96]
    assert common.DEFAULTS["run"]["K"] == 96 and common.DEFAULTS["run"]["policy"] == "D+P" and common.DEFAULTS["ours"]["rings"][0] == dict(R=70, a=160)


# ---------------------------------------------------------------------------- the jobs
def job(name):
    with open(os.path.join(REPO, "accuracy_tables", "hpc", name)) as f:
        return f.read()


JOBS = ["qwen3350.SBATCH", "qwen3350_merge.SBATCH", "qwen3350_tests.SBATCH"]


@pytest.mark.parametrize("name", JOBS)
def test_jobs_write_only_to_their_own_results_and_keep_the_rules(name):
    text = job(name)
    for folder in shards.PROTECTED:
        for start_of_line in (f"OUT=$GV_ROOT/results/{folder}\n", f"ROOT=$GV_ROOT/results/{folder} ", f"--out-dir $GV_ROOT/results/{folder}", f"--new $GV_ROOT/results/{folder}"):
            assert start_of_line not in text
    for banned in ("--no-parity", "nogate", "compare_budget", "seed_results", "afterok", "afterany"):
        assert banned not in text


def test_each_shard_is_its_own_one_gpu_job():
    text = job("qwen3350.SBATCH")
    assert "--gres=gpu:a100:1" in text and "GPU=A100" in text and "--time=08:00:00" in text and "--requeue" in text
    assert '--comment="preemption=yes;requeue=true"' in text and "--dependency=singleton" in text and "--nodes=1" in text
    assert "WANT=qwen3350-$MODE-$SHARD" in text and '"$SLURM_JOB_NAME" != "$WANT"' in text               # singleton only stops the same shard
    assert "S=$ROOT/shards/$SHARD" in text and "--out-dir $S --keep-going" in text and "repair_results $S/$M" in text
    assert "--expect $GPU --same-as $CAL_GPU --record $S/$M/gpu_types.txt" in text
    assert "shard_$SHARD.txt" in text and "--prefetch-workers $W" in text


def test_calibrate_runs_once_and_leaves_its_results_read_only():
    text = job("qwen3350.SBATCH")
    assert 'if [ -f "$TARGETS" ]' in text and 'chmod a-w "$TARGETS" "$BASE"' in text
    assert "--fractions --videos all" in text and "--counts $CAL/$M/kept_counts.jsonl" in text
    for scope in ("--all", "--list all350_pilot", "--smallest"):
        assert f"--shards $SHARDS {scope}" in text


def test_every_row_is_scored_and_merged():
    text, merge = job("qwen3350.SBATCH"), job("qwen3350_merge.SBATCH")
    assert "ROWS=uniform,blind,budget,$BASELINES" in text
    baselines = text.split("BASELINES=")[1].splitlines()[0].split(",")
    assert ["uniform", "blind", "budget"] + baselines == ROWS
    assert "ROWS=" + ",".join(ROWS) + "\n" in merge


def test_merge_compares_and_summarizes_as_the_steps_say():
    merge = job("qwen3350_merge.SBATCH")
    assert "--ref $ROOT/whole/$M --rows $ROWS --exact" in merge
    assert "--ref $GV_ROOT/results/hpc350_pilot/$M --rows uniform,blind,budget --report" in merge
    assert "--calibration-gpu $CAL/$M/gpu_types.txt" in merge and "--models $M --targets $TARGETS" in merge and "estimate_hours" in merge
    assert "gres" not in merge                                                           # no GPU held


def test_the_check_job_covers_section_11s_checks_and_the_small_set_whole():
    text = job("qwen3350.SBATCH")
    assert "--prefetch-workers 0 --out-dir $CK/off" in text and "--prefetch-workers $W --out-dir $CK/on" in text
    assert "compare_records --new $CK/on/$M --ref $CK/off/$M --rows $ROWS" in text and "check_identity --compare $CK/identity/$M" in text
    assert "--videos $SMALL" in text and "--out-dir $CK/whole" in text


# ---------------------------------------------------------------------------- section 11's path
SECTION_11_JOBS = dict(zip(("smolvlm350.SBATCH", "smolvlm350_summary.SBATCH", "smolvlm350_tests.SBATCH"),
                           ("ca0f693884d45ace7db43b7605501a7c3972f1868a6de1cd365f56978d7e9495",
                            "364e45557594b66639bd9b47d4c1fa005cce8384aee06ab40eca0dad60d6475f",
                            "58490cf45286266aa8f9518ac0c05d0fda8654f18100ff9c9717c32a86a83696")))


@pytest.mark.parametrize("name", sorted(SECTION_11_JOBS))
def test_section_11s_jobs_are_byte_for_byte_what_ran(name):
    assert shards.file_sha256(os.path.join(REPO, "accuracy_tables", "hpc", name)) == SECTION_11_JOBS[name]


def test_section_11s_calibration_doesnt_use_the_new_counts():
    assert "--counts" not in job("smolvlm350.SBATCH") and "--same-as" not in job("smolvlm350.SBATCH")
    assert "--exact" not in job("smolvlm350.SBATCH")


def test_calibrate_without_counts_never_touches_the_new_path(world, tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("the path section 11 ran must not use this")

    monkeypatch.setattr(cal, "resumable_counts", refuse)
    monkeypatch.setattr(cal, "fastvid_counts", refuse)
    monkeypatch.setattr(cal, "prunevid_counts", refuse)
    out = tmp_path / "plain"
    os.makedirs(out)
    cal.main(["--config", prepare.CONFIG, "--model", M, "--fractions", "--out-dir", str(out), "--write", str(out / "config.yaml")])
    assert (out / M / "targets.csv").exists()


def test_check_gpu_with_section_11s_arguments_needs_nothing_new(tmp_path):
    path = tmp_path / "gpu_types.txt"
    check_gpu.record(str(path), "NVIDIA B200")
    assert not check_gpu.wrong_type("NVIDIA B200", "B200") and path.read_text() == "NVIDIA B200\n"

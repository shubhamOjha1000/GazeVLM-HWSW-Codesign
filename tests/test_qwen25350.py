"""Qwen2.5-VL-3B on all 350 videos (runbook section 13). Section 13 has no GPU check job, so these tests are where threads on its baseline rows, its
shard split and a merged run against an unsharded one get checked, along with Qwen3's files staying as its running jobs need them. CPU only, no dataset,
no model: the encoder and the LLM are faked, and the helpers come from tests/test_qwen3350.py, because the code under test is the same."""
import collections
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import test_qwen3350 as shared
from accuracy_tables import common, models, run, summarize
from accuracy_tables.hpc.all350 import check_identity, merge_shards, prepare, report, shard_progress, shards
from accuracy_tables.hpc.newvideos import compare_records
from vlmeval import avgpool as ap
from vlmeval import budget_rule as br
from vlmeval import dtd, dycoke as dy, fastvid as fv
from vlmeval import gazevlm as gz
from vlmeval import gazevlm2 as gv2
from vlmeval import prunevid as pv
from vlmeval import qwen_fovea as qf
from vlmeval import streamingtom as stm

REPO = shared.REPO
M = "qwen2.5"
ROWS = shared.ROWS
MODULES = dict(avgpool=ap, gazevlm=gz, gazevlm2=gv2, streamingtom=stm, dycoke=dy, fastvid=fv, prunevid=pv, dtd=dtd)


# ---------------------------------------------------------------------------- a fake Qwen2.5
def fake_encode(vis, x, cells, side, mean, std):
    return shared.fake_encode(vis, x, cells, side, mean, std)[0]


def fake_probs(model, inputs, letter_ids):
    v = (float(torch.cat(list(inputs["feats"])).float().mean()) * 37) % 1
    return np.array([v, 1 - v, 0.0, 0.0, 0.0])


def fake_model(mp):
    mp.setattr(qf, "visual_of", lambda model: shared.VISION)
    mp.setattr(qf, "check_encoder", lambda vis: None)
    mp.setattr(qf, "encode_cells", fake_encode)
    mp.setattr(qf, "build_inputs", lambda model, proc, msgs, feats, cells, sizes: (dict(feats=feats), torch.zeros((1, 9), dtype=torch.long)))
    mp.setattr(qf, "letter_probs", fake_probs)
    mp.setattr(stm, "capture_saliency", shared.fake_capture)


class Qwen25(models.Qwen25):
    """The real adapter without a model. Adapters find their backends by class name, so it keeps this one."""
    def load(self):
        self.model, self.proc, self.letter_ids, self.model_id = None, shared.PROC, list(range(5)), "fake"
        return self

    def free(self):
        pass

    def blind_ids(self, messages):
        return torch.zeros((1, 9), dtype=torch.long)

    def blind_logits(self, ids):
        return torch.tensor([2.0, 1.0, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        yield shared.build_world(tmp_path_factory.mktemp("world25"), mp, M, Qwen25, fake_model, "qwen25_350")
    finally:
        mp.undo()


@pytest.fixture
def copy(world, tmp_path):
    out = tmp_path / "copy"
    shutil.copytree(world.root, out)
    return out


def merge(world, root):
    return merge_shards.merge(str(root), M, world.targets, world.base, ROWS, world.cal_gpu)


# ---------------------------------------------------------------------------- threads
def test_qwen25s_baseline_rows_take_threads(monkeypatch):
    monkeypatch.setattr(run, "ADAPTERS", models.ADAPTERS)
    assert models.Qwen25.loads_on_threads and models.Qwen25.baselines_load_on_threads
    assert run.baselines_without_threads([M], ROWS, 8) == []


def test_qwen25s_images_are_identical_with_threads_on_and_off(tmp_path, monkeypatch):
    fake_model(monkeypatch)
    frames = shared.write_frames(tmp_path, "a", 4)
    off, on = stm.Qwen25(None, shared.PROC, [0]), stm.Qwen25(None, shared.PROC, [0])
    on.prefetch(frames, 3)
    assert len(on.ready) == 4
    for fr in frames:
        one_at_a_time, from_threads = off.frame_input(fr), on.frame_input(fr)
        assert one_at_a_time.shape == (3, 448, 448) and torch.equal(one_at_a_time, from_threads)
        assert torch.equal(from_threads, qf.resize01(qf.load_frame01(fr["path"], qf.IMG), 448))        # what the code did before threads
    assert on.ready == {}


@pytest.mark.parametrize("row", sorted(MODULES))
def test_threads_change_no_record_and_load_each_view_once(row, tmp_path, monkeypatch):
    fake_model(monkeypatch)
    run_questions, _, conds, views = shared.BASELINES[row]
    backend = MODULES[row].Qwen25
    calls = shared.counted_loads(monkeypatch)
    items = shared.small_items(tmp_path)
    build = lambda *a: []
    run_questions(backend(None, shared.PROC, [0]), items, conds, str(tmp_path / "off.jsonl"), build)
    assert len(calls) == 2 * 3 * views                                                     # 2 videos, 3 frames, once for both conditions
    calls.clear()
    be = backend(None, shared.PROC, [0])
    run_questions(be, items, conds, str(tmp_path / "on.jsonl"), build, prefetch_workers=3)
    assert len(calls) == 2 * 3 * views
    assert be.ready == {}                                                                  # every loaded frame was used
    assert shared.without_time(tmp_path / "on.jsonl") == shared.without_time(tmp_path / "off.jsonl")
    assert len(shared.read_jsonl(tmp_path / "on.jsonl")) == 4 * len(conds)


def test_two_view_threads_keep_each_views_own_image(tmp_path, monkeypatch):
    """The two views of a frame share its key, so a lookup by key would give the crop the whole frame."""
    fake_model(monkeypatch)
    be = gv2.Qwen25(None, shared.PROC, [0])
    frame = shared.write_frames(tmp_path, "a", 1)[0]
    gv2.prefetch_views(be, [frame], [0.5], str(tmp_path), workers=2)
    crop = gv2.crop_file(str(tmp_path), frame, 0.5)
    assert sorted(be.ready) == sorted([(frame["path"], 448), (crop, 448)])
    assert not torch.equal(be.ready[(frame["path"], 448)], be.ready[(crop, 448)])


# ---------------------------------------------------------------------------- the shard split
def test_every_question_lands_in_one_shard_and_whole_videos_stay_together():
    counts = shared.like_the_dataset()
    questions = [v for v, n in counts.items() for _ in range(n)]                           # a qid is a position in the questions file
    manifest = shards.manifest_of(collections.Counter(questions), 3, "all")
    shard_of = {}
    for s in manifest["shards"]:
        for v in s["videos"]:
            assert v not in shard_of                                                       # a whole video, in one shard
            shard_of[v] = s["shard"]
    assert set(shard_of) == set(counts)
    per_shard = collections.Counter(shard_of[v] for v in questions)
    assert [per_shard[s["shard"]] for s in manifest["shards"]] == [s["questions"] for s in manifest["shards"]]
    assert sum(per_shard.values()) == len(questions) == manifest["n_questions"]
    assert max(per_shard.values()) - min(per_shard.values()) <= 1                          # balanced by questions


def test_the_shards_scored_each_question_once_and_kept_each_video_whole(world):
    manifest = shards.read_manifest(str(world.root))
    conditions = merge_shards.read_targets(world.targets)
    for row in ROWS:
        where, shards_of_video = {}, collections.defaultdict(set)
        for s in manifest["shards"]:
            for r in shared.read_jsonl(os.path.join(shards.shard_folder(str(world.root), s["shard"], M), f"{row}.jsonl")):
                key = (r["condition"], r["qid"])
                assert key not in where, (row, key)                                        # exactly one shard
                where[key] = s["shard"]
                assert r["video"] in s["videos"]
                shards_of_video[r["video"]].add(s["shard"])
        assert len(where) == len(conditions[row]) * len(shared.QUESTION_ORDER)             # every question, under every condition
        assert all(len(found) == 1 for found in shards_of_video.values()), row


# ---------------------------------------------------------------------------- calibrated, scored whole and in shards, merged
def test_the_settled_values_are_what_ran(world):
    c = common.load_config(world.base)
    assert c["run"]["K"] == 96 and c["run"]["policy"] == "D+P" and c["run"]["parity"] is True
    assert c["budget"]["fractions"] == [0.05, 0.1, 0.2] and c["budget"]["tokens_per_frame"]["qwen2.5"] == 51
    assert [br.frame_limit(b, 96 * 256, 51, 10 ** 6) for b in c["budget"]["fractions"]] == [24, 48, 96]
    assert models.Qwen25().setting_label(models.Qwen25().fovea_settings(c["ours"])[0]) == "R=70 a=160"
    for section, keys in (("streamingtom", ("tau", "knn", "beta", "released_dpc")), ("fastvid", ("c", "tau", "ignore", "d", "p", "beta")),
                          ("prunevid", ("gamma", "knn", "min_tokens")), ("gazevlm", ("sigma", "r_min", "thumb", "gaze")), ("gazevlm2", ("alpha", "beta"))):
        assert all(c[section][k] == common.DEFAULTS[section][k] for k in keys)
    assert {s["window"] for s in c["dycoke"]["settings"]} == {32} and {s["tau"] for s in c["prunevid"]["settings"]} == {0.8}
    t = pd.read_csv(world.targets)
    assert sorted(t[t.row == "budget"].condition) == sorted(f"budget rule {b}% (D+P gate, fovea R=70 a=160)" for b in (5, 10, 20))
    assert t[t.row == "uniform"].condition.tolist() == ["uniform K=96 (stock input)"]


def test_the_checkpoint_is_the_one_the_earlier_runs_used():
    assert models.Qwen25.default_ids == ("Qwen/Qwen2.5-VL-3B-Instruct",)
    with open(prepare.CONFIG) as f:
        assert "models" not in yaml.safe_load(f)                                           # no config points it elsewhere


def test_targets_land_on_the_token_rule_and_scoring_measures_what_was_predicted(world):
    t = pd.read_csv(world.targets)
    assert set(t.row) == set(ROWS) and len(t) == 2 + 3 + 8 * 3
    p = t.set_index(["row", "target_pct"]).predicted_pct
    for row, tokens in (("avgpool", (16, 25, 49)), ("gazevlm2", (12, 26, 52)), ("streamingtom", (13, 26, 51))):
        for f, n in zip((5, 10, 20), tokens):
            assert p[row, f] == pytest.approx(100 * n / 256), row
    measured = summarize.budgets(summarize.load_model_results(str(world.whole), M))
    lines = t[t.predicted_pct.notna()]
    assert len(lines) == 8 * 3
    for line in lines.itertuples():
        assert measured[line.condition] == pytest.approx(line.predicted_pct, abs=1e-9), (line.row, line.target_pct)


@pytest.mark.parametrize("row", ROWS)
def test_merged_files_match_an_unsharded_run_line_by_line(world, row):
    merged = shared.without_time(world.root / M / f"{row}.jsonl")
    assert merged == shared.without_time(world.whole / M / f"{row}.jsonl") and merged


def test_the_summary_gives_identical_numbers(world):
    config = lambda folder: dict(common.load_config(prepare.CONFIG), paths=dict(out_dir=str(folder)))
    a, b = summarize.summarize(config(world.whole)), summarize.summarize(config(world.root))
    assert len(a) == pd.read_csv(world.targets).condition.nunique() and "prompt_tokens_per_q" in a
    pd.testing.assert_frame_equal(a, b)


def test_the_report_gives_results_by_group(world, tmp_path, monkeypatch, capsys):
    root = tmp_path / "copy"
    shutil.copytree(world.root, root)
    monkeypatch.setattr(sys, "argv", ["report", "--out-dir", str(root), "--models", M, "--targets", world.targets])
    report.main()
    out = capsys.readouterr().out
    assert "EVERY n IS COMPLETE" in out and report.NOTE in out and shared.GPU in out
    table = pd.read_csv(root / "summary_by_group.csv")
    assert set(table.group) == {"all", "aea", "nymeria"} and "prompt_tokens_per_q" in table
    assert set(table[table.group == "aea"].n) == {5} and set(table[table.group == "nymeria"].n) == {6} and set(table[table.group == "all"].n) == {11}


def test_full_budget_rows_give_uniform_ks_answers(world, tmp_path):
    """Section 13 has no check job, so this is the only place this runs before the full run."""
    path = tmp_path / "identity.yaml"
    check_identity.write_config(prepare.CONFIG, str(path))
    cfg = common.load_config(str(path))
    cfg["paths"]["out_dir"] = str(tmp_path / "out")
    cfg["run"]["parity"] = False
    cfg["run"]["prefetch_workers"] = 2
    run.run_model(M, cfg, shared.data_of(world, ["loc_b", "2023_o1"]), ["uniform", *check_identity.ROWS])
    folder = tmp_path / "out" / M
    uniform = check_identity.read_records(folder / "uniform.jsonl")
    for row in check_identity.ROWS:
        r = check_identity.compare(uniform, check_identity.read_records(folder / f"{row}.jsonl"))
        assert r["records"] == 3 and check_identity.problems(r) == 0 and r["max_prob_diff"] == 0, row


def test_a_shard_resumes_after_a_kill_left_a_half_written_last_line(world, copy):
    path = copy / "shards" / "0" / M / "dtd.jsonl"
    lines = path.read_text().splitlines(True)
    path.write_text("".join(lines[:-1]) + lines[-1][:30])
    with pytest.raises(SystemExit, match="MERGE STOPPED"):
        merge(world, copy)
    shared.run_shard(world, copy, 0)
    merge(world, copy)
    for row in ROWS:
        assert shared.without_time(copy / M / f"{row}.jsonl") == shared.without_time(world.whole / M / f"{row}.jsonl")


def test_the_merge_stops_and_writes_nothing_on_a_missing_question(world, copy):
    before = (copy / M / "dtd.jsonl").read_text()
    path = copy / "shards" / "1" / M / "dtd.jsonl"
    path.write_text("".join(path.read_text().splitlines(True)[:-1]))
    with pytest.raises(SystemExit, match="MERGE STOPPED"):
        merge(world, copy)
    assert (copy / M / "dtd.jsonl").read_text() == before


# ---------------------------------------------------------------------------- comparing with the earlier pilot
def test_the_pilot_comparison_covers_only_the_overlap_and_reports_rather_than_fails(world, tmp_path):
    """The pilot ran on an L40S: with --report a different answer is counted and fails nothing, and only the pilot's records are compared."""
    pilot = tmp_path / "pilot"
    os.makedirs(pilot)
    kept = [r for r in shared.read_jsonl(world.root / M / "uniform.jsonl") if r["video"] == "loc_b"]
    assert len(kept) == 2
    kept[0] = dict(kept[0], pred="E" if kept[0]["pred"] != "E" else "A")
    (pilot / "uniform.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    r = compare_records.compare_records(compare_records.load_records(pilot / "uniform.jsonl"), compare_records.load_records(world.root / M / "uniform.jsonl"))
    assert (r["records"], r["missing"], r["pred_differs"], r["tokens_differ"]) == (2, 0, 1, 0) and r["ref_only"] == 9
    assert compare_records.problems(r, report=True) == 0


# ---------------------------------------------------------------------------- progress
def test_progress_counts_answers_per_shard_and_row(world, copy):
    full = shard_progress.progress(str(world.root), M)
    conditions = merge_shards.read_targets(world.targets)
    per_question = sum(len(names) for names in conditions.values())
    assert [(s["questions"], s["videos"]) for s in full] == [(4, 1), (4, 2), (3, 2)]
    for s in full:
        assert s["done"] == s["total"] == per_question * s["questions"]
        assert [row for row, _, _ in s["rows"]] == [r for r in shard_progress.RUN_ORDER]
    path = copy / "shards" / "2" / M / "dtd.jsonl"
    path.write_text("".join(path.read_text().splitlines(True)[:-2]))
    (copy / "shards" / "2" / M / "budget.jsonl").unlink()
    part = shard_progress.progress(str(copy), M, world.targets)[2]
    rows = {row: (done, total) for row, done, total in part["rows"]}
    assert rows["dtd"][0] == rows["dtd"][1] - 2 and rows["budget"] == (0, len(conditions["budget"]) * part["questions"])
    assert part["done"] == part["total"] - 2 - rows["budget"][1]


def test_progress_prints_a_line_per_shard(world, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["progress", "--root", str(world.root), "--model", M])
    shard_progress.main()
    out = capsys.readouterr().out
    assert "shard 2:" in out and "all shards:" in out and "(100.0%)" in out and "gazevlm2" in out


# ---------------------------------------------------------------------------- the jobs
def job(name):
    with open(os.path.join(REPO, "accuracy_tables", "hpc", name)) as f:
        return f.read()


JOBS = ["qwen25350.SBATCH", "qwen25350_merge.SBATCH", "qwen25350_tests.SBATCH"]


@pytest.mark.parametrize("name", JOBS)
def test_jobs_keep_the_rules(name):
    text = job(name)
    for banned in ("--no-parity", "nogate", "compare_budget", "seed_results", "afterok", "afterany"):
        assert banned not in text
    for folder in shards.PROTECTED:
        assert f"OUT=$GV_ROOT/results/{folder}" not in text and f"ROOT=$GV_ROOT/results/{folder}" not in text and f"--out-dir $GV_ROOT/results/{folder}" not in text


@pytest.mark.parametrize("name", JOBS[:2])
def test_jobs_never_name_qwen3s_results_or_job_names(name):
    assert "qwen3_350" not in job(name) and "qwen3350" not in job(name)


def test_each_shard_is_its_own_one_gpu_job_on_an_a100():
    text = job("qwen25350.SBATCH")
    assert "--gres=gpu:a100:1" in text and "GPU=A100" in text and "--time=08:00:00" in text and "--requeue" in text
    assert '--comment="preemption=yes;requeue=true"' in text and "--dependency=singleton" in text and "--nodes=1" in text
    assert "M=qwen2.5\n" in text and "OUT=$GV_ROOT/results/qwen25_350\n" in text
    assert "WANT=qwen25350-full-$SHARD" in text and '"$SLURM_JOB_NAME" != "$WANT"' in text
    assert "S=$OUT/shards/$SHARD" in text and "--out-dir $S --keep-going" in text and "repair_results $S/$M" in text
    assert "--expect $GPU --same-as $CAL_GPU --record $S/$M/gpu_types.txt" in text and "shard_$SHARD.txt" in text
    assert 'echo "SHARD DONE: full $SHARD"' in text and "--prefetch-workers $W" in text


def test_there_are_only_the_steps_the_user_asked_for():
    text = job("qwen25350.SBATCH")
    assert "<calibrate|full>" in text and "checkshard" not in text and "pilot)" not in text and "check)" not in text
    assert "accuracy_tables.run --config $BASE" in text and "--no-parity" not in text                   # run.py checks parity at each job's start


def test_calibrate_runs_once_and_leaves_its_results_read_only():
    text = job("qwen25350.SBATCH")
    assert 'if [ -f "$TARGETS" ]' in text and 'chmod a-w "$TARGETS" "$BASE"' in text
    assert "--fractions --videos all" in text and "--counts $CAL/$M/kept_counts.jsonl" in text and "--shards $SHARDS --all" in text


def test_every_row_is_scored_and_merged():
    for name in ("qwen25350.SBATCH", "qwen25350_merge.SBATCH"):
        assert "ROWS=" + ",".join(ROWS) + "\n" in job(name)


def test_the_merge_job_reports_the_pilot_comparison_without_failing_the_run():
    merge_job = job("qwen25350_merge.SBATCH")
    assert "merge_shards --root $ROOT --model $M" in merge_job and "--calibration-gpu $CAL/$M/gpu_types.txt" in merge_job
    assert "report --out-dir $ROOT --models $M --targets $TARGETS" in merge_job and "gpu_qwen25350-full-*_*.csv" in merge_job
    assert 'if ! run_in_env "python -u -m accuracy_tables.hpc.newvideos.compare_records --new $GV_ROOT/results/hpc350_pilot/$M --ref $ROOT/$M --rows uniform,blind,budget --report"' in merge_job
    assert "gres" not in merge_job                                                         # no GPU held


def test_section_13_starts_by_confirming_qwen3s_merge():
    with open(os.path.join(REPO, "HPC_ACCURACY_RUNBOOK.md")) as f:
        section = f.read().split("\n## 13.")[1]
    first = next(line for line in section.splitlines()[1:] if line.strip())
    assert "Qwen3" in first and "pull" in first and "merge" in first


# ---------------------------------------------------------------------------- what Qwen3's running jobs need
QWEN3_FILES = {
    "accuracy_tables/hpc/qwen3350.SBATCH": "bb3095bef53c52db0e7d4b8a091d68e819dd489fdc53b4a0501ed1e80059c4b3",
    "accuracy_tables/hpc/qwen3350_merge.SBATCH": "eabe8e4d0848f748da6a5bc1395b190da9c2bda0f204b720df6b04fbd5fe12ce",
    "accuracy_tables/hpc/qwen3350_tests.SBATCH": "27792f4c7e431f2e90c3d8032838dfc88b3cf5622f4e062aac026c5a26ddc51e",
    "accuracy_tables/hpc/all350/merge_shards.py": "840fa52339c37ce09031d4affd2074760b6d02824c4a52591934f4f53241f718",
    "accuracy_tables/hpc/all350/shards.py": "e61c28eb5b07a8ac23fd718c8400c2b93cb3643003956ce077f623fdc5745995",
    "accuracy_tables/hpc/all350/report.py": "90de29ed09ff931bd7be6247408e8f6dcd6d1749b84d374326b37792ae2986e1",
    "accuracy_tables/summarize.py": "909aa4f5f649c2f6bb6542cfcde42f252d5c6172d74db78956bd7de7ea9baeb0",
    "accuracy_tables/hpc/check_gpu.py": "afb278f3efc4d50a0041dc1ac1f1a57cec1d885935e56f8f0a9b9a6f00903080",
    "accuracy_tables/hpc/repair_results.py": "c1e94825e8d09e1f6d7ac2ac9ab50d293e015737270e770ea7ebbdb1f84ded2d",
    "accuracy_tables/hpc/newvideos/compare_records.py": "ef78ad2ac970d3bf0c3d8664996fe5268a5f9f053a31d12f33bd9905f5b18a68",
}


@pytest.mark.parametrize("path", sorted(QWEN3_FILES))
def test_qwen3s_jobs_and_merge_code_are_byte_for_byte_what_it_started_with(path):
    assert shards.file_sha256(os.path.join(REPO, path)) == QWEN3_FILES[path]


def test_a_merge_never_looks_at_the_code_now_on_disk(world, copy, monkeypatch):
    """So a merge that runs at a later commit than its shards still works, and only compares the jobs with each other."""
    def refuse(*a, **k):
        raise AssertionError("the merge must not depend on the commit or code on disk")

    monkeypatch.setattr(shards, "git_commit", refuse)
    monkeypatch.setattr(shards, "code_sha256", refuse)
    info = merge(world, copy)
    assert info["commit"] == "abc123" and info["gpu"] == shared.GPU


def test_qwen3s_settings_are_unchanged():
    assert models.Qwen3.key == "qwen3" and models.Qwen3.default_ids == ("Qwen/Qwen3-VL-4B-Instruct",) and models.Qwen3.baselines_load_on_threads
    with open(prepare.CONFIG) as f:
        assert yaml.safe_load(f)["budget"]["tokens_per_frame"] == {"qwen2.5": 51, "qwen3": 51, "phi": 285}
    assert not {"qwen3_350", "qwen3_350_pilot", "qwen3_350_check"} & set(shards.PROTECTED)    # its own merge writes there

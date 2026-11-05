# Accuracy runs on NYU Torch: runbook for the HPC agent

You know the cluster and wrote `EGOEVERYTHING_CONTEXT.md` (not in git); "context section N" below refers to it. This file
covers the project's side: what to run, in what order, and how it reads your dataset. It assumes no earlier knowledge of
the project.

**Everything runs as Slurm jobs, not notebooks.** The Python is already command-line scripts. The job files are in
`accuracy_tables/hpc/`, and the method detail is in `accuracy_tables/README.md`.

**Nothing here has run on Torch yet.** The job files have only had a bash syntax check, and the new Python and its tests
(`tests/test_hpc.py`) have never been executed, because Python isn't run on the machine they were written on. The tests
run as part of step 4. Expect to fix small problems, and report what you change.

## 1. The task

Score 11 methods on 4 VLMs on the EgoEverything multiple-choice questions, with each baseline set to the same token
budget as our method, then write one summary table. Accuracy only: latency, memory and energy are out of scope.

**Models for now: `qwen2.5`, `qwen3`, `phi`, `smolvlm`.** The user asked to skip `internvl` and `gemma` for now. To bring them
back, uncomment them in `MODELS` in `accuracy_tables/hpc/common.sh`: that one list drives the downloads, the smoke test and
`submit_all.sh`.

**Scope: the 12 videos below (834 questions).** The other 338 videos in `$EGO_ROOT` have no gate data, so the code skips
them. Don't extend to them without asking the user.

| Video | Questions | Set |
|---|---|---|
| `loc4_script1_seq1_rec1` | 111 | test |
| `loc1_script1_seq7_rec1` | 36 | test |
| `loc4_script3_seq2_rec2` | 34 | test |
| `loc2_script1_seq2_rec1` | 23 | test |
| `loc2_script1_seq4_rec1` | 111 | train |
| `loc3_script5_seq3_rec1` | 102 | train |
| `loc4_script2_seq4_rec1` | 84 | train |
| `loc2_script2_seq4_rec2` | 82 | train |
| `loc3_script3_seq5_rec1` | 79 | train |
| `loc2_script2_seq2_rec1` | 65 | train |
| `loc3_script5_seq2_rec1` | 62 | train |
| `loc1_script3_seq1_rec1` | 45 | train |

`test` = the 4 locked test videos (204 questions). `all` = all 12 (834 questions).

| Model key | Hugging Face id | Stock tokens per frame | Earlier runs |
|---|---|---|---|
| `qwen2.5` | `Qwen/Qwen2.5-VL-3B-Instruct` | 256 | full run on `test` |
| `internvl` | `OpenGVLab/InternVL3_5-4B-HF` | 256 | full run on `test`; **not run on Torch for now** |
| `smolvlm` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | 81 | full run on `test` |
| `qwen3` | `Qwen/Qwen3-VL-4B-Instruct` | 256 | 1-question smoke test only |
| `gemma` | `google/gemma-3-4b-it` (gated) | 256 | 1-question smoke test only; **not run on Torch for now** |
| `phi` | `microsoft/Phi-3.5-vision-instruct` | 757 | 1-question smoke test only |

New problems are most likely on `qwen3` and `phi`. No model has run on `all` through this pipeline yet.

| Row key | Method | What the model sees | Knob set by calibration |
|---|---|---|---|
| `uniform` | Uniform K (the 100% budget) | K = 96 evenly spaced frames per video (SmolVLM: 64) at stock input | none |
| `blind` | Blind (the 0% budget) | question and options, no frames | none |
| `ours` | Ours | frames chosen by the gaze + IMU gate (policy D+P), each foveated at its gaze point; 4 fovea settings | none: its budget is measured, the others match it |
| `gazevlm` | GazeVLM (Chen & Qi) | Uniform K frames: gaze-region tokens + 2x2 thumbnail | `rho` |
| `gazevlm2` | Two-view GazeVLM | Uniform K frames: global + gaze-centred view, both encoded | tokens per frame `B` |
| `avgpool` | Uniform K + average pooling | Uniform K frames: token grid pooled to s x s | `s` |
| `streamingtom` | StreamingTOM | Uniform K frames in time order, reduced per frame | `G/N` |
| `dycoke` | DyCoke | Uniform K frames in windows, pruned against a reference | pruning rate |
| `fastvid` | FastVID | Uniform K frames in segments, salient + merged tokens | `r` |
| `prunevid` | PruneVid | Uniform K frames in segments, static tokens merged | `beta` |
| `dtd` | DTD (TimeChat-Online) | Uniform K frames, tokens matching the previous frame dropped | drop share |

## 2. Rules from the user

- **Fair comparison.** Don't change `K`, the policy, the fovea settings or the baselines' other parameters (the defaults in
  `accuracy_tables/common.py`). Budgets are matched from token counts only, never from answers. Never tune anything on
  test accuracy.
- **Report every deviation:** a changed setting, a skipped method or model, a code change, a different library version.
  Never reduce the question count or drop a method without saying so.
- **Secrets.** `HF_TOKEN` stays in the environment: never print it, write it to a file, or ask for it in chat. Never open
  `urls/` or any links JSON.
- **This repo is public.** Commit only when the user asks, only to `eyehead-gate`. Never commit results or accuracy
  numbers, NetIDs or account names, tokens, or `*CONTEXT*.md`. The one exception, at the user's request, is the gate's
  training output in `accuracy_tables/hpc/eyehead_weights/`.
- **Replies.** The user is new to HPC: give short copy-paste commands, one step at a time, and one-line answers by default.
  Longer analysis only when they ask ("do the analysis").

## 3. The job files

All are in `accuracy_tables/hpc/`. Submit every one **from the repository root**: the jobs use `$SLURM_SUBMIT_DIR`.

| File | What it does | GPU |
|---|---|---|
| `common.sh` | paths (`EGO_ROOT`, `GV_ROOT`, `HF_HOME`, overlay, image), the `MODELS` list, `run_in_env`, `gpu_log`; sourced by the jobs | |
| `config.torch.yaml` | data paths; everything else keeps the protocol defaults | |
| `install_env.SBATCH` | installs the packages into the overlay (`:rw --fakeroot`) | no |
| `download_models.SBATCH` | downloads the `MODELS` list into `$HF_HOME` | no |
| `prepare_data.SBATCH` | unit tests, extracts the frames from `$EGO_ROOT`, checks the question counts | no |
| `smoke.SBATCH` | every method on every model in `MODELS` on 1 question | yes |
| `run_model.SBATCH <model> <test\|all>` | one model: Uniform K, Blind, Ours; calibration; the 8 baselines | yes |
| `submit_all.sh <test\|all> [models]` | one `run_model` job per model, then a summary job | no |
| `summarize.SBATCH <test\|all>` | writes `summary.csv` | no |

Paths default to `/scratch/$USER/...`. Set `EGO_ROOT`, `GV_ROOT`, `HF_HOME`, `OVERLAY` or `IMAGE` before `sbatch` to
override them. The `#SBATCH --output` lines assume `/scratch/<user>/GazeVLM/logs`.

The GPU jobs set `HF_HUB_OFFLINE=1`, so they never need internet. Nothing writes to `$EGO_ROOT`.

## 4. Data

| What | Where | Used for |
|---|---|---|
| Questions | `$EGO_ROOT/Egoeverything_VQA.json` | question text, options, answers |
| Videos | `$EGO_ROOT/videos/<video>/<video>.mp4` | 1 fps frames, extracted once by `prepare_data` |
| Gate CSVs | Drive `GazeVLM/folds/` | which frames exist, and one gaze point per frame |
| Gate predictions | Drive `GazeVLM/vlm/egoeverything/oof_preds_full.csv` | which frames the gate sends |
| Gate weights | this repo: `accuracy_tables/hpc/eyehead_weights/fold1.pt` ... `fold5.pt`, copied to scratch in step 2 | recomputing the predictions if that CSV is missing |

The accuracy run doesn't read the cluster's `general_eye_gaze.csv`, `imu.npz`, `calibration.json` or `timing.json`: the gaze
and IMU inputs are already inside the gate CSVs and predictions. So the open questions in context section 9 (bounding
boxes, keyframe timestamps, the 0.1 s AEA offset, projecting gaze) don't affect this run. Scoring uses only the question,
options and answer, and a keyframe time only picks the frame for the parity check.

Layout on the cluster:

```
/scratch/$USER/GazeVLM/                     <- GV_ROOT
  repo/                                     <- the git clone, submit jobs from here
  data/folds/train_val_fold1.csv ... train_val_fold5.csv, test.csv
  data/runs/eyehead/fold1.pt ... fold5.pt
  data/vlm/egoeverything/oof_preds_full.csv
  frames_1344/<video>/frame_00000.jpg ..., meta.json      <- written by prepare_data
  results/smoke/   results/hpc_test/   results/hpc_all/
  logs/
```

**Don't change these:**

- **The paths inside the gate CSVs and `oof_preds_full.csv`.** `feat_frame_1` and `feat_frame_2` still say
  `/content/drive/...`. They are never opened: they are matched as exact strings between the two files, and the frame
  number is parsed from them (`feat_<k>.npz`). Editing one file gives `KeyError: ... frame pairs have no prediction`.
- **The questions file.** A question's `qid` is its position in the file, and resuming and comparing depend on it.

**How frames are made.** `prepare_data` calls `vlmeval.frames.extract_1fps`: every 20th frame of the 20 fps mp4, long side
1344 px, JPEG quality 92, `frame_%05d.jpg` plus `meta.json`. The code doesn't
look for an mp4 or tar once `<video>/meta.json` exists.

## 5. Steps

Run in order, and wait for each job to finish and pass before the next. Logs are in `/scratch/$USER/GazeVLM/logs/`.

**Step 1: set up, on the login node.**

```bash
mkdir -p /scratch/$USER/GazeVLM/{data,logs,results}
cd /scratch/$USER/GazeVLM
git clone -b eyehead-gate https://github.com/shubhamOjha1000/GazeVLM-HWSW-Codesign.git repo
cd repo
export SBATCH_ACCOUNT=<account>      # context section 10; ask the user which tier. Needed in every new login session.
```

**Step 2: put the gate weights on scratch.** They are in the repo, so from the repo root:

```bash
mkdir -p /scratch/$USER/GazeVLM/data/runs/eyehead
cp accuracy_tables/hpc/eyehead_weights/fold*.pt /scratch/$USER/GazeVLM/data/runs/eyehead/
ls -l /scratch/$USER/GazeVLM/data/runs/eyehead/      # 5 files, fold1.pt ... fold5.pt
```

**The user places the Drive files by hand, so don't copy them yourself.** The run needs `data/folds/` and
`data/vlm/egoeverything/oof_preds_full.csv` at the paths in section 4, and step 4 stops with a clear message if they are
missing.

**Step 3: install and download.** No other job may have the overlay open during the install. The install keeps the overlay's
torch 2.15 and pins `transformers==5.20.1`. The package list comes from reading the imports and is untested in a clean environment: if an `ImportError` names
something else, install it and report it.

```bash
sbatch accuracy_tables/hpc/install_env.SBATCH       # log ends with "ok: python ... transformers 5.20.1"
```

Then the models (about 40 GB, an estimate). None of the 4 is gated, so no token is needed. If Gemma comes back, the user must
accept its licence on the hub and uncomment the `read` line, which keeps the token out of the shell history; the user types it
themselves.

```bash
# read -rs HF_TOKEN && export HF_TOKEN      # only for Gemma
sbatch accuracy_tables/hpc/download_models.SBATCH   # log lists the 4 models with no FAILED line
```

If these nodes have no internet, run the same steps where there is some (context section 10).

**Step 4: prepare and check the data.**

```bash
sbatch accuracy_tables/hpc/prepare_data.SBATCH      # log ends with "data check OK"
```

It stops if the question counts are wrong: 834 on `all`, 204 on `test`.

**Step 5: smoke test.**

```bash
sbatch accuracy_tables/hpc/smoke.SBATCH             # log ends with "=== 0 failed of 44" (11 methods x 4 models)
```

Expect a few minutes per model.

**Step 6: the 4 test videos, on the 4 models.**

```bash
bash accuracy_tables/hpc/submit_all.sh test
squeue -u $USER
```

Models run in parallel. When they end, the summary job writes `results/hpc_test/summary.csv`. Check `n` for every condition
(section 6) and show the user the table before going on.

**Step 7: all 12 videos.** Ask the user first, then:

```bash
bash accuracy_tables/hpc/submit_all.sh all
```

Results go to `results/hpc_all/`, kept apart from `hpc_test` because the knobs are calibrated per video set. To save time you
may first copy `uniform.jsonl`, `blind.jsonl` and `ours.jsonl` from `hpc_test/<model>/` into `hpc_all/<model>/`, so the 204 test
questions aren't scored again for those rows.

**Rerunning.** Results are appended after every question, so running the same `submit_all.sh` line continues where it stopped.
Use it after a timeout or a preemption that didn't requeue, naming the models if only some are unfinished. To redo a row,
delete its `.jsonl`.

**Time.** `run_model` asks for 8 hours. The only measured full run: Qwen2.5 took about 35 min on an A100, all rows, 4 test
videos. Scaling by question count, Qwen-class models should take about 2 to 3 hours on all 12 videos (an estimate). Phi will be
the slowest, at up to about 73,000 visual tokens per question at Uniform K. Other durations haven't been measured.

## 6. Known behaviour and pitfalls

**Cluster**

- **Low GPU use is the main risk.** On a T4 check, each image took about 130 ms whatever its GPU work, probably CPU
  preprocessing, so GPU use may be low for long stretches. `run_model` and `smoke` log GPU use once a minute to
  `logs/gpu_<job>_<id>.csv`. Watch the first real job. If jobs get cancelled, stop and tell the user; caching the preprocessed
  frames would help, but it isn't built.
- **GPU type.** `--gres=gpu:1` can land on any GPU. `check_gpu.py` stops the job unless the GPU has hardware bf16 and over 35 GB; the
  L40S (46 GB) passes. Add a constraint if jobs land on smaller ones.
- **Preemption.** The GPU jobs have `--requeue` and resume safely, and `repair_results.py` removes a half-written last line
  first. If your preemption settings need other flags, add them.

**Results**

- **Failed questions count as done.** A question that runs out of memory is written with `error: "oom"` (or `"nan"`), rerunning
  skips it, and the summary leaves it out. Check that `n` in `summary.csv` is 204 or 834 for every condition. To retry, delete
  the question's line from the row's `.jsonl`.
- **Two jobs must never write the same files.** `run_model` uses `--dependency=singleton` on the job name, which
  `submit_all.sh` sets per model and video set. Keep that naming if you submit by hand.
- **Back up** `results/` to long-term storage. `/scratch` has no backups (context section 10).
- **Don't claim a winner from the `test` table.** The 95% intervals span roughly 20 points on 204 questions, so differences
  between methods aren't significant.

**Parity**

- **Never score with `--no-parity`.** It compares our pipeline with the model's own forward, and it also sets how prompts are
  tokenized for this transformers version. A failure stops that model before anything is scored. Send the printed report to
  the user rather than bypassing it. torch 2.15 hasn't been tried, so this is the first thing to look at.
- `positions_max_abs_diff  not checked (TypeError)` in a Qwen parity report is known and is not a failure.

**Expected, not bugs**

- SmolVLM runs at K = 64 (8,192-token context), so its names say `K=64` and its budgets differ from the others.
- FastVID on SmolVLM can't reach the targets: `ignore=0.95` drops near-duplicate frames and the budget saturates near 13%, so it
  gets one condition instead of two. A fix was proposed (`ignore=1.0` and a wider `r` grid) but not made; ask the user first.
- Average pooling on SmolVLM lands well above its low target, because s x s is coarse on a 9x9 grid.
- FastVID's frames per question are lower than other baselines, because it leaves some frames without tokens.
- Blind beats Uniform K on Qwen2.5. That is an unexplained red flag, not a known bug. Report it if it repeats, and don't try to
  fix it.
- Harmless log lines: `Kwargs passed to processor.__call__ ...` and the tied-weights message.
- A job stuck on a `[y/N] trust_remote_code` prompt means something other than the repo's loader is loading a Phi tokenizer. Keep
  `run.phi_remote_parity: false`.

## 7. What to report back

Keep it short:

1. GPU model, Python, torch and transformers versions, and any deviation from this file.
2. The smoke test's last line, and each model's parity result.
3. The `summary.csv` path, any condition with `n` below 204 or 834, and any `oom` or `nan` records.

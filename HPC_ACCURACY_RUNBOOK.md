# Accuracy runs on NYU Torch: runbook for the HPC agent

You know the cluster and wrote `EGOEVERYTHING_CONTEXT.md` (not in git); "context section N" below refers to it. This file
covers the project's side: what to run, in what order, and how it reads your dataset. It assumes no earlier knowledge of
the project.

**Everything runs as Slurm jobs, not notebooks.** The Python is already command-line scripts. The job files are in
`accuracy_tables/hpc/`, and the method detail is in `accuracy_tables/README.md`.

**New task: section 17, SmolVLM2-2.2B's FLOPs-matched table on HD-EPIC's gaze questions. This round is its step 0 only, a read-only look at the data,
then STOP for the user's decisions D1 to D5. Don't pull while any section 12 to 16 job is queued or running.** Section 17 hasn't run.
Sections 1 to 7 cover the 12-video run, section 8 the budget rule on those videos, and section 9 gate data for the other 338 videos and SmolVLM on
Nymeria. Section 10, the budget rule on 4 models, has run up to its pilot, and its step 4 waits for the user. Section 11 is SmolVLM with every row at
5, 10 and 20%, section 12 is the same on Qwen3, section 13 the same on Qwen2.5, section 14 SmolVLM with every row matched on total FLOPs, and section 15 the same on Qwen2.5
and Qwen3. Section 16's job files have only had a bash syntax check, and its Python and tests have never been executed, because Python isn't run on the machine
they were written on. Expect to fix small problems, and report what you change.

## 1. The task

Score 11 methods on 4 VLMs on the EgoEverything multiple-choice questions, with each baseline set to the same token
budget as our method, then write one summary table. Accuracy only: latency, memory and energy are out of scope.

**Models for now: `qwen2.5`, `qwen3`, `phi`, `smolvlm`.** The user asked to skip `internvl` and `gemma` for now. To bring them
back, uncomment them in `MODELS` in `accuracy_tables/hpc/common.sh`: that one list drives the downloads, the smoke test and
`submit_all.sh`.

**Scope: the 12 videos below (834 questions).** Section 9 built gate data for the other 338 videos in `$EGO_ROOT`, and section 10
runs on all 350. Sections 1 to 7 stay on these 12.

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
| `run_budget.SBATCH <model> [questions]` | the frame budget, section 8: the budget rule and the no-gate control | yes |
| `newvideos_rebuild.SBATCH` | section 9, check 1: rebuild the 12 known videos and compare | no |
| `newvideos_pilot.SBATCH` | section 9, check 2: the 5 pilot Nymeria videos and the gaze sheets | no |
| `newvideos_build.SBATCH` | section 9: one array task per remaining video | no |
| `newvideos_gate.SBATCH` | section 9, check 3: gate predictions and the comparison with AEA | no |
| `newvideos_vlm.SBATCH <check\|pilot\|full>` | section 9: SmolVLM on Nymeria | yes |
| `newvideos_summary.SBATCH <pilot\|full>` | section 9: `summary.csv` and `budget_compare.csv` | no |
| `all350_prepare.SBATCH` | section 10: tests, the joined data and its checks, the pilot list, the token count | no |
| `all350_vlm.SBATCH <model> <check\|pilot\|full>` | section 10: one model, rows `uniform`, `blind`, `budget` | yes |
| `all350_summary.SBATCH <pilot\|full>` | section 10: GPU use and the results table | no |
| `smolvlm350_tests.SBATCH` | section 11: the CPU tests | no |
| `smolvlm350.SBATCH <calibrate\|check\|pilot\|full>` | section 11: SmolVLM with every row, on B200 only | yes |
| `smolvlm350_summary.SBATCH <pilot\|full>` | section 11: GPU use and the results table | no |
| `qwen3350_tests.SBATCH` | section 12: the CPU tests | no |
| `qwen3350.SBATCH <calibrate\|check>` | section 12: Qwen3's knobs, and its checks, on one A100 | yes |
| `qwen3350.SBATCH <checkshard\|pilot\|full> <0\|1\|2>` | section 12: one of Qwen3's 3 shards, on one A100 | yes |
| `qwen3350_merge.SBATCH <check\|pilot\|full>` | section 12: merges the 3 shards, then compares or summarizes | no |
| `qwen25350_tests.SBATCH` | section 13: the CPU tests | no |
| `qwen25350.SBATCH calibrate` | section 13: Qwen2.5's knobs and shard assignment, on one A100 | yes |
| `qwen25350.SBATCH full <0\|1\|2>` | section 13: one of Qwen2.5's 3 shards, on one A100 | yes |
| `qwen25350_merge.SBATCH` | section 13: merges the 3 shards, summarizes, then compares with the earlier pilot | no |
| `smolvlm_flops_tests.SBATCH` | section 14: the CPU tests | no |
| `smolvlm_flops.SBATCH <calibrate\|full>` | section 14: SmolVLM's FLOPs plan, then every planned condition, on B200 only | yes |
| `smolvlm_flops_summary.SBATCH` | section 14: GPU use and the FLOPs-matched table | no |
| `qwen_flops_tests.SBATCH` | section 15: every CPU test in `tests/` | no |
| `qwen_flops.SBATCH <qwen2.5\|qwen3> calibrate` | section 15: the split and one model's sealed FLOPs plan, on one A100 | yes |
| `qwen_flops.SBATCH <qwen2.5\|qwen3> full <0\|1\|2>` | section 15: one of a model's 3 FLOPs-matched shards, on one A100 | yes |
| `qwen_flops_merge.SBATCH <qwen2.5\|qwen3>` | section 15: merges a model's 3 shards, writes its FLOPs table, compares with section 12 or 13 | no |
| `smolvlm256_download.SBATCH` | section 16: SmolVLM2-256M at its pinned revision, into `$HF_HOME` | no |
| `smolvlm256_flops_tests.SBATCH` | section 16: every CPU test, the 256M's tokens per frame, `SAME FRAMES AS 2.2B` and the `PROPOSED` limits | no |
| `smolvlm256_flops_checks.SBATCH` | section 16: the 2.2B's regression, the 256M's parity, full-budget identity and threads, on one B200 | yes |
| `smolvlm256_flops.SBATCH <calibrate\|full>` | section 16: the 256M's FLOPs plan with a projected run time, then every planned condition, on one B200 | yes |
| `smolvlm256_flops_summary.SBATCH` | section 16: the records, GPU use, the FLOPs table and the side table against the 2.2B | no |
| `hdepic_explore.SBATCH` | section 17, step 0: HD-EPIC's gaze questions and their videos, read-only, for decisions D1 to D5 | no |

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
  results/smoke_budget/   results/hpc_budget/       <- section 8
  newvideos/                                        <- section 9: lists, gaze, rows, frames, one folder per group
  results/prefetch_check/   results/smoke_nymeria/   results/nymeria_pilot/   results/nymeria_smolvlm/    <- section 9
  newvideos/all350/                                 <- section 10: the joined fold files, predictions and frame links
  results/hpc350_check/   results/hpc350_pilot/   results/hpc_350/                                       <- section 10
  results/smolvlm_350_check/   results/smolvlm_350_pilot/   results/smolvlm_350/                         <- section 11
  results/qwen3_350_check/   results/qwen3_350_pilot/   results/qwen3_350/                               <- section 12
  results/qwen25_350/                                                                                    <- section 13
  results/smolvlm_flops/                                                                                 <- section 14
  results/qwen25_flops/   results/qwen3_flops/                                                           <- section 15
  results/smolvlm256_flops/                                                                              <- section 16
  data/hdepic/explore/                                <- section 17, step 0 (HD-EPIC itself stays in /scratch/$USER/hdepic/, read-only)
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

**Job count: 14 `sbatch` jobs, 9 on GPU and 5 on CPU,** with the 4 models in `MODELS`. A job that times out or fails is
submitted again on top of these.

| Step | Job | Jobs | Runs on |
|---|---|---|---|
| 1. Set up | none, login node | 0 | |
| 2. Weights on scratch | none, login node | 0 | |
| 3. Install | `install_env.SBATCH` | 1 | CPU |
| 3. Download the models | `download_models.SBATCH` | 1 | CPU |
| 4. Prepare the data | `prepare_data.SBATCH` | 1 | CPU |
| 5. Smoke test | `smoke.SBATCH` | 1 | GPU |
| 6. The 4 test videos | `run_model.SBATCH`, one per model | 4 | GPU |
| | `summarize.SBATCH` | 1 | CPU |
| 7. All 12 videos | `run_model.SBATCH`, one per model | 4 | GPU |
| | `summarize.SBATCH` | 1 | CPU |

Steps 6 and 7 are one `submit_all.sh` command each, run on the login node, which submits those 5 jobs.

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

## 8. The frame budget on SmolVLM, on the 12 videos

**This has run.** It passed with 0 failed of 6, all `n` = 834, no `oom` or `nan`, and parity OK. It took about 4.5 minutes, well
under the estimate below. Tokens were exactly 17 per frame in all 6 conditions. Section 9 builds on it.

### Why this exists

Uniform K stops at K frames however long the video is, and every baseline works on those frames, so their cost doesn't grow
with the video. The gate marks a frame every few seconds whatever the length, so its cost does. That barely matters on AEA
videos of a few minutes. On an 18-minute Nymeria video the gate may mark several hundred frames (an estimate from AEA, not
measured on Nymeria), while a 10% budget pays for about 51 on a 256-token model. To compare fairly at a low budget, Ours needs
a limit like every other method. This task tests that limit on data we already have, before any Nymeria work.

### The rule

The gate is unchanged. It labels each frame pair TRANSITION, PURSUIT, REFIXATION or STABLE, and policy D+P marks frames from
those labels:

- **Settle:** the first REFIXATION or STABLE frame after a TRANSITION or PURSUIT stretch.
- **Refixation:** a REFIXATION frame inside an already settled stretch.
- **Pursuit:** the first frame of a PURSUIT run.
- **Frame 0:** always marked.

Then, per video, in `vlmeval/budget_rule.py`:

1. **Limit** = budget x that video's Uniform K tokens / tokens per frame, rounded down.
2. **At most that many marks:** send them all and add nothing.
3. **More marks than that:** cut the video into that many equal slots and keep one frame per slot, the first that applies:
   - a settle frame (the longest moving stretch before it wins, then the earliest)
   - a refixation frame (earliest)
   - a pursuit frame (earliest)
   - frame 0
   - if the slot has no mark, the frame predicted not moving nearest the slot's middle
   - if every frame in the slot is predicted moving, the middle frame
4. **Foveate each kept frame** at its gaze point with the model's smallest fovea setting, and send the frames with the question.

The **no-gate control** (row `nogate`) uses the same slots and foveation, takes each slot's middle frame, never reads the
labels, and always sends the full limit.

### Why it is built this way

- **The limit is a ceiling, not a target.** When the gate stays quiet, that is its decision, so a quiet video isn't padded.
- **Equal slots** spread the frames over the whole video, because a question can ask about any moment.
- **Settle comes first** because it should give a new, steady view. The user chose this order in advance. It isn't tuned, and
  nothing yet shows that it helps accuracy. That is what this run starts to test.
- **One fovea setting at every budget,** so the budget changes how many frames are sent, not their resolution. Only the frame
  choice differs between the two rows.
- **The no-gate row is the real test.** Uniform K differs from the rule in frames, tokens and resolution all at once, so
  beating it wouldn't show that the gate helped. The no-gate row differs only in which frame each slot keeps.
- **SmolVLM first:** it is the fastest model, and in earlier runs frames helped on it, while on Qwen2.5 Blind beat Uniform K.
- **The 12 AEA videos first:** they are the only videos with gate predictions, and they need no new data.

**This run checks the code. It can't judge the rule.** On SmolVLM, Uniform K is up to 64 x 81 = 5,184 tokens and the smallest
fovea is 17 tokens, so the limits are 15, 30 and 60 frames. The exception is `loc4_script3_seq2_rec2`, which has 59 frames, so
Uniform K is 59 frames there and its limits are 14, 28 and 56. These videos are short, so their slots are far shorter than a long
video's. The real test is Nymeria, in section 9.

### What changed in the code

| File | Change |
|---|---|
| `vlmeval/budget_rule.py` | new: the limit, the send reasons, the slots and the pick per slot |
| `accuracy_tables/common.py` | new `budget_frames` and a `budget` config section; `load_data` also returns `clips`, `labels`, `frame`; `_gate_labels` split out of `_gate_selection` |
| `accuracy_tables/run.py` | new rows `budget` and `nogate`, one condition per budget, run only when named in `--rows` |
| `accuracy_tables/summarize.py` | reads the two new result files |
| `accuracy_tables/compare_budget.py` | new: rule minus no gate per budget, on the same questions, with a video-level 95% interval |
| `accuracy_tables/hpc/run_budget.SBATCH` | new: the job |
| `tests/test_budget_rule.py` | new: CPU tests |

**The 11 existing rows don't change.** `ROWS` is the same, so `smoke.SBATCH` still expects 44 and `run_model.SBATCH` runs what it
ran before. `smoke.SBATCH` and `run_model.SBATCH` aren't touched, so your local `--comment` lines stay.

**The existing results stay untouched.** The real run writes only to `results/hpc_budget/`. It reads `uniform.jsonl`,
`blind.jsonl` and `ours.jsonl` from `results/hpc_all/smolvlm/` and copies them there, because the token budgets are measured
against Uniform K. No Uniform K, Blind or Ours question is scored again.

### Steps

Set `SBATCH_ACCOUNT` as in step 1, and submit from `repo/`.

**1. Check that nothing is running, then pull.**

```bash
squeue -u $USER                                 # no job of yours may be using the checkout
git status --short                              # expect only your 2 local --comment edits
git pull --ff-only origin eyehead-gate
```

**2. Quick check: tests, parity, and 6 questions in `results/smoke_budget/`.** Wait for the user's OK first.

```bash
sbatch --job-name=acc-budget-check-smolvlm accuracy_tables/hpc/run_budget.SBATCH smolvlm 6
```

The log should show the tests passing and `parity OK`, and end with `=== 0 failed of 6`. Show the user the `--- budget rule` and
`--- no gate` lines: they give the frames per question and the frames kept by reason.

**3. The real run on all 12 videos.** Ask the user first.

```bash
sbatch --job-name=acc-budget-smolvlm accuracy_tables/hpc/run_budget.SBATCH smolvlm
```

That is 6 conditions x 834 questions. Expect about 15 to 30 minutes (an estimate from the earlier SmolVLM run, not a measurement);
the job asks for 2 hours. It ends by writing `results/hpc_budget/summary.csv` and `results/hpc_budget/smolvlm/budget_compare.csv`.
Rerunning continues where it stopped.

**Job count: 2 `sbatch` jobs, both on GPU.**

### Expected, not bugs

- **The rule can send fewer frames than no gate,** because its limit is a ceiling and no gate always sends the full limit. Its
  measured budget can therefore be lower, and more so at 20%, where more videos fit. Per-video matching was rejected earlier as
  biased, so leave this as it is and report both budgets.
- **`fill` in the log:** frames from slots with no mark. These aren't gate frames.
- **A video just over its limit can lose many marks,** because each slot keeps one frame even if a neighbouring slot is empty, and
  that empty slot then gets a filler frame. The user chose the rule as written. Report the `fill` counts, and don't change it.

### Rules for this task

- **Don't change anything after seeing accuracy:** not the budgets, the limit, the preference order, the fovea setting or the slots.
- **Don't add other controls without asking.** The user decided against a head-speed control.
- **Don't claim a winner.** `outcome` in `budget_compare.csv` only says whether the interval excludes 0, and this run checks the code.
- **Never write to `results/hpc_all/`.** Never commit results.
- **`more than one clip`:** a video's frame pairs don't chain. Stop and tell the user.
- **`set budget.tokens_per_frame.<model>`:** only SmolVLM has a token count. Another model needs a measured value and the user's OK.

### What to report back

1. Any code you changed, and the quick check's last line.
2. The `--- budget rule` and `--- no gate` log lines.
3. For the 6 new conditions: `n` (should be 834), measured budget, frames per question, and any `oom` or `nan` records.
4. The `budget_compare.csv` table and the GPU use log.

## 9. Gate data for Nymeria, then the budget rule on SmolVLM

**This has run.** Rows, gate predictions and 1 fps frames exist for all 350 videos, and SmolVLM scored all 325 Nymeria videos, which took
29:59 on a B200 with GPU use averaging 27.7%. Section 10 builds on it. Step 1 first stopped at `REBUILD DIFFERS`, from a missing crop of
the calibration, and step 2 found a 30 Hz gaze problem. Both were fixed before the steps below ran to the end (see "How the gate data is
built").

Where this section differs from context section 13, this section is right: 13 came partly from web summaries of
the code, and this was checked against the code.

### Why this exists

Section 8 only checked the code, because AEA videos are too short for the limit to matter. Nymeria is the real test of the budget
rule, but the gate has no data for those videos, so the code skips them. This task builds that data from the files on the cluster,
checks it, and then scores SmolVLM.

### Scope, set by the user

- **Model:** SmolVLM only.
- **Rows:** `uniform`, `blind`, `budget`, `nogate`. Ask before adding Ours or other baselines.
- **Videos:** a pilot of 5 Nymeria videos, then all 325.
- **The 13 other AEA videos:** built and put through check 3, not scored on the VLM. No gate model trained on them, and they are the
  same kind of video as the training data, so they can separate a builder bug from a real Nymeria difference.
- **Unchanged:** the budgets, the limit, the preference order, the fovea setting and the slots.
- **No AUC check:** the user decided against it, so DINOv2 isn't used anywhere here.

### How the gate data is built, and why

- **It follows the existing builder.** `src/dataprep/build_feature_csv.py` can't run because it needs VRS files and DINOv2, so
  `build_rows.py` calls the same window functions and writes the same formats.
- **Each stream is rebased to its own first sample,** and frames sit at i / native fps, because the gate trained on that alignment.
  `timing.json` isn't used. The streams start within about 0.07 s of each other (context section 7).
- **Projection needs the `aria` environment,** so `project_gaze.py` runs there and saves the result, and `build_rows.py` reads it in the
  base environment. Depth is a fixed 1 m, as in the builder, so context section 8's per-sample depth doesn't apply.
- **The calibration needs Aria's crop rule.** `calibration.json` describes the 2880 px sensor and the videos are 1408 px. Aria's VRS loader
  matches them before the builder ever sees the camera, so `project_gaze.py` does the same with `rescale([1408, 1408], 0.5, [32, 32])`, and
  stops on any other sizes. Skipping it puts gaze about 2.3% too close to the image centre, which changed SmolVLM's fovea cell on 39 of
  1,738 frames. With it, all 1,738 windows matched the original text on the 12 videos. Context section 8's `rescale(..., 1408 / 2880)` has no
  offset, so it wouldn't fix this. **Limit:** this is proven on those 12 AEA videos only. No Nymeria VRS is on the cluster to compare with,
  so Nymeria relies on its 1408 px MP4s using the same crop, and a 2% scale can't be seen on the gaze sheets.
- **30 Hz gaze:** 34 of the 325 Nymeria videos have 30 Hz gaze, and everything else has 10 Hz. The gate trained on 10 Hz and reads only the
  first 9 steps of a window, while its gyro is spread over the whole window, so at 30 Hz it would see only the first 0.3 s of each second,
  misaligned with the gyro. On those videos `gaze_rates_window` uses every third sample, with no averaging. Any other rate stops the build.
  `gaze_xy_window`, which only the fovea reads, keeps every sample, so `n_gaze_in_gap` is about 30 there while `n_velocities` is about 9.
- **Nymeria's yaw:** its CSV comes from a newer gaze model with a left and a right yaw and no single yaw. The combined yaw comes from
  `mps.read_eyegaze(...).yaw`, and the plain mean is saved only to compare. AEA's single yaw is read straight from the CSV.
- **All 6 IMU columns are written.** With only the 3 gyro columns, `_gyro_at_steps` returns `None` and head speed silently becomes 0.
- **Placeholders:** `load_rows` needs `frame_similarity`, `gaze_patch_token_sim` and `quad`, so the first two are empty and `quad` is 3.
  Nothing may score against them, including the `D: ground truth` condition that `select_conditions` computes.
- **Feature names hold the video,** as in `<video>/feat_00012.npz`. No such file exists, but predictions are matched by this exact
  string, so it must be unique across videos.
- **A separate data folder:** `newvideos/nymeria/folds/` has an empty `train_val_fold1.csv` and the rows in `test.csv`, and `paths.oof_cache`
  points at its own `preds.csv`. So `oof_predictions` is never called and `common.py` needs no new data path.
- **Full 1 fps frames,** with `extract_1fps` unchanged. Decoding is most of the cost either way, and the parity check needs a frame
  near the keyframe.

### What the code does with no gaze, GPU use and batching

- **A kept frame with no gaze:** `_fovea_layout` places the fovea at the image centre, with the same 17 tokens. Frame 0 never has gaze.
  Any other frame loses it if the second before it has one sample out of view, or no gaze at all. Records already hold `n_gaze_missing`;
  the log, `summary.csv` and `budget_compare.csv` now report it.
- **Low GPU use:** from reading the code, most likely CPU image loading. Each frame is decoded and processed once per budget and row, and
  startup and summaries also hold the GPU idle. Nymeria needs about 89,000 image loads (210 for the two rows' budgets plus 64 for
  Uniform K, per video) with few questions per video, so GPU use could be lower than on AEA.
- **Cancellation:** Torch's documented limits are 50% on `ga*`, `gl*` and `gr*` nodes and 60% on `gh*`. Its page gives no time window and
  says "Enforcement will be very aggressive", so treat a long run at low GPU use as at risk.
- **Two fixes are in:** frames load on 8 threads (`--prefetch-workers 8`, off by default, step 5 checks that answers don't change), and
  summaries run in a CPU job.
- **Batching:** questions run one at a time, and encoded frames are reused within a video and condition. Batching isn't part of this
  task, because the GPU is mostly waiting on the CPU.

### What changed in the code

| File | Change |
|---|---|
| `accuracy_tables/hpc/newvideos/plan.py` | new: the video lists, and a stop if the groups aren't 12, 13 and 325 |
| `.../extract_frames.py`, `.../project_gaze.py`, `.../build_rows.py` | new: frames, gaze and rows per video |
| `.../compare_rows.py` | new: check 1 |
| `.../draw_gaze.py` | new: check 2 |
| `.../score_gate.py`, `.../check_gate.py` | new: gate predictions, and check 3 |
| `.../compare_records.py` | new: are two runs' answers the same? |
| `accuracy_tables/hpc/config.newvideos.yaml`, 6 `newvideos_*.SBATCH` files | new: Nymeria paths and the jobs |
| `src/common/gaze_geometry.py` | `calibration_projector` split out of `make_gaze_projector`, so a calibration JSON can use the same code, with the camera matched to the video |
| `accuracy_tables/hpc/common.sh` | new `run_in_aria` |
| `vlmeval/smolvlm_fovea.py`, `accuracy_tables/models.py`, `run.py`, `common.py` | loading frames on threads, off by default |
| `accuracy_tables/run.py`, `summarize.py`, `compare_budget.py` | report frames with no gaze |
| `tests/test_newvideos.py` | new: CPU tests |

**The existing rows and jobs don't change.** `smoke.SBATCH` and `run_model.SBATCH` aren't touched, so your local `--comment` lines stay.
`summary.csv` gains one column, `gaze_missing_pct`.

**Existing data stays untouched.** Everything is written under `$GV_ROOT/newvideos/` and the new `results/` folders. `data/folds/`,
`oof_preds_full.csv`, `frames_1344/`, `results/hpc_all/` and `results/hpc_budget/` are only read.

### Steps

**Job count: 9 `sbatch` jobs, 3 on GPU and 6 on CPU,** one of them an array of 333 tasks. A job that times out or fails is submitted again
on top of these. All times are estimates.

| Step | Job | Runs on |
|---|---|---|
| 1. Rebuild and compare the 12 videos | `newvideos_rebuild.SBATCH` | CPU |
| 2. 5 pilot videos and gaze sheets | `newvideos_pilot.SBATCH` | CPU |
| 3. The other 320 Nymeria and 13 AEA videos | `newvideos_build.SBATCH`, an array | CPU |
| 4. Gate predictions and the comparison | `newvideos_gate.SBATCH` | CPU |
| 5. Thread check and 6 Nymeria questions | `newvideos_vlm.SBATCH check` | GPU |
| 6. VLM on the 5 pilot videos | `newvideos_vlm.SBATCH pilot`, `newvideos_summary.SBATCH pilot` | GPU, CPU |
| 7. VLM on all 325 videos | `newvideos_vlm.SBATCH full`, `newvideos_summary.SBATCH full` | GPU, CPU |

Set `SBATCH_ACCOUNT` as in section 5, and submit from `repo/`. **Before step 1,** on the login node:

```bash
squeue -u $USER                                 # no job of yours may be using the checkout
git status --short                              # expect only your 2 local --comment edits
git pull --ff-only origin eyehead-gate
myquota                                         # the frames need about 70 to 80 GB and 360,000 files
```

**Step 1: check 1, the rebuild.** Mostly minutes.

```bash
sbatch accuracy_tables/hpc/newvideos_rebuild.SBATCH
```

This job always rebuilds the 12 videos, so it replaces the rows from the failed run. The log should show the group sizes, the tests passing,
and end with `REBUILD MATCHES`. Each video is compared on pairs, the 4 count
columns, `gaze_rates_window`, `imu_window`, `gaze_xy_window` and predicted classes, using the same model as its existing predictions. Show the
user `compare_rows.csv`'s table either way. **On `REBUILD DIFFERS`, stop and tell the user. Don't adjust a tolerance to make it pass.**

**Step 2: check 2, the pilot.** Roughly 3 to 5 minutes per video for the frames, which also measures the real cost of step 3.

```bash
sbatch accuracy_tables/hpc/newvideos_pilot.SBATCH
```

A rerun skips finished frames and gaze, and rebuilds rows only for a video whose gaze step changed, which on the pilot is the one 30 Hz
video. Each build line in the log gives the gaze rate and step. It rescores all 5 pilot videos, because their predictions share one file.
Show the user the yaw lines, the `check_gate` table and the sheet paths in `newvideos/check_gaze/`. **The user looks at the sheets by eye
and must approve them before step 3:** the red circle is where the fovea goes, and it should sit where the person is plausibly looking
in every video, with the control video as the reference. If a large yaw difference appears, tell the user.

**Step 3: build the rest.** Use the range `plan.py` printed.

```bash
sbatch --array=0-332%20 accuracy_tables/hpc/newvideos_build.SBATCH
```

About 20 to 30 task-hours, so about 1.5 hours with 20 running at once. Then count the results: `ls /scratch/$USER/GazeVLM/newvideos/rows/*.csv
| wc -l` should print 350. To retry failures, resubmit with `--array=<their task ids>`; finished steps are skipped.

**Step 4: check 3, Nymeria against AEA.**

```bash
sbatch accuracy_tables/hpc/newvideos_gate.SBATCH
```

Show the user the whole table, with `aea12`, `aea13` and `nymeria` side by side: class mix, marks per minute, real marks per video, frames with
no gaze, input speeds and `vor_*`. **The gate trained only on AEA, so an odd pattern means stop and tell the user.** Examples:
a class nearly vanishing, marks per minute far off, many frames with no gaze, much higher or lower eye speed, or `vor_median_cos` no longer
clearly negative, which would mean the head rotation fitted on AEA doesn't hold. `aea13` far from `aea12` points at the builder. `marks_*.csv` has
the real marks per video, replacing section 8's estimate.

**Step 5: thread check, then 6 Nymeria questions.** Ask the user first, because this is the first VLM job. Use the GPU type of `results/hpc_budget/`
if the cluster allows it.

```bash
sbatch --job-name=new-vlm-check accuracy_tables/hpc/newvideos_vlm.SBATCH check
```

The log should show `parity OK`, `=== 0 failed of 6`, `SAME ANSWERS`, and end with `=== 0 failed of 8`. Tiny probability differences are possible on
another GPU type, so report `max_prob_diff`. **On `ANSWERS DIFFER`, stop and tell the user.**

**Step 6: the pilot.** Ask the user first.

```bash
ID=$(sbatch --parsable --job-name=new-vlm-pilot accuracy_tables/hpc/newvideos_vlm.SBATCH pilot)
sbatch --dependency=afterok:${ID%%;*} accuracy_tables/hpc/newvideos_summary.SBATCH pilot
```

Then check GPU use, with the job's id:

```bash
awk -F', ' '{s+=$2; n++} END {print s/n "% over " n " minutes"}' /scratch/$USER/GazeVLM/logs/gpu_new-vlm-pilot_<id>.csv
squeue -u $USER --format="%N" | head -3         # while a job runs: its node, which sets the limit
```

**If the mean is under the node's limit, stop and tell the user before step 7.** The next option is caching preprocessed frames in a CPU job,
at about 30 to 115 GB. It isn't built.

**Step 7: all 325 videos.** Ask the user first.

```bash
ID=$(sbatch --parsable --job-name=new-vlm-full accuracy_tables/hpc/newvideos_vlm.SBATCH full)
sbatch --dependency=afterok:${ID%%;*} accuracy_tables/hpc/newvideos_summary.SBATCH full
```

About 1 to 2 hours. The job asks for 8, and rerunning continues where it stopped. If the job has to be resubmitted, submit the summary again too.

### Expected, not bugs

- **Long slots:** at about 1,100 s, the limits of 15, 30 and 60 frames mean slots of about 73, 37 and 18 s.
- **The rule can send fewer frames than no gate,** as in section 8. Report both measured budgets.
- **`D: ground truth` and the other conditions in `select_conditions`** are computed and never used. On these videos they mean nothing.
- **A `FutureWarning` about concatenating empty frames,** from the empty `train_val_fold1.csv`, is harmless.
- **With threads on, a block's first `sec`** no longer includes loading its frames.
- **A 5-minute `load_data` at 0% GPU** at the start of the full job, and again after a requeue, is expected.

### Rules for this task

- **Don't change anything after seeing accuracy:** not the budgets, the limit, the preference order, the fovea setting or the slots.
- **Don't add rows, controls, models or videos without asking,** and keep any batching as a separate change.
- **Don't claim a winner.** Report AEA (`results/hpc_budget/`) and Nymeria (`results/nymeria_smolvlm/`) separately, with both rows' measured budgets.
- **Never write to `results/hpc_all/`, `results/hpc_budget/` or `data/folds/`.** Never commit results, and keep the gaze sheets out of the repo.
- **Commit only when the user asks.** Never commit accuracy numbers, NetIDs, account names, tokens or `*CONTEXT*.md`.
- **Nothing runs on the user's Mac.** Give one read-only command at a time when you need them to look at something.

### What to report back

1. Any code you changed, and each step's last log line.
2. Check 1's table, check 2's yaw lines and sheet paths, and check 3's table.
3. The `--- budget rule` and `--- no gate` log lines, including frames with no gaze.
4. For each condition: `n`, measured budget, frames per question, `gaze_missing_pct`, and any `oom` or `nan` records.
5. The `budget_compare.csv` table, the GPU type, and mean GPU use with the job's length.

## 10. Next task: the budget rule on all 350 videos and 4 models

**Don't submit any job until the user approves it, one step at a time.** Nothing here has been run.

Where this section differs from context section 13, this section is right: it was checked against the code, and 13 is an older plan for a
different set of rows.

### Why this exists

Section 9 scored the budget rule on Nymeria with SmolVLM only. This task runs it on every EgoEverything video (350 videos, 5,985 questions)
on the 4 models, so it can be read per model, with AEA and Nymeria apart.

### Scope, set by the user

- **Models:** `qwen2.5`, `qwen3`, `smolvlm`, `phi`.
- **Rows:** `budget`, plus `uniform` and `blind` as references. Uniform K is also what the budgets are measured against.
- **No `nogate` row.** So these results can't show whether the gate itself helps, and `report.py` says that on every line of its table.
  `compare_budget.py` needs that row, so these jobs don't call it. Existing SmolVLM no-gate records stay where they are and aren't copied.
- **Ask before adding Ours or the other baselines.**
- **Unchanged:** the budgets (5%, 10%, 20%), the limit, the preference order, the slots, `K`, and each model using its smallest fovea setting.

### How the three groups become one run

The existing code reads one fold folder, one predictions file and one frames folder, so `prepare.py` makes one of each under
`newvideos/all350/`, and `config.all350.yaml` points at them. Everything else is only read, and `prepare_data.py` isn't used.

| Group | Videos, questions | Rows from | Predictions from | Frames in |
|---|---|---|---|---|
| `aea12` | 12, 834 | `data/folds/` | `oof_preds_full.csv` (held-out folds, and the 5-model average on the 4 test videos) | `frames_1344/` |
| `aea13` | 13, 1,478 | `newvideos/aea13/folds/` | `newvideos/aea13/preds.csv` (5-model average) | `newvideos/frames/` |
| `nymeria` | 325, 3,673 | `newvideos/nymeria/folds/` | `newvideos/nymeria/preds.csv` (5-model average) | `newvideos/frames/` |

- **Each video keeps its existing predictions.** The 12 keep their original rows too, not their rebuilt ones, because predictions are matched
  by the exact frame-name strings, and `results/hpc_all/` and `results/hpc_budget/` were scored on them.
- **Only the columns the code reads are kept,** so the file stays small: `sequence`, `feat_frame_1`, `feat_frame_2`, `quad`, `gaze_xy_window`,
  `n_oof_in_gap`.
- **Frames are symlinks,** so no frame is copied, and `ensure_frames` sees `meta.json` through them and never writes.
- **All rows sit in `test.csv`,** so `test` means nothing in this config. Use `all` or a video list.
- **The check replaces `prepare_data.py`'s 834 and 204:** 5,985 questions on 350 videos, and 834, 1,478 and 3,673 per group. Then each video's
  labels, Uniform K frames, gate frames and SmolVLM budget frames, with their gaze, must match what its own group gives when read from where it
  was built.

### Tokens per frame

On SmolVLM the smallest fovea is always 17 tokens. On the other 3 models the layout's rings depend on where the gaze falls, so the count varies
by frame. The rule takes one number per model.

| Model | Tokens per frame, possible range | On the 12 videos (`results/hpc_all/`, per question) | Limits at 5 / 10 / 20% if that holds |
|---|---|---|---|
| Qwen2.5, Qwen3 | 40 to 64 | mean 48.8, 47.2 to 50.5 | 25 / 50 / 100 (at 49) |
| Phi | 253 to 320 | mean 280.4, 277.6 to 284.9 | 12 / 25 / 51 (at 281) |
| SmolVLM | exactly 17 | 17 | 15 / 30 / 60 |

- **The value:** `measure_tokens.py` counts every frame of all 350 videos from real gaze, with a frame with no gaze counted at the image centre,
  and proposes the mean over all frames, rounded up. It reads no answers. **The value goes into `config.all350.yaml` only with the user's OK,** and
  until then a budget row on these models stops with a clear error.
- **The count is checked first:** on the 12 videos it must predict `n_image_tokens` exactly on every Ours record of `results/hpc_all/` at this fovea setting.
- **Why the mean and not the maximum:** the maximum would guarantee no video goes over, but the limits would fall to 19 / 38 / 76 and 11 / 22 / 45, spending
  only about 3.8 / 7.6 / 15.2% and 4.2 / 8.4 / 17.2%. With the mean, a video can land slightly over, and the measured budget shows it.
- **Qwen3 shares Qwen2.5's count:** its layout is the same one scaled up.

**What a frame costs on Phi:** 84 to 147 crop tokens, 12 to 18 row breaks, a separator and a 156-token global view. So 157 tokens are always
there, which is why Phi gets about half of Qwen's frames at the same budget. This is the fovea setting as ruled, not a bug, and `n_image_tokens` counts all of it.

**The 24 tokens on Phi's baselines in `results/hpc_all/`:** `calibrate.py` counts only kept tokens for StreamingTOM, DyCoke, DTD and PruneVid, but
their Phi `pack` adds a row break after each token row that has tokens. Once nearly all 24 rows have a token, that is about 24 per frame, or
3.2 points of budget above Ours. Average pooling counts its breaks. FastVID and GazeVLM have the same gap in their counts, and how far they
landed hasn't been looked at. This comes from reading the code, and one smoke row fits it. It doesn't touch this run, because the budget rule's count includes the
breaks and no baseline runs here. Don't change `calibrate.py`: the user decides that if the baselines come back.

### Threads, parity and low GPU use

- **Threads were SmolVLM only,** and `--prefetch-workers` was silently ignored on the others, so a threads check there would have tested nothing. Qwen2.5, Qwen3
  and Phi now load and resize frames on threads, off by default. Other models stop with an error if it's set. `cells_to_patches` and all GPU work stay on the main thread.
- **`OMP_NUM_THREADS=1` in the GPU job,** because 8 loading threads each starting 8 OpenMP threads would fight over the 8 cores.
- **Parity depends on the question:** Qwen must give the same answer and Phi's next token must be a letter. It has only passed on one AEA question on these models, and
  the first question sorted is now a Nymeria one. The check job tests one of each. **If it fails, send the user the report. Don't loosen it and never use `--no-parity`.**
- **Low GPU use:** Torch's limits are 50% on `ga*`, `gl*` and `gr*` nodes, 60% on `gh*`, and 10% by default, with no time window given. Which applies to B200
  nodes isn't known, so the pilot records the node. `gpu_summary.py` reports the mean and the lowest 30-minute mean, because a mean can hide a long idle stretch.

### Reuse and GPU type

| Model | `uniform`, `blind` | `budget` | Still to score |
|---|---|---|---|
| SmolVLM | `hpc_all` (12), `nymeria_smolvlm` (325) | `hpc_budget` (12), `nymeria_smolvlm` (325) | the 13 other AEA videos, 1,478 questions |
| Qwen2.5, Qwen3, Phi | `hpc_all` (12) | nothing | `uniform` and `blind` on 5,151 questions, `budget` on all 5,985 |

- **`seed_results.py` copies them** into `results/hpc_350/` at the start of the full run, so the run skips them. It stops on an error record, a duplicate, a record of another
  video, another condition name, or a wrong count. Step 1 also checks every one has the frames and tokens the joined data gives today.
- **Why it's safe:** since `hpc_all`, the only model-code change on those rows is the thread loading, which the check job tests.
- **Every new GPU job requests `--gres=gpu:b200:1`,** so all new records share one GPU type. Reused records stay on the GPUs they ran on, which were mixed. Tokens and frames don't depend on
  the GPU. Answers can differ only through rounding on near-ties, which favours no row, and each check job measures it on 30 reused questions. Report the GPU types either way.

### What changed in the code

| File | Change |
|---|---|
| `accuracy_tables/hpc/all350/prepare.py` | new: joins the groups, picks the pilot, checks the result and the reused records |
| `.../measure_tokens.py` | new: tokens per frame on real gaze, checked against recorded runs, and the proposed values |
| `.../seed_results.py` | new: copies reusable records, with checks |
| `.../report.py` | new: the results per model, for all videos, AEA and Nymeria |
| `.../gpu_summary.py` | new: mean and lowest 30-minute GPU use |
| `accuracy_tables/hpc/config.all350.yaml`, 3 `all350_*.SBATCH` files | new: paths and the jobs |
| `vlmeval/qwen_fovea.py`, `qwen3_fovea.py`, `phi_fovea.py`, `accuracy_tables/models.py`, `run.py`, `common.py` | loading frames on threads, off by default |
| `accuracy_tables/hpc/newvideos/compare_records.py` | new `--report`: only different frames or tokens fail, and it counts different answers |
| `accuracy_tables/summarize.py` | skips a model with no results in the chosen videos |
| `tests/test_all350.py` | new: CPU tests |

**The existing rows and jobs don't change.** `smoke.SBATCH` and `run_model.SBATCH` aren't touched, so your local `--comment` lines stay. `compare_budget.py`,
`budget_rule.py` and `prepare_data.py` aren't touched.

### Steps

**Job count: 14 `sbatch` jobs, 11 on GPU and 3 on CPU,** plus an expected 1 or 2 reruns for Phi, and perhaps 1 for Qwen3. All times are estimates.

| Step | Job | Runs on |
|---|---|---|
| 1. Tests, joined data, checks, token count | `all350_prepare.SBATCH` | CPU |
| 2. Parity, threads and reused answers | `all350_vlm.SBATCH <model> check`, for Qwen2.5, Qwen3 and Phi | 3 GPU |
| 3. The pilot, on 7 videos | `all350_vlm.SBATCH <model> pilot`, for all 4; `all350_summary.SBATCH pilot` | 4 GPU, CPU |
| 4. All 350 videos | `all350_vlm.SBATCH <model> full`, for all 4; `all350_summary.SBATCH full` | 4 GPU, CPU |

Set `SBATCH_ACCOUNT` as in section 5, and submit from `repo/`. **Before step 1,** on the login node:

```bash
squeue -u $USER                                 # no job of yours may be using the checkout
git status --short                              # expect only your 2 local --comment edits
git pull --ff-only origin eyehead-gate
myquota                                         # the joined files need under 1 GB, and the links 350 files
```

**Step 1: tests, joined data, checks and the token count.** About 30 to 60 minutes.

```bash
sbatch accuracy_tables/hpc/all350_prepare.SBATCH
```

The log should show the tests passing, the 7 pilot videos with their question counts, each group's counts, then `COMBINED MATCHES`, `REUSED RECORDS MATCH` and
`TOKEN COUNT MATCHES RECORDS`, and end with the token table, the limits and a `PROPOSED` line. **Any other ending: stop and tell the user.** **Show the user the table, the limits and the
`PROPOSED` line, and wait.** They decide the values, and they get committed to `config.all350.yaml`. Then pull again before step 2.

**Step 2: check each new model.** Ask the user first. A few minutes on Qwen, up to about 30 on Phi.

```bash
for M in qwen2.5 qwen3 phi; do
    sbatch --job-name=all350-check-$M accuracy_tables/hpc/all350_vlm.SBATCH $M check
done
```

Each log should show `parity OK` three times, `=== 0 failed of 5` twice, `SAME ANSWERS`, `=== 0 failed of 2`, a `TOKENS MATCH` line, and end with `CHECK DONE`. Threads on and off run on the same
GPU in one job, so **`ANSWERS DIFFER` means stop and tell the user.** The `TOKENS MATCH` line says how many of 60 reused answers changed against `results/hpc_all/`. Show it to the user even if it's 0: it's how
much the GPU type matters, and they decide whether reuse stands. **`TOKENS DIFFER`: stop and tell the user.**

**Step 3: the pilot.** Ask the user first. The 7 videos are 2 Nymeria videos with many questions, 4 with one, and 1 new AEA video, picked by position in the sorted names. That's about 440 questions: Phi takes
roughly 1 to 2 hours, and SmolVLM a few minutes.

```bash
IDS=""
for M in qwen2.5 qwen3 phi smolvlm; do
    ID=$(sbatch --parsable --job-name=all350-pilot-$M accuracy_tables/hpc/all350_vlm.SBATCH $M pilot)
    IDS=$IDS:${ID%%;*}
done
sbatch --dependency=afterany$IDS accuracy_tables/hpc/all350_summary.SBATCH pilot
```

The summary's log gives each job's GPU use, then the table, and ends with `EVERY n IS COMPLETE`. Then show the user the node and length of each job, and the time per row:

```bash
sacct -u $USER -S now-7days -X --name=all350-pilot-qwen2.5,all350-pilot-qwen3,all350-pilot-phi,all350-pilot-smolvlm --format=JobName%24,NodeList,Elapsed,State
grep -h "^--- \|^done:" /scratch/$USER/GazeVLM/logs/all350-pilot-phi_*.log
```

**If a job's lowest 30-minute mean is under its node's limit, stop and tell the user before step 4.** SmolVLM's job will be too short for a 30-minute reading. Its `TOKENS MATCH` line compares the Nymeria videos with
`results/nymeria_smolvlm/`, and its `missing` count is the new AEA video, which that folder doesn't have. Rescale the time table below from the pilot and show the user. If a job would need more than 8 hours, say so.

**Step 4: all 350 videos.** Ask the user first.

```bash
for M in qwen2.5 qwen3 phi smolvlm; do
    sbatch --job-name=all350-full-$M accuracy_tables/hpc/all350_vlm.SBATCH $M full
done
```

| Model | New answers | At the speeds of `results/hpc_all/`, no threads | If B200 and threads help as assumed | Passes 8 hours? | Low-GPU risk |
|---|---|---|---|---|---|
| SmolVLM | 7,390 | about 9 min, plus about 5 min startup | shorter | No | Low: short, and it ran 30 min at 27.7% uncancelled |
| Qwen2.5 | 28,257 | 6.7 h | 2.4 h | No | Medium: about 60% of the time is loading and encoding frames |
| Qwen3 | 28,257 | 7.8 h | 2.6 h | Only close if B200 gives no speed-up | Medium: about 75% |
| Phi | 28,257 | 22.7 h | 10.3 h | **Yes: 2 or 3 jobs** | Low on Uniform, medium on the budget rows |

The assumption divides frame encoding by 3.7, which is what SmolVLM gained on Nymeria on a B200 with threads, and model answering by 2, which is a guess. Phi's Uniform row alone is about 15.5 hours at those speeds, at about 10.5 s per question. **A job that ends in `TIMEOUT` isn't requeued: submit the same line again,
and it continues where it stopped.** The copy step leaves existing result files alone. Never run two jobs of one model at once: `singleton` prevents it if you keep the job name.

When every job has finished:

```bash
sbatch accuracy_tables/hpc/all350_summary.SBATCH full
```

The log should end with `EVERY n IS COMPLETE`. It writes `results/hpc_350/summary_by_group.csv`. **`INCOMPLETE`: rerun the unfinished model, or tell the user.** A question that ran out of memory counts as done, so it shows up here as a low `n`.

### Expected, not bugs

- **`350 test videos` in the log:** every row sits in `test.csv` in this config, so the word means nothing here.
- **The rule can send fewer frames than the limit:** the limit is a ceiling, so a quiet video sends only its marks.
- **Qwen's limit at 20% can pass K:** about 100 foveated frames against Uniform K's 96 stock frames. Only SmolVLM has a frame cap.
- **A measured budget slightly above its target,** on Qwen or Phi: the limit uses the mean, and the kept frames can cost a little more. Report it, and don't adjust anything.
- **A `FutureWarning` about concatenating empty frames,** from the empty `train_val_fold1.csv`, is harmless.
- **A 5-minute `load_data` at 0% GPU** at the start of every job, including reruns.
- **With threads on, the first `sec` of a video or block** no longer includes loading its frames, so `sec` isn't a latency. Use the `done:` lines and the job's length for time.
- **`D: ground truth` and the other conditions in `select_conditions`** are computed and never used. On the new videos they mean nothing.

### Rules for this task

- **Don't change anything after seeing accuracy:** not the budgets, the limit, the preference order, the fovea setting, the slots or `K`.
- **Don't add rows, controls, models or videos without asking,** and keep any batching as a separate change.
- **Don't claim a winner.** Without a no-gate row, these results can't show whether the gate itself helps.
- **Never write to `data/folds/`, `results/hpc_all/`, `results/hpc_budget/` or `results/nymeria_smolvlm/`.** Never score with `--no-parity`.
- **Commit only when the user asks.** Never commit results, accuracy numbers, NetIDs, account names, tokens or `*CONTEXT*.md`.
- **Nothing runs on the user's Mac.** Give one read-only command at a time when you need them to look at something.

### What to report back

1. Any code you changed, and each step's last log line.
2. Step 1's token table, limits and `PROPOSED` line.
3. For each new model: parity, `SAME ANSWERS` with its `max_prob_diff`, and how many reused answers changed.
4. For each job: the GPU type and node, its length, its mean and lowest 30-minute GPU use, and any `oom` or `nan` records.
5. Per model, for all 350 videos, AEA and Nymeria: `n`, accuracy, measured budget, frames per question and `gaze_missing_pct`, with the note that the results can't show whether the gate itself helps.

## 11. Next task: SmolVLM on all 350 videos, with every row at 5, 10 and 20%

**Don't submit any job until the user approves it, one step at a time.** Nothing here has been run.

### Why this exists

SmolVLM is the smallest of the 4 models, so the user wants its whole table first: the budget rule, its references and all 8 baselines at the same
budgets, on every EgoEverything video, with AEA and Nymeria apart. The other 3 models come later, if the user asks.

### Scope, set by the user

- **Model:** `smolvlm` only.
- **Videos:** all 350, 5,985 questions, on the joined data from section 10's step 1.
- **Rows at 5, 10 and 20%:** the budget rule (`budget`), Uniform K + average pooling (`avgpool`), GazeVLM (`gazevlm`), GAZEVLM_EFFICIENT (`gazevlm2`), `streamingtom`, `dycoke`, `fastvid`,
  `prunevid` and `dtd`.
- **Reference rows:** `uniform` (the 100% budget) and `blind` (the 0% budget).
- **No `ours` and no `nogate`,** so these results can't show whether the gate itself helps. The report says that on every line, and `compare_budget.py` isn't used.
- **Do this before section 10's step 4.** Don't start that step, or any other model's baselines, unless the user says so.
- **Unchanged:** the budgets, the rule's limit, preference order and slots, `K` (64 on SmolVLM), the fovea setting, and every baseline parameter except its budget parameter.

### One GPU type, and nothing reused

Answers can change with the GPU type: SmolVLM differed on 60 of 1,760 answers between an L40S and a B200, and on 0 of 60 where the type matched on other models. So every row of this run comes
from one GPU type, **B200**.

- **Why B200:** it has run SmolVLM for 30 minutes at 27.7% GPU use without being cancelled, which would break the 50% limit on `gl*` nodes.
- **Nothing is copied.** `seed_results.py` isn't called, and every row is scored fresh into `results/smolvlm_350/`. Scoring the reference rows and the budget rule again costs about an hour, and it means no row depends on which GPU an old run landed on.
- **It's enforced.** The job asks for `--gres=gpu:b200:1`, which a requeue keeps. `check_gpu --expect B200` stops the job on anything else, and `--record` writes the GPU's name to `results/smolvlm_350/smolvlm/gpu_types.txt`. The report fails if that file shows two types.
- **If B200s stay busy, ask the user.** Don't override `--gres`. Changing the type is their decision, and it means starting the results folder again.
- **The old results are only compared with.** The pilot compares `uniform`, `blind` and `budget` with `results/nymeria_smolvlm/`, which also ran on a B200. A difference there isn't from the GPU type, so report the count even if it's 0.

### How each baseline reaches its budget

The target is a share of the question-weighted Uniform K tokens over all 350 videos, measured the way `summarize.py` does it. On a 64-frame video that's 259.2, 518.4 and 1,036.8 tokens per question, or 4.05, 8.1 and 16.2 tokens per frame. Each row gets one knob per budget for the
whole dataset, chosen on token counts only, with no answer read and no LLM run. `calibrate.py --fractions` does this, and it uses the same code as `results/hpc_all/`.

| Row | Budget parameter | Counted from | Fixed |
|---|---|---|---|
| `avgpool` | grid side `s` | the grid | adaptive pooling |
| `gazevlm2` | tokens per frame `B` | frame counts | `alpha`, `beta` |
| `gazevlm` | `rho` | gaze on every frame | `sigma`, `R_min`, 2 x 2 thumbnail, fixation gaze |
| `streamingtom` | `G/N` | frame counts | `tau`, `k`, `beta` |
| `dycoke` | pruning rate | frame counts | window 32 |
| `dtd` | drop share | frame counts | relative mode |
| `fastvid` | `r` | its plans on encoded frames, encoder only | `c`, `tau`, `ignore`, `d`, `p`, `beta` |
| `prunevid` | `beta` | its plans on encoded frames, encoder only | `tau` 0.8, `gamma`, `k`, sets of 14 or fewer not merged |

**Row breaks don't arise here.** They only exist on Phi, where these counts would land too high, so `--fractions` stops on Phi until the user decides how to count them.

### What each row is expected to reach

These are landings in percent at 5, 10 and 20%, from each method's token rule. A row takes the closest it can reach on either side. **Bold** is more than 0.5 points off, which the report flags as `nearest_only`. **Nothing is forced:** no other parameter changes to close a gap.

| Row | 5% | 10% | 20% | Why it can't get closer |
|---|---|---|---|---|
| `avgpool` | 4.94 (2 x 2) | **11.11** (3 x 3) | 19.75 (4 x 4) | square grids only: 10% is 4.94 or 11.11 |
| `gazevlm2` | 4.94 | 9.88 | 19.75 | |
| `gazevlm` | **about 10.6** | **about 10.6** | about 20 | the 2 x 2 thumbnail and the smallest gaze region are its floor, so 5% and 10% are one condition |
| `streamingtom` | 4.94 | 9.88 | 19.75 | |
| `dycoke` | **5.52** | 10.30 | 19.87 | 2 whole frames, then whole tokens per frame: 5% is 4.32 or 5.52 |
| `dtd` | 5.21 | 10.07 | 19.79 | |
| `fastvid` | about 4.94 | about 9.88 | **not known** | near-duplicate frames give up tokens, which sets a ceiling. It had one condition on SmolVLM before, which suggests a ceiling under 16%. Only the calibration can tell |
| `prunevid` | **not known** | about 10 | about 20 | sets of 14 tokens or fewer never merge, which sets a floor that depends on the video |

No baseline has run at 5% before: the earlier targets were Ours' budgets, 16 to 31% on SmolVLM. **Step 2 prints the real values. Show them to the user whatever they are.**

### Threads, parity and low GPU use

- **Baseline rows ignored `--prefetch-workers`, so a thread check would have tested nothing.** On SmolVLM they now load frames on threads, off by default. On the other models `run.py` stops with an error instead of ignoring it.
- **The two views of `gazevlm2`:** a frame's views share its key, so loaded frames are looked up by path. Each local view is a lossless crop in `/dev/shm`, about 86 MB per video at most, removed once it's encoded.
- **Parity is per model, not per row.** Parity runs at the start of every run. To reach the baselines too, `check_identity.py` sets 4 rows to keep every token: `avgpool` 9 x 9, `dtd` dropping none, `dycoke` pruning none and `streamingtom` keeping all. They must then give Uniform K's answers and tokens. The other 4 rows can't reproduce stock input, so they get parity and the thread check only.
- **Low GPU use:** this will be a low-use job, because SmolVLM's answers are short at these budgets. Which limit applies to B200 nodes isn't known. **If a job is cancelled for low GPU use, stop and tell the user.** Don't move it to another GPU type.

### What changed in the code

| File | Change |
|---|---|
| `accuracy_tables/calibrate.py` | new `--fractions`: targets from `budget.fractions`, nothing scored first, writes `targets.csv`, stops on Phi. New `--prefetch-workers`. GazeVLM's count now runs one video at a time, because keeping every frame's sorted heatmap would take about 45 GB on 350 videos |
| `vlmeval/gazevlm.py`, `streamingtom.py` | SmolVLM's backends load frames through `prefetch` and `frame_input` |
| `vlmeval/avgpool.py`, `gazevlm2.py`, `dycoke.py`, `dtd.py`, `fastvid.py`, `prunevid.py` | `prefetch_workers`, off by default. `gazevlm2` now names each crop's file by frame and reads the frame's side from its header |
| `accuracy_tables/models.py`, `run.py` | pass the workers to the baseline rows, and stop on models without them |
| `accuracy_tables/hpc/check_gpu.py` | `--expect` and `--record`. With neither it behaves as before |
| `accuracy_tables/hpc/all350/check_identity.py` | new: the full-budget check |
| `accuracy_tables/hpc/all350/report.py` | `--models` and `--targets`: target and measured budget per row and budget, `nearest_only`, the GPU types, a fixed row order |
| `accuracy_tables/summarize.py` | a new column, `prompt_tokens_per_q` |
| `accuracy_tables/hpc/smolvlm350.SBATCH`, `smolvlm350_tests.SBATCH`, `smolvlm350_summary.SBATCH` | new: the jobs |
| `tests/test_smolvlm350.py` | new: CPU tests with a fake encoder and LLM, through the real run loops and `calibrate.py` |

**Section 10's jobs and rows don't change.** `all350_vlm.SBATCH`, `seed_results.py`, `config.all350.yaml`, `budget_rule.py` and `compare_budget.py` aren't touched, and neither are your local `--comment` lines.

### Steps

**Job count: 7 `sbatch` jobs, 4 on GPU and 3 on CPU.** All times are estimates, and the answers number 173,565 at most (29 conditions x 5,985). That's 167,580 if `gazevlm` merges its 5% and 10%, and fewer if `fastvid` or `prunevid` merge any too.

| Step | Job | Runs on | Time |
|---|---|---|---|
| 1. Tests | `smolvlm350_tests.SBATCH` | CPU | minutes |
| 2. The baselines' knobs | `smolvlm350.SBATCH calibrate` | GPU | about 1 hour |
| 3. Parity, threads, full budget | `smolvlm350.SBATCH check` | GPU | about 15 minutes |
| 4. The pilot, on section 10's 7 videos | `smolvlm350.SBATCH pilot`, `smolvlm350_summary.SBATCH pilot` | GPU, CPU | about 20 minutes |
| 5. All 350 videos | `smolvlm350.SBATCH full`, `smolvlm350_summary.SBATCH full` | GPU, CPU | about 5 hours |

Set `SBATCH_ACCOUNT` as in section 5, and submit from `repo/`. **Before step 1,** on the login node:

```bash
squeue -u $USER                                 # no job of yours may be using the checkout, so wait for Phi's pilot
git status --short                              # expect only your 2 local --comment edits
git pull --ff-only origin eyehead-gate
ls /scratch/$USER/GazeVLM/newvideos/lists/all350_pilot.txt     # from section 10's step 1
```

**Step 1: tests.**

```bash
sbatch accuracy_tables/hpc/smolvlm350_tests.SBATCH
```

The log should end with every test passing. **If one fails, fix it if it's a small problem in the new code and report what you changed.** Don't loosen a test to make it pass.

**Step 2: the baselines' knobs.** Ask the user first.

```bash
sbatch --job-name=smolvlm350-calibrate accuracy_tables/hpc/smolvlm350.SBATCH calibrate
```

The log prints the GPU, the budgets and a table of every row's target, predicted budget, gap and knob, and ends with `TARGETS WRITTEN`. It writes `results/smolvlm_350/config_baselines.yaml`, `calibration.json` and `targets.csv`. **Show the user the table, and wait.** A job on another GPU type stops at once.

**Step 3: parity, threads and full budget.** Ask the user first.

```bash
sbatch --job-name=smolvlm350-check accuracy_tables/hpc/smolvlm350.SBATCH check
```

The log should show `parity OK` three times, `=== 0 failed of 13` twice, `SAME ANSWERS`, `=== 0 failed of 5`, `SAME AS UNIFORM K`, and end with `CHECK DONE: smolvlm`. **`ANSWERS DIFFER` or `NOT THE SAME AS UNIFORM K`: stop and tell the user.** If parity fails, send them the report.

**Step 4: the pilot.** Ask the user first. These are section 10's 7 videos, 386 questions.

```bash
ID=$(sbatch --parsable --job-name=smolvlm350-pilot accuracy_tables/hpc/smolvlm350.SBATCH pilot)
sbatch --dependency=afterany:${ID%%;*} accuracy_tables/hpc/smolvlm350_summary.SBATCH pilot
```

- **The pilot's log** ends with a `TOKENS MATCH` line, which counts how many answers differ from `results/nymeria_smolvlm/`. Its `missing` count is the AEA video, which that folder doesn't have. Show the user this line even if it's 0. **`TOKENS DIFFER`: stop and tell the user.**
- **The summary's log** gives the GPU use, then the table per group, and ends with `EVERY n IS COMPLETE`.
- **Show the user the measured budget against the target for every row and budget, and wait.** That's the check before any full run. The table holds accuracy too, and nothing may change because of it.

**Step 5: all 350 videos.** Ask the user first.

```bash
sbatch --job-name=smolvlm350-full accuracy_tables/hpc/smolvlm350.SBATCH full
```

**A job that ends in `TIMEOUT` isn't requeued: submit the same line again,** and it continues where it stopped. Never run two at once: `singleton` prevents it if you keep the job name. When it has finished:

```bash
sbatch accuracy_tables/hpc/smolvlm350_summary.SBATCH full
```

The log should end with `EVERY n IS COMPLETE`, and it writes `results/smolvlm_350/summary_by_group.csv`. **`INCOMPLETE`: rerun the job, or tell the user.** A question that ran out of memory counts as done, so it shows up here as a low `n`.

### Expected, not bugs

- **`nearest_only` is True** on the cells in bold above. It's a flag, not a failure.
- **One condition fewer than 29 or more,** where budgets share a condition. The row still has a line per budget, with the same results.
- **`fastvid` or `prunevid` landing apart on AEA and Nymeria:** one knob for the whole dataset, and their tokens depend on the video. Report it, and don't add a knob per group.
- **`budget` measuring under its target:** the limit is a ceiling, so a quiet video sends only its marks.
- **`gaze_missing_pct` is blank** on rows that don't use gaze. Only `budget`, `gazevlm` and `gazevlm2` have it.
- **Under 64 frames per question** on `fastvid`, `prunevid` and on short videos: frames left with no tokens leave the prompt.
- **`prompt_tokens_per_q` differs between rows at one budget:** the budget counts visual tokens only, and each frame sent also adds a time label to the prompt. Report it.
- **About 10 minutes at 0% GPU at the start of the calibrate job:** loading the data, then counting GazeVLM's tokens from the gaze.
- **`350 test videos` in the log, a `FutureWarning` about empty frames, and `sec` not being a latency:** as in section 10.

### Rules for this task

- **Don't change anything after seeing accuracy:** not the budgets, the limit, the preference order, the slots, the fovea setting, `K` or any baseline parameter.
- **Don't rerun the calibrate job after step 3 without asking.** It rewrites the knobs, and results under the old ones would stop matching `targets.csv`.
- **Calibrate on token counts only.** Never pick a knob by reading answers.
- **Don't force a budget:** if a row can't reach one, report the nearest. Changing another parameter or the frame count is the user's decision.
- **Don't add rows, controls, models or videos without asking,** and keep any batching as a separate change.
- **Don't claim a winner.** Without a no-gate row, these results can't show whether the gate itself helps.
- **One GPU type.** Don't override `--gres`, and don't copy records in from another folder.
- **Never write to `data/folds/`, `results/hpc_all/`, `results/hpc_budget/` or `results/nymeria_smolvlm/`.** Never score with `--no-parity`.
- **Commit only when the user asks.** Never commit results, accuracy numbers, NetIDs, account names, tokens or `*CONTEXT*.md`.
- **Nothing runs on the user's Mac.** Give one read-only command at a time when you need them to look at something.

### What to report back

1. Any code you changed, and each step's last log line.
2. Step 2's table: every row's target, predicted budget, gap and knob, and which rows only reached the nearest.
3. Step 3: parity, `SAME ANSWERS` with each row's `max_prob_diff`, and the `SAME AS UNIFORM K` lines.
4. Step 4: the `TOKENS MATCH` line, and the measured budget against the target for every row and budget.
5. For each GPU job: the GPU type and node, its length, its mean and lowest 30-minute GPU use, and any `oom` or `nan` records.
6. For all 350 videos, AEA and Nymeria, per row and budget: `n`, accuracy, target and measured budget, frames per question and `gaze_missing_pct`, with the note that the results can't show whether the gate itself helps.

## 12. Next task: Qwen3-VL-4B on all 350 videos, split across 3 A100s

**Don't submit any job until the user approves it, and stop for their OK after each step.** Nothing here has been run.

### Why this exists

Section 11's table on SmolVLM took one job. The same table on Qwen3 is about 60 GPU hours, mostly encoding, so it's split across 3 separate one-GPU jobs and merged into one table that reads exactly like an unsharded run's.

### Scope, set by the user

- **Model:** `qwen3`, which is `Qwen/Qwen3-VL-4B-Instruct`. A test checks that, and that no config points it elsewhere.
- **Videos:** all 350, 5,985 questions, on the joined data from section 10's step 1, with AEA and Nymeria reported separately.
- **Rows:** `uniform`, `blind`, `budget`, `avgpool`, `gazevlm`, `gazevlm2`, `streamingtom`, `dycoke`, `fastvid`, `prunevid` and `dtd`. Everything but `uniform` and `blind` runs at 5, 10 and 20%.
- **The rule on Qwen3:** the approved 51 tokens per frame, so limits of 24, 48 and 96 frames. `K` stays 96, the policy D+P and the fovea `R=70 a=160`, with the same preference order and slots.
- **No `ours` and no `nogate`,** so these results can't show whether the gate itself helps. The report says that on every line, and `compare_budget.py` isn't used.
- **One GPU type, and nothing reused in the reported table.**
- **This is the only run in this section.** Don't start section 10's step 4 or any other model unless the user says so.

### How the split works

- **3 separate jobs, each with `--gres=gpu:a100:1`,** not one job holding 3 GPUs. They don't need to start together, and one crashing doesn't touch the others.
- **Whole videos go to shards, balanced by question count.** The videos with the most questions go first, each to the shard with the fewest so far. `shards.py` does this from the questions file alone, so it's the same on every rerun.
- **The manifest is written once,** by the calibrate job, to `<results>/manifest/`. Writing it again changes nothing, and a different assignment stops the job.
- **Each shard writes only to its own folder.** The final folder is written only by the merge:

```
results/qwen3_350/
  manifest/manifest.json, shard_0.txt, shard_1.txt, shard_2.txt, all_videos.txt
  calibrate/config_baselines.yaml                        read-only once calibrate has finished
  calibrate/qwen3/targets.csv, calibration.json, kept_counts.jsonl, gpu_types.txt
  shards/<n>/qwen3/<row>.jsonl, gpu_types.txt, shard_info.jsonl, run_info.json        each written only by shard <n>
  qwen3/<row>.jsonl, gpu_types.txt, merge_info.json      written only by the merge
  summary_by_group.csv                                   written only by the report
```

- **Each shard has its own job name,** such as `qwen3350-full-1`, so `singleton` only stops the same shard running twice. A job refuses to run under any other name, because shards sharing a name would run one at a time.
- **A rerun continues from the shard's own files.** The job cuts a half-written last line, then skips every question already there. You can rerun one shard alone.
- **Each job checks itself before it scores anything.** `shards.py start` compares its commit, code, targets, settings, GPU name and library versions with every earlier job of all 3 shards, stops on any difference, and otherwise adds a line to `shard_info.jsonl`. A `git pull` between shard jobs therefore stops the next one at once, rather than after hours of work the merge would refuse.

### One GPU type, and nothing reused

GPU type changes answers in this project, so everything here runs on A100s, including the calibration.

- **Enforced three ways:** the job asks for `--gres=gpu:a100:1`, which a requeue keeps. `check_gpu --expect A100` stops it on another type. `check_gpu --same-as` stops it unless its full GPU name is the one the calibrate job recorded, so a shard can't spend hours on an A100 variant the merge would refuse.
- **If A100s stay busy, ask the user.** Don't override `--gres` or use another GPU type. If a job stops because its A100 has a different full name, tell the user.
- **Nothing is copied in.** Every row is scored fresh. `results/hpc350_pilot/` is only compared with, in step 4.

### Calibration

- **The same calculation as section 11:** a share of the question-weighted Uniform K tokens over all 350 videos, on token counts only, with no answer read and no LLM run. On a 96-frame video that's 12.8, 25.6 and 51.2 tokens per frame. Only each row's budget parameter changes.
- **It can continue after a time limit.** Encoding about 33,600 frames, then running FastVID's and PruneVid's whole plans at every candidate value, would take about 12 hours on an A100, and section 11's version starts again from nothing if it's stopped. So `calibrate.py --counts` saves each video's counts as soon as they're known, and a rerun skips those videos.
- **It's faster, with the same numbers.** A frame's kept tokens come straight from FastVID's budgets and PruneVid's set sizes, without selecting or merging anything. Tests compare this with the whole plans at every candidate value, and each calibrate job does the same on its first 2 real videos and stops if they differ. `--counts` is used only here: section 11's path doesn't change.
- **It runs once.** The job refuses to start if `targets.csv` exists, and it makes `targets.csv` and `config_baselines.yaml` read-only when it finishes. Shards record both files' hashes. **Don't rerun calibrate after the check step without asking the user.**

### What each row is expected to reach

These are landings in percent at 5, 10 and 20%, from each method's token rule on a 96-frame video. The few AEA videos with fewer frames shift them slightly. A row takes the closest it can reach on either side, and **bold** is more than 0.5 points off, which the report flags as `nearest_only`. **Nothing is forced.**

| Row | 5% | 10% | 20% | Why it can't get closer |
|---|---|---|---|---|
| `avgpool` | **6.25** (4 x 4) | 9.77 (5 x 5) | **19.14** (7 x 7) | square grids only: 5% is 3.52 or 6.25, and 20% is 19.14 or 25.00 |
| `gazevlm2` | 4.69 | 10.16 | 20.31 | |
| `gazevlm` | about 5.0 | about 10 | about 20 | 5% is right at its floor, the 2 x 2 thumbnail and the smallest gaze region |
| `streamingtom` | 5.08 | 10.16 | 19.92 | |
| `dycoke` | 5.02 | 9.94 | 20.15 | |
| `dtd` | 4.91 | 9.93 | 19.98 | |
| `fastvid` | about 4.69 | about 10.16 | about 19.92 | near-duplicate frames give up tokens, which could set a ceiling. Only the calibration can tell |
| `prunevid` | about 5 | about 10 | about 20 | one search step of `beta` is about 1 point here, and sets of 14 tokens or fewer never merge |

All 8 baselines already have Qwen3 backends, which ran in `results/hpc_all/` at higher budgets. **No baseline has run at 5 or 10% on Qwen3.** Step 2 prints the real values.

### The merge

- **By hand, after all 3 shards finish.** It doesn't depend on the shard jobs, because a shard that hits the time limit gets resubmitted and would break that dependency. One job does the merge, then the comparison or summary.
- **It checks everything first and writes only if all of it passes.** It stops, leaving the final folder as it was, on a missing question or condition, a duplicate, a half-written or unreadable line, an error record, a record of a video that isn't that shard's, a manifest that doesn't split the videos cleanly, or any disagreement in GPU name, commit, code, targets file, settings or library versions.
- **An out-of-memory or `nan` record stops it.** Section 11 counted those as done and showed them as a low `n`. Here the merge names them, so tell the user.
- **Its output is the shards' own lines, unchanged, in the order an unsharded run writes them.** That order matters: `video_bootstrap` draws from videos in the order they first appear, so a different order would change the confidence intervals. `summarize.py` and `report.py` don't change, and they read the merged folder like any other.

### Threads, parity and low GPU use

- **Baseline rows ignored `--prefetch-workers` on Qwen3, so a thread check would have tested nothing.** They now load frames on threads, off by default, and the check job compares on against off. Models without this still stop with an error.
- **Parity runs at the start of every job, on that job's first question,** so the 3 shards check 3 different questions. **If it fails, send the user the report. Don't loosen it and never use `--no-parity`.**
- **Low GPU use:** `ga*` nodes cancel jobs below 50%. Encoding keeps the GPU busy and frames load on threads, but this has only been measured on section 10's rows, so step 4 measures it per shard. **If any shard's mean is under 50%, stop and tell the user before step 5.**

### What changed in the code

| File | Change |
|---|---|
| `accuracy_tables/hpc/all350/shards.py` | new: the assignment and manifest, and each job's check and record |
| `accuracy_tables/hpc/all350/merge_shards.py` | new: the merge and everything it refuses |
| `accuracy_tables/hpc/all350/estimate_hours.py` | new: a full-run estimate from the pilot's timings. Reads no answers |
| `accuracy_tables/hpc/qwen3350.SBATCH`, `qwen3350_merge.SBATCH`, `qwen3350_tests.SBATCH` | new: the jobs |
| `tests/test_qwen3350.py` | new: CPU tests with a fake encoder and LLM, through the real calibrate, run, merge and report code |
| `accuracy_tables/calibrate.py` | new `--counts`: resumable, with FastVID's and PruneVid's counts from their token arithmetic. Without it, nothing changes |
| `accuracy_tables/hpc/check_gpu.py` | new `--same-as`. Without it, nothing changes |
| `accuracy_tables/hpc/newvideos/compare_records.py` | new `--exact`, and a `ref_only` count in each line it prints. Without `--exact`, what fails a job doesn't change |
| `vlmeval/gazevlm.py`, `streamingtom.py` | Qwen3's backends load frames through `prefetch` and `frame_input` |
| `accuracy_tables/models.py`, `run.py` | Qwen3's baseline rows take `--prefetch-workers`, and the help says so |
| `accuracy_tables/hpc/all350/check_identity.py` | its description no longer names SmolVLM's grid |

**Section 11's jobs don't change:** a test checks that their 3 job files are byte for byte what ran. `report.py`, `summarize.py`, `config.all350.yaml`, `budget_rule.py`, `seed_results.py` and your local `--comment` lines aren't touched.

### Steps

**Job count: 15 `sbatch` jobs, 11 on GPU and 4 on CPU,** plus an expected 6 resubmissions in step 5. All times are estimates, and step 4 replaces the last one. Answers number at most 173,565 (29 conditions x 5,985), and fewer if budgets share a condition.

| Step | Job | Runs on | Time |
|---|---|---|---|
| 1. Tests | `qwen3350_tests.SBATCH` | CPU | a few minutes |
| 2. The knobs and the assignments | `qwen3350.SBATCH calibrate` | 1 GPU | 3 to 5 hours |
| 3. Checks, then the small set in 3 shards | `qwen3350.SBATCH check`, `checkshard 0, 1, 2`, `qwen3350_merge.SBATCH check` | 4 GPU, CPU | about 1 hour, then up to 30 minutes per shard |
| 4. The pilot, on section 11's 7 videos | `qwen3350.SBATCH pilot 0, 1, 2`, `qwen3350_merge.SBATCH pilot` | 3 GPU, CPU | about 30 minutes per shard |
| 5. All 350 videos | `qwen3350.SBATCH full 0, 1, 2`, `qwen3350_merge.SBATCH full` | 3 GPU, CPU | about 20 hours per shard: 3 jobs each |

Set `SBATCH_ACCOUNT` as in section 5, and submit from `repo/`. **Before step 1,** on the login node:

```bash
squeue -u $USER                                 # no job of yours may be using the checkout
git status --short                              # expect only your 2 local --comment edits
git pull --ff-only origin eyehead-gate
ls /scratch/$USER/GazeVLM/newvideos/lists/all350_pilot.txt     # from section 10's step 1
```

**Step 1: tests.**

```bash
sbatch accuracy_tables/hpc/qwen3350_tests.SBATCH
```

The log should end with every test passing, including section 11's. **Show the user the test result and the log's last line, and stop.** If one fails, fix it if it's a small problem in the new code and report what you changed. **Don't loosen a test to make it pass.**

**Step 2: the knobs and the assignments.** Ask the user first.

```bash
sbatch --job-name=qwen3350-calibrate accuracy_tables/hpc/qwen3350.SBATCH calibrate
```

If it ends in `TIMEOUT`, submit the same line again and it continues. The log prints the GPU, each manifest's questions and videos per shard, `the token arithmetic matches the methods' own plans` twice per job, a table of every row's target, predicted budget, gap, `nearest_only` and knob, and ends with `TARGETS WRITTEN`.

**Report the predicted budget and `nearest_only` for every row and target, and the questions and videos per shard, and stop.** If a baseline doesn't support Qwen3 or can't reach a target, flag it and ask the user. **Don't tune it or drop it.**

**Step 3: checks, then the small set in 3 shards.** Ask the user first. The small set is 5 videos chosen from the questions file: the AEA video with the fewest questions, the Nymeria video with the fewest above one, and the first 3 Nymeria videos with one. First, on one A100:

```bash
sbatch --job-name=qwen3350-check accuracy_tables/hpc/qwen3350.SBATCH check
```

The log should show `parity OK` four times, `=== 0 failed of 13` three times, `SAME ANSWERS`, `=== 0 failed of 5`, `SAME AS UNIFORM K`, and end with `CHECK DONE: qwen3`. **`ANSWERS DIFFER` or `NOT THE SAME AS UNIFORM K`: stop and tell the user.** Then the same small set in 3 shards:

```bash
for N in 0 1 2; do
    sbatch --job-name=qwen3350-checkshard-$N accuracy_tables/hpc/qwen3350.SBATCH checkshard $N
done
```

Each log ends with `SHARD DONE: checkshard <n>`. When all 3 have:

```bash
sbatch accuracy_tables/hpc/qwen3350_merge.SBATCH check
```

The log should show `MERGED`, then one line per row, and end with `SAME ANSWERS AND PROBABILITIES`. **Expect 0 differing answers and `max_prob_diff` 0.0 on every row. Anything else fails the job: stop and tell the user.** Report each line, and stop.

**Step 4: the pilot.** Ask the user first. These are section 11's 7 videos, 386 questions.

```bash
for N in 0 1 2; do
    sbatch --job-name=qwen3350-pilot-$N accuracy_tables/hpc/qwen3350.SBATCH pilot $N
done
```

Each log ends with `SHARD DONE: pilot <n>`. When all 3 have:

```bash
sbatch accuracy_tables/hpc/qwen3350_merge.SBATCH pilot
sacct -u $USER -S now-7days -X --name=qwen3350-pilot-0,qwen3350-pilot-1,qwen3350-pilot-2 --format=JobName%24,NodeList,Elapsed,State
```

The merge's log shows, in order:

- **`MERGED`,** with each shard's questions, videos and node, and the full GPU name.
- **A `TOKENS MATCH` line,** which counts how many answers differ from section 10's Qwen3 pilot in `results/hpc350_pilot/`, also on an A100. Report it even if it's 0. **`TOKENS DIFFER`: stop and tell the user.**
- **The estimate of the full run's hours.** Correct it as its last line says, using the job lengths from `sacct`.
- **Each job's GPU mean and lowest 30-minute mean.** A short shard may have no 30-minute reading, so say so.
- **The table per group,** ending with `EVERY n IS COMPLETE`.

**Report each shard's time, GPU mean and lowest 30-minute mean, the estimate, the differing answers, and the measured budget against the target for every row and budget. If any shard's mean is under 50%, stop and tell the user before step 5.** `MERGE STOPPED` or `INCOMPLETE`: stop and tell the user. The table holds accuracy too, and nothing may change because of it.

**Step 5: all 350 videos.** Ask the user first.

```bash
for N in 0 1 2; do
    sbatch --job-name=qwen3350-full-$N accuracy_tables/hpc/qwen3350.SBATCH full $N
done
```

**A shard that ends in `TIMEOUT` isn't requeued: submit that shard's line again,** with its own number, and it continues where it stopped. Each is expected to need about 3 jobs. Never run two of one shard at once: `singleton` prevents it. To see how far each has got, without any answers:

```bash
wc -l /scratch/$USER/GazeVLM/results/qwen3_350/shards/*/qwen3/*.jsonl
```

**Only when all 3 logs end with `SHARD DONE: full <n>`:**

```bash
sbatch accuracy_tables/hpc/qwen3350_merge.SBATCH full
```

The log should end with `EVERY n IS COMPLETE`, and it writes `results/qwen3_350/summary_by_group.csv`. **`MERGE STOPPED` or `INCOMPLETE`: stop and tell the user.** It's safe to rerun once the cause is fixed.

### Expected, not bugs

- **Shards finishing hours apart.** They're balanced by questions, but encoding is most of the time on Qwen3 and follows the number of videos, which can differ by about 20% between shards. The manifest shows both.
- **A rerun of a finished shard** does nothing after about 5 minutes of loading, and ends with `SHARD DONE`.
- **Several lines per job in `shard_info.jsonl` and `gpu_types.txt`:** one per job. They must all agree.
- **`nearest_only` is True** on the cells in bold above. It's a flag, not a failure.
- **Fewer than 29 conditions,** if two budgets share one. The row still has a line per budget.
- **`fastvid` or `prunevid` landing apart on AEA and Nymeria:** one knob for the whole dataset, and their tokens depend on the video. Report it, and don't add a knob per group.
- **`budget` measuring under its target:** the limit is a ceiling, so a quiet video sends only its marks. It can also land slightly over, because 51 tokens per frame is a mean rounded up.
- **`gaze_missing_pct` is blank** on rows that don't use gaze. Only `budget`, `gazevlm` and `gazevlm2` have it.
- **Under 96 frames per question** on some rows and on short videos: frames left with no tokens leave the prompt.
- **`prompt_tokens_per_q` differs between rows at one budget:** the budget counts visual tokens only, and each frame sent also adds a time label to the prompt. Report it.
- **`--same-as` in the check job against calibrate's GPU name:** every A100 job must match it, even though only the shards are merged.
- **A `FutureWarning` about empty frames, `<n> test videos` in the log, and `sec` not being a latency:** as in section 10.

### Rules for this task

- **Don't change anything after seeing accuracy:** not the budgets, the limit, the preference order, the fovea setting, the slots or `K`.
- **Don't add rows, controls, models or videos without asking,** and keep any batching as a separate change.
- **Calibrate on token counts only,** once. Don't rerun it after the check step without asking the user.
- **Don't force a budget:** if a row can't reach one, flag it and ask. Don't tune a baseline or drop it.
- **Don't claim a winner.** Without a no-gate row, these results can't show whether the gate itself helps.
- **Never score with `--no-parity`.**
- **If A100s stay busy, ask the user.** Don't override `--gres`, use another GPU type or copy records in from another folder.
- **Don't pull between shard jobs of one run.** A different commit stops the next job, and the shards can only merge if they all ran the same code.
- **Never write to `data/folds/`, `results/hpc_all/`, `results/hpc_budget/`, `results/nymeria_smolvlm/`, `results/hpc350_check/`, `results/hpc350_pilot/`, `results/smolvlm_350/`, `results/smolvlm_350_pilot/` or `results/smolvlm_350_check/`.**
- **Commit only when the user asks.** This repo is public: never commit results or accuracy numbers, NetIDs or account names, tokens or `*CONTEXT*.md`.
- **Report every code change you make to existing files, and every deviation.**
- **Nothing runs on the user's Mac.** Give one read-only command at a time when you need them to look at something.

### What to report back

1. Any code you changed, and each step's last log line.
2. Step 2: every row's target, predicted budget and `nearest_only`, and the questions and videos per shard.
3. Step 3: parity, `SAME ANSWERS` with each row's `max_prob_diff`, the `SAME AS UNIFORM K` lines, and each row's line from the shards against the whole run.
4. Step 4: the `TOKENS MATCH` line, the estimate of full-run hours, and the measured budget against the target for every row and budget.
5. For each shard: its questions and videos, node, full GPU name, elapsed time, GPU mean and lowest 30-minute mean, and any `oom` or `nan` records.
6. For all 350 videos, AEA and Nymeria, per row and budget: `n`, accuracy, target and measured budget, frames per question, `gaze_missing_pct` and `prompt_tokens_per_q`, flagged rows and every deviation, with the note that the results can't show whether the gate itself helps. **Don't claim a winner.**

## 13. Next task: Qwen2.5-VL-3B on all 350 videos, split across 3 A100s

**Before you pull anything, confirm that Qwen3's full merge in section 12 has finished.** Pulling changes the code on disk, so a Qwen3 shard resubmit or its merge could run on a different commit from the one its shards started with. Read this section on GitHub until then. The first command block below does the check, and only pulls if it passes.

**The user submits every job.** Nothing here has been run.

### Why this exists

It's the same table as sections 11 and 12, on Qwen2.5. Section 12's Python is reused as it is: `shards.py`, `merge_shards.py`, `report.py`, `calibrate.py`, `check_identity.py` and `gpu_summary.py` all take the model as a setting. Only 3 small job files are new, because Qwen3's must stay as they are while its jobs run.

### Scope, set by the user

- **Model:** `qwen2.5`, which is `Qwen/Qwen2.5-VL-3B-Instruct`, the same id as every earlier run. Step 1 checks the cluster's copy.
- **Videos:** all 350, 5,985 questions, with AEA and Nymeria reported separately.
- **Rows:** `uniform`, `blind`, `budget`, `avgpool`, `gazevlm`, `gazevlm2`, `streamingtom`, `dycoke`, `fastvid`, `prunevid` and `dtd`. Everything but `uniform` and `blind` runs at 5, 10 and 20%. **No earlier result is reused.**
- **GPUs:** 3 one-GPU jobs, all on A100 (`--gres=gpu:a100:1`), with whole videos per shard, balanced by question count.
- **Results:** `results/qwen25_350/`, laid out as in section 12, with `qwen2.5` in place of `qwen3`.
- **Steps:** the tests, calibrate, then the full run straight away. **There is no check job and no pilot.** Parity still runs at the start of every shard job, and `--no-parity` is never used.

### The settled values

None of these changes. They come from `common.py`'s defaults and from commit `98c383a`.

| Setting | Value |
|---|---|
| `K` | 96 Uniform K frames per video, or all of a shorter video's frames |
| Policy | D+P |
| Fovea | `R=70 a=160`, the smallest setting, on the 448 px frame |
| Budgets | 5, 10 and 20% of each video's Uniform K tokens |
| Stock tokens per frame | 256 |
| Rule's tokens per frame | 51 (`config.all350.yaml`, approved in `98c383a`) |
| Rule's frame limits | 24, 48 and 96 on a video with at least 96 frames |
| Preference order and slots | as in `budget_rule.py` |
| Baselines' other parameters | `common.py`'s defaults; only each row's budget parameter is calibrated |
| Calibration | token counts only. A row that can't reach a budget is reported as `nearest_only`, with nothing tuned to fix it |

### What's different from section 12

| | Section 12, Qwen3 | Section 13, Qwen2.5 |
|---|---|---|
| Check job and small set in shards | yes | **no** |
| Pilot | yes | **no** |
| `max_prob_diff` 0.0 between shards and a whole run, on the GPU | measured | **not measured:** only on the fake model in the unit tests |
| Threads against no threads on baseline rows, on the GPU | measured | **not measured:** only on the fake model in the unit tests |
| GPU use before the full run | measured | **not measured:** watch it in the first hour |
| Comparison with an earlier pilot | before the full run, and it can stop the job | after the merge, and it only reports |

### Threads

Qwen2.5's `uniform` and `budget` rows already loaded frames on threads, from section 10. **Its 8 baseline rows didn't, and ignored the setting,** which would have put the jobs at risk on `ga*` nodes, where GPU use under 50% gets a job cancelled. They now use the same threaded loading as section 12. Tests check that its images are identical with threads on and off, and that all 8 rows write the same records.

### What changed in the code

| File | Change |
|---|---|
| `accuracy_tables/hpc/qwen25350.SBATCH`, `qwen25350_merge.SBATCH`, `qwen25350_tests.SBATCH` | new: the jobs |
| `accuracy_tables/hpc/all350/shard_progress.py` | new: answers done per shard and row. Standard library only, so it runs on a login node, and it reads no answers |
| `tests/test_qwen25350.py` | new: CPU tests on a fake Qwen2.5, through the real calibrate, run, merge and report code |
| `vlmeval/gazevlm.py`, `streamingtom.py` | Qwen2.5's backends load frames through `prefetch` and `frame_input` |
| `accuracy_tables/models.py`, `run.py` | Qwen2.5's baseline rows take `--prefetch-workers`, and the help says so |
| `tests/test_qwen3350.py` | its small-dataset fixture became a function that takes the model, so these tests can reuse it. One test renamed, and `qwen2.5` dropped from its list of models without threads |
| `tests/test_smolvlm350.py` | 2 tests that used `qwen2.5` as the model without threads now use `internvl`, which still has none. They test the same thing |

**Qwen3 isn't affected.** A test checks that its 3 job files and the code its merge runs (`merge_shards.py`, `shards.py`, `report.py`, `summarize.py`, `check_gpu.py`, `repair_results.py`, `compare_records.py`) are byte for byte what `e3d6803` had, and another that a merge never looks at the commit or code now on disk. `smoke.SBATCH` and `run_model.SBATCH` aren't touched, so your 2 local `--comment` edits can't conflict.

### Steps

**Job count: 6 `sbatch` jobs, 4 on GPU and 2 on CPU,** plus an expected 3 resubmissions. **All times are estimates, with no pilot behind them on this GPU.**

Set `SBATCH_ACCOUNT` as in section 5. Every command assumes `/scratch/$USER/GazeVLM`.

**Step 1: confirm Qwen3 has finished, pull, and run the tests.**

```bash
cd /scratch/$USER/GazeVLM/repo
if squeue -h -u $USER -o "%j" | grep -q "^qwen3350"; then
    echo "STOP: a Qwen3 job is still queued or running. Don't pull."
elif [ ! -s ../results/qwen3_350/qwen3/merge_info.json ] || ! tail -n 1 "$(ls -t ../logs/qwen3350-merge_*.log | head -1)" | grep -q "EVERY n IS COMPLETE"; then
    echo "STOP: Qwen3's full merge hasn't finished. Don't pull."
else
    git status --short
    git pull --ff-only origin eyehead-gate
    git log --oneline | head -1
    ls $HF_HOME/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/
    grep -h '"model_id"' ../results/hpc_all/qwen2.5/run_info.json ../results/hpc350_pilot/qwen2.5/run_info.json
    sbatch accuracy_tables/hpc/qwen25350_tests.SBATCH
fi
```

- **`STOP`:** do nothing else, and try again later.
- **Expect** `git status` to show only your 2 local `--comment` edits, one snapshot folder, and `"model_id": "Qwen/Qwen2.5-VL-3B-Instruct"` twice.
- **More than one snapshot, or another `model_id`: stop and tell the user,** because the checkpoint may not be the earlier runs' one.
- **The test log** is `logs/qwen25350-tests_<job>.log`. Its last line should say how many passed, with no `failed` or `error`. **Stop on any failure.** Fix it only if it's a small problem in the new code, and never loosen a test to make it pass.

**Step 2: the knobs and the assignment.** Estimated 2 to 3 hours.

```bash
cd /scratch/$USER/GazeVLM/repo
sbatch --job-name=qwen25350-calibrate accuracy_tables/hpc/qwen25350.SBATCH calibrate
```

- **If it ends in `TIMEOUT`, submit the same line again.** It continues from the videos it has already counted.
- **The log** prints the GPU, each shard's questions and videos, `the token arithmetic matches the methods' own plans` twice per job, a table of every row's target, predicted budget, gap, `nearest_only` and knob, and ends with `TARGETS WRITTEN`.
- **Expected `nearest_only`:** `avgpool` at 5% (6.25%) and 20% (19.14%), and perhaps `fastvid` or `prunevid`, because their tokens depend on the frames. Report them, and don't tune anything.
- **Stop on:** a job on another GPU type, `the token arithmetic differs`, or `the assignment is wrong`.
- **Don't rerun it once it has finished.** The job refuses, because the shards read its results.

**Step 3: all 350 videos.** Estimated 10 to 14 hours per shard, so about 2 jobs each.

```bash
cd /scratch/$USER/GazeVLM/repo
for N in 0 1 2; do
    sbatch --job-name=qwen25350-full-$N accuracy_tables/hpc/qwen25350.SBATCH full $N
done
```

- **A shard's log starts with** the GPU's full name, `shard <n>: commit ...`, and `parity OK`. It then has one block per row, and ends with `=== 0 failed of 13` and `SHARD DONE: full <n>`.
- **`TIMEOUT`, or a preemption that didn't requeue:** resubmit only that shard, with its own number. It repairs a half-written last line and continues from its own files:

```bash
N=0      # the shard that stopped
sbatch --job-name=qwen25350-full-$N accuracy_tables/hpc/qwen25350.SBATCH full $N
```

- **Stop on:** parity failing (send the user the report, and never use `--no-parity`), a line that says `failed of 13` with anything but `0 failed`, a job cancelled for low GPU use, a job refusing its GPU or `doesn't match the shards before it`, or A100s staying busy. **Don't override `--gres`, and don't pull between shard jobs.**

**Then the merge, by hand, only when all 3 logs show `SHARD DONE: full <n>`:**

```bash
cd /scratch/$USER/GazeVLM/repo
for N in 0 1 2; do grep -l "SHARD DONE: full $N" ../logs/qwen25350-full-${N}_*.log | tail -n 1; done
sbatch accuracy_tables/hpc/qwen25350_merge.SBATCH
```

The first line must print a log for all 3 shards. The merge's log, `logs/qwen25350-merge_<job>.log`, shows in order:

1. **`MERGED`,** with each shard's questions, videos and node, and the full GPU name. **`MERGE STOPPED`: stop and tell the user.** It names what's wrong and writes nothing.
2. **Each job's GPU mean and lowest 30-minute mean.**
3. **The table per group** (all, AEA, Nymeria), with `prompt_tokens_per_q`, ending with `EVERY n IS COMPLETE`. **`INCOMPLETE`: stop and tell the user.** It writes `results/qwen25_350/summary_by_group.csv`.
4. **A comparison with the earlier Qwen2.5 pilot** in `results/hpc350_pilot/`, ending with `TOKENS MATCH: answers differ on <N> of 1930 records compared`. That pilot ran on an L40S, so some answers will differ: between L40S and B200 it was 3 of 60. **Report the count. It doesn't fail the run.** If frames or tokens differ, it prints `TOKENS DIFFER` and then a line saying the comparison didn't end cleanly. The job still succeeds, and the merged results stand, but tell the user, because frames and tokens shouldn't depend on the GPU.

### Checking GPU use and progress

All of this only reads. Run it any time from a login node:

```bash
cd /scratch/$USER/GazeVLM/repo
squeue -u $USER -o "%.12i %.24j %.2t %.10M %R"
python3 -m accuracy_tables.hpc.all350.shard_progress --root ../results/qwen25_350 --model qwen2.5
python3 -m accuracy_tables.hpc.all350.gpu_summary ../logs/gpu_qwen25350-full-*_*.csv
for N in 0 1 2; do
    L=$(ls -t ../logs/qwen25350-full-${N}_*.log 2>/dev/null | head -1)
    [ -n "$L" ] || continue
    echo "--- shard $N"
    grep -v processor_kwargs "$L" | tail -n 1 | cut -c1-170
    cut -d, -f2 ../logs/gpu_$(basename "$L" .log).csv | tail -n 15 | tr -d '\n'; echo
done
```

- **`shard_progress` is the true progress.** It counts answers written, for the whole shard. An answer is one question under one condition, so a shard's total is its questions times the conditions, at most 29.
- **The counter in the log, such as `4675/11970  0.06 s each  ~7 min left`, covers less than it looks like.**
    - It counts answers in **the current row only,** and starts again at every row.
    - After a resubmission, its total is only **what that row had left,** not the row's size.
    - `~7 min left` is for **that row, not the shard.**
    - The name in brackets is the condition and video it has reached.
- **Rows run in this order,** whatever order the job lists them in: `uniform`, `blind`, `gazevlm`, `streamingtom`, `dycoke`, `fastvid`, `prunevid`, `dtd`, `avgpool`, `gazevlm2`, then `budget` at 5, 10 and 20%.
- **GPU use:** `gpu_summary` gives each job's mean and lowest 30-minute mean, and the last line of the loop shows the last 15 minutes. The node's name is in the `squeue` output. **On a `ga*` node, tell the user if a job's lowest 30-minute mean nears 50%.** About 5 minutes near 0% at the start of each job is normal, while it loads the data. **Check this in the first hour,** because there's no pilot behind it.

### Expected, not bugs

- **Shards finishing hours apart.** They're balanced by questions, but much of the time is encoding, which follows the number of videos.
- **A rerun of a finished shard** does nothing after about 5 minutes of loading, and ends with `SHARD DONE`.
- **Several lines per job in `shard_info.jsonl` and `gpu_types.txt`:** one per job. They must all agree.
- **Fewer than 29 conditions,** if two budgets share one. The row still has a line per budget.
- **`fastvid` or `prunevid` landing apart on AEA and Nymeria:** one knob for the whole dataset.
- **`budget` measuring under its target:** the limit is a ceiling, so a quiet video sends only its marks. It can also land slightly over, because 51 tokens per frame is a mean rounded up.
- **`gaze_missing_pct` is blank** on rows that don't use gaze.
- **`prompt_tokens_per_q` differing between rows at one budget:** the budget counts visual tokens only, and each frame sent also adds a time label.

### Rules for this task

- **Don't pull until Qwen3's full merge has finished.**
- **Don't change anything after seeing accuracy:** not `K`, the policy, the fovea, the baselines, the budgets, the limits or the preference order.
- **Don't add rows, controls, models or videos without asking.**
- **Calibrate on token counts only,** once.
- **Don't claim a winner.** Without a no-gate row, these results can't show whether the gate itself helps.
- **Never score with `--no-parity`.**
- **If A100s stay busy, ask the user.** Don't override `--gres` or use another GPU type.
- **Don't pull between shard jobs of this run.** A different commit stops the next job.
- **Never write to `results/qwen3_350/` or any folder section 12 lists as read-only.** `results/hpc350_pilot/` is only read.
- **Commit only when the user asks.** This repo is public: never commit results, NetIDs, account names, tokens or `*CONTEXT*.md`.
- **Nothing runs on the user's Mac.**

### What to report back

1. Any code you changed, and each step's last log line.
2. Step 1: the test result, the snapshot folder and the two `model_id` lines.
3. Step 2: every row's target, predicted budget and `nearest_only`, and the questions and videos per shard.
4. For each shard: its questions and videos, node, full GPU name, elapsed time, GPU mean and lowest 30-minute mean, and any `oom` or `nan` records.
5. The pilot comparison: the `TOKENS MATCH` line and each row's `max_prob_diff`.
6. For all 350 videos, AEA and Nymeria, per row and budget: `n`, accuracy, target and measured budget, frames per question, `gaze_missing_pct` and `prompt_tokens_per_q`, flagged rows and every deviation, with the note that the results can't show whether the gate itself helps. **Don't claim a winner.**

## 14. Next task: SmolVLM on all 350 videos, with every row matched on total FLOPs

**Section 14 comes in the same pull as section 13, so don't pull it until Qwen3's full merge has finished, and never while a Qwen3 or Qwen2.5 job is queued or running.**
Step 1 checks this. It can run alongside section 13's shards, because it uses a B200 and its own results folder.

**The user submits every job.** Nothing here has been run. The job files have had a bash syntax check only, and the Python and its tests have never been executed. Expect to fix small problems, and report what you change.

### Why this exists

Sections 11 to 13 match every row on **tokens**. But the 8 baselines encode all 64 frames at full resolution before they prune, so at an equal token budget they still spend the whole vision encoder, while the budget rule encodes only the frames it keeps, at fovea size. The user wants the other comparison: **every row given the same total FLOPs per question as ours,** and then their accuracy.

### Scope, set by the user

- **Model:** `smolvlm` only, on **B200** only, with every row scored fresh into `results/smolvlm_flops/`. Nothing is copied from section 11.
- **Videos:** all 350, 5,985 questions, with AEA and Nymeria reported separately.
- **Ours sets the targets and doesn't change.** The budget rule runs at its token budgets of 5, 10 and 20%, with the same limit, slots, preference order and `k=1` fovea as section 11. Whatever share of Uniform K's FLOPs each of these costs becomes that budget's FLOPs target.
- **Each baseline's knob is its number of input frames, `K'`.** It runs its own method on `K'` evenly spaced frames, with `K'` set so that its FLOPs come closest to the target.
- **Tokens per frame:** each baseline keeps ours' tokens per frame, which is 17 of SmolVLM's 81. Its budget parameter is set to reach that, the same way `calibrate.py` sets it. Every other parameter stays at `common.py`'s defaults.
    - The authors give a default only for some of the 8 methods, so one rule for all of them is used instead.
    - With equal tokens per frame and equal FLOPs, the rows differ only in how many frames they can afford, and which ones.
- **Two controls the user agreed to:**
    - `uniform_flops`: Uniform K' at stock input, with all 81 tokens per frame, at the same FLOPs.
    - `nogate`: the no-gate control at ours' token budgets, as in section 8. It has no FLOPs target of its own.
- **References:** `uniform` (K=64, which is 100% of the FLOPs) and `blind`.

### What a question's FLOPs are

Counted with `torch.utils.flop_counter`, which counts matrix products, convolutions and attention. The definitions are in `accuracy_tables/flops.py`.

| Part | Counted as | Who pays it |
|---|---|---|
| Gate | 5 fold models × one decision's FLOPs × the video's decisions | `budget` only. Every video is counted as the 5-model ensemble, the most it can cost |
| Vision encoder | frames encoded × one frame's FLOPs: a stock 384 px frame, or ours' `k=1` fovea | every row but `blind`. `gazevlm2` pays twice per frame, for its 2 views. A baseline pays for every input frame, even frames it then leaves out of the prompt |
| LLM | its one forward over the prompt, `c0 + c1 n + c2 n²` for `n` tokens | every row. The tables read the first answer token's logits, so nothing is decoded |

- **The baselines' own work** between the encoder and the LLM (similarities, clustering, saliency) is counted as zero, which favours them. The calibrate job measures it, and stops if it's more than 2% of their vision FLOPs.
- **Measured once, then checked.**
    - The calibrate job measures one stock frame, one foveated frame, the LLM at 5 prompt lengths and one gate decision.
    - It then scores every planned condition on 2 questions under the counter. Vision and LLM FLOPs must match `flops.py`: exactly for `uniform`, `uniform_flops`, `budget` and `nogate`, and within 2% above for the baselines.
- **The report recounts from each record,** using the frames and prompt tokens of every answer, so a row's measured FLOPs share isn't an estimate.

### What each row is expected to reach

These are estimates, not results. **Step 2 prints the real values. Show them to the user whatever they are.**

| | 5% | 10% | 20% |
|---|---|---|---|
| Ours' frame limit | 15 | 30 | 60 |
| Target, as a share of Uniform K's FLOPs | probably within a point of 5% | within a point of 10% | within a point of 20% |
| `K'` of the 7 single-view baselines | about 4 | about 8 | about 16 |
| `K'` of `uniform_flops` | about 3 | about 7 | about 14 |
| `K'` of `gazevlm2` | about 2 | about 4 | about 8 |

- **One stock frame should cost about 670 GFLOPs, and one foveated frame about 130,** as the earlier system check counted them. The count doesn't depend on the GPU, so report any value far from these.
- **`nearest_only` at 5% is likely,** for any row. One frame there is worth about 1 point of the FLOPs share, so the nearest `K'` can miss by up to half of that. It's a flag, not a failure.
- **`dycoke` and `dtd` keep their first frame whole,** so at about 4 frames they can't get down to 17 tokens per frame. They land at a higher token share and slightly above their FLOPs target.
- **`gazevlm2` needs an even token count,** 2 equal views, so it gets 16 or 18 tokens per frame. Both are as near 17.

### What changed in the code

**No existing file changed.** Section 11 to 13's code and job files are byte for byte as before, and the Qwen3 hash test still holds.

| File | What it is |
|---|---|
| `accuracy_tables/flops.py` | what a question's FLOPs are, and the measuring functions (SmolVLM only for now) |
| `accuracy_tables/flops_calibrate.py` | the plan: units, prompt lengths, targets, `K'`, the baselines' budget parameters, the check on real questions |
| `accuracy_tables/flops_run.py` | scores the plan through `run.py`'s own jobs, with only `K` and the budget parameter changed |
| `accuracy_tables/hpc/all350/flops_report.py` | the table per group, with measured FLOPs share against target, accuracy and its interval |
| `accuracy_tables/hpc/smolvlm_flops.SBATCH`, `smolvlm_flops_summary.SBATCH`, `smolvlm_flops_tests.SBATCH` | the jobs |
| `tests/test_smolvlm_flops.py` | CPU tests: the arithmetic, a whole plan on a fake SmolVLM, the run's jobs against the plan, the report, the job files |

### Steps

**Job count: 4 `sbatch` jobs, 2 on GPU and 2 on CPU.** All times are estimates.

| Step | Job | Runs on | Time |
|---|---|---|---|
| 1. Tests | `smolvlm_flops_tests.SBATCH` | CPU | minutes |
| 2. The plan | `smolvlm_flops.SBATCH calibrate` | B200 | about 1 hour |
| 3. All 350 videos | `smolvlm_flops.SBATCH full` | B200 | about 3 to 5 hours |
| 4. The table | `smolvlm_flops_summary.SBATCH` | CPU | under an hour |

Set `SBATCH_ACCOUNT` as in section 5. Every command assumes `/scratch/$USER/GazeVLM`.

**Step 1: pull only if it's safe, then run the tests.**

```bash
cd /scratch/$USER/GazeVLM/repo
if [ -f accuracy_tables/flops_run.py ]; then
    echo "section 14 is already here, from section 13's pull: don't pull again"
    git log --oneline | head -1
    ls ../data/runs/eyehead/fold1.pt $HF_HOME/hub/models--HuggingFaceTB--SmolVLM2-2.2B-Instruct/snapshots/
    sbatch accuracy_tables/hpc/smolvlm_flops_tests.SBATCH
elif squeue -h -u $USER -o "%j" | grep -Eq "^(qwen3350|qwen25350)"; then
    echo "STOP: a Qwen3 or Qwen2.5 job is still queued or running. Don't pull."
elif [ ! -s ../results/qwen3_350/qwen3/merge_info.json ] || ! tail -n 1 "$(ls -t ../logs/qwen3350-merge_*.log | head -1)" | grep -q "EVERY n IS COMPLETE"; then
    echo "STOP: Qwen3's full merge hasn't finished. Don't pull."
elif [ -d ../results/qwen25_350 ] && [ ! -s ../results/qwen25_350/qwen2.5/merge_info.json ]; then
    echo "STOP: section 13 has started and isn't merged. Don't pull."
else
    git status --short
    git pull --ff-only origin eyehead-gate
    git log --oneline | head -1
    ls ../data/runs/eyehead/fold1.pt $HF_HOME/hub/models--HuggingFaceTB--SmolVLM2-2.2B-Instruct/snapshots/
    sbatch accuracy_tables/hpc/smolvlm_flops_tests.SBATCH
fi
```

- **`STOP`:** do nothing else, and try again later.
- **Expect** `git status` to show only your 2 local `--comment` edits, the gate checkpoint, and one snapshot folder. **No `fold1.pt`: stop and tell the user,** because the plan counts one gate decision on it.
- **The test log** is `logs/smolvlm-flops-tests_<job>.log`. Its last line should say how many passed, with no `failed` or `error`. **Stop on any failure.** Fix it only if it's a small problem in the new code, and never loosen a test to make it pass.

**Step 2: the plan.** Ask the user first.

```bash
cd /scratch/$USER/GazeVLM/repo
sbatch --job-name=smolvlm-flops-calibrate accuracy_tables/hpc/smolvlm_flops.SBATCH calibrate
```

The log, `logs/smolvlm-flops-calibrate_<job>.log`, shows in order:

1. **The GPU's full name, the data** (5,985 questions on 350 videos) **and `parity OK`.**
2. **`--- step 1`:** the gate's FLOPs per decision, the LLM's quadratic, and one stock frame's and one foveated frame's GFLOPs.
3. **`--- step 2`:** a video prompt as its blind prompt, plus an intro and a number of tokens per frame, with the largest miss in tokens.
4. **`--- step 3`:**
    - ours' FLOPs share at each budget, and the no-gate control's;
    - `counting the baselines' tokens on Uniform K'=<n> frames` once per frame count;
    - `the token arithmetic matches the methods' own plans`, twice for each frame count.
5. **`--- step 4`:** one `ok` line per condition and question, with its vision and prefill ratios.
6. **A table of every row and budget:** target, planned share, gap, `nearest_only`, `K'`, budget parameter and token share. Then `PLAN WRITTEN`.

**Stop and tell the user on:**
- `FLOPS DIFFER`, `aren't a quadratic` or `prefill is ... x the fitted one`;
- `the token arithmetic differs`;
- a job on another GPU type.

**`ATTENTION NOT COUNTED`** doesn't stop the job, but report it with the table: the torch version's attention kernel has no FLOP formula, so attention is missing from every row's count.

**If it ends in `TIMEOUT` or is preempted, submit the same line again.** It continues the token counts it has already saved. It writes `results/smolvlm_flops/smolvlm/flops_plan.json`, `flops_targets.csv` and `flops_counts_K<n>.jsonl`.

**Show the user the step 3 lines and the table, and wait.** The targets and every `K'` are fixed here, from FLOPs and token counts only, before any answer is read.

**Step 3: all 350 videos.** Ask the user first.

```bash
cd /scratch/$USER/GazeVLM/repo
sbatch --job-name=smolvlm-flops-full accuracy_tables/hpc/smolvlm_flops.SBATCH full
```

- **The log starts with** the GPU's full name and `parity OK`, then `<n> jobs, <m> conditions`. It has one block per condition, and ends with `=== 0 failed of <n>`.
- **The job stops before scoring** if the config changed since the plan, if a condition's name differs from the plan's, or if the GPU isn't the one the plan ran on. **Stop and tell the user.**
- **`TIMEOUT`, or a preemption that didn't requeue: submit the same line again.** It repairs a half-written last line and continues. Never run two at once: `singleton` prevents it if you keep the job name.
- **Progress,** any time, from a login node (an answer is one line):

```bash
wc -l /scratch/$USER/GazeVLM/results/smolvlm_flops/smolvlm/*.jsonl
python3 -m accuracy_tables.hpc.all350.gpu_summary /scratch/$USER/GazeVLM/logs/gpu_smolvlm-flops-full_*.csv
```

**Step 4: the table,** when step 3's log ends with `=== 0 failed`.

```bash
cd /scratch/$USER/GazeVLM/repo
sbatch accuracy_tables/hpc/smolvlm_flops_summary.SBATCH
```

The log gives each job's GPU mean and lowest 30-minute mean, the units, then a table per group (all, AEA, Nymeria), and ends with `EVERY n IS COMPLETE`. It writes `results/smolvlm_flops/summary_flops_by_group.csv`. **`INCOMPLETE`: rerun step 3, or tell the user.**

### Expected, not bugs

- **`nogate` above ours' FLOPs:** it always sends the limit, while ours sends fewer frames on a quiet video. It has no target.
- **`budget` with its planned share equal to its target:** the target is its own share.
- **`flops_pct` differing slightly from `planned_pct`:** the plan used fitted prompt lengths, and the report uses each answer's real one.
- **A baseline's vision ratio of 1.000 to about 1.002 in step 4:** that's its own work, counted as zero. StreamingTOM's saliency makes it the largest.
- **Two budgets sharing one condition** in a row, if they get the same `K'`. The row still has a line per budget.
- **`frames_per_q` under `K'`** on `fastvid`, `prunevid` and `dtd`: frames left with no tokens leave the prompt, but `frames_encoded_per_q` still counts them.
- **About 10 minutes at 0% GPU at the start of each job:** loading the data. Step 2 is mostly CPU work throughout. **If a job is cancelled for low GPU use, stop and tell the user.** Don't move it to another GPU type.

### Rules for this task

- **Don't pull while a Qwen3 or Qwen2.5 job is queued or running,** or before Qwen3's full merge has finished.
- **Don't change anything after seeing accuracy:** not the targets, any `K'`, the tokens per frame, the budgets, the limit, the preference order, the slots, the fovea or any baseline parameter.
- **Don't rerun step 2 once step 3 has started.** It rewrites the plan, and step 3 would then refuse its own results.
- **Plan on FLOPs and token counts only.** Never pick a `K'` or a parameter by reading answers.
- **Don't force a target.** If a row can't reach one, report the nearest.
- **Don't add rows, controls, models or videos without asking.**
- **Don't claim a winner.**
- **One GPU type, B200.** Don't override `--gres`. If B200s stay busy, ask the user.
- **Never score with `--no-parity`.**
- **Write only to `results/smolvlm_flops/`.** Never write to `data/folds/` or to any results folder of sections 8 to 13.
- **Commit only when the user asks.** This repo is public: never commit results, NetIDs, account names, tokens or `*CONTEXT*.md`.
- **Nothing runs on the user's Mac.**

### What to report back

1. Any code you changed, and each step's last log line.
2. Step 1: the test result, the snapshot folder and `fold1.pt`.
3. Step 2:
    - the 4 units (gate, LLM quadratic, stock frame, foveated frame);
    - the prompt fit and its largest miss;
    - ours' and the no-gate control's FLOPs shares;
    - the whole table: every row's target, planned share, `K'`, budget parameter and token share, and which rows are `nearest_only`;
    - every step 4 line that isn't exactly 1.00000 on vision or prefill.
4. For each GPU job: the GPU type and node, its length, its mean and lowest 30-minute GPU use, and any `oom` or `nan` records.
5. For all 350 videos, AEA and Nymeria, per row and budget:
    - `n`, accuracy and its interval;
    - target and measured FLOPs share, and the token share;
    - GFLOPs, frames, frames encoded and prompt tokens per question.

    **Don't claim a winner.**

## 15. Next task: Qwen2.5-VL-3B and Qwen3-VL-4B on all 350 videos, every row matched on total FLOPs, 3 A100s per model

**Don't pull until section 14's summary exists, and never while a SmolVLM FLOPs, Qwen3 or Qwen2.5 job is queued or running.** Step 1 checks this.
**The user submits every job, and stops for an OK after each step.** Nothing here has been run. The job files have had a bash syntax check only, and the Python and its tests have never been executed. Expect to fix small problems, and report what you change.

### Why this exists

Section 14 ran on SmolVLM and finished cleanly (`=== 0 failed of 35`, `EVERY n IS COMPLETE`). The user wants the same comparison on Qwen2.5 and Qwen3: **every row given the same total FLOPs per question as ours,** and then its accuracy.

### Scope, set by the user

- **Models:** `qwen2.5` (`Qwen/Qwen2.5-VL-3B-Instruct`, snapshot `66285546d2b821cf421d4f5eb2576359d3770cd3`) and `qwen3` (`Qwen/Qwen3-VL-4B-Instruct`, the one snapshot section 12 ran on). Same model ids, prompts and answer scoring as sections 12 and 13.
- **Videos:** all 350, 5,985 questions (AEA 2,312 on 25 videos, Nymeria 3,673 on 325), reported as all, AEA and Nymeria.
- **Design: section 14's, unchanged.** Ours at its token budgets of 5, 10 and 20% sets the FLOPs targets. `uniform_flops` and the 8 baselines get the frame count `K'` whose FLOPs come closest, with section 14's planning rule: `choose_k`, the same knob matching, ours' tokens per frame, `nearest_only` when more than 0.5 points off. `nogate` keeps ours' token budget and has no target.
- **35 lines per model:** `uniform` (Uniform K), `blind`, then `budget`, `nogate`, `uniform_flops`, `gazevlm`, `streamingtom`, `dycoke`, `fastvid`, `prunevid`, `dtd`, `avgpool` and `gazevlm2`, each at 5, 10 and 20%.
- **Nothing is reused.** Every condition is rerun, so a model's 35 lines come from one code version on one GPU type (A100).
- **No winner is claimed.** The gate itself is only compared against `nogate`, the same frames' budget spread evenly. These results can't show more than that about the gate.

### The settled values

None of these changes. They are sections 12 and 13's, from `common.py`'s defaults and `config.all350.yaml`.

| Setting | Value |
|---|---|
| `K` | 96 Uniform K frames per video, or all of a shorter video's frames |
| Stock tokens per frame | 256 (Qwen2.5 448 px, Qwen3 512 px) |
| Rule's tokens per frame | 51, so frame limits 24, 48 and 96 on a video with 96 or more frames |
| Fovea | `R=70 a=160`, with the same preference order and slots |
| Gate | the 5 fold models averaged (`fold1.pt` to `fold5.pt`), counted as 5 models on every video |
| Baselines | `common.py`'s defaults. Only `K'` and each row's budget parameter are planned |

### What's new on Qwen: how its FLOPs are counted

The counter is section 14's: `torch.utils.flop_counter`, run on the real model, through the code every row scores with.

- **The LLM.** Its answer forward is counted at 5 prompt lengths, and fitted as `c0 + c1 n + c2 n²`. Stop on `aren't a quadratic`.
    - **New check:** `c2` against `4 x layers x heads x head_dim`, read from the model's config. That's the counter's formula for attention over the whole prompt, and section 14's SmolVLM `c2` matched it exactly.
    - Expect about **294,912** on Qwen2.5 (36 layers, 16 heads x 128) and **589,824** on Qwen3 (36 layers, 32 heads x 128).
    - **More than 1% off stops the job with `c DIFFERS FROM CONFIG`.** `c1` is printed against 2 x the LLM's non-embedding parameters, for information only.
    - **Grouped-query attention.** Both LLMs have fewer key/value heads than query heads (16 / 2 and 32 / 8), and transformers can pass them to SDPA unrepeated. torch's own FLOP formula for SDPA expects equal head counts.
        - While counting, `flops.gqa_attention` counts those calls over the query's heads, as if the keys and values were repeated, which is what eager attention does.
        - Calls with equal heads keep torch's formula unchanged, so SmolVLM's count is as in section 14.
        - If a count still fails with an `AssertionError` from `torch/utils/flop_counter.py`, stop and send the user the traceback.
- **The vision encoder.** It is counted as the model runs it, with no closed form assumed.
    - Every frame is its own image, its picture repeated over the 2 frames of a temporal patch, as sections 12 and 13 send it. So a frame's cost doesn't depend on how many frames there are, odd or even. **Step 4 checks this:** a pairing of frames would cost something other than frames x one frame, and stop the job with `FLOPS DIFFER`.
    - **A stock frame** costs the same every time. The plan measures it while scoring one question.
    - **A foveated frame's cost follows its layout, which follows the gaze.** Its cell count changes. On Qwen2.5, so does how its patches fall into the 112 px window-attention blocks (it also has full-attention blocks).
    - So step 1 counts the encoder on 16 layouts, nothing scored at them, and fits each frame's count to its layout's terms:
        - the layouts: the stock frame; uniform frames at 2 other sides (224 and 336 px on Qwen2.5, 256 and 384 px on Qwen3); the run's fovea `R=70 a=160` at 7 gaze points; the other 3 fovea settings of `common.py` at 2 gaze points each;
        - why 16: the run's fovea alone keeps the patch count in a narrow band (about 160 to 232 patches, following the gaze), far from the stock frame's 1,024. Over that band a cost that curves in the patch count is mostly absorbed by the fit. The extra layouts spread it over about 14 values, which makes such a cost about 6 times more visible;
        - the terms, which are in `flops.py`:
            - Qwen2.5: patches, the sum of each window's patches squared, and patches squared;
            - Qwen3, whose encoder attends over the whole frame in every block, with its DeepStack mergers: patches and patches squared.
    - **The fit must reproduce every counted layout within 0.01%,** or the job stops: `aren't linear in its patches and attention sizes`, `don't tell the ... terms apart`, or `the fit can't be checked` (too few layouts or term values for the check to mean anything).
    - Ours' and `nogate`'s vision FLOPs and image tokens are then **summed over each video's own frames.** They're stored in the plan, and the report uses them per record.
- **Attention the counter can't see.** Every count runs under a watch on the ops that execute.
    - If an attention op runs that has no FLOP formula, every count is redone with eager attention, which runs attention as matrix products the counter sees. Only counting passes use it. Knob counting and all scoring keep the model's own attention.
    - If one is still missed, `ATTENTION NOT COUNTED` is printed. It doesn't stop the job, but report it.
- **The baselines' own work** (similarities, clustering, saliency) counts as zero, as in section 14, and step 4 stops if it's more than 2% of their vision FLOPs.

### What each row is expected to reach

**Rough estimates from the models' sizes, not results.** Step 3's plan prints the real values. Show them to the user whatever they are.

| | Qwen2.5 | Qwen3 |
|---|---|---|
| One stock frame | about 1,340 GFLOPs | about 780 GFLOPs |
| One foveated frame | about 260 GFLOPs, varying with the gaze | about 140 GFLOPs, varying with the gaze |
| Uniform K's prompt, about 25,000 tokens | its `n²` attention is about half the question's FLOPs | about 60% |
| Targets at 5 / 10 / 20% tokens | about 3 / 6.5 / 13.5% of the FLOPs | about 2.5 / 5 / 10.5% |
| `K'` of `uniform_flops` | about 5 / 10 / 19 | about 5 / 10 / 19 |
| `K'` of the 7 single-view baselines | about 8 / 17 / 35 | about 12 / 24 / 50 |
| `K'` of `gazevlm2` | about 4 / 9 / 18 | about 7 / 15 / 32 |

- **The targets are likely below the token budgets,** unlike SmolVLM's. Uniform K's long prompt pays attention that grows with the square of its length, and ours' short prompts barely do.
- **`nearest_only` at 5% is likely,** mostly on `uniform_flops` and `gazevlm2`. One frame there is worth a point or more of the FLOPs share. It's a flag, not a failure.
- **`avgpool` keeps 7 x 7 = 49 tokens per frame, and `gazevlm2` an even count near 51,** as in sections 12 and 13.

### How the split works

- **The split is sections 12 and 13's, read and never recomputed.** The calibrate job reads `results/qwen25_350/manifest/` for Qwen2.5 and `results/qwen3_350/manifest/` for Qwen3. It checks that split against the questions file, copies it byte for byte into this section's folder, and prints it.
    - Expect **1,995 questions on each shard, on 114, 116 and 120 videos.**
    - **A wrong or missing split stops the job.**
- **One calibrate job per model writes the plan once.** It then seals it, recording the plan's and targets' hashes in `flops_plan.sha256`, and makes all three read-only.
    - A shard refuses to start if the plan is missing, isn't the sealed one, or was made with another config.
    - Then `shards.py start` runs, as in sections 12 and 13. It checks the commit, code, targets, settings, GPU and library versions against every earlier job of all 3 shards.
- **Each shard writes only to its own folder.** The final folder is written only by the merge:

```
results/qwen25_flops/              (results/qwen3_flops/ the same, with qwen3)
  manifest/manifest.json, shard_<n>.txt, all_videos.txt     copied from results/qwen25_350/manifest/ by calibrate
  calibrate/qwen2.5/flops_plan.json, flops_targets.csv, flops_plan.sha256, flops_counts_K<n>.jsonl, gpu_types.txt
  shards/<n>/qwen2.5/<row>.jsonl, gpu_ledger.jsonl, gpu_types.txt, shard_info.jsonl, run_info_flops.json
  qwen2.5/<row>.jsonl, gpu_types.txt, merge_info.json        written only by the merge
  summary_flops_by_group.csv                                 written only by the report
```

- **GPU type per record.** Each shard's `gpu_ledger.jsonl` records the GPU before every condition, and which line of the row's file that condition starts at. That puts every record on one GPU name.
    - The merge stops if any record isn't on the ledger, or if there's more than one name. It also stops on a name that differs from the calibrate job's, as `check_gpu --same-as` does at every job's start.
- **Threads** load frames ahead for every Qwen row, as in sections 12 and 13. GPU use is logged every minute to `logs/gpu_<job name>_<job id>.csv`.
    - **The baselines now encode far fewer frames,** so GPU use may be lower than sections 12 and 13's 67 to 73% mean (lowest 30-minute mean 39.5 to 44%).
    - **`ga*` nodes cancel jobs below 50% and warn below 70%.** If a job is cancelled or warned for low GPU use, stop and tell the user. Don't move it to another GPU type.

### Answers expected

- **Per model: 2 x 5,985 + 11 x 17,955 = 209,475 answers.** The files for `uniform` and `blind` hold 5,985 lines. Each other row's file holds 17,955, because its 3 budgets share one file, as in section 14.
- **Per shard: 2 x 1,995 + 11 x 5,985 = 69,825.**
- **Fewer if two budgets of a row get the same `K'`:** that condition is scored once, and the plan's table shows it.

### What changed in the code

| File | Change |
|---|---|
| `accuracy_tables/flops.py` | the user's `choose_k` tie fix, committed as it is on the cluster. Then measuring functions for Qwen2.5 and Qwen3, the vision fit per layout, the attention watch, the eager fallback for counting, and the `c2` check against the config |
| `accuracy_tables/flops_calibrate.py` | takes the model as an argument, all from section 14's code: per-layout vision for Qwen's foveated rows, counting under the watch, the config check. SmolVLM's plan and table are as before. Its plan file gains 2 fields, and its log 2 lines |
| `accuracy_tables/flops_run.py` | writes `gpu_ledger.jsonl` |
| `accuracy_tables/hpc/all350/flops_report.py` | new `--plan`, needed because the merge rewrites `<model>/`. Uses Qwen's per-layout vision, and flags records whose image tokens aren't the plan's. Same columns as section 14 |
| `accuracy_tables/hpc/all350/flops_shards.py` | new: the split, the seal, each shard's check, and the merge with the ledger check |
| `accuracy_tables/hpc/all350/flops_progress.py` | new: answers done per shard and row, standard library only |
| `accuracy_tables/hpc/qwen_flops.SBATCH`, `qwen_flops_merge.SBATCH`, `qwen_flops_tests.SBATCH` | new: the jobs |
| `tests/test_qwen_flops.py` | new: both models on fake encoders and LLMs, through the real calibrate, split, shard, merge and report code |
| `tests/test_smolvlm_flops.py` | its test that an unmeasured model is refused now uses `internvl`, because `qwen2.5` is measured now. It tests the same thing |

**Unchanged:** sections 11 to 13's job files, `shards.py`, `merge_shards.py`, `shard_progress.py`, `report.py`, `summarize.py`, `check_gpu.py`, `repair_results.py`, `compare_records.py` and `config.all350.yaml`. The hash tests on Qwen3's and section 11's files still hold.

### Steps

**Job count: 11 `sbatch` jobs: 8 on GPU (2 calibrate, 6 shards) and 3 on CPU,** plus resubmissions after a time limit. All times are estimates.

| Step | Job | Runs on | Time |
|---|---|---|---|
| 1. Pull and tests | `qwen_flops_tests.SBATCH` | CPU | 20 minutes to 1 hour |
| 2. The plans | `qwen_flops.SBATCH <model> calibrate`, both models at once | 2 A100s | 2 to 4 hours each |
| 3. All 350 videos | `qwen_flops.SBATCH <model> full 0, 1, 2`, both models | 6 A100s | 3.5 to 6 hours per shard, Qwen3 the longer |
| 4. Merge and table | `qwen_flops_merge.SBATCH <model>`, one model after the other | CPU | 2 to 4 hours each |

- **Time limits:** 8 hours for each GPU job and the merge, 3 hours for the tests.
- **Shard estimate:** sections 12 and 13's shards took about 5 to 6.5 hours, with every baseline on 96 frames. Here the baselines encode about 60 to 90% as many frames, but each budget separately, and their prompts are shorter.

Set `SBATCH_ACCOUNT` as in section 5. Every command assumes `/scratch/$USER/GazeVLM`.

**Step 1: pull only if it's safe, then run the tests.**

```bash
cd /scratch/$USER/GazeVLM/repo
FIX='-        if best is None or abs(s - target) < abs(best[1] - target):
+        if best is None or abs(s - target) < abs(best[1] - target) - 1e-12:'
GOT=$(git diff -U0 -- accuracy_tables/flops.py | grep -E '^[-+]' | grep -vE '^(---|\+\+\+) ')
if squeue -h -u $USER -o "%j" | grep -Eq "^(smolvlm-flops|qwen3350|qwen25350)"; then
    echo "STOP: a section 12, 13 or 14 job is still queued or running. Don't pull."
elif [ ! -s ../results/smolvlm_flops/summary_flops_by_group.csv ]; then
    echo "STOP: section 14's results/smolvlm_flops/summary_flops_by_group.csv isn't there. Don't pull."
elif [ -n "$GOT" ] && [ "$GOT" != "$FIX" ]; then
    echo "STOP: accuracy_tables/flops.py has local changes other than the choose_k fix. Don't pull. Send the user this diff:"
    git diff -- accuracy_tables/flops.py
else
    if [ -n "$GOT" ]; then
        git checkout -- accuracy_tables/flops.py
        echo "restored accuracy_tables/flops.py: the same one-line fix comes in the pull"
    fi
    git status --short
    git pull --ff-only origin eyehead-gate
    git log --oneline | head -3
    ls ../data/runs/eyehead/fold1.pt
    ls $HF_HOME/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/ $HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/
    sbatch accuracy_tables/hpc/qwen_flops_tests.SBATCH
fi
```

- **`STOP`:** do nothing else, and tell the user.
- **Expect:**
    - `git status` showing only your 2 local `--comment` edits;
    - `fold1.pt`;
    - exactly one snapshot folder per model, Qwen2.5's being `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- **Stop and tell the user on** more than one snapshot of a model, or another one.
- **The test log** is `logs/qwen-flops-tests_<job>.log`. It runs every test in `tests/`. Its last line should be `<n> passed`, with no `failed` or `error`. **Stop on any failure.** Fix it only if it's a small problem in the new code, and never loosen a test to make it pass.

**Step 2: the two plans.** Ask the user first. They can run at the same time:

```bash
cd /scratch/$USER/GazeVLM/repo
sbatch --job-name=qwen25-flops-calibrate accuracy_tables/hpc/qwen_flops.SBATCH qwen2.5 calibrate
sbatch --job-name=qwen3-flops-calibrate  accuracy_tables/hpc/qwen_flops.SBATCH qwen3 calibrate
```

Each log, `logs/qwen25-flops-calibrate_<job>.log` or `logs/qwen3-flops-calibrate_<job>.log`, shows in order:

1. **The GPU's full name, then the split:** `the split sections 12 and 13 ran on ... copied`, and `shard <n>: 1995 questions on <114|116|120> videos` three times.
2. **The data** (5,985 questions on 350 videos), then **`parity OK`.**
3. **`--- step 1`:**
    - the gate's FLOPs per decision;
    - the LLM's quadratic, then `n^2 term ... from the model's config: +0.000%` and `n term ... for information only`;
    - `one stock frame ... GFLOPs, one foveated frame ... GFLOPs`;
    - `the encoder per frame = ...`, exact on all 16 layouts, with the run's foveated range in GFLOPs and tokens (about 40 to 58 tokens at the 7 gaze points).
4. **`--- step 2`:** the prompt fit and its largest miss.
5. **`--- step 3`:**
    - per budget, ours' and the no-gate control's image tokens and GFLOPs per foveated frame;
    - ours' FLOPs share at each budget, and the no-gate control's;
    - `counting the baselines' tokens on Uniform K'=<n> frames`, once per frame count;
    - `the token arithmetic matches the methods' own plans`.
6. **`--- step 4`:** one `ok` line per condition and question, with its vision and prefill ratios.
7. **The table of every row and budget,** `PLAN WRITTEN`, `sealed: ...`, then `CALIBRATE DONE: <model>`.

**Stop and tell the user on any of these:**
- `FLOPS DIFFER`, `aren't a quadratic`, `prefill is ... x the fitted one`, `c DIFFERS FROM CONFIG`;
- `aren't linear in its patches and attention sizes`, `don't tell the ... terms apart`, `the fit can't be checked`;
- `the token arithmetic differs`;
- the split's `is wrong`, `holds a different assignment` or `no .../manifest.json`;
- `parity failed`;
- a GPU other than A100, or another A100 name on a rerun.

**`ATTENTION NOT COUNTED`** doesn't stop the job. Report it with the table, and say whether `redone with eager attention` appeared before it.

- **`TIMEOUT` or a preemption: submit the same line again.** It continues the token counts it has saved.
- **If it then refuses because `flops_plan.json` exists but there's no `flops_plan.sha256`, stop and tell the user.**

**Show the user both plans' step 1 and step 3 lines and both tables, and wait.** The targets and every `K'` are fixed here, from FLOPs and token counts only, before any answer is read.

**Step 3: all 350 videos, 6 shard jobs from one commit.** Ask the user first. **Don't pull between them, or between a shard and its resubmission.**

```bash
cd /scratch/$USER/GazeVLM/repo
for N in 0 1 2; do sbatch --job-name=qwen25-flops-full-$N accuracy_tables/hpc/qwen_flops.SBATCH qwen2.5 full $N; done
for N in 0 1 2; do sbatch --job-name=qwen3-flops-full-$N  accuracy_tables/hpc/qwen_flops.SBATCH qwen3 full $N; done
```

- **6 A100s at once may wait on `QOSGrpGRES` or `QOSMaxGRESPerUser`,** so they may not all start together. That's fine. If they stay pending for a long time, tell the user. Don't override `--gres`.
- **Each log starts with** the GPU's full name, `shard <n>: commit ...` and `parity OK`, then `<n> jobs, <m> conditions`. It has one block per condition, then `=== 0 failed of <n>` and **`SHARD DONE: full <n>`.**
- **The job stops before scoring** if the plan isn't the sealed one, the config changed, a condition's name differs from the plan's, the GPU isn't the calibrate job's, or an earlier job of any shard ran at another commit or code. **Stop and tell the user.**
- **`TIMEOUT`, or a preemption that didn't requeue: resubmit that shard's same line.** It cuts a half-written last line and skips every record already scored. For example:

```bash
sbatch --job-name=qwen25-flops-full-1 accuracy_tables/hpc/qwen_flops.SBATCH qwen2.5 full 1
```

- **Progress,** any time, from a login node (an answer is one line):

```bash
cd /scratch/$USER/GazeVLM/repo
python3 -m accuracy_tables.hpc.all350.flops_progress --root /scratch/$USER/GazeVLM/results/qwen25_flops --model qwen2.5
python3 -m accuracy_tables.hpc.all350.flops_progress --root /scratch/$USER/GazeVLM/results/qwen3_flops --model qwen3
wc -l /scratch/$USER/GazeVLM/results/qwen25_flops/shards/*/qwen2.5/*.jsonl
python3 -m accuracy_tables.hpc.all350.gpu_summary /scratch/$USER/GazeVLM/logs/gpu_qwen25-flops-full-*_*.csv /scratch/$USER/GazeVLM/logs/gpu_qwen3-flops-full-*_*.csv
```

**Step 4: merge each model, by hand, once its 3 logs show `SHARD DONE`.**

```bash
cd /scratch/$USER/GazeVLM/repo
sbatch accuracy_tables/hpc/qwen_flops_merge.SBATCH qwen2.5
sbatch accuracy_tables/hpc/qwen_flops_merge.SBATCH qwen3
```

They share a job name, so the second waits for the first. Each log, `logs/qwen-flops-merge_<job>.log`, shows in order:

1. **`oom / nan records: 0`,** or one line per shard and row that has any.
2. **The merge:**
    - per shard, its questions, videos, nodes and jobs;
    - the GPU name, the commit, `GPU per record: {'NVIDIA A100-SXM4-80GB': 209475}`;
    - `MERGED: 209475 records of 13 rows`.
3. **Each shard job's GPU mean and lowest 30-minute mean.**
4. **The units, then the FLOPs table per group** (all, AEA, Nymeria), `GPU types recorded: {'<model>': ['NVIDIA A100-SXM4-80GB']}` and **`EVERY n IS COMPLETE`.** It writes `results/<folder>/summary_flops_by_group.csv`, with section 14's columns.
5. **The comparison with section 13's (Qwen2.5) or section 12's (Qwen3) records** for `uniform`, `blind` and `budget`, which also ran on A100s and are only read:
    - per row, `records`, `missing`, `ref_only`, `pred_differs`, `tokens_differ` and `max_prob_diff`;
    - then `TOKENS MATCH: answers differ on <x> of <y> records compared`.
    - These are counts to report, not a pass or fail. A difference here never fails the job.
6. **`MERGE JOB DONE: <model>`.**

- **`MERGE STOPPED, INCOMPLETE`:** nothing was written. Rerun the shard it names, or tell the user if it names an `error record`, the ledger or another GPU.
- **`INCOMPLETE` from the report:** tell the user.
- **The merge job is safe to rerun.**

### Expected, not bugs

- **`nogate` above ours' FLOPs:** it always sends the limit, while ours sends fewer frames on a quiet video. It has no target.
- **`budget` with its planned share equal to its target:** the target is its own share.
- **`flops_pct` differing slightly from `planned_pct`:** the plan used fitted prompt lengths, and the report uses each answer's real one.
- **A baseline's vision ratio of 1.000 to about 1.02 in step 4:** that's its own work, counted as zero. StreamingTOM's saliency makes it the largest.
- **Foveated frames averaging other than 51 tokens:** 51 is the approved budget-rule value. A frame's real count follows its gaze, and the plan and the report use the real count.
- **Two budgets sharing one condition** in a row, if they get the same `K'`.
- **`frames_per_q` under `K'`** on `fastvid`, `prunevid` and `dtd`: frames left with no tokens leave the prompt, but `frames_encoded_per_q` still counts them.
- **About 10 minutes at low GPU use at the start of each job,** loading the data. Step 2 is mostly counting.

### Rules for this task

- **Don't pull** while a `smolvlm-flops*`, `qwen3350*` or `qwen25350*` job is queued or running, or before section 14's summary exists. Never pull between the shard jobs of step 3.
- **Don't change anything after seeing accuracy:** not `K`, the tokens per frame, the limits, the budgets, the fovea, the preference order, the slots, the gate, the planning rule, any `K'`, or any baseline parameter.
- **Don't rerun calibrate once a shard has started.** The job refuses to while the plan exists.
- **Plan on FLOPs and token counts only.** Don't force a target. If a row can't reach one, report the nearest.
- **Don't add rows, controls, models or videos without asking.** **Don't claim a winner.**
- **One GPU type, A100.** Don't override `--gres`. If A100s stay busy, ask the user. **Never score with `--no-parity`.**
- **Write only to `results/qwen25_flops/` and `results/qwen3_flops/`.**
    - Never write to `data/folds/` or to any results folder of sections 8 to 14: `results/smolvlm_flops/`, `results/qwen3_350/` and `results/qwen25_350/` are only read.
    - So are `results/hpc_all/`, `hpc_budget/`, `nymeria_smolvlm/`, `hpc350_check/`, `hpc350_pilot/`, `smolvlm_350/`, `smolvlm_350_pilot/` and `smolvlm_350_check/`.
- **This repo is public:** never commit results, logs, NetIDs, account names, tokens or `*CONTEXT*.md`. Commit only when the user asks.
- **Nothing runs on the user's Mac.**

### What to report back

1. Any code you changed, and each job's last log line.
2. **Step 1:**
    - the test result;
    - whether `flops.py` was restored;
    - `git status`;
    - `fold1.pt` and both snapshot folders.
3. **Step 2, per model:**
    - the split;
    - the 4 units, the `n^2` term against the config, and the `n` term against the parameters;
    - the encoder fit and its foveated range;
    - the prompt fit and its largest miss;
    - ours' and the no-gate control's tokens and FLOPs per foveated frame and FLOPs shares;
    - the whole table, with every row's target, planned share, `K'`, budget parameter and token share, and which rows are `nearest_only`;
    - every step 4 line that isn't exactly 1.00000, and any `ATTENTION NOT COUNTED`.
4. **For each GPU job:**
    - the GPU type and node, and its length;
    - its mean and lowest 30-minute GPU use, and any low-use warning;
    - any `oom` or `nan` records.
5. **Step 4, per model:**
    - the `MERGED` line and GPU per record;
    - the comparison counts and `max_prob_diff` per row.
6. **For all 350 videos, AEA and Nymeria, per model, row and budget:**
    - `n`, accuracy and its interval;
    - target and measured FLOPs share, and the token share;
    - GFLOPs, frames, frames encoded and prompt tokens per question.

    **Don't claim a winner.**

## 16. Next task: SmolVLM2-256M on all 350 videos, every row matched on total FLOPs, one B200

**Don't pull while any job of sections 12 to 16 is queued or running, or before section 14's summary exists.** Step 1 checks this.
**The user submits every job, and stops for an OK after each step.** Nothing here has been run. The job files have had a bash syntax check only, and the Python and its tests have never been executed. Expect to fix small problems, and report what you change.

### Why this exists

Section 14 matched every row on FLOPs on SmolVLM2-2.2B, and section 15 on Qwen2.5 and Qwen3. The user wants the same experiment on the smallest SmolVLM2, SmolVLM2-256M, with the same 35 conditions, the same matching and the same table.

### Scope, set by the user

- **Model:** `smolvlm256`, which is `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` pinned at revision `9cda5c165b28c68a52e9bb542d2b89f0f6963709`. Results only in `results/smolvlm256_flops/`.
- **Videos:** all 350, 5,985 questions (AEA 2,312 on 25 videos, Nymeria 3,673 on 325), reported as all, AEA and Nymeria.
- **Rows:** section 14's 35 conditions:
    - `uniform` and `blind`;
    - at 5, 10 and 20%: `budget` (ours), `nogate`, `uniform_flops` and the 8 baselines.
- **Matching:** section 14's, unchanged. Ours' measured FLOPs share at each token budget is the target. `choose_k` sets each row's `K'`, and the baselines' own knob matching sets their budget parameters.
- **One GPU type, B200,** as section 14. Every row is scored fresh: nothing is reused from earlier runs.
- **Reused as they are:** the gate (5 folds averaged), its row files for all 350 videos, the 1 fps frames and the questions. Nothing is rebuilt.
- **No winner is claimed.** The gate itself is only compared against `nogate`.

### What the 256M's own config gives, and what that changed

Read from its pinned `config.json` and `preprocessor_config.json`, and checked against the loaded model at every job's start (`check_geometry`):

| | SmolVLM2-2.2B (section 14) | SmolVLM2-256M |
|---|---|---|
| Frame | 384 px, 14 px patches, 27 x 27 | **512 px, 16 px patches, 32 x 32** |
| Pixel shuffle | 3: 9 x 9 = **81 tokens** of 42 px | **4: 8 x 8 = 64 tokens** of 64 px |
| LLM | SmolLM2-1.7B | **SmolLM2-135M: 30 layers, 9 query heads against 3 key/value heads of 64**, 8,192 positions |
| Chat template, answer letters, prompt | | the same |

**Settings that couldn't be reused as they were, and what the user approved on 2026-11-13:**

| Setting | On the 2.2B | On the 256M | Why |
|---|---|---|---|
| **The fovea** (ours and `nogate`) | 3 x 3 grid of 126 px cells, the nearest kept full, the rest shrunk 3x: **17 of 81** | **2 x 2 grid of 256 px cells, the nearest kept full (16 tokens), the other 3 shrunk 4x: 19 of 64** | An 8 x 8 grid has no 3 x 3 split. The 2.2B's fovea uses its own pixel-shuffle step (3x), so this uses the 256M's (4x). It covers 1/4 of the frame (the 2.2B's covers 1/9). 19 on every frame, whatever the gaze |
| Tokens per frame of the budget rule | 17 | **19** (`config.smolvlm256.yaml`) | follows from the fovea |
| Frame limits at 5 / 10 / 20% | 15 / 30 / 60 | **10 / 21 / 43** | the same proposer, `floor(budget x 64 x 64 / 19)`. **Step 2 prints them as PROPOSED and stops for the user's OK** |
| A shrunk cell's patch positions | the centre patch of each 3 x 3 block | the patch just up and left of each 4 x 4 block's centre | the 2.2B port's rule, `(k - 1) // 2`. A 4 x 4 block has no centre patch |
| StreamingTOM's and FastVID's saliency | keys pooled over each token's 3 x 3 patches | over its 4 x 4 patches | the same rule, on this model's patches per token |
| GazeVLM's 2 x 2 thumbnail and `r_min` 0.125 | 4 of 81 tokens, `r_min` about 1.1 tokens | **kept:** 4 of 64 tokens, `r_min` exactly 1 token | the user kept both definitions |
| StreamingTOM knn 7, FastVID c 8, p 4, knn 4, PruneVid knn 7, min_tokens 14 | | **kept** | the authors' values, already used unchanged on 81- and 256-token models |
| Frame loading threads | 8 on 8 cores | **16 on 16 cores** | the B200's low-GPU-use limit. Step 2 checks they change no answer |

Nothing else changes: K = 64 Uniform K frames (64 x 64 image tokens + text, about 4,900 of the 8,192 positions), the budgets, the slots, the preference order, the gate and every baseline parameter.

**Kept apart from the earlier runs.** The 256M's settings live in their own `accuracy_tables/hpc/config.smolvlm256.yaml`, which is `config.all350.yaml` plus three entries. `config.all350.yaml`, the defaults and section 14's job files are byte for byte as before, and a test pins them.

### FLOPs on the 256M

- **The counter and the definition are section 14's:** `torch.utils.flop_counter` on the real model, gate + vision encoder + LLM prefill.
- **The LLM** is counted at 5 prompt lengths and fitted as `c0 + c1 n + c2 n²`.
    - `c2` must equal `4 x 30 layers x 9 heads x 64 = 69,120`, **within 0.01%, not section 15's 1%**.
    - Grouped-query attention (9 query heads against 3 key/value heads) is counted over the query's heads, as in `d0f6def`.
    - A miss prints `ATTENTION NOT COUNTED` and stops with `c DIFFERS FROM CONFIG`.
- **The vision encoder** is fitted on 16 layouts, as in `8de72e8`, to patches and patches squared (SigLIP attends over the whole frame):
    - the stock frame;
    - uniform frames of 128, 256, 384 and 448 px;
    - the run's fovea at 7 gaze points;
    - the 2- and 3-cell fovea at 2 gaze points each.
    - It must reproduce every layout within 0.01%. It stops on `aren't linear` or `the fit can't be checked`.
- **Step 4** scores every planned condition on 2 questions under the counter, as before. `FLOPS DIFFER` stops it.

### What each row is expected to reach

**Rough estimates from the model's size, not results.** Step 3 prints the real values. Show them to the user whatever they are.

| | Estimate |
|---|---|
| One stock frame | about 215 GFLOPs |
| One foveated frame (19 tokens, 304 patches) | about 56 GFLOPs, the same for every gaze |
| The LLM per prompt token | about 0.2 GFLOPs, so **the vision encoder is about 85% of a Uniform K question** |
| Targets at 5 / 10 / 20% tokens | about 4 / 8 / 17% of the FLOPs, or lower where a video sends fewer frames than its limit |
| `K'` of the 7 single-view baselines | about 3 / 6 / 12 |
| `K'` of `uniform_flops` | about 3 / 6 / 12 |
| `K'` of `gazevlm2` | about 1 or 2 / 3 / 6 |

- **Few frames for the baselines.** On this model the encoder dominates, and a baseline encodes every input frame at full size. So it gets few frames at an equal cost, more so than on the 2.2B.
- **`nearest_only` is likely at 5%,** and on `gazevlm2` at every budget: one frame is worth 1 to 3 points of the FLOPs share there. It's a flag, not a failure.
- **`avgpool` keeps 4 x 4 = 16 of 64 tokens,** the nearest square to 19, and **`gazevlm2` an even count, 18 or 20.**

### Low GPU use: the main risk

`gb*` nodes cancel jobs under 10% GPU use and warn under 50%. The 2.2B's calibrate averaged 11.8%, and the 256M asks far less of the GPU, so loading frames on the CPU sets the pace.

- **16 loading threads on 16 cores,** checked in step 2 to give the same answers as none.
- **Calibrate prints this job's GPU mean and lowest 30-minute mean, and the projected full-run time.** These come from a timing run on every 25th video.
- **Stop rule: if calibrate's GPU mean is under 15%, stop before step 4 and ask the user.**
- **Nothing that can change an answer was changed:** no batching of questions, no precision, attention or generation change.

### What changed in the code

| File | What it is |
|---|---|
| `vlmeval/smolvlm256_fovea.py` | new: the 256M's wrapper: its geometry, fovea, encoder path, parity and scoring. It imports the 2.2B module's geometry-free code (prompt, letters, merge, probabilities) rather than copying it |
| `vlmeval/gazevlm.py`, `streamingtom.py`, `dycoke.py`, `fastvid.py`, `prunevid.py`, `dtd.py`, `avgpool.py`, `gazevlm2.py` | a `SmolVLM256` backend each, on the 256M's module. StreamingTOM's saliency now reads the patches per token from the module (`SHUFFLE`), which is 3 on the 2.2B, so the 2.2B's behaviour is unchanged |
| `vlmeval/smolvlm_fovea.py` | one new constant, `SHUFFLE = 3`. Nothing else |
| `accuracy_tables/models.py` | the `SmolVLM256` adapter; `download_kwargs`, for its pinned revision |
| `accuracy_tables/flops.py`, `flops_calibrate.py` | the 256M's measuring functions; its exact `c2` check; `ATTENTION NOT COUNTED` before every `c DIFFERS FROM CONFIG` stop |
| `accuracy_tables/hpc/config.smolvlm256.yaml` | new: `config.all350.yaml` plus the 256M's frame cap, fovea list and tokens per frame |
| `accuracy_tables/hpc/all350/smolvlm256_checks.py` | new: picks the check videos, the tokens and `PROPOSED` limits, `SAME FRAMES AS 2.2B`, the strict answer comparison, `SAME AS UNIFORM K` at max_prob_diff 0.0, the projection, one job's records, the side table |
| `accuracy_tables/hpc/all350/flops_shards.py` | the ledger count per folder, shared with the check above |
| `accuracy_tables/hpc/download_models.py` | passes a model's pinned revision and file patterns, and prints the snapshot against the pin |
| `accuracy_tables/hpc/smolvlm256_download.SBATCH`, `smolvlm256_flops_tests.SBATCH`, `smolvlm256_flops_checks.SBATCH`, `smolvlm256_flops.SBATCH`, `smolvlm256_flops_summary.SBATCH` | new: the jobs |
| `tests/test_smolvlm256_flops.py` | new: 63 CPU tests |

### Steps

**Job count: 6 `sbatch` jobs: 3 on B200 (checks, calibrate, full) and 3 on CPU (download, tests, summary),** plus resubmissions after a time limit. All times are estimates.

| Step | Job | Runs on | Time |
|---|---|---|---|
| 1. Pull and download | `smolvlm256_download.SBATCH` | CPU, with internet | a few minutes |
| 2a. Tests, tokens, frames | `smolvlm256_flops_tests.SBATCH` | CPU | 1 to 2 hours |
| 2b. Regression, parity, identity, threads | `smolvlm256_flops_checks.SBATCH` | B200 | 20 to 40 minutes |
| 3. The plan | `smolvlm256_flops.SBATCH calibrate` | B200 | 1 to 1.5 hours |
| 4. All 350 videos | `smolvlm256_flops.SBATCH full` | B200 | 45 minutes to 1.5 hours (the 2.2B took 1:05) |
| 5. The table | `smolvlm256_flops_summary.SBATCH` | CPU | 1.5 to 2 hours (section 14's took about 1.5) |

Set `SBATCH_ACCOUNT` as in section 5. Every command assumes `/scratch/$USER/GazeVLM`. Keep your own `HF_HOME`: no job here sets it.

**Step 1: pull only if it's safe, then download the model.**

```bash
cd /scratch/$USER/GazeVLM/repo
if squeue -h -u $USER -o "%j" | grep -Eq "^(smolvlm-flops|smolvlm256|qwen3350|qwen25350|qwen25-flops|qwen3-flops|qwen-flops)"; then
    echo "STOP: a job of sections 12 to 16 is queued or running. Don't pull."
elif [ ! -s ../results/smolvlm_flops/summary_flops_by_group.csv ]; then
    echo "STOP: section 14's results/smolvlm_flops/summary_flops_by_group.csv isn't there, and step 2 compares with section 14. Don't pull."
else
    git status --short
    git pull --ff-only origin eyehead-gate
    git log --oneline | head -3
    sbatch accuracy_tables/hpc/smolvlm256_download.SBATCH
fi
```

- **`STOP`:** do nothing else, and tell the user.
- **Expect** `git status` to show only your 2 local `--comment` edits.
- **The download log,** `logs/smolvlm256-download_<job>.log`, ends with:
    - `smolvlm256: HuggingFaceTB/SmolVLM2-256M-Video-Instruct -> .../snapshots/9cda5c165b28c68a52e9bb542d2b89f0f6963709`;
    - `smolvlm256: snapshot 9cda5c165b28c68a52e9bb542d2b89f0f6963709, pinned 9cda5c165b28c68a52e9bb542d2b89f0f6963709: the same`.
- **`DIFFERENT`, or a failed download: stop and tell the user.** Then list it:

```bash
ls $HF_HOME/hub/models--HuggingFaceTB--SmolVLM2-256M-Video-Instruct/snapshots/
```

**Step 2: the tests and the checks.** Ask the user first. The two jobs can run at the same time:

```bash
cd /scratch/$USER/GazeVLM/repo
sbatch accuracy_tables/hpc/smolvlm256_flops_tests.SBATCH
sbatch accuracy_tables/hpc/smolvlm256_flops_checks.SBATCH
```

**2a, `logs/smolvlm256-flops-tests_<job>.log` (CPU), shows in order:**

1. **`853 passed, 3 skipped`:** the 790 tests that passed at `d0f6def` and 63 new ones. A different total is fine if nothing fails. **Any `failed` or `error`: stop.**
2. **The geometry:** `SmolVLM2-256M at 9cda5c...: 512 px, 16 px patches, pixel shuffle 4: 64 stock tokens per frame (SmolVLM2-2.2B: 81); text context 8192 positions`.
3. **`foveated tokens per frame at k = 1: 19 on every one of <n> frames`**.
4. **The limits table,** for both models.
5. **`the 2.2B's budget frames recomputed here give the frame count of all 17955 budget records`**, then **`SAME FRAMES AS 2.2B: ...`**.
    - The check applies only where a video's limit is the same on both models. Those are short videos, where both clamp to the same count.
    - Everywhere else it says the check doesn't apply, because the limits differ.
6. **`PROPOSED, needs the user's OK: smolvlm256 tokens per frame: stock 64, foveated 19; frame limits 10/21/43 at 5/10/20% ...`**, ending in **`STOP`**.

**2b, `logs/smolvlm256-flops-checks_<job>.log` (B200), shows in order:**

1. The GPU's full name.
2. **The 2.2B's regression:**
    - `parity OK`;
    - the uniform, blind and budget rows (all 3 budgets, which includes 10%) on 20 questions;
    - `SAME ANSWERS: 0 of <n> records differ, max_prob_diff 0.0` against section 14's records.
3. **The 256M's `parity OK`,** with `geometry: 512 px, 16 px patches, pixel shuffle 4: 64 tokens per frame` and `tokens_per_image_in_stock 64`.
    - It compares our path with Hugging Face's own processor and forward, and their logits, on the stock input.
    - It doesn't compare with `generate`, because no row decodes: every row reads the first answer token's logits, as on the 2.2B.
4. **`SAME AS UNIFORM K at max_prob_diff 0.0`:** avgpool, dtd, dycoke and streamingtom at full budget on 30 questions.
5. **The threads check:** all 12 rows on 50 questions, run once with no loading threads and once with 16, then `SAME ANSWERS: 0 of <n> records differ, max_prob_diff 0.0`.
6. This job's GPU mean and lowest 30-minute mean, then `CHECKS DONE: smolvlm256`.

**Stop and tell the user on:**
- `ANSWERS DIFFER`, `NOT THE SAME AS UNIFORM K`, `FRAMES DIFFER FROM 2.2B`, `FOVEATED TOKENS DIFFER`;
- `parity failed`, `geometry isn't`, `the 2.2B's budget frames recomputed here don't`;
- a GPU other than B200.

**Send the user both logs' lines above, and wait for the OK on the PROPOSED limits.** Nothing is calibrated before it.

**Step 3: the plan.** Only after the user's OK:

```bash
cd /scratch/$USER/GazeVLM/repo
sbatch --job-name=smolvlm256-flops-calibrate accuracy_tables/hpc/smolvlm256_flops.SBATCH calibrate
```

The log, `logs/smolvlm256-flops-calibrate_<job>.log`, shows in order:

1. **The GPU's full name, the data** (5,985 questions on 350 videos) **and `parity OK`.**
2. **`--- step 1`:**
    - the gate's FLOPs;
    - the LLM's quadratic;
    - `n^2 term 69,120 ... from the model's config: +0.000%`;
    - `one stock frame ... GFLOPs, one foveated frame ... GFLOPs`;
    - `the encoder per frame = ... exact on all 16 layouts`.
3. **`--- step 2`:** the prompt fit.
4. **`--- step 3`:**
    - `budget 5%: 19.00 image tokens ... per foveated frame`, and the same for the other budgets and `nogate`;
    - ours' and the no-gate control's FLOPs shares;
    - the `K'` counts.
5. **`--- step 4`:** one `ok` line per condition and question.
6. **The table, `<n> conditions for 35 lines`, and `PLAN WRITTEN`.**
7. **The timing run** on every 25th video, then **`PROJECTED full run: about <a> to <b> hours ...`**.
8. **This job's GPU mean and lowest 30-minute mean, then `CALIBRATE DONE: smolvlm256`.**

**Stop and tell the user on any of these:**
- `FLOPS DIFFER`, `aren't a quadratic`, `prefill is ... x the fitted one`;
- `ATTENTION NOT COUNTED` or `c DIFFERS FROM CONFIG`;
- `aren't linear`, `the fit can't be checked`, `the token arithmetic differs`;
- a non-B200 GPU.

**Stop before step 4 and ask the user if this job's GPU mean is under 15%.** If the job is cancelled for low GPU use, tell the user. Don't move it to another GPU type.

- **`TIMEOUT` or a preemption: submit the same line again.** It continues the token counts it has saved. If the plan was written before the timing run finished, it refuses to start again: tell the user.
- **Send the user** the step 1 and step 3 lines, the table, the projection and the GPU lines, **and wait.**

**Step 4: all 350 videos.** Only after the user's OK:

```bash
cd /scratch/$USER/GazeVLM/repo
sbatch --job-name=smolvlm256-flops-full accuracy_tables/hpc/smolvlm256_flops.SBATCH full
```

- **The log starts with** the GPU's full name and `parity OK`, then `<n> jobs, <m> conditions`. It has one block per condition.
- **It ends with `=== 0 failed of <n>`** (35, or fewer if two budgets share a condition), then this job's GPU mean and lowest 30-minute mean, then **`RUN DONE: full smolvlm256`**. One job, so this line stands for section 15's `SHARD DONE`.
- **`TIMEOUT`, or a preemption that didn't requeue: submit the same line again.** It cuts a half-written last line and skips every record already scored.
- **Progress,** any time, from a login node:

```bash
wc -l /scratch/$USER/GazeVLM/results/smolvlm256_flops/smolvlm256/*.jsonl
python3 -m accuracy_tables.hpc.all350.gpu_summary /scratch/$USER/GazeVLM/logs/gpu_smolvlm256-flops-full_*.csv
```

**Step 5: the table,** when step 4's log ends with `RUN DONE`.

```bash
cd /scratch/$USER/GazeVLM/repo
sbatch accuracy_tables/hpc/smolvlm256_flops_summary.SBATCH
```

The log, `logs/smolvlm256-flops-summary_<job>.log`, shows in order:

1. **`oom / nan records: 0`.**
2. **`GPU per record: {'NVIDIA B200': 209475}`.**
3. **`RECORDS: 209475 of 209475 planned answers in .../smolvlm256 (one job, so nothing to merge)`.** This stands for section 15's `MERGED`. The total is 2 x 5,985 + 11 x 17,955, or less if two budgets share a condition.
4. **The full job's GPU mean and lowest 30-minute mean.**
5. **The units, then the table per group** (all, AEA, Nymeria), and **`EVERY n IS COMPLETE`.**
6. **The side table** against SmolVLM2-2.2B: per group, row and budget, each model's accuracy, interval and FLOPs share. No paired test, no winner.
7. **`TOKENS MATCH: not applicable, different model`.** The 256M sends 64 tokens per frame, the 2.2B 81, through another LLM, so no record can match.
8. **`written .../summary_flops_by_group.csv and .../summary_vs_2p2b.csv`.**

**`INCOMPLETE`: rerun step 4, or tell the user.**

**Step 6: the report.** Per group (all, AEA, Nymeria), every row and budget:
- `n`, accuracy and its 95% interval;
- target and measured FLOPs share, and the token share;
- GFLOPs, frames, frames encoded and prompt tokens per question.

Plus Uniform minus Blind per group, the GPU job table (type, node, length, mean and lowest 30-minute mean) and every deviation from this section.

### Expected, not bugs

- **`nogate` above ours' FLOPs:** it always sends the limit. It has no target.
- **`budget` with its planned share equal to its target:** the target is its own share.
- **`flops_pct` differing slightly from `planned_pct`:** the report uses each answer's real prompt length.
- **A baseline's vision ratio from 1.000 to about 1.02 in step 4:** its own work, counted as zero.
- **`frames_per_q` under `K'`** on `fastvid`, `prunevid` and `dtd`: frames left with no tokens leave the prompt, but they were encoded.
- **SAME FRAMES AS 2.2B on few pairs:** the 256M's limits (10 / 21 / 43) differ from the 2.2B's (15 / 30 / 60) on any video with 64 frames or more, so only short videos are compared.

### Rules for this task

- **Don't pull** while a job of sections 12 to 16 is queued or running, or between step 3 and step 4.
- **Don't calibrate before the user approves the PROPOSED limits,** and don't change anything after seeing accuracy:
    - not K, the tokens per frame, the limits, the budgets or the fovea;
    - not the preference order, the slots or the gate;
    - not the planning rule, any `K'` or any baseline parameter.
- **Plan on FLOPs and token counts only.** Don't force a target.
- **Don't add rows, controls, models or videos without asking.** **Don't claim a winner.**
- **One GPU type, B200.** Don't override `--gres`. If B200s stay busy, ask the user. **Never score with `--no-parity`.**
- **Write only to `results/smolvlm256_flops/`.** Never write to `data/folds/` or to any results folder of sections 8 to 15; `results/smolvlm_flops/` is only read.
- **This repo is public:** never commit results, logs, NetIDs, account names, tokens or `*CONTEXT*.md`. Commit only when the user asks.
- **Nothing runs on the user's Mac.**

### What to report back

1. Any code you changed, and each job's last log line.
2. **Step 1:** `git status`, the snapshot line and the folder listing.
3. **Step 2:**
    - the test result;
    - the geometry line, the tokens per frame, the limits table, the frame check and the PROPOSED line;
    - the regression, parity, identity and threads lines with their counts and max_prob_diff.
4. **Step 3:**
    - the units, the `n^2` term against the config and the encoder fit;
    - the targets, the no-gate control's shares and the whole table with `K'`, knobs, token shares and `nearest_only`;
    - every step 4 line that isn't exactly 1.00000;
    - the projection and the GPU mean.
5. **Steps 4 and 5:** the GPU job table, the records lines, both CSV paths, and the report of step 6.

    **Don't claim a winner.**

## 17. Next task: SmolVLM2-2.2B's FLOPs-matched table on HD-EPIC's gaze questions

**This round is step 0 only: a read-only look at the data, then STOP for the user's decisions D1 to D5.** Steps 1 to 9 are written after the user decides, because every one of them depends on D1 to D5. **The user submits every job and approves each `sbatch`.** Nothing here has been run.

### Scope, set by the user

- **Questions: HD-EPIC's gaze group only.** That's its 2 types:
    - "where looking" (`gaze_gaze_estimation.json`);
    - "what's next" (`gaze_interaction_anticipation.json`).
    - About 2,000 questions in all. **No other group's file is ever opened,** and no other question is loaded, given frames or scored.
- **Model and method: section 14's, exactly.**
    - SmolVLM2-2.2B at snapshot `482adb537c021c86670beed01cd58990d01e72e4`.
    - Uniform K = 64; the budget rule at 17 tokens per foveated frame with limits 15 / 30 / 60.
    - The gate as the 5 fold models averaged (HD-EPIC is unseen), the same slots and preference order, every baseline setting, and `choose_k`.
    - The same 35 conditions, every row matched on total FLOPs per question.
    - Calibrate recomputes the targets on these questions. That's the method, not tuning. **Nothing changes after accuracy is seen.**
- **One GPU type, B200.** Never `--no-parity`, and never override `--gres`.
- **Folders:**
    - `/scratch/$USER/hdepic/` is read-only: nothing is ever written inside it.
    - Derived data goes to `/scratch/$USER/GazeVLM/data/hdepic/`, outside the repository, and only what the gaze questions need.
    - Results go to `results/hdepic_gaze_smolvlm_flops/` only.
    - Never write to `data/folds/` or to any results folder of sections 8 to 16.
- **HD-EPIC's licence is non-commercial.** **Never commit** HD-EPIC data, question text, answers or the logs that show them. Step 0's log prints 2 example questions per type, for the user only.
- **No winner,** anywhere.

### The leakage on "where looking"

That type's answer is where the wearer's own gaze is. Ours, `nogate`, GazeVLM and GazeVLM2 place their fovea or region from that same gaze, so they see information the other rows don't.
- **No method changes because of this.**
- Step 0 reports whether gaze is recorded at the asked time, inside the span every row's frames come from. Step 1's overlay sheets show where it lands.
- The summary reports each type alone, both together and the mean of the 2. The "where looking" table carries the note: **"gaze-driven rows see the wearer's gaze, which defines this type's answer".**

### The decisions, and what step 0 prints for each

The user makes all of them, before any accuracy exists, and each applies to every row equally.

- **D1. Short clips.**
    - **Step 0 prints, per type:**
        - the input form (one `time`, or `start_time` and `end_time`) and the span lengths;
        - the MP4 frames and 1 fps frames inside each span, and the MP4 frames within 1, 5 and 10 s of its end;
        - for each option, what Uniform K and the budget rule's limits become, and the candidate spacing, with 10 Hz gaze samples and IMU bins per gap.
    - **The options:**
        - (a) 64 candidate frames evenly over each question's span, at native fps through the time CSV;
        - (b) the 1 fps frames inside the span, so K drops to what the clip holds;
        - (c) only questions whose span is at least 64 s.
    - **What the code already says:**
        - The gate was trained on 1 s frame pairs. Each window gives it 9 gaze steps at 10 Hz and 10 IMU bins (`build_rows.pair_rows`).
        - So under (a), a span shorter than 63 s puts candidates less than 1 s apart, and the gate's windows would be shorter than any it was trained on. Step 0 prints how often.
        - Under (b), a few-second clip gives K of a few frames, and limits of 1 to 2 at every budget.
        - Under (c), most short questions are excluded.
    - **"What's next" must never see a frame after its span's end:** that would show the answer. Step 0 prints how far each span ends from the video's end.
    - The user decides once the numbers exist.
- **D2. Inputs per question.** Step 0 counts the inputs (segments) and videos per question, by type. If any question has more than one, the proposal is:
    - frames drawn in time order, split across segments in proportion to their duration;
    - the gate run per segment, never across a gap.
- **D3. `<TIME>` and `<BBOX>` tags.** Step 0 counts them, by type, in questions and in choices.
    - The proposal is to keep them verbatim, as the official benchmark does, unless its evaluation code does otherwise.
    - If any boxes exist, step 1 tests 0 / 90 / 180 / 270° rotations and original-versus-MP4 resolution on 20 of them, and shows which fits.
    - No question is dropped silently.
- **D4. Gaze rate.**
    - Step 0 prints every touched video's gaze rate, and whether a whole step reaches the gate's 10 Hz. The pilot saw 10 Hz on P01 and P03, and 60 Hz on P04.
    - `450298f` handles only 30 Hz, by taking every third sample with no averaging.
    - **The proposal:** the same rule generalised to any rate that is a whole multiple of 10 Hz (every `rate / 10`-th sample).
        - Step 1 will prove it gives the existing 30 Hz path's rows exactly, on the videos that path already handles.
        - A rate with no whole step stops the build, and is shown to the user.
- **D5. The 2 videos without gaze** (P02-20240210-113925 and P02-20240211-182630).
    - Step 0 prints how many gaze questions sit on them.
    - **The recommendation:** exclude them from every row, and report the count.

### Steps 1 to 9, planned

They're written once D1 to D5 are decided. Each will have an exact command, the lines to expect and its stop phrases, as in sections 14 to 16.

1. **The loader.**
    - HD-EPIC's gaze CSV, `imu.npz`, `calibration.json` and time CSV, fed to the existing row builder unchanged. Rebuilding the 12 AEA videos must still print `REBUILD MATCHES`.
    - The IMU axes, units and right-IMU choice checked against AEA's, with gyro columns 3 to 5.
    - The D4 resampler and its proof.
    - Gaze overlay sheets for at least 5 videos, at least 1 of them at 60 Hz, with the share of gaze inside the frame. **STOP.**
2. **The gate.**
    - Rows and scores on the touched videos only.
    - The statistics, as in section 9's step 4. A `vor_median_cos` that isn't clearly negative prints `VOR CHECK FAILED` and stops (AEA and Nymeria: about -0.88).
    - **STOP** for the user's OK.
3. **Frames.** Exactly the frames D1 and D2 call for, through the time CSV, within the quota step 0 printed. Ends with `FRAMES MATCH PLAN`.
4. **The manifest.** The questions kept under D1 to D5, with every exclusion counted and its reason, then a separate HD-EPIC `data check OK`. The EgoEverything checks don't change.
5. **The tests on the CPU, and checks on a B200, in parallel:**
    - every unit test;
    - section 14's regression, `SAME ANSWERS: 0 of 100`;
    - `parity OK` on HD-EPIC inputs;
    - `SAME AS UNIFORM K` at full budget;
    - the threads check;
    - 17 foveated tokens on every kept frame.
6. **Calibrate on a B200.**
    - Section 14's output, a timing run with `PROJECTED full run`, and the GPU mean.
    - **STOP AND ASK** if the GPU mean is under 15%.
7. **Full, on one B200.** About 2,000 questions x 35 conditions, about a third of section 14. No shards unless the projection says over 6 hours.
8. **The summary.**
    - Tables for both types together (question-weighted), each type, and the mean of the 2.
    - The chance level per type, Uniform minus Blind per type, section 14's interval method, and a side table against section 14.
    - No gap called a difference. With about 1,000 questions per type, expect intervals of roughly ±3 points.
9. **The report.**

### Step 0: pull only if it's safe, then look at the data

```bash
cd /scratch/$USER/GazeVLM/repo
if squeue -h -u $USER -o "%j" | grep -Eq "^(smolvlm-flops|smolvlm256-flops|qwen3350|qwen25350|qwen25-flops|qwen3-flops|qwen-flops|hdepic)"; then
    echo "STOP: a section 12 to 17 job is still queued or running. Don't pull."
else
    git status --short
    git pull --ff-only origin eyehead-gate
    git log --oneline | head -3
    myquota
    sbatch accuracy_tables/hpc/hdepic_explore.SBATCH
fi
```

- **`STOP`:** do nothing else, and tell the user.
- **Expect** `git status` to show only your 2 local `--comment` edits.
- **The log,** `logs/hdepic-explore_<job>.log`, shows in order:
    1. `HDEPIC_ROOT=...`, then `myquota`'s lines.
    2. `HD-EPIC: 156 video folders; gaze group only: gaze_estimation (...), interaction_anticipation (...)`.
    3. **The questions per type:** count, choices per question (expect 5), input forms, inputs per question, multi-video questions, videos, the tag counts and the spread of `correct_idx`.
    4. **`the gaze questions touch <n> of 156 videos`,** with a note if that's fewer than about 20.
    5. **The per-video table:**
        - fps, frames, resolution and duration;
        - the time CSV: rows against MP4 frames, repeated rows, backward steps and the largest offset from `i / fps`;
        - the gaze file, its rate and step, and its coverage at the start and end and its longest gap;
        - the IMU rates and magnitudes;
        - then the calibration's structure, the IMU arrays and `status.json`.
    6. **The D1 tables, then D2, D3, D4 and D5,** then the leakage line.
    7. **The disk table:** frames, files and gigabytes per D1 option, against `myquota`.
    8. **2 example questions per type,** for the user only.
    9. **`STOP: D1 to D5 are the user's. Send them this log.`**
- **It writes** `data/hdepic/explore/per_video.csv` and `per_question.csv`: ids, times and counts, never question text.
- **Stop and tell the user on:**
    - a missing video folder (`NOT ON SCRATCH`);
    - a time CSV without the 3 columns;
    - an input with neither `time` nor `start_time` / `end_time`;
    - any traceback.

**STOP. Send the user the whole log and `myquota`'s output, and wait for D1 to D5.**

### Rules for this task

- **Raw HD-EPIC is read-only.** **Never commit** HD-EPIC data, question or choice text, answers, these logs, NetIDs, account names, tokens or `*CONTEXT*.md`.
- **Don't open any HD-EPIC group's file but the gaze group's 2.**
- **Nothing runs past step 0 until the user has decided D1 to D5.**
- **Don't claim a winner.**
- **Nothing runs on the user's Mac.**

### What to report back

1. Step 0's whole log, `myquota`, and any traceback.
2. Anything unexpected in the time CSVs, gaze rates, IMU arrays or calibration fields.

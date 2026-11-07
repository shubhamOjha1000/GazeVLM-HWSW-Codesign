# Accuracy runs on NYU Torch: runbook for the HPC agent

You know the cluster and wrote `EGOEVERYTHING_CONTEXT.md` (not in git); "context section N" below refers to it. This file
covers the project's side: what to run, in what order, and how it reads your dataset. It assumes no earlier knowledge of
the project.

**Everything runs as Slurm jobs, not notebooks.** The Python is already command-line scripts. The job files are in
`accuracy_tables/hpc/`, and the method detail is in `accuracy_tables/README.md`.

**New task: section 13, Qwen2.5-VL-3B on all 350 videos, split across 3 A100s. Don't pull it until Qwen3's full merge in section 12 has finished.**
Sections 1 to 7 cover the 12-video run, section 8 the budget rule on those videos, and section 9 gate data for the other 338 videos and SmolVLM on
Nymeria. Section 10, the budget rule on 4 models, has run up to its pilot, and its step 4 waits for the user. Section 11 is SmolVLM with every row at
5, 10 and 20%, section 12 is the same on Qwen3, and section 13 is the same on Qwen2.5. Section 13 hasn't run: its job files have only had a bash syntax
check, and its Python and tests have never been executed, because Python isn't run on the machine they were written on. Expect to fix small problems,
and report what you change.

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

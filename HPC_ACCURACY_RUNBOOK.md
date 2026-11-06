# Accuracy runs on NYU Torch: runbook for the HPC agent

You know the cluster and wrote `EGOEVERYTHING_CONTEXT.md` (not in git); "context section N" below refers to it. This file
covers the project's side: what to run, in what order, and how it reads your dataset. It assumes no earlier knowledge of
the project.

**Everything runs as Slurm jobs, not notebooks.** The Python is already command-line scripts. The job files are in
`accuracy_tables/hpc/`, and the method detail is in `accuracy_tables/README.md`.

**New task: section 10, the budget rule on all 350 videos and 4 models.** Sections 1 to 7 cover the 12-video run, section 8 the
budget rule on those videos, and section 9 gate data for the other 338 videos and SmolVLM on Nymeria. All of those have run.
Section 10 hasn't: its job files have only had a bash syntax check, and its Python and tests have never been executed, because
Python isn't run on the machine they were written on. Expect to fix small problems, and report what you change.

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

# accuracy_tables — Uniform K, Blind and Ours on six VLMs

This folder produces three rows of the paper's accuracy tables (Table 1, EgoEverything) for every
VLM in the project:

| row in the paper | condition name in the result files | what the model sees |
|---|---|---|
| Uniform *K* frames (stock input), the 100% budget | `uniform K=96 (stock input)` | *K* = 96 frames per video, evenly spaced over the 1 fps stream (all frames if the video is shorter), each at the model's stock input |
| Blind (question and options only), 0% | `blind (question and options only)` | the same prompt without any frame |
| Ours (gaze/IMU gate + foveation) | `ours D+P gate + fovea R=70 a=160` (one per fovea setting) | the frames the eye + head gate sends under policy D+P, each foveated at that frame's gaze point |

Every row uses the same prompt (`vlmeval.qwen_mcq.build_messages`: each frame labelled
"Frame at m:ss", then the question, the five options and "Answer with the option's letter"),
and the same scoring: the answer is the argmax of the next-token logits over A–E, with the full
distribution kept in `p`.

The models (keys used everywhere in this folder):

| key | model | module in `vlmeval/` | stock tokens per frame |
|---|---|---|---|
| `qwen2.5` | Qwen/Qwen2.5-VL-3B-Instruct | `qwen_fovea.py` | 256 (448 px) |
| `qwen3` | Qwen/Qwen3-VL-4B-Instruct | `qwen3_fovea.py` | 256 (512 px) |
| `internvl` | OpenGVLab/InternVL3_5-4B-HF | `internvl_fovea.py` | 256 (448 px tile) |
| `gemma` | google/gemma-3-4b-it (gated) | `gemma_fovea.py` | 256 (896 px, 4 x 4 pooling) |
| `phi` | microsoft/Phi-3.5-vision-instruct | `phi_fovea.py` | 757 (2 x 2 crops of 336 px + global view) |
| `smolvlm` | HuggingFaceTB/SmolVLM2-2.2B-Instruct | `smolvlm_fovea.py` | 81 (384 px) |

## Files

| file | role |
|---|---|
| `config.example.yaml` | every data path and run option; copy it to `config.yaml` and edit |
| `common.py` | config loading (path resolution), and the data every model is scored on: questions, gate frames, Uniform K frames, gaze per frame, items |
| `models.py` | one adapter per VLM over its `vlmeval/*_fovea.py` module: load, parity check, stock / fovea condition, scoring, text-only prompt for Blind |
| `run.py` | command line: scores the chosen rows for the chosen models, resumably |
| `summarize.py` | command line: accuracy, 95% interval, frames, visual tokens and token budget per row and model → `summary.csv` |
| `colab_accuracy_tables.ipynb` | the same steps on Colab (Google Drive paths) |
| `../tests/test_accuracy_tables.py` | CPU tests of the config, frame selection and budget code |

No model code lives here: the foveation, the token assembly and each model's parity check are in
`vlmeval/` and are shared with the per-model notebooks in `notebooks/colab_vlm_egoeverything*.ipynb`.

## Data and paths

All paths are in `config.yaml` under `paths`. Rules: `${name}` is replaced by another key of
`paths` or else by an environment variable; `~` is expanded; relative paths are relative to the
folder of the config file; `null` means not available.

| key | needed | content |
|---|---|---|
| `folds_dir` | yes | the gate's fold CSVs: `train_val_fold1.csv` … `train_val_fold5.csv`, `test.csv` (labels, frame pairs, `gaze_xy_window`, used for the gate inputs and for the gaze point per frame) |
| `gate_ckpt_dir` | yes (read only if `oof_cache` is missing) | the eye + head gate checkpoints `fold1.pt` … `fold5.pt` |
| `oof_cache` | optional | CSV of the gate's held-out predictions (`eyehead.oof.oof_predictions`); read if it exists, written otherwise |
| `questions_json` | optional | EgoEverything `Egoeverything_VQA.json`; `null` downloads it from the hub (`roxqtang/EgoEverything`) |
| `frames_cache_dir` | yes | one `<video>_px1344.tar` per video (1 fps frames at 1344 px + `meta.json` with timestamps); missing videos are downloaded and written here |
| `links_json` | only for missing videos | the AEA download-links JSON (`aea_download_urls.json`, links expire after ~14 days) |
| `frames_local_dir` | yes | where the tars are unpacked (local disk) |
| `raw_download_dir` | only for missing videos | temporary mp4 files |
| `out_dir` | yes | results (below) |

On the project's Google Drive everything sits under `MyDrive/GazeVLM/`, which is the default
`data_root` in `config.example.yaml`. On another machine only `data_root` (and the two local
folders) usually change.

Hugging Face access: Gemma-3 is gated (accept its licence on the hub); set the `HF_TOKEN`
environment variable or log in with `huggingface-cli login`. SmolVLM2's processor needs the pip
package `num2words`.

## Running

From the repository root (the code imports `vlmeval/` and `eyehead/`):

```bash
pip install -r requirements.txt transformers huggingface_hub num2words scipy pytest
cp accuracy_tables/config.example.yaml accuracy_tables/config.yaml      # then edit paths
python -m pytest tests/test_accuracy_tables.py -q
python -m accuracy_tables.run --config accuracy_tables/config.yaml --model qwen2.5
python -m accuracy_tables.run --config accuracy_tables/config.yaml --model all --rows uniform,blind
python -m accuracy_tables.summarize --config accuracy_tables/config.yaml
```

Options of `run.py`: `--model` one key, a comma list or `all` (one model at a time is loaded and
freed); `--rows` any of `uniform,blind,ours` (default all three); `--videos test|all` overrides
`run.videos`; `--no-parity` skips the parity check.

What a run does, per model:

1. `common.load_data` (once): questions on the chosen videos (`run.videos`: `test` = the gate's 4
   locked test videos, 204 questions; `all` = the 12 gate videos with EgoEverything questions,
   834), the gate's frames under `run.policy` (held-out fold predictions for training videos,
   the 5-fold ensemble for test videos), the Uniform K frames, and one gaze point per frame
   (`vlmeval.spatial.gaze_fixation`: the densest fixation cluster in the preceding second).
2. Load the model, then **parity** (`run.parity`, on by default): the model module's
   `parity_report` on the first question's keyframe, compared with the model's own processor and
   forward, with the same pass criteria as the per-model notebooks. A failed parity stops that
   model before anything is scored. Parity also fixes how the prompt is tokenized for that
   transformers version (the module's `TOKENIZE` setting), so keep it on.
3. Score the rows; each question's result is appended to the row's file right away.

Every row is resumable: rerunning skips (condition, question) pairs already in the file. To redo
a row, delete its file. Rough A100 times for 204 questions: Blind a minute, Ours ~10–30 min per
fovea setting, Uniform K 30–90 min (Phi-3.5, at 757 tokens per frame, is the slowest).

## Results

`<out_dir>/<model>/`:

- `uniform.jsonl`, `blind.jsonl`, `ours.jsonl`: one JSON object per (condition, question):
  `condition, qid, video, answer, pred, correct, p` (probabilities over A–E), `n_frames`,
  `n_image_tokens` (visual tokens entering the LLM, every token counted: Phi's row breaks and
  global view, SmolVLM's per-frame tokens), `n_tokens` (whole prompt), `sec`, `error`
  (`null`, `"oom"` or `"nan"`; rows with an error are left out of the summary), plus
  model-specific fields (cell sizes, gaze-missing counts).
- `run_info.json`: model id, versions, the config used, the frame cap, and the parity report.

`<out_dir>/summary.csv` (from `summarize.py`): per model and condition `n, accuracy, ci_low,
ci_high` (95% interval from resampling whole videos), `frames_per_q, tokens_per_q, budget_pct`
(visual tokens per question / the Uniform K row's, on the same questions), `videos`.

## Protocol notes and caveats

- **The budget of "ours" is measured, not set**: it depends on how many frames the gate sends
  (content-dependent) and on the fovea setting. The paper's protocol sweeps the settings on the
  held-out training videos, picks the ≈5% and ≈10% operating points, and only then evaluates on
  the test videos. Hence the four ring settings (`ours.rings`, ~48 / 96 / 150 / 205 tokens on a
  256-token frame; SmolVLM: `ours.smolvlm_k`). Run with `run.videos: all` for the sweep and read
  `budget_pct` per setting.
- **SmolVLM2** has an 8,192-token context and was trained on at most 64 video frames, so
  `run.max_frames.smolvlm: 64` caps both Uniform K (K = 64 for it; the condition name says so)
  and the gate's frames (evenly subsampled when more were sent).
- **Phi-3.5** at K = 96 is ~73k visual tokens per question; it fits a 40 GB A100 but is slow. A
  question that runs out of memory is recorded with `error: "oom"`.
- **Missing gaze**: a frame without a fixation estimate is foveated at the image centre and counted
  (`n_gaze_missing`).
- transformers 5.20 breaks SmolVLM's own image-token merge; `vlmeval/smolvlm_fovea.py` patches it
  for the parity reference (our scoring path does not use it).
- The other rows of Table 1 (Uniform K + average pooling, GazeVLM, StreamingTOM, DyCoke, FastVID,
  PruneVid, DTD) and HD-EPIC are not implemented here yet.

## Extending

- **Another fovea setting**: add it to `ours.rings` (or `ours.smolvlm_k`); rerun `--rows ours`.
  Existing settings are skipped.
- **Another model**: write its `vlmeval/<name>_fovea.py` (a `run_questions` taking items as built by
  `common.make_items`, a `parity_report`, a prompt builder), then an `Adapter` subclass in
  `models.py` and an entry in `ADAPTERS`.
- **Another policy**: `run.policy: D` or `D+P all`; condition names include the policy, so results
  of different policies can share a file.

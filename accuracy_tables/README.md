# accuracy_tables — Uniform K, Blind, Ours and GazeVLM on six VLMs

This folder produces four rows of the paper's accuracy tables (Table 1, EgoEverything) for every
VLM in the project:

| row in the paper | condition name in the result files | what the model sees |
|---|---|---|
| Uniform *K* frames (stock input), the 100% budget | `uniform K=96 (stock input)` | *K* = 96 frames per video, evenly spaced over the 1 fps stream (all frames if the video is shorter), each at the model's stock input |
| Blind (question and options only), 0% | `blind (question and options only)` | the same prompt without any frame |
| Ours (gaze/IMU gate + foveation) | `ours D+P gate + fovea R=70 a=160` (one per fovea setting) | the frames the eye + head gate sends under policy D+P, each foveated at that frame's gaze point |
| GazeVLM (Chen & Qi) | `GazeVLM rho=0.3 (K=96, sigma=0.1, R_min=0.125, thumb 2x2, fixation gaze)` (one per rho) | the Uniform K frames, each reduced to the tokens of its gaze region plus a 2 x 2-token thumbnail (section GazeVLM below) |

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
| `models.py` | one adapter per VLM over its `vlmeval/*_fovea.py` module: load, parity check, stock / fovea condition, scoring, text-only prompt for Blind, its GazeVLM backend |
| `run.py` | command line: scores the chosen rows for the chosen models, resumably |
| `summarize.py` | command line: accuracy, 95% interval, frames, visual tokens and token budget per row and model → `summary.csv` |
| `colab_accuracy_tables.ipynb` | the same steps on Colab (Google Drive paths) |
| `../tests/test_accuracy_tables.py` | CPU tests of the config, frame selection and budget code |
| `../vlmeval/gazevlm.py` | GazeVLM: the method (heatmap, rho-mass box, token-grid mapping, thumbnail) and one backend per VLM |
| `../tests/test_gazevlm.py` | CPU tests of GazeVLM: the box, the token layouts per model family, positions, the run loop |

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
python -m accuracy_tables.run --config accuracy_tables/config.yaml --model phi --rows gazevlm
python -m accuracy_tables.summarize --config accuracy_tables/config.yaml
```

Options of `run.py`: `--model` one key, a comma list or `all` (one model at a time is loaded and
freed); `--rows` any of `uniform,blind,ours,gazevlm` (default all four); `--videos test|all` overrides
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

## GazeVLM

[Chen & Qi, "Eye Gaze Tells You Where to Compute: Gaze-Driven Efficient VLMs", arXiv 2509.16476]
(not Yoon et al.'s GazeVLM). Code: `vlmeval/gazevlm.py`; settings: `gazevlm` in the config.

**The original** (single images, Qwen2.5-VL): the gaze trace is smoothed with a Gaussian into a
heatmap over the image (sum 1); pixels are taken in descending heat until a fraction rho of the gaze
mass is covered; their bounding box, grown to at least R_min, is the region of interest (ROI). The
VLM gets two images: a 28 x 28 px global thumbnail of the whole image, then the ROI crop at high
resolution. rho sets the size of the ROI, hence the token count.

**The token-map version used here** (the paper's protocol, so that all six models get the same
method): a native-resolution crop saves tokens only on the Qwen models; on InternVL, Phi, SmolVLM
and Gemma (fixed inputs) a crop plus a thumbnail costs more than the frame. So each frame is encoded
once at the model's stock input, the ROI is mapped onto the model's token grid, the tokens it
touches are kept, and the whole grid average-pooled to 2 x 2 tokens is the thumbnail. Per frame the
LLM gets [thumbnail, 4 tokens] then [ROI tokens, raster order], the original's order.

| model | token grid the ROI is mapped on | model-specific handling |
|---|---|---|
| Qwen2.5-VL | 16 x 16 tokens of 28 px (448 px frame) | M-RoPE: ROI tokens at their own (row, column), each thumbnail token at the centre of the 8 x 8 tokens it averages |
| Qwen3-VL | 16 x 16 tokens of 32 px (512 px frame) | as Qwen2.5-VL, and each token's DeepStack features are kept / pooled the same way |
| InternVL3.5 | 16 x 16 tokens of 28 px (448 px tile) | 1D positions: tokens in the order above |
| Gemma-3 | 16 x 16 tokens of 56 px | pooling where Gemma pools: on its 64 x 64 SigLIP patch grid before its projector (ROI tokens are its own 4 x 4-patch tokens, thumbnail tokens 32 x 32 patches) |
| Phi-3.5 | 24 x 24 crop tokens of 28 px (672 px view) | Phi's layout: ROI rows each followed by a row break, the separator, the thumbnail rows (with row breaks) as the global view; Phi's own 156-token global view is dropped (21% of its 757 tokens, more than the budgets compared) |
| SmolVLM2 | 9 x 9 tokens of 42 px (384 px image) | 1D positions; on a 9-token side the 2 x 2 thumbnail bins overlap by one row / column (adaptive pooling) |

Settings (`gazevlm` in the config):

- `rho`: the budget knob, one condition per value. On a 256-token frame (estimates for one gaze
  point; GazeVLM uses the same K frames as Uniform K, so tokens per frame / 256 is its budget):
  rho 0.05–0.1 → about 8–13 tokens per frame (3–5%), 0.2–0.3 → 13–20 (5–8%), 0.4–0.5 → 20–29
  (8–11%), 0.6 → 29–40 (11–16%). `summarize.py` reports the measured budget; pick the rho that
  matches ours.
- `sigma` (0.1 of the frame side), `r_min` (0.125 of the frame side, 56 px on a 448 px frame,
  Qwen's smallest image) and the 224-cell heatmap grid are our choices: the authors' code is not
  public and the paper gives no values. With one gaze point per frame, sigma and rho together only
  set the ROI size, so sweeping rho covers sigma too.
- `gaze`: `fixation` (default, the protocol: our fixation estimate per frame, the same gaze as
  "ours") or `trace` (the raw gaze samples of the second before the frame, closer to the original's
  heatmap of a trace).
- `thumb`: thumbnail side in tokens (2 per the protocol).

GazeVLM is image-only: it does not select frames, so it runs on the Uniform K frames, and a frame
without a gaze estimate uses the image centre (counted in `n_gaze_missing`). Its floor is the
thumbnail plus the smallest ROI: about 8 tokens on a 256-token frame (≈3%) but 8 of 81 on SmolVLM2
(≈10%), so SmolVLM2 has no GazeVLM point near 5%.

## Results

`<out_dir>/<model>/`:

- `uniform.jsonl`, `blind.jsonl`, `ours.jsonl`, `gazevlm.jsonl`: one JSON object per (condition, question):
  `condition, qid, video, answer, pred, correct, p` (probabilities over A–E), `n_frames`,
  `n_image_tokens` (visual tokens entering the LLM, every token counted: Phi's row breaks and
  global view, SmolVLM's per-frame tokens), `n_tokens` (whole prompt), `sec`, `error`
  (`null`, `"oom"` or `"nan"`; rows with an error are left out of the summary), plus
  model-specific fields (cell sizes, gaze-missing counts). GazeVLM rows add `n_roi_tokens`,
  `roi_area` (mean ROI area, fraction of the frame) and `gazevlm` (its parameters).
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
- The other rows of Table 1 (Uniform K + average pooling, StreamingTOM, DyCoke, FastVID, PruneVid,
  DTD) and HD-EPIC are not implemented here yet.

## Extending

- **Another fovea setting**: add it to `ours.rings` (or `ours.smolvlm_k`); rerun `--rows ours`.
  Existing settings are skipped.
- **Another model**: write its `vlmeval/<name>_fovea.py` (a `run_questions` taking items as built by
  `common.make_items`, a `parity_report`, a prompt builder), then an `Adapter` subclass in
  `models.py` and an entry in `ADAPTERS`.
- **Another policy**: `run.policy: D` or `D+P all`; condition names include the policy, so results
  of different policies can share a file.
- **Another GazeVLM setting**: add a rho to `gazevlm.rho`, or change `sigma`, `r_min`, `thumb`,
  `gaze`; every parameter is in the condition name, so old and new settings never mix.

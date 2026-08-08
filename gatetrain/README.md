# gatetrain — the design PDF's pipeline, cross-validated

Trains and tests exactly what the slides specify, over the 5 fold CSVs.

| PDF | What |
|---|---|
| **p2** | `L_NCE` — InfoNCE (τ = 0.1) between `z_M` and `ẑ_M`, the gaze embedding predicted from the two frozen frame features |
| **p3** | `L_sim` — MSE regression of `[S_frame, S_gaze]` from `z_M` |
| **p9** | `L = L_NCE + L_similarity` |
| **p4** | `FilterFrameForVLM` on the predicted scores, at the thresholds that made the labels |

```
frame t-1 ──► [DINOv2 CLS, frozen] ──► z_I0 ─┐
                                              ├─► h ──► ẑ_M ─┐
frame t   ──► [DINOv2 CLS, frozen] ──► z_IT ─┘               │ L_NCE
                                                             │
gaze rates (3,9) ──► f_M ──► z_M ────────────────────────────┘
                              │
                              └──► head ──► [S_frame, S_gaze] ──► L_sim
```

## Run

```bash
python -m gatetrain.train --folds_dir /content/drive/MyDrive/GazeVLM/folds \
                          --out_dir runs/joint

python -m gatetrain.infer --folds_dir /content/drive/MyDrive/GazeVLM/folds \
                          --ckpt_dir runs/joint --out_csv runs/joint/test_preds.csv
```

| Flag | |
|---|---|
| `--sim_only` | Loss 2 alone — skips the `.npz` entirely; also a clean ablation of what `L_NCE` is worth |
| `--nce_weight` | p9 says plain sum, so this defaults to 1.0 |
| `--rates_col gaze_vec3d_rates_window` | the sphere-exact speed signal |
| `--shuffle_control` | pairs each row's frames with **another row's** gaze; must fail |
| `--feat_root` | re-root the `.npz` paths if the features moved |

## Why this folder exists outside `src/`

`src/loss1`, `src/loss2` and `src/inference` already implement the three losses and the
gate rule against the slides, and are unit-tested. This package **imports** them and adds
only the fold-aware experiment layer — so there is one implementation of each loss, not two.

`src/foldtrain` remains as a **baseline**: cross-entropy on hard `quad` labels, gaze-only.
It is not the PDF's method, and comparing against it shows what the joint objective buys.

## Three implementation points that matter

**The thresholds are loaded, never re-derived.** `thresholds.json` holds the τ_frame /
τ_gaze that produced the labels. Both `train.py` and `infer.py` assert that applying them
to the *true* similarities reproduces the stored `quad` column, and refuse to run
otherwise. Without that check, re-running the labelling notebook with different
percentiles would silently score every model against the wrong boundary.

**Predictions are un-standardised before thresholding.** The loss trains on standardised
targets; τ lives in raw cosine units. Getting this wrong produces a model that looks
trained and a gate that is nonsense.

**The frame features are read once.** Loss 1 needs `z_I0`/`z_IT` from one `.npz` per
frame on Drive. `FeatureCache` loads each *unique* frame into a `(n_frames, 384)` matrix —
~20 MB for 13,000 frames — which every fold then indexes. Drive is paid once per session,
optionally cached to local disk so a rerun skips even that.

## Metrics

The head emits two regressions, not a probability, so AUC needs an explicit ranking:

| Metric | Score |
|---|---|
| `AUC_frame` | `Ŝ_frame` vs `frame_similarity > τ_f` |
| `AUC_gaze` | `Ŝ_gaze` vs `gaze_patch_token_sim > τ_g` |
| **`AUC_gate`** | `min(Ŝ_frame − τ_f, Ŝ_gaze − τ_g)` vs DISCARD — the margin on the weaker axis |

Reported alongside: **R² against predict-the-mean** — the number to read first, because it
has a baseline and AUC does not; per-quadrant recall, so `REFIXATION` is visible; and the
false-skip / false-send split, which are not symmetric.

**All figures are the mean over the last 5 epochs, not the best epoch.** A maximum over a
noisy validation curve is a multiple-comparisons trap: measured on this project, a shuffle
control with no signal by construction still reached AUC 0.90 at its best epoch. The
checkpoint keeps the best epoch — legitimate model selection, and test stays independent.

## Reading the result

| | |
|---|---|
| **R² ≤ 0** | no better than predicting the training mean. AUC can still look fine when this is true — read R² first |
| **`REFIXATION` recall ≈ 0** | the quadrant that justifies two thresholds is not recovered. That *is* the finding |
| **control ≉ chance** | the evaluation leaks; nothing else is credible |
| **fold spread > 0.05** | the headline is mostly which videos landed in validation |

## What is deployed

Only `f_M` + the head. `h` exists solely to create the Loss-1 target and is discarded at
inference — `count_params` reports the deployed subset separately for that reason. No
frame is encoded at test time.

# Loss 2 — spatially-aware similarity prediction

Predicts the two DINOv2 teacher scores from the gaze-rate signal alone.

```
gaze rates (9, 3) ──► f_M ──► z_M ──► encoder ──► refinement (↻) ──► [S_frame, S_gaze]
                       ▲                                                     │
             shared with Loss 1                                          MSE │
                                                                             ▼
frame t-1, t ──► DINOv2 (frozen) ──► [frame_similarity, gaze_patch_token_sim]
```

`f_M` is the **same** `GazeRateEncoder` Loss 1 trains — that sharing is what makes
`L = L_NCE + L_similarity` a joint objective rather than two separate models.

## The targets

The Loss-2 slide shows a two-element teacher vector `[0.7, 0.6]`. The inference slide puts
exactly two heads on `z_M`, one for `S_frame` and one for `S_gaze` — so the vector is a
global frame similarity and a gaze-region similarity. Those are already columns in the
feature CSV:

| slide | CSV column |
|---|---|
| `S_frame` | `frame_similarity` |
| `S_gaze` | `gaze_patch_token_sim` |

**Loss 2 therefore needs no new data.**

## Files

| File | Contents |
|---|---|
| `models.py` | `SimilarityHead` — the slide's encoder + residual refinement |
| `losses.py` | standardised MSE, plus R², Pearson r and the predict-the-mean baseline |
| `dataset.py` | `Loss2Dataset` (subclasses `Loss1Dataset`), target statistics |
| `train.py` | training loop / CLI, with `--joint` and `--shuffle_control` |

## Run

```bash
python -m src.loss2.train --csv data/feature_dataset.csv --out_dir runs/loss2
python -m src.loss2.train --csv ... --joint                 # L = L_sim + L_NCE
python -m src.loss2.train --csv ... --shuffle_control       # negative control
python -m src.loss2.train --csv ... --init_from runs/loss1/best.pt   # warm start
```

## Why MSE and not KL

The slide offers "KL divergence or MSE". KL measures the distance between probability
**distributions**. `[S_frame, S_gaze]` is two independent cosine similarities: it does not
sum to 1 and is not a distribution. Softmaxing it to force one would discard the absolute
level and keep only the ratio — and the absolute level is exactly what the gate thresholds
on. `[0.70, 0.60]` and `[0.35, 0.30]` would become identical while meaning completely
different things to `FilterFrameForVLM`.

If calibrated uncertainty is ever wanted, bin each score into ~10 buckets and predict a
categorical distribution per target. That keeps the absolute level. Only worth the extra
machinery if MSE plateaus.

Targets are **standardised** using train-split statistics. On real data
`frame_similarity` has std ≈ 0.13 and `gaze_patch_token_sim` ≈ 0.22, so raw MSE would let
the gaze term dominate the gradient roughly 3× purely because it varies more.

## Why this is the more informative experiment

Loss 2 is a regression with a **trivial baseline**: predict the training mean for every
row. A model that cannot beat that on held-out videos has found no signal — no ambiguity
about memorisation, no best-epoch cherry-picking, no dependence on batch composition.
Loss 1's contrastive retrieval offers no such reference point.

It is also the more direct test of the premise. The gate needs `S_frame` and `S_gaze`
predicted from gaze alone, and that is exactly this task; Loss 1 only shapes the
representation feeding it.

Every metric prints alongside `base_mse` and a `[beats mean]` / `[NO BETTER THAN MEAN]`
flag, so the comparison is never left implicit.

## Reading the output

| Final val R² | Meaning |
|---|---|
| **> 0** | gaze rates carry real information about visual change — verify with `--shuffle_control` before believing it |
| **≤ 0** | no better than predicting the mean. At small scale this is expected and is **not** evidence against the premise: it says the experiment cannot answer the question yet |

The verdict deliberately uses the **final** epoch, not the best. Taking the max over 50
noisy epochs is a multiple-comparisons trap — a run with no signal will still show a
slightly positive best-epoch R² by chance. Both numbers are printed.

## Verified

The implementation was checked on synthetic data with known answers:

| Case | Final val R² | Expected |
|---|---|---|
| targets **are** a function of the rates | **+0.49** (r +0.71), beats mean on both | learns |
| targets are pure noise | **−0.26**, no better than mean | fails |
| learnable targets + `--shuffle_control` | **−0.19**, no better than mean | fails |

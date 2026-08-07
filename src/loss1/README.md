# Loss 1 — self-supervised gaze-rate feature learning

Adaptation of **EgoDistill §3.4** (*Self-supervised IMU feature learning*) with the IMU
stream replaced by the 3-channel gaze-rate signal.

```
frame t-1 ──► [DINOv2 CLS, FROZEN] ──► z_I0 (384) ─┐
                                                    ├─► h ──► ẑ_M (256) ─┐
frame t   ──► [DINOv2 CLS, FROZEN] ──► z_IT (384) ─┘                     │
                                                                    L_NCE (τ=0.1)
gaze rates (9, 3) ─────────────────► f_M ─────────► z_M  (256) ──────────┘
```

`ẑ_M` is a function of the two frame features **only**, so pulling it towards `z_M`
forces `f_M` to encode exactly the part of eye motion that explains visual change —
which is what the gate needs at inference, where no picture is available.

## Files

| File | Contents |
|---|---|
| `models.py` | `GazeRateEncoder` (f_M, 1D dilated CNN), `ChangePredictor` (h, MLP) |
| `losses.py` | `info_nce` (EgoDistill Eq. 7), `retrieval_metrics` |
| `dataset.py` | `Loss1Dataset`, sequence-level split, channel stats, video-balanced sampler |
| `train.py` | training loop / CLI |

## Run

```bash
python -m src.loss1.train --csv /content/build/feature_dataset.csv --out_dir runs/loss1
python -m src.loss1.train --csv ... --dry_run          # one batch, sanity check
```

Input is the CSV from `notebooks/colab_build_feature_csv.ipynb`. Only `sequence`,
`feat_frame_1`, `feat_frame_2` and `gaze_rates_window` are read; the two similarity
columns belong to Loss 2.

## Defaults

Straight from EgoDistill §4.1's pretraining setup: AdamW, lr 1e-4, batch 64, 50 epochs,
τ = 0.1, fusion hidden width 1024.

## Four decisions worth knowing

**The image encoder is not trained.** Features are precomputed on disk, so `f_I` is
frozen by construction. EgoDistill footnote 2 reports that finetuning it causes mode
collapse, so this is a requirement rather than a convenience.

**`L1` is dropped.** EgoDistill pretrains with `L_NCE + L1`, where `L1` matches a heavy
video model's clip feature. This project has no video model. The grounding role `L1`
played is served later by Loss 2, which regresses the DINOv2 similarities already in the
CSV — so the full objective remains `L = L_NCE + L_similarity` as the design slides say.

**Splits are by video, never by row.** Consecutive windows within one fixation are nearly
identical, so a random row split leaks the validation set and makes the retrieval metric
meaningless.

**Batches are balanced across videos.** In-batch negatives are the point of InfoNCE, but
two windows seconds apart in the same video can have near-identical gaze rates — false
negatives the loss actively pushes apart. `VideoBalancedBatchSampler` caps rows per video
per batch; set `--max_per_video >= batch_size` to disable.

## What to watch

`retrieval_metrics` reports **top-1**: given `ẑ_M`, does the correct `z_M` rank first in
the batch? Chance is `1/batch` (1.6% at batch 64). Climbing well above chance means gaze
rates really are predictable from visual change — the project's core premise, testable
long before any gate exists.

## Scale

The 5-video CSV (445 rows) is enough to debug the loop, not to draw conclusions.
EgoDistill pretrained on thousands of clips. Raise `N_VIDEOS` and `MAX_SECONDS` in the
build notebook before reading anything into the numbers.

## Known limitation

The window is **9 steps**; EgoDistill's IMU tensor was 422, and the project's own
architecture slide specifies 200×3. Dilations 1/2/4 give a receptive field of 15, so the
whole window is seen — but if 9 proves too thin, widen the clip by pairing frames *t* and
*t+K* instead of *t* and *t+1*, giving `10K-1` steps. That changes how the CSV rows are
built, not anything in this folder.

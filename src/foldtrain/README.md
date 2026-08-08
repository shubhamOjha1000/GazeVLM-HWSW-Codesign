# foldtrain — cross-validated gate classification from gaze alone

Consumes the fold CSVs from `notebooks/colab_label_quadrants.ipynb` and predicts the gate
decision from the gaze-rate signal, with **no frame features at any point** — the
constraint the deployed gate actually operates under.

```
gaze rates (B, 3, 9) ──► GazeRateEncoder (f_M) ──► z_M ──► head ──► logits
```

`f_M` is the *same* encoder Loss 1 and Loss 2 train, imported rather than reimplemented,
so checkpoints are interchangeable and `--init_from` can warm-start it.

## Files

| File | Contents |
|---|---|
| `dataset.py` | `FoldData` — parses a fold CSV once into GPU-resident tensors |
| `models.py` | `GazeClassifier` — `f_M` + encoder + residual refinement + linear head |
| `metrics.py` | AUC (binary + macro OvR), balanced accuracy, confusion, gate costs |
| `train.py` | K-fold training loop with early stopping, CV summary |
| `infer.py` | test-set inference, ensembling the K fold models |

## Run

```bash
python -m src.foldtrain.train --folds_dir /content/drive/MyDrive/GazeVLM/folds \
                              --out_dir runs/gate --target gate

python -m src.foldtrain.infer --folds_dir /content/drive/MyDrive/GazeVLM/folds \
                              --ckpt_dir runs/gate --out_csv runs/gate/test_preds.csv
```

| Flag | |
|---|---|
| `--target gate` | binary SEND / DISCARD (default) |
| `--target quad` | the 4 quadrants; AUC becomes macro one-vs-rest |
| `--rates_col gaze_vec3d_rates_window` | the sphere-exact speed signal |
| `--shuffle_control` | permutes training labels; **must** land at AUC ≈ 0.5 |
| `--init_from runs/loss2/best.pt` | warm-start `f_M` |

## Why there is no DataLoader

The gate reads only the gaze signal, and the labels are already CSV columns. So no `.npz`
file is ever opened, and the whole dataset is a `(N, 3, 9)` float array — **about 1.4 MB
for 13,000 rows.**

That makes the right design "no DataLoader at all": parse once, push everything to the
device, batch by indexing. There is no host-to-device traffic per step and no
worker/pin_memory/prefetch surface to tune.

`Loss1Dataset` is deliberately *not* reused, because it loads two `.npz` files per row —
precisely the cost this task does not need to pay.

## T4 settings, and what actually matters

| Setting | Value | Why |
|---|---|---|
| `batch_size` | 512 | the limit is kernel-launch overhead, not memory |
| `amp` | on | T4 has tensor cores; the gain is modest at this size but free |
| workers | none | there is no DataLoader to have workers |
| `patience` | 20 | early stopping is the real runtime lever |

The model is ~300 k parameters over a 9-step sequence. It is **not** matmul-bound, so a
bigger GPU, more workers or pinned memory would change nothing. If you want it faster, cut
epochs.

## Metrics

**AUC is the headline**, and it is implemented here rather than imported — a five-line
rank statistic, verified against sklearn to 2e-16 including heavy ties.

- **binary** (`--target gate`): AUC on P(SEND)
- **4-class** (`--target quad`): macro one-vs-rest. Macro, not micro, so `REFIXATION` —
  the rare class the two-threshold design exists for — counts as much as `STABLE`

Also reported: **balanced accuracy**, because with skewed quadrants plain accuracy is
dominated by `STABLE` and `TRANSITION` and can look respectable while the model never
predicts `REFIXATION` at all.

## The two gate errors are not symmetric

`infer.py` reports them separately, and this is the distinction accuracy hides:

| Error | Cost |
|---|---|
| **FALSE SKIP** — gate discards, oracle would send | content **lost**, unrecoverable |
| **FALSE SEND** — gate sends, oracle would discard | compute wasted, nothing lost |

A gate that skips nothing is useless but harmless; one that skips the *wrong* frames is
worse than no gate. Random skipping at the same send rate prints alongside as a floor.

## Reading the output

| Result | Meaning |
|---|---|
| fold AUC spread > 0.05 | most of the headline number is which videos landed in validation. Report the spread |
| mean AUC ≈ 0.5 | gaze is not separating these classes at this scale — check the control lands there too |
| control AUC ≫ 0.5 | **the evaluation is leaking.** Nothing else in the run is credible |

Run `--shuffle_control` before believing any positive result. It is the check that makes
the others mean something.

## `quad` is a target, never a feature

The labels come from the true DINOv2 similarities. The gate has to predict them from gaze
alone, so `quad`, `gate`, `frame_similarity` and `gaze_patch_token_sim` must never enter
the model's input. `FoldData` reads only the rates column, which enforces this structurally.

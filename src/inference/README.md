# Inference — the gating module

Implements the Inference slide. At runtime **no frame is encoded and DINOv2 never runs**:

```
eye tracker ──► gaze rates (L,3) ──► f_M ──► z_M ──► head ──► [S_frame, S_gaze]
                                                                     │
                                                       FilterFrameForVLM
                                                                     │
                                                          SEND ◄─────┴─────► DISCARD
```

`f_M` and the head are exactly what Loss 1 and Loss 2 train. Nothing is learned here.

## The decision rule

Transcribed from the slide, in `gate.py`:

```python
def filter_frame_for_vlm(s_frame, threshold_frame, s_gaze, threshold_gaze):
    if s_frame > threshold_frame:
        if s_gaze > threshold_gaze:
            return DISCARD, "Frame not sent (High frame AND high gaze similarity)"
        return SEND, "Frame sent to VLM (High frame similarity, but low gaze similarity)"
    return SEND, "Frame sent to VLM (Low frame similarity)"
```

Only one of the four quadrants discards. The interesting branch is **high frame + low
gaze** — the scene looks unchanged but the attended region moved, which a single global
threshold would miss. That branch is the entire justification for two scores.

## Files

| File | Contents |
|---|---|
| `gate.py` | `filter_frame_for_vlm`, `GazeGate`, `StreamingGate`, cost report |
| `calibrate.py` | threshold selection: from true labels, or to hit a compute budget |
| `evaluate.py` | CLI comparing the gate against the oracle, with baselines |

## Use

```python
from src.inference import GazeGate, StreamingGate

gate = GazeGate.from_checkpoint("runs/loss2/best.pt",
                                threshold_frame=0.80, threshold_gaze=0.55)

# one decision from one window of gaze rates
d = gate.decide(rates)          # rates: (L, 3) = omega_yaw, omega_pitch, omega_mag
d.action, d.reason, d.s_frame, d.s_gaze

# or the streaming loop from the system-architecture slide
s = StreamingGate(gate)
for t_us, yaw, pitch in eye_tracker:
    s.push_gaze(t_us, yaw, pitch)
    if frame_boundary:
        d = s.on_frame(t_us)    # SEND or DISCARD -- no pixels involved
```

```bash
python -m src.inference.evaluate --csv data/feature_dataset.csv \
                                 --ckpt runs/loss2/best.pt --val_only
```

## Thresholds are set, not learned

They encode a trade — compute saved against content missed — which is a deployment
decision no loss can settle. Two ways to choose:

| | |
|---|---|
| `from_labels(csv, frame_pct, gaze_pct)` | percentiles of the **true** DINOv2 similarities. Defines the **oracle** gate: what a perfect predictor would decide |
| `for_keep_rate(pred, target_keep)` | search predicted scores for the thresholds hitting a target send rate. Use when the compute budget is fixed |

Many threshold pairs give the same send rate, since only the upper-right quadrant is
discarded. Ties break toward a **lower gaze threshold**, the more conservative choice: it
sends more frames when the attended region changes.

## Evaluation: two errors, not one

`evaluate.py` compares the gate against the oracle. The two error types are **not
symmetric** and are reported separately:

| Error | Meaning | Cost |
|---|---|---|
| **FALSE SKIP** | gate discards, oracle would send | content **lost**, unrecoverable |
| **FALSE SEND** | gate sends, oracle would discard | compute wasted, nothing lost |

A gate that skips nothing is useless but harmless; a gate that skips the *wrong* frames is
worse than no gate at all. Accuracy alone would hide that distinction.

Two baselines print alongside: **random skipping** at the same send rate, and **always
send**. A gate that does not beat random skipping has learned nothing useful, however
respectable its raw agreement looks.

## Cost

`gate.cost_report()` returns the gate's parameters and MACs per decision. At the default
sizes that is roughly **306 k parameters, ~0.9 M MACs** per frame.

No encoder cost is quoted, deliberately: the saving is simply that a skipped frame is
never encoded, so the compute avoided equals whatever the image encoder or VLM would have
cost on that frame. Quoting a made-up ratio would only obscure that.

## Verified

- All four quadrants of `filter_frame_for_vlm` match the slide exactly.
- End-to-end on synthetic data with a known answer: checkpoint loading, scoring,
  thresholding, oracle comparison and the streaming loop all behave — the gate reached
  77% oracle agreement against 45% for random skipping at the same send rate.

Those numbers come from synthetic data and mean nothing about real performance; they
confirm the plumbing.

# eyehead — eye + head 4-way gate

Classifies each frame pair into TRANSITION / PURSUIT / REFIXATION / STABLE from eye and
head motion. No frames and no DINOv2 features are used, at training or at inference.
STABLE is skipped and everything else is sent to the VLM.

```
eye_speed  (9) ─► causal CNN ─► f_eye(k)  ┐
head_speed (9) ─► causal CNN ─► f_head(k) ├─ concat per step ─► MLP ─► masked mean+max ─► 4 classes
cos(eye,head)(9) ► causal CNN ─► f_cos(k)  ┘
```

| signal | source column | meaning |
|---|---|---|
| eye_speed | `gaze_rates_window` cols 0–1 | √(eye yaw rate² + eye pitch rate²), rad/s |
| head_speed | `imu_window` gyro, mapped to the eye's yaw/pitch axes | √(head yaw rate² + head pitch rate²), rad/s |
| cos(eye, head) | both | +1 same direction (gaze shift), −1 counter-rotation (VOR); 0 when either is still |

Loss: class-weighted cross-entropy. Folds and labels: the files written by
`notebooks/colab_label_quadrants.ipynb`.

## Run

```bash
python -m eyehead.train --folds_dir $FOLDS --out_dir runs/eyehead
python -m eyehead.infer --folds_dir $FOLDS --ckpt_dir runs/eyehead \
                        --out_csv runs/eyehead/test_preds.csv \
                        --baseline_preds $DRIVE/release/gate_v1/test_preds.csv
```

Or run `notebooks/colab_train_test_eyehead.ipynb`, which does the training, the ablations,
the control and the test.

| flag | |
|---|---|
| `--streams eye` / `eye,head` | ablations: what the IMU adds, and what `cos` adds |
| `--shuffle_control` | features paired with the wrong labels; balanced accuracy must stay ≈ 0.25 |
| `--imu_rot calib --imu_calib imu_calib.json` | exact IMU→CPF rotation from the factory calibration (`python -m eyehead.imu_calib --vrs <file.vrs> --out imu_calib.json`) |
| `--imu_rot fit` (the default without a calib file) | estimates the IMU→eye-axes map from the training rows' inputs, assuming the eyes mostly counter-rotate against the head |
| `--missing_imu zero` | keep rows without IMU, with head = 0, instead of dropping them |
| `--class_weight sqrt` / `none` | softer or no class rebalancing |

## Things worth knowing

- **Head axes.** The gyro is stored in the IMU's own frame. In CPF (+X left, +Y up,
  +Z forward), `head_yaw = (R w)_y` and `head_pitch = −(R w)_x`, which matches the eye's
  convention (+yaw = left, +pitch = up). Every run prints a **VOR sign check**: the median
  cos on steps where the head turns and the eye moves smoothly should be clearly negative.
  In `fit` mode that check is circular.
- **Alignment is approximate.** The CSV has ~10 equal-width IMU bins and 9 gaze steps per
  window, with no per-sample timestamps. The gyro is interpolated onto the step centres.
- **10 Hz gaze.** A saccade (20–80 ms) fits inside one step, so it shows up as one fast
  step rather than a velocity profile.
- **No net-displacement feature, by design.** Speed alone cannot tell a saccade that
  returns to the same spot (STABLE) from one that lands somewhere new (REFIXATION).
  Watch REFIXATION precision.
- **Statistics come from training rows only.** Per fold, the IMU map (fit mode), the
  normalisation and the class weights are estimated on that fold's training rows.

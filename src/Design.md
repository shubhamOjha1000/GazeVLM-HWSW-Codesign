
The `common/` encoders and gaze geometry are shared so the future **spatial**
axis and **hardware–software co-design** stages reuse them — matching the
project's stated scope of working along both spatial and temporal axes and
deploying to glasses/VR for egocentric settings.

---

## File-by-File Summary

### `src/common/aria_io.py`
- **`download_sequence(seq, meta, raw_dir, want_calibration=False)`** — downloads
  the RGB preview MP4 + eye-gaze zip per sequence, and (when calibration
  projection is requested) the **`main_vrs`** file for device calibration
  (Option B). Robustness bundle folded in: fails loudly if `main_vrs` is missing,
  verifies the file exists after download, and **returns the VRS path** so the
  caller uses it directly. Returns `(seq_dir, mp4, vrs_path)`.
- **`load_gaze_raw(seq_dir)`** — returns the RAW high-rate gaze `(ts_us, yaw, pitch)`;
  it is **not** downsampled, so speed can be integrated over all in-window samples.
- **`load_imu(seq_dir)`** — best-effort IMU loader for the IMU CSV column.
- **`subsample_frames(mp4, out_dir, target_fps)`** — decodes the video to ~1 FPS,
  saves frames to Drive, returns per-frame timestamps.
- **`nearest_imu(...)`** — nearest IMU reading per frame (guards `None`).

### `src/common/gaze_geometry.py`
- **`yawpitch_to_unit_vec(yaw, pitch)`** — the 3D unit-vector ("unit sphere")
  gaze representation.
- **`yawpitch_to_norm_xy(...)`** — prototype-grade pinhole projection to
  normalized `(x, y)` (kept as a fallback).
- **`measure_gaze_rate(g_ts_us)`** — measures the **actual** gaze sampling rate
  from timestamps (verified ~10 Hz vs. the 20 fps video) rather than assuming it.
- **`window_gaze_motion(...)`** — the **fine-grained** speed computation: gathers
  ALL raw gaze samples within each inter-frame window and integrates the
  per-sample angular steps on the sphere (Option 1: `int_speed_3d`), plus
  per-sample instantaneous-speed stats (Option 2: `inst_mean/max/std`). The
  denominator is the **measured gaze sampling period**, not the frame period.
- **`make_gaze_projector(projection, vrs_path, ...)`** — pluggable projector:
  `"pinhole"` (approx) or `"calibration"` (accurate, reads the RGB camera model
  from the VRS via `projectaria_tools`). The calibration branch flags out-of-FOV
  gaze via a returned boolean so it can be logged.

### `src/common/encoders.py`
- **`frame_embedding(...)`** — DINOv2 CLS embedding of the **full frame**
  → `frame_similarity`.
- **`gaze_patch_token(...)`** — the DINOv2 **patch token at the gaze location**
  (not a re-encoded crop) → `gaze_patch_token_sim`.
- **`gaze_crop_rgb(...)`, `show_pair(...)`** — visualization helper to display
  consecutive frames + gaze patches for the visual sanity check.

### `src/temporal/build_similarity_csv.py`
The main driver. Per sequence: download (calibration VRS if needed) → measure
gaze rate → build the calibration/pinhole projector → subsample frames →
compute per-frame DINOv2 frame embedding + gaze-patch token + projected gaze →
per consecutive frame-pair, compute frame similarity, gaze-patch-token
similarity, fine-grained gaze speeds, 3D vector, yaw/pitch, x/y, IMU → write CSV.
Supports `--verify_only` for the scanpath/pair visual check, and all paths are
CLI arguments so it is **Colab-agnostic**.

---

## CSV Columns (matches the hand-drawn spec)

| Column | Meaning |
|---|---|
| `idx`, `sequence` | row index, sequence name |
| `path_frame_1`, `path_frame_2` | consecutive frame image paths |
| `gaze_frame_1`, `gaze_frame_2` | projected gaze per frame |
| `n_gaze_in_gap` | # fine-grained gaze samples between the two frames |
| `vec_3d` | 3D unit gaze vector `[x, y, z]` |
| `yaw`, `pitch` | raw gaze angles |
| `x`, `y` | normalized image gaze coordinates |
| `gaze_patch_token_sim` | DINOv2 patch-token cosine similarity (at gaze) |
| `frame_similarity` | DINOv2 full-frame cosine similarity |
| `gaze_speed_int3d` | integrated 3D angular speed (Option 1) |
| `gaze_inst_max`, `gaze_inst_mean` | per-sample speed stats (Option 2) |
| `imu` | nearest IMU reading |

---

## Alignment with the Architecture & Literature

- **Gaze-only inference premise.** The pipeline isolates the two signals the
  two-threshold `FilterFrameForVLM` rule depends on — a **frame-similarity**
  signal (here DINOv2 frame + gaze-patch cosine, the teacher signal) and a
  **gaze-speed** signal — to test whether gaze motion predicts visual change,
  the prerequisite for an image-free gate.
- **Gaze-patch token = DINOv2 patch token at the gaze cell**, consistent with the
  Loss-2 schematic's "Frame & Gaze Tokens" (accepted stand-in).
- **Fine-grained gaze speed.** Integrating over all high-rate samples per
  inter-frame window (rather than a two-point chord) respects that the gaze
  stream (~10 Hz) is denser and more continuous than the ~1 FPS frames.
- **Calibration-based projection** (from the VRS) is the accurate route for the
  fisheye Aria RGB camera, aligning with how the AR-efficiency literature
  projects gaze for token/region selection [3][4]; the pinhole path is a
  prototype-grade fallback.
- **HW-SW co-design scope.** The `common/` reusables and the placeholder
  `spatial/` and `hwsw_codesign/` modules set up the broader goal of reducing
  memory/energy on glasses — the motivation shared by the efficiency literature
  [3][4], with the temporal axis being the current focus.

---

## Honest Caveats
- The frame↔gaze timing on the preview MP4 is approximate (derived from fps);
  rigorous alignment would use VRS sensor timestamps.
- The pinhole projection is prototype-grade on a fisheye camera; verify the
  scanpath visually and prefer the calibration route for trustworthy numbers.
- Log the out-of-FOV projection rate and exclude clips where gaze is often
  out-of-frame, since gaze reliability affects downstream quality [4].
- This stage measures **frame-vs-gaze similarity relationships**, not FLOPs or
  answer accuracy — those belong to later stages, and real on-device savings
  ultimately require token/KV-cache reduction inside the VLM [4].
# GazeVLM — temporal subsampling for egocentric video

Work in progress on gaze-driven temporal subsampling for egocentric video understanding.

**The idea.** On AR/VR headsets with eye tracking, the vestibulo-ocular reflex makes the
eyes counter-rotate against head motion, so eye-in-head angular velocity acts as a proxy
for head motion — and head motion is what makes consecutive frames differ. If that holds,
a gate can decide *"has the scene changed enough to be worth encoding?"* from the gaze
stream alone, without ever looking at a frame.

This repository currently contains the **data pipeline** that produces the supervision for
that idea, and the **Loss-1 training scaffold** that consumes it.

## Pipeline

Aria Everyday Activities sequence → one row per consecutive frame pair:

| Stage | What happens |
|---|---|
| 0 | download the RGB preview MP4, the MPS eye-gaze zip, and the VRS (calibration only) |
| 1 | decode to 1 FPS |
| 2 | project gaze onto each frame, encode with frozen DINOv2 (CLS + patch grid) |
| 3 | per pair: whole-frame cosine and gaze-patch cosine — the teacher labels |
| 4 | per pair: a 3-channel gaze-rate signal (ω_yaw, ω_pitch, ω_mag) — the model input |

The two cosines are labels used at training time only. The gaze rates are the only thing
the deployed gate would see.

## Layout

```
src/
├── common/          shared I/O, gaze geometry, DINOv2 encoders
├── dataprep/        build the feature CSV that training consumes
├── temporal/        the original similarity-CSV builder
├── loss1/           self-supervised gaze-rate feature learning (InfoNCE)
├── spatial/         placeholder
└── hwsw_codesign/   placeholder
configs/             pipeline configuration
scripts/             example invocations
notebooks/           Colab notebooks for running and verifying each stage
```

## Usage

Obtain `*_download_urls.json` from
[projectaria.com/datasets/aea](https://www.projectaria.com/datasets/aea/) — links expire
after ~14 days and are **not** committed here.

```bash
pip install -r requirements.txt

# raw sequences -> feature CSV + per-frame .npz features
python -m src.dataprep.build_feature_csv \
  --urls_json aea_download_urls.json \
  --out_csv data/feature_dataset.csv \
  --raw_dir data/raw --frames_dir data/frames_1fps --feat_dir data/features \
  --n_videos 5 --cleanup_raw

# Loss 1: self-supervised gaze-rate feature learning
python -m src.loss1.train --csv data/feature_dataset.csv --out_dir runs/loss1
```

Every video is used in full by default. AEA sequences average ~193 s (67–456 s), so at
1 FPS that is ~190 rows each. `--max_seconds N` trims a debug run.

Features are encoded once and cached as `.npz`, so training never touches the video, the
VRS, or DINOv2 again.

## Notebooks

Each notebook verifies one part of the pipeline in isolation; several run without the
2.5 GB VRS download.

| Notebook | Covers |
|---|---|
| `colab_test_load_gaze_raw.ipynb` | gaze loading and windowing (~60 KB download) |
| `colab_check_stage0_download.ipynb` | stages 0–4, including gaze-projection correctness |
| `colab_build_feature_csv.ipynb` | builds the feature CSV over N sampled videos |

## Notes

Loss 1 follows **EgoDistill** (Tan, Nagarajan, Grauman — arXiv:2301.02217) §3.4, with the
IMU stream replaced by gaze rates. The gaze projection follows Project Aria's own
[MPS eye-gaze documentation](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/mps_eye_gaze).

Known limitations are documented in the relevant module docstrings and in
`src/loss1/README.md` — notably the fixed 1 m gaze depth, the coarse 7×7 patch grid, and
the fact that `ω_mag` is not sphere-exact.

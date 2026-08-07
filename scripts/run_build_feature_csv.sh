#!/usr/bin/env bash
# Build the feature CSV that src/loss1 consumes.
#
# Downloads ~2.5 GB per video (the VRS dominates, and is needed for the calibration
# projection). --cleanup_raw deletes each video's raw files once its features are
# written, keeping peak disk at roughly one video instead of n_videos.

set -euo pipefail

python -m src.dataprep.build_feature_csv \
  --urls_json   "aea_download_urls.json" \
  --out_csv     "data/feature_dataset.csv" \
  --raw_dir     "data/raw" \
  --frames_dir  "data/frames_1fps" \
  --feat_dir    "data/features" \
  --n_videos    5 \
  --seed        0 \
  --max_seconds 90 \
  --cleanup_raw

# then:
#   python -m src.loss1.train --csv data/feature_dataset.csv --out_dir runs/loss1

# --- variations -------------------------------------------------------------------
# pin specific sequences instead of sampling:
#   --seqs loc1_script1_seq6_rec1 loc2_script4_seq7_rec1
#
# whole videos rather than the first 90 s (~212 rows each instead of 89):
#   --max_seconds 0

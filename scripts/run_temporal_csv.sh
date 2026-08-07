#!/usr/bin/env bash
python -m src.temporal.build_similarity_csv \
  --urls_json "/content/drive/MyDrive/aea/aea_download_urls.json" \
  --seqs loc5_script4_seq6_rec1 loc4_script1_seq1_rec1 loc3_script5_seq6_rec1 \
  --raw_dir   "/content/drive/MyDrive/aea_gaze_check/raw" \
  --frames_dir "/content/drive/MyDrive/aea_gaze_check/frames_1fps" \
  --out_csv   "/content/drive/MyDrive/aea_gaze_check/frame_vs_gaze_similarity.csv"

# # accurate
# python -m src.temporal.build_similarity_csv ... \
#     --projection calibration \
#     --calibration_path "/content/drive/MyDrive/aea_gaze_check/raw/<seq>/..._mps_slam_calibration.json"
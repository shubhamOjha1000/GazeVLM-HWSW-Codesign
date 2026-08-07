"""Loss 1 -- self-supervised gaze-rate feature learning.

Adapts EgoDistill Sec 3.4 (self-supervised IMU feature learning) with the IMU stream
replaced by the 3-channel gaze-rate signal produced by
`src.common.gaze_geometry.window_gaze_rates`.
"""

from src.loss1.dataset import (Loss1Dataset, VideoBalancedBatchSampler,
                               compute_channel_stats, parse_rates, split_by_sequence)
from src.loss1.losses import info_nce, retrieval_metrics
from src.loss1.models import ChangePredictor, GazeRateEncoder, build_models

__all__ = [
    "GazeRateEncoder", "ChangePredictor", "build_models",
    "info_nce", "retrieval_metrics",
    "Loss1Dataset", "VideoBalancedBatchSampler",
    "compute_channel_stats", "split_by_sequence", "parse_rates",
]

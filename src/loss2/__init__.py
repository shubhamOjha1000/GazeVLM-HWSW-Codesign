"""Loss 2 -- spatially-aware similarity prediction.

Regresses the two DINOv2 teacher scores, [S_frame, S_gaze], from the gaze-rate signal
alone. Those are the `frame_similarity` and `gaze_patch_token_sim` columns of the feature
CSV, so no new data is needed beyond what Loss 1 already uses.
"""

from src.loss2.dataset import Loss2Dataset, compute_target_stats
from src.loss2.losses import (TARGETS, format_metrics, regression_metrics,
                              similarity_loss)
from src.loss2.models import SimilarityHead, build_head

__all__ = [
    "SimilarityHead", "build_head",
    "similarity_loss", "regression_metrics", "format_metrics", "TARGETS",
    "Loss2Dataset", "compute_target_stats",
]

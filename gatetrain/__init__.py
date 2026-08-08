"""Training and testing for the temporal-subsampling gate, as specified in the design PDF.

    p2  Loss 1   InfoNCE between the gaze embedding z_M and the one predicted from the
                 two frozen frame features
    p3  Loss 2   MSE regression of [S_frame, S_gaze] from z_M
    p9  Joint    L = L_NCE + L_similarity
    p4  Gate     FilterFrameForVLM applied to the predicted scores at the SAME thresholds
                 that created the labels

    python -m gatetrain.train --folds_dir <dir> --out_dir runs/joint
    python -m gatetrain.infer --folds_dir <dir> --ckpt_dir runs/joint

Lives outside `src/` on purpose: this is the fold-aware experiment layer. The components
themselves stay in `src/loss1`, `src/loss2` and `src/inference` and are imported, not
copied, so there is one implementation of each loss.
"""

from gatetrain.data import (FeatureCache, FoldSet, check_thresholds, load_thresholds,
                            train_stats)
from gatetrain.losses import evaluate, gate_scores, joint_loss
from gatetrain.model import GateModel, build, count_params

__all__ = ["FoldSet", "FeatureCache", "load_thresholds", "check_thresholds",
           "GateModel", "build", "count_params", "joint_loss", "evaluate", "gate_scores"]

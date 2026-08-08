"""Cross-validated classification of the gate decision from gaze alone.

Consumes the fold CSVs written by `notebooks/colab_label_quadrants.ipynb`:

    folds/train_val_fold1..K.csv    all non-test rows, `split` marks val per fold
    folds/test.csv                  held out of every fold

and predicts either the binary gate decision (SEND / DISCARD) or the 4-way quadrant
(TRANSITION / PURSUIT / REFIXATION / STABLE) from the gaze-rate signal only. No frame
features are read at any point -- that is the constraint the deployed gate operates under.

    python -m src.foldtrain.train --folds_dir <dir> --out_dir runs/gate
    python -m src.foldtrain.infer --folds_dir <dir> --ckpt_dir runs/gate
"""

from src.foldtrain.dataset import FoldData, GATE_NAMES, QUAD_NAMES
from src.foldtrain.metrics import (accuracy, balanced_accuracy, confusion, gate_costs,
                                   macro_ovr_auc, roc_auc, summarise)
from src.foldtrain.models import GazeClassifier, build, count_params

__all__ = [
    "FoldData", "GATE_NAMES", "QUAD_NAMES",
    "GazeClassifier", "build", "count_params",
    "roc_auc", "macro_ovr_auc", "accuracy", "balanced_accuracy", "confusion",
    "gate_costs", "summarise",
]

"""Loss-2 dataset: everything Loss 1 loads, plus the two regression targets.

Subclasses `Loss1Dataset` so the windowing, feature loading, rate standardisation, the
padding mask and the shuffle control are shared rather than duplicated.
"""

import numpy as np
import pandas as pd
import torch

from src.loss1.dataset import Loss1Dataset
from src.loss2.losses import TARGETS


class Loss2Dataset(Loss1Dataset):
    """Adds `y` (standardised targets, what the model regresses) and `y_raw`
    (the original similarities, used for interpretable metrics)."""

    def __init__(self, *args, target_stats=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_stats = target_stats          # (mean(2,), std(2,)) or None
        missing = [c for c in TARGETS if c not in self.df.columns]
        if missing:
            raise KeyError(f"CSV is missing Loss-2 target columns: {missing}")

    def __getitem__(self, i):
        item = super().__getitem__(i)
        r = self.df.iloc[i]
        y_raw = np.array([float(r[c]) for c in TARGETS], dtype=np.float32)
        y = y_raw.copy()
        if self.target_stats is not None:
            mean, std = self.target_stats
            y = (y - mean) / std
        item["y"] = torch.from_numpy(y)
        item["y_raw"] = torch.from_numpy(y_raw)
        return item


def compute_target_stats(csv_path, sequences=None, eps=1e-6):
    """Per-target mean/std over the TRAIN split only.

    Needed because the two targets have very different spreads -- on real data
    frame_similarity has std ~0.13 and gaze_patch_token_sim ~0.22 -- so unstandardised
    MSE would weight them unequally for no principled reason.
    """
    df = pd.read_csv(csv_path)
    if sequences is not None:
        df = df[df["sequence"].isin(sequences)]
    y = df[list(TARGETS)].to_numpy(dtype=np.float64)
    mean = y.mean(axis=0).astype(np.float32)
    std = np.maximum(y.std(axis=0), eps).astype(np.float32)
    return mean, std

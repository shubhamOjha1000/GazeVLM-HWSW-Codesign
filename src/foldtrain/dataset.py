"""Fold CSV -> GPU-resident tensors.

The single most important performance fact about this task: **the gate never reads the
frame features**. It predicts the quadrant from the gaze signal alone, and the labels are
already columns in the CSV. So none of the .npz files are touched, and the entire dataset
is a (N, 3, 9) float array -- about 1.4 MB for 13,000 rows.

That means the right design is not "a faster DataLoader" but no DataLoader at all: parse
once, push the whole thing to the GPU, and batch by indexing. On a T4 this turns an
I/O-bound job into a compute-bound one whose epochs take milliseconds, and it removes
every worker/pin_memory/prefetch knob from the tuning surface.

`Loss1Dataset` is deliberately not reused here: it loads two .npz files per row, which is
exactly the cost this task does not need to pay.
"""

import os

import numpy as np
import pandas as pd
import torch

from src.loss1.dataset import parse_rates

# quad codes, from the labelling notebook
QUAD_NAMES = {0: "TRANSITION", 1: "PURSUIT", 2: "REFIXATION", 3: "STABLE"}
GATE_NAMES = {0: "DISCARD", 1: "SEND"}

RATE_COLS = ("gaze_rates_window", "gaze_vec3d_rates_window")


def _needed_columns(csv_path, rates_col, target):
    """Columns to read, checked against the file's actual header.

    `split` is optional on purpose: the fold files carry it, test.csv does not. Requesting
    it unconditionally made pandas raise a usecols error on the test set -- a confusing
    failure for what is simply a different kind of file.
    """
    have = set(pd.read_csv(csv_path, nrows=0).columns)
    cols = ["sequence", rates_col, "quad"]
    if target == "gate":
        cols.append("gate")
    missing = [c for c in cols if c not in have]
    if missing:
        raise KeyError(f"{os.path.basename(csv_path)} is missing {missing}. "
                       f"Was it produced by notebooks/colab_label_quadrants.ipynb? "
                       f"(found: {sorted(have)})")
    if "split" in have:
        cols.append("split")
    return cols


def pack_rates(series, seq_len, stats=None):
    """'a,b,c;a,b,c;...' x N  ->  (N, 3, seq_len) float32 plus a (N, seq_len) bool mask.

    Padded steps are masked rather than zeroed-and-forgotten: a zero is a legitimate
    angular rate, so the pooling has to know which steps are real.
    """
    n = len(series)
    X = np.zeros((n, seq_len, 3), dtype=np.float32)
    M = np.zeros((n, seq_len), dtype=bool)
    for i, s in enumerate(series):
        r = parse_rates(s)
        k = min(len(r), seq_len)
        if k:
            X[i, :k] = r[:k]
            M[i, :k] = True
    if stats is not None:
        mean, std = stats
        X = (X - mean) / std
    return np.ascontiguousarray(X.transpose(0, 2, 1)), M      # (N, 3, L) for Conv1d


def channel_stats(X, M, eps=1e-6):
    """Per-channel mean/std over REAL steps only.

    Including the padding would drag both statistics toward zero by an amount that depends
    on how many short windows happen to be in the split -- a silent coupling between
    preprocessing and data composition.
    """
    real = X.transpose(0, 2, 1)[M]                            # (n_real_steps, 3)
    mean = real.mean(axis=0).astype(np.float32)
    std = np.maximum(real.std(axis=0), eps).astype(np.float32)
    return mean, std


class FoldData:
    """One fold file, parsed once, held on the device.

    Attributes are plain tensors; there is no __getitem__ and no DataLoader. Use
    `batches()` to iterate.
    """

    def __init__(self, csv_path, target="gate", rates_col="gaze_rates_window",
                 seq_len=9, device="cpu", stats=None, drop_ambiguous=True):
        if rates_col not in RATE_COLS:
            raise ValueError(f"rates_col must be one of {RATE_COLS}, got {rates_col!r}")
        if target not in ("gate", "quad"):
            raise ValueError(f"target must be 'gate' or 'quad', got {target!r}")

        df = pd.read_csv(csv_path, usecols=_needed_columns(csv_path, rates_col, target))
        self.target, self.rates_col, self.seq_len = target, rates_col, seq_len
        self.device = device

        if drop_ambiguous:
            # rows inside the notebook's DEAD_ZONE have no meaningful class
            n0 = len(df)
            if target == "gate":
                df = df[df["gate"].isin(["SEND", "DISCARD"])]
            df = df.reset_index(drop=True)
            if len(df) < n0:
                print(f"   dropped {n0-len(df):,} AMBIGUOUS rows")

        self.df = df
        self.n_classes = 2 if target == "gate" else 4

        y = (df["gate"].map({"DISCARD": 0, "SEND": 1}).to_numpy()
             if target == "gate" else df["quad"].to_numpy())
        self.y_np = y.astype(np.int64)

        X, M = pack_rates(df[rates_col].tolist(), seq_len)
        self.stats = stats if stats is not None else channel_stats(X, M)
        mean, std = self.stats
        X = (X - mean[None, :, None]) / std[None, :, None]

        self.X = torch.from_numpy(X).to(device)
        self.M = torch.from_numpy(M).to(device)
        self.y = torch.from_numpy(self.y_np).to(device)
        self.split = df["split"].to_numpy() if "split" in df.columns else None
        self.sequence = df["sequence"].to_numpy()

    # ---------------------------------------------------------------- views
    def subset(self, mask):
        """A view of the rows where `mask` is True, sharing the same tensors."""
        idx = np.flatnonzero(np.asarray(mask))
        v = object.__new__(FoldData)
        v.__dict__.update(self.__dict__)
        t = torch.from_numpy(idx).to(self.device)
        v.X, v.M, v.y = self.X[t], self.M[t], self.y[t]
        v.y_np = self.y_np[idx]
        v.sequence = self.sequence[idx]
        v.split = None if self.split is None else self.split[idx]
        v.df = self.df.iloc[idx].reset_index(drop=True)
        return v

    def train_val(self):
        if self.split is None:
            raise KeyError("this CSV has no `split` column -- is it test.csv?")
        return self.subset(self.split == "train"), self.subset(self.split == "val")

    # ---------------------------------------------------------------- batching
    def __len__(self):
        return int(self.X.shape[0])

    def batches(self, batch_size, shuffle=False, generator=None):
        n = len(self)
        idx = (torch.randperm(n, device=self.device, generator=generator)
               if shuffle else torch.arange(n, device=self.device))
        for s in range(0, n, batch_size):
            j = idx[s:s + batch_size]
            yield self.X[j], self.M[j], self.y[j]

    def class_counts(self):
        return np.bincount(self.y_np, minlength=self.n_classes)

    def class_weights(self):
        """Inverse-frequency weights, normalised to mean 1.

        Worth doing here: with median thresholds the quadrants are typically 2:1 or worse,
        and an unweighted loss will happily never predict REFIXATION -- the one class the
        two-threshold design exists to capture.
        """
        c = self.class_counts().astype(np.float64)
        w = np.where(c > 0, c.sum() / np.maximum(c, 1), 0.0)
        w = w / w[w > 0].mean()
        return torch.tensor(w, dtype=torch.float32, device=self.device)

    def describe(self, name=""):
        c = self.class_counts()
        names = GATE_NAMES if self.target == "gate" else QUAD_NAMES
        parts = [f"{names[i]} {c[i]:,} ({100*c[i]/max(1,len(self)):.1f}%)"
                 for i in range(self.n_classes)]
        print(f"   {name:<6} {len(self):6,} rows  {self.df['sequence'].nunique():3d} videos"
              f"   " + "  ".join(parts))

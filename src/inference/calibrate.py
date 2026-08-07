"""Choosing threshold_frame and threshold_gaze.

The thresholds are not learned. They set a trade -- how much compute is saved against how
much content is missed -- and that trade is a deployment decision, not something a loss
can settle. This module makes it explicit.

Two ways to pick them:

`from_labels`   percentiles of the TRUE DINOv2 similarities. Defines the "oracle" gate:
                what a perfect predictor would decide. Use this as the reference the
                learned gate is measured against.

`for_keep_rate` search predicted scores for the thresholds that hit a target fraction of
                frames sent. Use this when a compute budget is fixed and the question is
                how well the gate spends it.
"""

import numpy as np
import pandas as pd

from src.inference.gate import SEND, filter_frame_for_vlm
from src.loss1.dataset import parse_rates
from src.loss2.losses import TARGETS


def from_labels(csv_path, frame_pct=50.0, gaze_pct=50.0, sequences=None):
    """Percentiles of the true similarity columns.

    frame_pct=50, gaze_pct=50 puts the thresholds at the medians, which is a neutral
    starting point: roughly a quarter of frames land in the DISCARD quadrant (both scores
    high). Raise a percentile to send more frames.
    """
    df = pd.read_csv(csv_path)
    if sequences is not None:
        df = df[df["sequence"].isin(sequences)]
    return dict(
        threshold_frame=float(np.percentile(df[TARGETS[0]], frame_pct)),
        threshold_gaze=float(np.percentile(df[TARGETS[1]], gaze_pct)),
    )


def predict_scores(gate, csv_path, sequences=None, batch=256):
    """Run the gate over every row of a CSV. Gaze only -- no features are loaded."""
    df = pd.read_csv(csv_path)
    if sequences is not None:
        df = df[df["sequence"].isin(sequences)]
    R = [parse_rates(s) for s in df["gaze_rates_window"]]
    L = gate.seq_len
    X = np.zeros((len(R), L, 3), np.float32)
    for i, r in enumerate(R):
        n = min(len(r), L)
        if n:
            X[i, :n] = r[:n]
    out = np.concatenate([gate.scores(X[i:i+batch]) for i in range(0, len(X), batch)]) \
        if len(X) else np.zeros((0, 2), np.float32)
    return df.reset_index(drop=True), out


def for_keep_rate(pred, target_keep=0.5, n_grid=60):
    """Grid-search thresholds on PREDICTED scores to hit a target send fraction.

    Many (threshold_frame, threshold_gaze) pairs give the same keep rate -- only the
    upper-right quadrant is discarded, so the two thresholds trade against each other.
    The pair closest to the target is returned; ties break toward a lower gaze threshold,
    which is the more conservative choice (a low gaze threshold sends more frames when the
    attended region changes).
    """
    f, g = pred[:, 0], pred[:, 1]
    fs = np.quantile(f, np.linspace(0.02, 0.98, n_grid))
    gs = np.quantile(g, np.linspace(0.02, 0.98, n_grid))
    best, best_err = None, np.inf
    for tf in fs:
        for tg in gs:
            keep = float(np.mean(~((f > tf) & (g > tg))))
            err = abs(keep - target_keep)
            if err < best_err - 1e-12 or (abs(err - best_err) <= 1e-12 and
                                          best is not None and tg < best[1]):
                best, best_err = (float(tf), float(tg)), err
    return dict(threshold_frame=best[0], threshold_gaze=best[1],
                achieved_keep_rate=float(np.mean(~((f > best[0]) & (g > best[1])))))


def decisions(scores, threshold_frame, threshold_gaze):
    """Vectorised FilterFrameForVLM: True where the frame is SENT."""
    s = np.asarray(scores)
    return ~((s[:, 0] > threshold_frame) & (s[:, 1] > threshold_gaze))


def reasons(scores, threshold_frame, threshold_gaze):
    return [filter_frame_for_vlm(float(a), threshold_frame, float(b), threshold_gaze)[1]
            for a, b in np.asarray(scores)]

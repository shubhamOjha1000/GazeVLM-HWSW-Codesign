"""Held-out 4-class predictions for every gate video.

Each pooled video sits in exactly one fold's validation split, so scoring it with THAT
fold's model gives a prediction from a model that never trained on it. Test videos are
scored by the 5-fold ensemble, as in eyehead.infer. Together: clean predictions for all
70 videos, with no retraining.

    from eyehead.oof import oof_predictions
    df = oof_predictions(folds_dir, ckpt_dir)      # one row per frame pair, all videos
"""

import os

import numpy as np
import pandas as pd
import torch

from eyehead.features import NAMES, build, load_rows, normalise
from eyehead.infer import load_ckpt, model_from_ckpt
from eyehead.train import predict_proba

PAIR_COLS = ["feat_frame_1", "feat_frame_2"]


def _score(ck, df, device):
    f = build(df, np.asarray(ck["head_map"]), ck["seq_len"], ck["eye_min"], ck["head_min"])
    Z = normalise(f["X"], f["M"], ck["stats"])
    return predict_proba(model_from_ckpt(ck, device), Z, f["M"], device)


def _rows(csv_path):
    df = load_rows(csv_path)
    pairs = pd.read_csv(csv_path, usecols=PAIR_COLS)
    return pd.concat([df.reset_index(drop=True), pairs.reset_index(drop=True)], axis=1)


def oof_predictions(folds_dir, ckpt_dir, k=5, device=None, verbose=True):
    """-> DataFrame: sequence, feat_frame_1, feat_frame_2, quad, source ('fold<i>' or
    'test ensemble'), p_<CLASS> x 4, pred (argmax class id)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cks = [load_ckpt(os.path.join(ckpt_dir, f"fold{i}.pt"), device) for i in range(1, k + 1)]
    parts = []
    for i, ck in enumerate(cks, 1):
        df = _rows(os.path.join(folds_dir, f"train_val_fold{i}.csv"))
        df = df[df["split"] == "val"].reset_index(drop=True)
        p = _score(ck, df, device)
        parts.append(df.assign(source=f"fold{i}", _p=list(p)))
        if verbose:
            print(f"   fold{i}: {len(df):,} held-out rows, {df.sequence.nunique()} videos")

    te = _rows(os.path.join(folds_dir, "test.csv"))
    p = np.mean([_score(ck, te, device) for ck in cks], axis=0)
    parts.append(te.assign(source="test ensemble", _p=list(p)))
    if verbose:
        print(f"   test : {len(te):,} rows, {te.sequence.nunique()} videos (5-fold ensemble)")

    out = pd.concat(parts, ignore_index=True)
    P = np.stack(out.pop("_p").to_numpy())
    for c in range(4):
        out[f"p_{NAMES[c]}"] = P[:, c]
    out["pred"] = P.argmax(1)
    dup = out.duplicated("feat_frame_2").sum()
    if dup:
        raise ValueError(f"{dup} frame pairs predicted twice -- a video sits in two "
                         "validation splits, which the folds should never allow")
    keep = ["sequence", *PAIR_COLS, "quad", "source", *[f"p_{NAMES[c]}" for c in range(4)], "pred"]
    return out[keep]

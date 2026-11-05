"""A group's rows as fold files, and the gate's predictions on them from the 5 fold models averaged. CPU only.

    python -u -m accuracy_tables.hpc.newvideos.score_gate --group nymeria

None of the 5 models trained on these videos, so all can be used. Writes newvideos/<group>/folds/test.csv, an empty
train_val_fold1.csv beside it (load_labels and gaze_tracks read both) and newvideos/<group>/preds.csv, with the columns of
eyehead.oof.oof_predictions. Pointing paths.oof_cache at preds.csv means oof_predictions is never called.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from accuracy_tables.hpc.newvideos import paths
from eyehead.features import NAMES
from eyehead.infer import load_ckpt
from eyehead.oof import PAIR_COLS, _rows, _score


def merge_rows(group, videos):
    """Joins the videos' rows into <group>/folds/ -> the test.csv path."""
    missing = [v for v in videos if not os.path.exists(paths.rows_csv(v))]
    if missing:
        sys.exit(f"{group}: {len(missing)} videos have no rows, for example {missing[:5]}. Rerun the build for them.")
    d = paths.folds_dir(group)
    os.makedirs(d, exist_ok=True)
    rows = pd.concat([pd.read_csv(paths.rows_csv(v)) for v in videos], ignore_index=True)
    rows.insert(0, "idx", range(len(rows)))
    test = os.path.join(d, "test.csv")
    rows.to_csv(test, index=False)
    rows.head(0).to_csv(os.path.join(d, "train_val_fold1.csv"), index=False)
    return test


def load_models(ckpt_dir, device="cpu", k=5):
    return [load_ckpt(os.path.join(ckpt_dir, f"fold{i}.pt"), device) for i in range(1, k + 1)]


def score_ensemble(csv_path, cks, device="cpu"):
    """Same columns as eyehead.oof.oof_predictions, every row from all the models averaged."""
    df = _rows(csv_path)
    p = np.mean([_score(ck, df, device) for ck in cks], axis=0)
    out = df.assign(source="test ensemble")
    for c in range(4):
        out[f"p_{NAMES[c]}"] = p[:, c]
    out["pred"] = p.argmax(1)
    return out[["sequence", *PAIR_COLS, "quad", "source", *[f"p_{NAMES[c]}" for c in range(4)], "pred"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True, help="a list written by plan.py")
    a = ap.parse_args()
    test = merge_rows(a.group, paths.read_list(a.group))
    out = score_ensemble(test, load_models(paths.ckpt_dir()))
    path = paths.preds_csv(a.group)
    tmp = path + ".tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)
    print(f"{a.group}: {len(out)} predictions on {out.sequence.nunique()} videos -> {path}", flush=True)


if __name__ == "__main__":
    main()

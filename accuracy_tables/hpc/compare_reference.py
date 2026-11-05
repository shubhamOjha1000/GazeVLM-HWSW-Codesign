"""Compares a run's summary.csv with the Colab runs' summary.csv files, on the same (model, condition):

    python -m accuracy_tables.hpc.compare_reference <summary.csv> <folder with */summary.csv>

The condition names hold the calibrated knobs and token counts don't depend on the GPU, so a missing condition or a
different token count means the frames, gaze or gate data differ. The exceptions are PruneVid and FastVID, whose plans
come from the encoder's output, so a different GPU or torch can move them a little: they are shown but don't fail the
check. Accuracy may move by a question or two.
"""
import glob
import os
import sys

import pandas as pd

KEYS = ["model", "condition"]
COLS = ["n", "accuracy", "frames_per_q", "tokens_per_q"]
SOFT_ROWS = ("prunevid", "fastvid")


def compare(new, ref):
    """One row per reference condition -> with missing, counts_match (n, frames, tokens) and accuracy_diff."""
    m = ref[KEYS + ["row"] + COLS].merge(new[KEYS + COLS], on=KEYS, how="left", suffixes=("_colab", "_new"))
    m["missing"] = m["n_new"].isna()
    m["counts_match"] = ((m["n_new"] == m["n_colab"]) & ((m["frames_per_q_new"] - m["frames_per_q_colab"]).abs() < 0.01)
                         & ((m["tokens_per_q_new"] - m["tokens_per_q_colab"]).abs() < 0.01))
    m["accuracy_diff"] = m["accuracy_new"] - m["accuracy_colab"]
    return m


def main(summary, folder):
    files = sorted(glob.glob(os.path.join(folder, "*", "summary.csv")))
    if not files:
        sys.exit(f"no */summary.csv under {folder}")
    m = compare(pd.read_csv(summary), pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
    pd.set_option("display.width", 200, "display.max_colwidth", 70)
    print(m[KEYS + ["n_new", "counts_match", "accuracy_colab", "accuracy_new", "accuracy_diff"]].round(2).to_string(index=False))
    done = m[~m["missing"]]
    print(f"\n{len(done)} of {len(m)} reference conditions found; counts differ on {int((~done['counts_match']).sum())}; "
          f"largest accuracy difference {done['accuracy_diff'].abs().max():.2f} points")
    hard = m[~m["row"].isin(SOFT_ROWS)]
    if hard["missing"].any() or not hard["counts_match"].all():
        sys.exit("DIFFERS from the Colab runs: see the rows above")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

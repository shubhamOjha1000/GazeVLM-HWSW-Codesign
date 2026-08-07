"""Evaluate the gate against the oracle it is trying to imitate.

    python -m src.inference.evaluate --csv data/feature_dataset.csv \
                                     --ckpt runs/loss2/best.pt

The oracle applies FilterFrameForVLM to the TRUE DINOv2 similarities: what a perfect
predictor would decide. The gate applies the same rule to scores predicted from gaze
alone. The gap between them is the only thing worth measuring.

Two errors, and they are not symmetric:

  FALSE SKIP  gate discards, oracle would have sent   -- content is LOST, unrecoverable
  FALSE SEND  gate sends, oracle would have discarded -- compute wasted, nothing lost

A gate that skips nothing is useless but harmless; a gate that skips the wrong frames is
worse than no gate at all. Both rates are reported separately for that reason.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.inference import calibrate
from src.inference.gate import GazeGate
from src.loss1.dataset import split_by_sequence
from src.loss2.losses import TARGETS


def parse_args():
    ap = argparse.ArgumentParser("Evaluate the gaze gating module")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt", required=True, help="best.pt from src.loss2.train")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--device", default="cpu")

    ap.add_argument("--val_only", action="store_true",
                    help="evaluate on held-out videos only (recommended)")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--frame_pct", type=float, default=50.0,
                    help="percentile of true frame_similarity used as threshold_frame")
    ap.add_argument("--gaze_pct", type=float, default=50.0)
    ap.add_argument("--target_keep", type=float, default=None,
                    help="instead, calibrate the gate's own thresholds to this send rate")
    return ap.parse_args()


def confusion(gate_send, oracle_send):
    tp = int(np.sum(gate_send & oracle_send))       # sent, should send
    tn = int(np.sum(~gate_send & ~oracle_send))     # skipped, should skip
    fs = int(np.sum(~gate_send & oracle_send))      # FALSE SKIP  -- content lost
    fx = int(np.sum(gate_send & ~oracle_send))      # FALSE SEND  -- compute wasted
    n = len(gate_send)
    return dict(
        agreement=(tp + tn) / n,
        false_skip_rate=fs / n,
        false_send_rate=fx / n,
        # of the frames the oracle would have sent, what fraction did the gate lose?
        recall_of_needed=tp / (tp + fs) if (tp + fs) else float("nan"),
        counts=dict(sent_correctly=tp, skipped_correctly=tn,
                    false_skip=fs, false_send=fx, n=n),
    )


def main():
    a = parse_args()
    gate = GazeGate.from_checkpoint(a.ckpt, device=a.device)

    seqs = None
    if a.val_only:
        _, val_seqs = split_by_sequence(a.csv, a.val_frac, a.seed)
        seqs = val_seqs
        print(f"evaluating on HELD-OUT videos only: {val_seqs}")

    df, pred = calibrate.predict_scores(gate, a.csv, seqs)
    truth = df[list(TARGETS)].to_numpy(dtype=np.float64)
    print(f"{len(df)} rows from {df['sequence'].nunique()} videos\n")

    # -- thresholds ---------------------------------------------------------------
    th = calibrate.from_labels(a.csv, a.frame_pct, a.gaze_pct)
    print(f"ORACLE thresholds (percentiles {a.frame_pct}/{a.gaze_pct} of the true labels)")
    print(f"   threshold_frame {th['threshold_frame']:.4f}   "
          f"threshold_gaze {th['threshold_gaze']:.4f}")

    gate_th = dict(th)
    if a.target_keep is not None:
        gate_th = calibrate.for_keep_rate(pred, a.target_keep)
        print(f"\nGATE thresholds calibrated to a {a.target_keep:.0%} send rate")
        print(f"   threshold_frame {gate_th['threshold_frame']:.4f}   "
              f"threshold_gaze {gate_th['threshold_gaze']:.4f}   "
              f"achieved {gate_th['achieved_keep_rate']:.1%}")

    oracle_send = calibrate.decisions(truth, th["threshold_frame"], th["threshold_gaze"])
    gate_send = calibrate.decisions(pred, gate_th["threshold_frame"],
                                    gate_th["threshold_gaze"])

    # -- results ------------------------------------------------------------------
    c = confusion(gate_send, oracle_send)
    print(f"\n{'='*72}\nSEND RATES")
    print(f"   oracle sends : {oracle_send.mean():6.1%}   "
          f"(skips {1-oracle_send.mean():.1%} -> that much encoder compute avoided)")
    print(f"   gate   sends : {gate_send.mean():6.1%}   (skips {1-gate_send.mean():.1%})")

    print(f"\nAGREEMENT WITH THE ORACLE")
    print(f"   overall agreement : {c['agreement']:6.1%}")
    print(f"   FALSE SKIP  (content LOST)    : {c['false_skip_rate']:6.1%}   "
          f"{c['counts']['false_skip']} frames")
    print(f"   FALSE SEND  (compute wasted)  : {c['false_send_rate']:6.1%}   "
          f"{c['counts']['false_send']} frames")
    print(f"   of frames the oracle would send, the gate kept {c['recall_of_needed']:.1%}")

    # -- baselines: the numbers that stop a bad gate looking good ------------------
    rng = np.random.default_rng(a.seed)
    rand_send = rng.random(len(df)) < gate_send.mean()
    always = np.ones(len(df), bool)
    print(f"\nBASELINES (same send rate where applicable)")
    for nm, d in (("random skipping", rand_send), ("always send", always)):
        cb = confusion(d, oracle_send)
        print(f"   {nm:16s} agreement {cb['agreement']:6.1%}   "
              f"false skip {cb['false_skip_rate']:6.1%}")
    print("   A gate that does not beat 'random skipping' has learned nothing useful.")

    # -- how well the underlying scores track the truth ---------------------------
    print(f"\nSCORE QUALITY (before any thresholding)")
    for k, name in enumerate(TARGETS):
        r = np.corrcoef(pred[:, k], truth[:, k])[0, 1]
        print(f"   {name:22s} pearson r {r:+.3f}")

    cost = gate.cost_report()
    print(f"\nGATE COST")
    print(f"   {cost['params']:,} params, ~{cost['macs_per_decision']:,} MACs per decision")
    print(f"   every skipped frame avoids the FULL image-encoder / VLM cost for that frame")
    print(f"{'='*72}")

    if c["false_skip_rate"] > 0.25:
        print("\n  HIGH FALSE-SKIP RATE. The gate is discarding frames the oracle would\n"
              "  have kept, which loses content irrecoverably. Raise the thresholds\n"
              "  (send more) until the false-skip rate is acceptable for the task.")

    if a.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(a.out_json)) or ".", exist_ok=True)
        json.dump(dict(thresholds=th, gate_thresholds=gate_th,
                       oracle_send_rate=float(oracle_send.mean()),
                       gate_send_rate=float(gate_send.mean()),
                       **{k: v for k, v in c.items()}, cost=cost),
                  open(a.out_json, "w"), indent=2)
        print(f"\nwrote {a.out_json}")


if __name__ == "__main__":
    main()

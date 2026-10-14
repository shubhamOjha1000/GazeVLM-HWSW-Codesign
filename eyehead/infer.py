"""Test the eye + head gate once: 5-fold ensemble on the held-out videos, full evaluation.

    python -m eyehead.infer --folds_dir /content/drive/MyDrive/GazeVLM/folds \
                            --ckpt_dir  /content/drive/MyDrive/GazeVLM/runs/eyehead \
                            --out_csv   /content/drive/MyDrive/GazeVLM/runs/eyehead/test_preds.csv \
                            --baseline_preds /content/drive/MyDrive/GazeVLM/release/gate_v1/test_preds.csv

Each fold model scores the test rows with ITS OWN IMU map and normalisation (both were
estimated from that fold's training rows); the five probability vectors are averaged.

Every test row is kept, including rows without IMU (head = 0, flagged `imu_ok = 0`), so
the row order matches test.csv and the numbers are comparable with the previous gate's
`test_preds.csv`. Metrics are also printed on the IMU-present subset.

Read this once. Every extra look at the test set spends part of the only unbiased
estimate there is.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from gatetrain.data import check_thresholds, load_thresholds
from eyehead.features import NAMES, TARGETS, build, load_rows, normalise
from eyehead.metrics import confusion_matrix, evaluate, report, skip_sweep
from eyehead.model import EyeHeadGate
from eyehead.train import predict_proba

INV = {v: k for k, v in NAMES.items()}


def parse_args():
    ap = argparse.ArgumentParser("Test the eye + head gate")
    ap.add_argument("--folds_dir", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--test_csv", default=None)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--baseline_preds", default=None,
                    help="the previous gate's test_preds.csv (pred_quad / true_quad "
                         "columns) for a side-by-side comparison on the same rows")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_ckpt(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:                       # torch < 1.13 has no weights_only
        return torch.load(path, map_location=device)


def model_from_ckpt(ck, device):
    ar = ck["arch"]
    m = EyeHeadGate(ck["streams"], ar["d"], ar["h"], ar["k"], dropout=0.0).to(device)
    m.load_state_dict(ck["model"])
    return m.eval()


def main():
    a = parse_args()
    test_csv = a.test_csv or os.path.join(a.folds_dir, "test.csv")
    paths = sorted(p for p in os.listdir(a.ckpt_dir)
                   if p.startswith("fold") and p.endswith(".pt"))
    if not paths:
        raise FileNotFoundError(f"no fold*.pt in {a.ckpt_dir}")
    cks = [load_ckpt(os.path.join(a.ckpt_dir, p), a.device) for p in paths]

    th = a.thresholds or os.path.join(os.path.dirname(a.folds_dir.rstrip("/")),
                                      "thresholds.json")
    tau_f, tau_g, _ = load_thresholds(th)
    for p, ck in zip(paths, cks):
        if abs(ck["tau_frame"] - tau_f) > 1e-9 or abs(ck["tau_gaze"] - tau_g) > 1e-9:
            raise ValueError(f"{p} was trained on labels from different thresholds")

    df = load_rows(test_csv)
    y = df["quad"].to_numpy().astype(int)
    check_thresholds(df[list(TARGETS)].to_numpy(np.float32), y, tau_f, tau_g, "test.csv")
    ref = cks[0]
    print(f"{len(cks)} fold models   streams {ref['streams']}   "
          f"IMU map {ref['head_map_info'].get('mode')}   {ref['n_params']:,} params each")
    print(f"test: {len(df):,} rows from {df.sequence.nunique()} videos\n")

    per_model, feats = [], None
    for p, ck in zip(paths, cks):
        f = build(df, np.asarray(ck["head_map"]), ck["seq_len"], ck["eye_min"], ck["head_min"])
        Z = normalise(f["X"], f["M"], ck["stats"])
        per_model.append(predict_proba(model_from_ckpt(ck, a.device), Z, f["M"], a.device))
        feats = feats or f                    # the raw channels do not depend on stats
    imu_ok, no_gaze = feats["imu_ok"], feats["no_gaze"]
    print(f"rows without usable IMU: {int((~imu_ok).sum())} (scored with head = 0)   "
          f"without gaze: {int(no_gaze.sum())}\n")

    print(f"{'model':>8} {'bal_acc':>8} {'macroF1':>8} {'AUC_gate':>9} "
          f"{'rec_T':>6} {'rec_P':>6} {'rec_R':>6} {'rec_S':>6}")
    for p, pm in zip(paths, per_model):
        m, _ = evaluate(pm, y)
        print(f"{p.replace('.pt',''):>8} {m['bal_acc']:>8.3f} {m['macro_f1']:>8.3f} "
              f"{m['auc_gate']:>9.4f} {m['recall_0']:>6.3f} {m['recall_1']:>6.3f} "
              f"{m['recall_2']:>6.3f} {m['recall_3']:>6.3f}")

    proba = np.mean(per_model, axis=0)
    met, pred = evaluate(proba, y)
    print("-" * 66)
    print(report(met, "\nENSEMBLE, all test rows"))
    if (~imu_ok).any() and imu_ok.any():
        msub, _ = evaluate(proba[imu_ok], y[imu_ok])
        print(report(msub, f"\nENSEMBLE, rows with IMU only ({int(imu_ok.sum()):,})"))

    cm = confusion_matrix(y, pred)
    print("\nconfusion (rows = truth, columns = predicted):")
    print(pd.DataFrame(cm, index=list(NAMES.values()), columns=list(NAMES.values()))
          .to_string())

    sweep = skip_sweep(proba[:, 3], y)
    print("\nskip iff P(STABLE) > t   (argmax is one point on this curve)")
    print(f"   {'t':>5} {'skip%':>7} {'false_skip%':>12} {'needed_kept%':>13} {'STABLE_caught%':>15}")
    for r in sweep:
        print(f"   {r['t']:>5.2f} {r['skip_pct']:>7.1f} {r['false_skip_pct']:>12.1f} "
              f"{r['needed_kept_pct']:>13.1f} {r['stable_caught_pct']:>15.1f}")

    base = None
    if a.baseline_preds and os.path.exists(a.baseline_preds):
        b = pd.read_csv(a.baseline_preds, usecols=["sequence", "true_quad", "pred_quad"])
        if len(b) != len(df) or (b["sequence"].to_numpy() != df["sequence"].to_numpy()).any():
            print(f"\n!! {a.baseline_preds} is not row-aligned with {test_csv}; skipping")
        elif (b["true_quad"].map(INV).to_numpy() != y).any():
            print(f"\n!! {a.baseline_preds} has different true labels; skipping")
        else:
            bp = b["pred_quad"].map(INV).to_numpy()
            onehot = np.eye(4)[bp]              # hard predictions -> AUCs are degenerate
            base, _ = evaluate(onehot, y)
            print(f"\n{'='*70}\nSIDE BY SIDE on the same {len(y):,} test rows\n{'='*70}")
            print(f"   {'metric':<22} {'previous gate':>14} {'eye+head':>10}")
            for key, nm in (("bal_acc", "balanced accuracy"), ("macro_f1", "macro F1"),
                            ("acc", "accuracy"), ("recall_0", "TRANSITION recall"),
                            ("recall_1", "PURSUIT recall"), ("recall_2", "REFIXATION recall"),
                            ("recall_3", "STABLE recall"),
                            ("gate_false_skip_rate", "false skip"),
                            ("gate_false_send_rate", "false send"),
                            ("gate_send_rate", "send rate")):
                print(f"   {nm:<22} {base[key]:>14.3f} {met[key]:>10.3f}")
            print("   (the previous gate's AUCs are not comparable from its hard labels)")

    if a.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)) or ".", exist_ok=True)
        out = df[["sequence", *TARGETS, "quad"]].copy()
        out["true_quad"] = [NAMES[i] for i in y]
        for c in range(4):
            out[f"p_{NAMES[c]}"] = proba[:, c]
        out["pred_quad"] = [NAMES[i] for i in pred]
        out["pred_gate"] = np.where(pred == 3, "DISCARD", "SEND")
        out["correct"] = (pred == y).astype(int)
        out["imu_ok"] = imu_ok.astype(int)
        X, M = feats["X"], feats["M"]
        out["eye_speed_max"] = np.where(M, X[:, 0], 0).max(1)
        out["head_speed_mean"] = (X[:, 1] * M).sum(1) / np.maximum(M.sum(1), 1)
        out["cos_mean"] = (X[:, 2] * M).sum(1) / np.maximum(M.sum(1), 1)
        out.to_csv(a.out_csv, index=False)
        summ = os.path.splitext(a.out_csv)[0] + "_summary.json"
        json.dump(dict(ensemble=met, confusion=cm.tolist(), skip_sweep=sweep,
                       baseline=base, n_models=len(cks), streams=ref["streams"],
                       head_map=ref["head_map_info"].get("mode"),
                       tau_frame=tau_f, tau_gaze=tau_g),
                  open(summ, "w"), indent=2, default=str)
        print(f"\nwrote {a.out_csv}\nwrote {summ}")


if __name__ == "__main__":
    main()

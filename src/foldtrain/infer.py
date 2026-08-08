"""Inference on the held-out test set, ensembling the K fold models.

    python -m src.foldtrain.infer --folds_dir /content/drive/MyDrive/GazeVLM/folds \
                                  --ckpt_dir runs/gate --out_csv runs/gate/test_preds.csv

The K checkpoints are averaged in probability space. That is not a trick to inflate the
number: each was trained on a different 4/5 of the same pool, so averaging is the natural
way to use all of them, and it is what the cross-validation was measuring the variance of.
Per-model scores are printed alongside so the ensemble gain is visible rather than assumed.

Read this once. Every extra look at the test set spends a little of the only unbiased
estimate the project has.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from src.foldtrain.dataset import FoldData, GATE_NAMES, QUAD_NAMES
from src.foldtrain.metrics import confusion, gate_costs, roc_auc, summarise
from src.foldtrain.models import build
from src.foldtrain.train import predict


def parse_args():
    ap = argparse.ArgumentParser("Test-set inference for the gate classifier")
    ap.add_argument("--folds_dir", required=True)
    ap.add_argument("--ckpt_dir", required=True, help="directory of fold1.pt .. foldK.pt")
    ap.add_argument("--test_csv", default=None, help="defaults to <folds_dir>/test.csv")
    ap.add_argument("--out_csv", default=None, help="write per-row predictions here")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_fold_models(ckpt_dir, device):
    """Every foldN.pt in the directory, plus the args they were trained with."""
    paths = sorted(p for p in os.listdir(ckpt_dir)
                   if p.startswith("fold") and p.endswith(".pt"))
    if not paths:
        raise FileNotFoundError(f"no fold*.pt in {ckpt_dir}")
    models, metas = [], []
    for p in paths:
        ck = torch.load(os.path.join(ckpt_dir, p), map_location=device)
        a = ck["args"]
        m = build(n_classes=ck["n_classes"], width=a["width"], dim=a["dim"],
                  hidden=a["hidden"], dropout=0.0).to(device)
        m.load_state_dict(ck["model"]); m.eval()
        models.append(m); metas.append(ck)
    return models, metas, paths


def main():
    a = parse_args()
    test_csv = a.test_csv or os.path.join(a.folds_dir, "test.csv")
    models, metas, names_pt = load_fold_models(a.ckpt_dir, a.device)

    ref = metas[0]["args"]
    target, rates_col, seq_len = ref["target"], ref["rates_col"], ref["seq_len"]
    names = GATE_NAMES if target == "gate" else QUAD_NAMES
    n_classes = metas[0]["n_classes"]

    # every fold must agree about what it was predicting, or averaging is meaningless
    for m, p in zip(metas, names_pt):
        assert m["args"]["target"] == target, f"{p}: target differs"
        assert m["args"]["rates_col"] == rates_col, f"{p}: rates_col differs"

    print(f"{len(models)} fold models from {a.ckpt_dir}")
    print(f"target {target}   signal {rates_col}   classes {list(names.values())}")
    print(f"test   {test_csv}")

    # standardise the test rows with each fold's own TRAIN statistics -- the test set
    # never contributes to its own normalisation
    per_model = []
    for m, ck in zip(models, metas):
        stats = (np.array(ck["stats"][0], np.float32), np.array(ck["stats"][1], np.float32))
        d = FoldData(test_csv, target=target, rates_col=rates_col, seq_len=seq_len,
                     device=a.device, stats=stats)
        per_model.append(predict(m, d, a.batch_size))
    data = d                                     # same rows/labels for all folds
    y = data.y_np
    print(f"\n{len(y):,} test rows from {data.df['sequence'].nunique()} videos")
    data.describe("test")

    print(f"\n{'model':>8} {'AUC':>8} {'acc':>8} {'bal_acc':>9}")
    for p, pm in zip(names_pt, per_model):
        s = summarise(y, pm, target)
        print(f"{p.replace('.pt',''):>8} {s['auc']:>8.4f} {s['acc']:>8.4f} {s['bal_acc']:>9.4f}")

    proba = np.mean(per_model, axis=0)
    s = summarise(y, proba, target)
    print("-" * 38)
    print(f"{'ENSEMBLE':>8} {s['auc']:>8.4f} {s['acc']:>8.4f} {s['bal_acc']:>9.4f}")

    solo = np.array([summarise(y, pm, target)["auc"] for pm in per_model])
    print(f"\nensemble vs mean single model: {s['auc']:+.4f} vs {solo.mean():+.4f} "
          f"({s['auc']-solo.mean():+.4f})")

    pred = proba.argmax(axis=1)
    print(f"\nconfusion (rows = truth, cols = predicted):")
    cm = confusion(y, pred, n_classes)
    print(pd.DataFrame(cm, index=[names[i] for i in range(n_classes)],
                       columns=[names[i] for i in range(n_classes)]).to_string())

    print("\nper class:")
    for c in range(n_classes):
        n_c = int((y == c).sum())
        rec = float((pred[y == c] == c).mean()) if n_c else float("nan")
        prec = float((y[pred == c] == c).mean()) if (pred == c).any() else float("nan")
        auc_c = roc_auc(y == c, proba[:, c])
        print(f"   {names[c]:<12} n {n_c:6,}   recall {rec:.3f}   precision {prec:.3f}"
              f"   AUC {auc_c:.4f}")

    # ---- what the gate would actually cost -----------------------------------
    if target == "gate":
        send_true, send_pred = (y == 1), (pred == 1)
    else:
        send_true, send_pred = (y != 3), (pred != 3)     # DISCARD only on STABLE
    g = gate_costs(send_true, send_pred)
    print(f"\n{'='*62}\nGATE BEHAVIOUR ON TEST\n{'='*62}")
    print(f"   oracle sends     : {100*g['oracle_send_rate']:5.1f}%")
    print(f"   gate sends       : {100*g['send_rate']:5.1f}%  "
          f"(skips {100*(1-g['send_rate']):.1f}% -> that much encoder compute avoided)")
    print(f"   agreement        : {100*g['agreement']:5.1f}%")
    print(f"   FALSE SKIP       : {100*g['false_skip_rate']:5.1f}%  "
          f"{g['counts']['false_skip']:,} frames -- content LOST, unrecoverable")
    print(f"   FALSE SEND       : {100*g['false_send_rate']:5.1f}%  "
          f"{g['counts']['false_send']:,} frames -- compute wasted, nothing lost")
    print(f"   of frames the oracle would send, the gate kept {100*g['recall_of_needed']:.1f}%")

    rng = np.random.default_rng(0)
    rand = rng.random(len(y)) < g["send_rate"]
    gr = gate_costs(send_true, rand)
    print(f"\n   baseline, random skipping at the same rate:")
    print(f"      agreement {100*gr['agreement']:.1f}%   false skip {100*gr['false_skip_rate']:.1f}%")
    print(f"   {'BEATS' if g['agreement'] > gr['agreement'] else 'DOES NOT BEAT'} random skipping")
    if g["false_skip_rate"] > 0.25:
        print("\n   HIGH FALSE-SKIP RATE. Raise the decision threshold (send more) until")
        print("   this is acceptable; lost content cannot be recovered downstream.")
    print("=" * 62)

    if a.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)) or ".", exist_ok=True)
        out = data.df.copy()
        for c in range(n_classes):
            out[f"p_{names[c]}"] = proba[:, c]
        out["pred"] = [names[int(i)] for i in pred]
        out["truth"] = [names[int(i)] for i in y]
        out["correct"] = (pred == y).astype(int)
        out.to_csv(a.out_csv, index=False)
        print(f"\nwrote per-row predictions -> {a.out_csv}")
        json.dump(dict(target=target, rates_col=rates_col, n_models=len(models),
                       ensemble=s, per_model=[float(x) for x in solo], gate=g),
                  open(os.path.splitext(a.out_csv)[0] + "_summary.json", "w"),
                  indent=2, default=str)


if __name__ == "__main__":
    main()

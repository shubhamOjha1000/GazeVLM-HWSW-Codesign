"""Test-set inference: predict [S_frame, S_gaze], then apply FilterFrameForVLM (PDF p4).

    python -m gatetrain.infer --folds_dir /content/drive/MyDrive/GazeVLM/folds \
                              --ckpt_dir runs/joint --out_csv runs/joint/test_preds.csv

The decision uses the SAME tau_frame / tau_gaze that created the labels, loaded from
thresholds.json rather than re-derived from the predictions. That is what makes the gate's
decisions directly comparable with the oracle's: both are read off the same boundary, so
a disagreement is the model's error rather than a difference of definition.

At no point is a frame encoded. The head runs on z_M alone; the frame features exist in
the checkpoint's training history only, and h is not even loaded.

Read this once. Every extra look spends part of the only unbiased estimate available.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from gatetrain.data import FoldSet, check_thresholds, load_thresholds
from gatetrain.losses import evaluate, gate_scores
from gatetrain.model import build
from src.foldtrain.metrics import roc_auc
from src.inference.gate import filter_frame_for_vlm

QUAD = {0: "TRANSITION", 1: "PURSUIT", 2: "REFIXATION", 3: "STABLE"}


def parse_args():
    ap = argparse.ArgumentParser("Test the joint model through FilterFrameForVLM")
    ap.add_argument("--folds_dir", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--test_csv", default=None)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    a = parse_args()
    test_csv = a.test_csv or os.path.join(a.folds_dir, "test.csv")
    paths = sorted(p for p in os.listdir(a.ckpt_dir)
                   if p.startswith("fold") and p.endswith(".pt"))
    if not paths:
        raise FileNotFoundError(f"no fold*.pt in {a.ckpt_dir}")

    cks = [torch.load(os.path.join(a.ckpt_dir, p), map_location=a.device) for p in paths]
    ref = cks[0]["args"]
    th_path = a.thresholds or os.path.join(
        os.path.dirname(a.folds_dir.rstrip("/")), "thresholds.json")
    tau_f, tau_g, meta = load_thresholds(th_path)

    # the checkpoints must have been trained against these same thresholds
    for p, ck in zip(paths, cks):
        if abs(ck.get("tau_frame", tau_f) - tau_f) > 1e-9 or \
           abs(ck.get("tau_gaze", tau_g) - tau_g) > 1e-9:
            raise ValueError(
                f"{p} was trained with tau ({ck['tau_frame']:.4f}, {ck['tau_gaze']:.4f}) "
                f"but thresholds.json says ({tau_f:.4f}, {tau_g:.4f}). Scoring against a "
                f"different boundary than the one that made the labels is meaningless.")

    print(f"{len(cks)} fold models from {a.ckpt_dir}")
    print(f"tau_frame {tau_f:.4f}   tau_gaze {tau_g:.4f}   (from {th_path})")
    print(f"signal {ref['rates_col']}   objective "
          f"{'L_sim only' if ref.get('sim_only') else 'L_sim + L_NCE'}")
    print(f"test   {test_csv}\n")

    per_model = []
    for ck in cks:
        rs = (np.array(ck["rate_stats"][0], np.float32),
              np.array(ck["rate_stats"][1], np.float32))
        ts = (np.array(ck["target_stats"][0], np.float32),
              np.array(ck["target_stats"][1], np.float32))
        # test rows are standardised with each fold's own TRAIN statistics; the test set
        # never contributes to its own normalisation
        d = FoldSet(test_csv, None, ref["rates_col"], ref["seq_len"], a.device,
                    rate_stats=rs, target_stats=ts)
        m = build(feat_dim=ck.get("feat_dim", 384), dim=ref["dim"], hidden=ref["hidden"],
                  width=ref["width"], fusion=ref["fusion"], dropout=0.0).to(a.device)
        m.load_state_dict(ck["model"]); m.eval()
        out = []
        with torch.no_grad():
            for j in d.batches(a.batch_size):
                X, M, *_ = d.gather(j)
                out.append(m.head(m.f_m(X, M)).float().cpu().numpy())
        # back to RAW cosine units -- tau lives there, the loss does not
        per_model.append(np.concatenate(out) * ts[1] + ts[0])
    data = d

    check_thresholds(data.y_raw_np, data.quad_np, tau_f, tau_g, "test.csv")
    print(f"\n{len(data):,} test rows from {data.df.sequence.nunique()} videos")

    print(f"\n{'model':>8} {'AUC_gate':>10} {'AUC_frame':>10} {'R2':>8} {'quad_acc':>9}")
    for p, pm in zip(paths, per_model):
        m_, _ = evaluate(pm, data.y_raw_np, data.quad_np, tau_f, tau_g)
        print(f"{p.replace('.pt',''):>8} {m_['auc_gate']:>10.4f} {m_['auc_frame']:>10.4f} "
              f"{m_['r2_mean']:>8.3f} {m_['quad_acc']:>9.3f}")

    pred = np.mean(per_model, axis=0)
    met, quad_pred = evaluate(pred, data.y_raw_np, data.quad_np, tau_f, tau_g)
    print("-" * 50)
    print(f"{'ENSEMBLE':>8} {met['auc_gate']:>10.4f} {met['auc_frame']:>10.4f} "
          f"{met['r2_mean']:>8.3f} {met['quad_acc']:>9.3f}")

    print(f"\n{'='*68}\nREGRESSION (PDF p3) -- raw cosine units\n{'='*68}")
    for nm in ("frame", "gaze"):
        beat = met[f"mse_{nm}"] < met[f"mse_base_{nm}"]
        print(f"   S_{nm:<6} mse {met[f'mse_{nm}']:.5f}  base {met[f'mse_base_{nm}']:.5f}"
              f"   R2 {met[f'r2_{nm}']:+.3f}   r {met[f'r_{nm}']:+.3f}"
              f"   AUC {met[f'auc_{nm}']:.4f}"
              f"   [{'beats mean' if beat else 'NO BETTER THAN MEAN'}]")
    print("   `base` is the error of predicting the split mean. R2 <= 0 means the model")
    print("   found nothing, whatever the AUC says.")

    # ---- the decision, straight from PDF p4 --------------------------------------
    reasons = [filter_frame_for_vlm(float(p[0]), tau_f, float(p[1]), tau_g)
               for p in pred]
    sent = np.array([r[0] == "SEND" for r in reasons])
    oracle_sent = data.quad_np != 3

    print(f"\n{'='*68}\nFilterFrameForVLM (PDF p4)\n{'='*68}")
    from collections import Counter
    for r, c in Counter(x[1] for x in reasons).most_common():
        print(f"   {c:6,}  {100*c/len(pred):5.1f}%   {r}")

    print(f"\n{'='*68}\nQUADRANTS -- predicted vs oracle\n{'='*68}")
    cm = pd.crosstab(pd.Series([QUAD[i] for i in data.quad_np], name="truth"),
                     pd.Series([QUAD[i] for i in quad_pred], name="predicted"))
    cm = cm.reindex(index=list(QUAD.values()), columns=list(QUAD.values()), fill_value=0)
    print(cm.to_string())
    print()
    for c in range(4):
        n = int((data.quad_np == c).sum())
        prec = float((data.quad_np[quad_pred == c] == c).mean()) if (quad_pred == c).any() \
            else float("nan")
        print(f"   {QUAD[c]:<12} n {n:6,}  recall {met[f'recall_q{c}']:.3f}  "
              f"precision {prec:.3f}")
    if met["recall_q2"] < 0.1:
        print("\n   REFIXATION recall is near zero -- the quadrant that justifies having")
        print("   two thresholds is not being recovered. That is the finding, not a detail.")

    print(f"\n{'='*68}\nGATE COST\n{'='*68}")
    print(f"   oracle sends   : {100*oracle_sent.mean():5.1f}%")
    print(f"   gate sends     : {100*sent.mean():5.1f}%   "
          f"(skips {100*(1-sent.mean()):.1f}% -> encoder compute avoided)")
    print(f"   agreement      : {100*met['gate_agreement']:5.1f}%")
    print(f"   FALSE SKIP     : {100*met['gate_false_skip_rate']:5.1f}%  content LOST")
    print(f"   FALSE SEND     : {100*met['gate_false_send_rate']:5.1f}%  compute wasted")
    print(f"   of frames the oracle would send, the gate kept "
          f"{100*met['gate_recall_of_needed']:.1f}%")

    rng = np.random.default_rng(0)
    rand = rng.random(len(pred)) < sent.mean()
    print(f"\n   random skipping at the same rate: agreement "
          f"{100*float((rand == oracle_sent).mean()):.1f}%, "
          f"false skip {100*float((~rand & oracle_sent).mean()):.1f}%")
    print(f"   AUC_gate {met['auc_gate']:.4f} on min(S_frame-tau_f, S_gaze-tau_g)")
    print("=" * 68)

    if a.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)) or ".", exist_ok=True)
        out = data.df.copy()
        out["pred_S_frame"], out["pred_S_gaze"] = pred[:, 0], pred[:, 1]
        out["pred_quad"] = [QUAD[i] for i in quad_pred]
        out["true_quad"] = [QUAD[i] for i in data.quad_np]
        out["pred_gate"] = np.where(sent, "SEND", "DISCARD")
        out["reason"] = [r[1] for r in reasons]
        out["gate_score"] = gate_scores(pred, tau_f, tau_g)
        out["correct"] = (quad_pred == data.quad_np).astype(int)
        out.to_csv(a.out_csv, index=False)
        print(f"\nwrote {a.out_csv}")
        json.dump(dict(tau_frame=tau_f, tau_gaze=tau_g, n_models=len(cks),
                       ensemble=met), open(os.path.splitext(a.out_csv)[0] + "_summary.json",
                                           "w"), indent=2, default=str)


if __name__ == "__main__":
    main()

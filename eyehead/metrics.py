"""Metrics for the 4-way gate, read two ways: as a classifier and as a SEND/SKIP gate.

As a classifier, plain accuracy is dominated by TRANSITION and STABLE, so balanced
accuracy (mean per-class recall) and macro F1 are the headline numbers, and every class
gets its own recall and precision -- REFIXATION and PURSUIT are the classes this model
exists to recover.

As a gate, only STABLE is skipped, so TRANSITION / PURSUIT / REFIXATION confusions cost
nothing. What matters there is the FALSE SKIP rate (content lost) against the skip rate
(compute saved), and AUC of P(STABLE) for ranking.
"""

import numpy as np

from src.foldtrain.metrics import confusion, gate_costs, roc_auc

from eyehead.features import NAMES

SKIP = 3        # STABLE is the only class the gate discards


def evaluate(proba, y):
    """proba (N, 4) class probabilities, y (N,) true class -> (flat dict, predictions)."""
    p = np.asarray(proba, dtype=np.float64)
    y = np.asarray(y).astype(int)
    pred = p.argmax(1)
    out = dict(n=int(len(y)), acc=float((pred == y).mean()))

    recalls, f1s, aucs = [], [], []
    for c in range(4):
        t, q = y == c, pred == c
        rec = float((q & t).sum() / t.sum()) if t.any() else float("nan")
        prec = float((q & t).sum() / q.sum()) if q.any() else float("nan")
        f1 = (2 * rec * prec / (rec + prec)
              if np.isfinite(rec) and np.isfinite(prec) and rec + prec > 0 else 0.0)
        auc = roc_auc(t, p[:, c])
        out.update({f"recall_{c}": rec, f"precision_{c}": prec, f"f1_{c}": f1,
                    f"auc_{c}": auc, f"n_true_{c}": int(t.sum()), f"n_pred_{c}": int(q.sum())})
        if t.any():
            recalls.append(rec); f1s.append(f1)
        if np.isfinite(auc):
            aucs.append(auc)
    out["bal_acc"] = float(np.mean(recalls)) if recalls else float("nan")
    out["macro_f1"] = float(np.mean(f1s)) if f1s else float("nan")
    out["macro_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    out["min_recall"] = float(np.min(recalls)) if recalls else float("nan")

    skip_true, skip_pred = y == SKIP, pred == SKIP
    out["auc_gate"] = roc_auc(skip_true, p[:, SKIP])
    out.update({f"gate_{k}": v for k, v in gate_costs(~skip_true, ~skip_pred).items()
                if k != "counts"})
    return out, pred


def confusion_matrix(y, pred):
    return confusion(np.asarray(y).astype(int), np.asarray(pred).astype(int), 4)


def skip_sweep(p_stable, y, thresholds=None):
    """Skip a frame iff P(STABLE) > t. Shows the compute-saved vs content-lost trade that
    argmax hides: argmax is just one point on this curve."""
    y = np.asarray(y).astype(int)
    need = y != SKIP
    rows = []
    for t in (np.round(np.arange(0.1, 0.95, 0.05), 2) if thresholds is None else thresholds):
        skip = np.asarray(p_stable) > t
        rows.append(dict(t=float(t),
                         skip_pct=100 * float(skip.mean()),
                         false_skip_pct=100 * float((skip & need).mean()),
                         needed_kept_pct=100 * float((~skip & need).sum() / max(1, need.sum())),
                         stable_caught_pct=100 * float((skip & ~need).sum() / max(1, (~need).sum()))))
    return rows


def report(m, title=""):
    """Human-readable block for one evaluated split."""
    lines = [f"{title}" if title else "",
             f"   acc {m['acc']:.4f}   balanced acc {m['bal_acc']:.4f}   macro F1 "
             f"{m['macro_f1']:.4f}   macro AUC {m['macro_auc']:.4f}",
             f"   {'class':<12} {'n':>6} {'recall':>8} {'precision':>10} {'F1':>7} {'AUC':>7}"]
    for c in range(4):
        lines.append(f"   {NAMES[c]:<12} {m[f'n_true_{c}']:>6,} {m[f'recall_{c}']:>8.3f} "
                     f"{m[f'precision_{c}']:>10.3f} {m[f'f1_{c}']:>7.3f} {m[f'auc_{c}']:>7.3f}")
    lines += [f"   gate: skips {100*(1-m['gate_send_rate']):.1f}% (oracle "
              f"{100*(1-m['gate_oracle_send_rate']):.1f}%)   agreement "
              f"{100*m['gate_agreement']:.1f}%   FALSE SKIP {100*m['gate_false_skip_rate']:.1f}%"
              f"   FALSE SEND {100*m['gate_false_send_rate']:.1f}%   AUC_gate {m['auc_gate']:.4f}"]
    return "\n".join(l for l in lines if l != "" or title)

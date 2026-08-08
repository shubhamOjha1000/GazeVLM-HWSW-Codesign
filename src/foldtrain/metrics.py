"""Metrics for the gate classifier, AUC first.

AUC is implemented here rather than imported from sklearn for two reasons: it is a
five-line rank statistic, and it keeps the training path free of a dependency that is not
in requirements.txt. The implementation is checked against sklearn in the tests.
"""

import numpy as np


def _avg_ranks(x):
    """Ranks 1..n with ties averaged -- what the Mann-Whitney form of AUC requires.

    Ties matter: a model that outputs the same score for many rows (early training, or a
    collapsed model) would otherwise get a misleadingly high or low AUC depending on the
    arbitrary order the sort happened to produce.
    """
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(xs):                      # average within each run of equal values
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def roc_auc(y_true, score):
    """Binary ROC AUC via the Mann-Whitney U statistic.

    Returns NaN when one class is absent -- AUC is undefined there, and returning 0.5
    would quietly look like "no better than chance" instead of "not measurable".
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(score, dtype=np.float64)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _avg_ranks(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def macro_ovr_auc(y_true, proba, n_classes=None):
    """One-vs-rest AUC averaged over classes, for the 4-way quadrant target.

    Macro rather than micro on purpose: REFIXATION is the rare class and the one the
    two-threshold design exists for, so it must count as much as STABLE.
    """
    y = np.asarray(y_true)
    p = np.asarray(proba, dtype=np.float64)
    k = n_classes or p.shape[1]
    per = [roc_auc(y == c, p[:, c]) for c in range(k)]
    good = [a for a in per if np.isfinite(a)]
    return (float(np.mean(good)) if good else float("nan")), per


def accuracy(y_true, y_pred):
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def balanced_accuracy(y_true, y_pred, n_classes=None):
    """Mean per-class recall. With skewed quadrants, plain accuracy is dominated by
    STABLE and TRANSITION and can look respectable while the model never predicts
    REFIXATION at all."""
    y, p = np.asarray(y_true), np.asarray(y_pred)
    k = n_classes or int(max(y.max(), p.max())) + 1
    rec = [float((p[y == c] == c).mean()) for c in range(k) if (y == c).any()]
    return float(np.mean(rec)) if rec else float("nan")


def confusion(y_true, y_pred, n_classes):
    m = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        m[int(t), int(p)] += 1
    return m


def gate_costs(y_true_send, y_pred_send):
    """The two errors a gate can make, which are NOT symmetric.

    FALSE SKIP  gate discards a frame the oracle would send -- content lost, unrecoverable
    FALSE SEND  gate sends one the oracle would discard    -- compute wasted, nothing lost

    Accuracy averages these together and hides the distinction, so they are reported
    separately.
    """
    t = np.asarray(y_true_send).astype(bool)
    p = np.asarray(y_pred_send).astype(bool)
    n = len(t)
    fs = int((~p & t).sum())
    fx = int((p & ~t).sum())
    return dict(
        agreement=float(((p == t).sum()) / n),
        false_skip_rate=fs / n,
        false_send_rate=fx / n,
        recall_of_needed=float((p & t).sum() / max(1, t.sum())),
        send_rate=float(p.mean()),
        oracle_send_rate=float(t.mean()),
        counts=dict(false_skip=fs, false_send=fx, n=n),
    )


def summarise(y_true, proba, target):
    """Everything worth printing for one split, as a flat dict."""
    p = np.asarray(proba, dtype=np.float64)
    y = np.asarray(y_true)
    pred = p.argmax(axis=1)
    out = dict(n=len(y), acc=accuracy(y, pred),
               bal_acc=balanced_accuracy(y, pred, p.shape[1]))
    if target == "gate":
        # class 1 == SEND (see dataset.py); AUC on the positive-class probability
        out["auc"] = roc_auc(y == 1, p[:, 1])
        out.update({f"gate_{k}": v for k, v in
                    gate_costs(y == 1, pred == 1).items() if k != "counts"})
    else:
        out["auc"], per = macro_ovr_auc(y, p)
        for c, a in enumerate(per):
            out[f"auc_class{c}"] = a
    return out

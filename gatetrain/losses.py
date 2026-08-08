"""The joint objective, PDF p9:  L = L_NCE + L_similarity

Both terms are imported rather than rewritten -- `info_nce` is EgoDistill Eq. 7 as the
slide states it (τ = 0.1, in-batch negatives), `similarity_loss` is MSE on standardised
targets.

On the scales: L_NCE starts near ln(batch) -- about 6.2 at batch 512 -- while L_sim starts
near 1.0 on standardised targets. The slide writes a plain sum, so `nce_weight` defaults
to 1.0 and the two terms are always reported separately, because a plain sum at those
scales means the contrastive term dominates the gradient early on. That may be fine, or
it may need weighting; either way it should be visible rather than inferred.
"""

import numpy as np

from src.loss1.losses import info_nce, retrieval_metrics
from src.loss2.losses import similarity_loss

from src.foldtrain.metrics import gate_costs, roc_auc   # verified against sklearn


def joint_loss(pred, y, z_hat=None, z_m=None, tau=0.1, nce_weight=1.0):
    """L = L_sim + nce_weight * L_NCE, plus the parts for logging.

    z_hat is None under --sim_only, and the NCE term needs at least two rows to have a
    negative, so it is skipped on a trailing batch of one rather than producing a
    degenerate loss.
    """
    l_sim, per_target = similarity_loss(pred, y)
    parts = dict(sim=float(l_sim.detach()),
                 sim_frame=float(per_target[0].detach()),
                 sim_gaze=float(per_target[1].detach()), nce=float("nan"), nce_top1=float("nan"))
    total = l_sim
    if z_hat is not None and z_m is not None and z_m.shape[0] >= 2:
        l_nce, logits = info_nce(z_hat, z_m, tau)
        total = total + nce_weight * l_nce
        parts["nce"] = float(l_nce.detach())
        parts["nce_top1"] = retrieval_metrics(logits.detach())["top1"]
    parts["total"] = float(total.detach())
    return total, parts


def gate_scores(pred_raw, tau_f, tau_g):
    """Continuous score for ranking the DISCARD decision.

    AUC needs an ordering, and the head emits two regressions rather than a probability.
    DISCARD requires BOTH scores above their thresholds, so the natural margin is the
    smaller of the two excesses: how comfortably a row clears the bar on its weaker axis.
    Higher = more confidently skippable.
    """
    p = np.asarray(pred_raw, dtype=np.float64)
    return np.minimum(p[:, 0] - tau_f, p[:, 1] - tau_g)


def evaluate(pred_raw, y_raw, quad_true, tau_f, tau_g):
    """Every number worth printing for one split, in raw cosine units.

    Reported at the thresholds that CREATED the labels, not at thresholds re-derived from
    the predictions -- otherwise the gate would be scored against a moving target and
    could not be compared with the oracle.
    """
    p = np.asarray(pred_raw, dtype=np.float64)
    y = np.asarray(y_raw, dtype=np.float64)
    quad_true = np.asarray(quad_true)

    out = {}
    for k, nm in enumerate(("frame", "gaze")):
        tau = tau_f if k == 0 else tau_g
        ss_res = float(((y[:, k] - p[:, k]) ** 2).sum())
        ss_tot = float(((y[:, k] - y[:, k].mean()) ** 2).sum())
        out[f"r2_{nm}"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        out[f"mse_{nm}"] = ss_res / len(y)
        # the number that matters: error of simply predicting the split's mean
        out[f"mse_base_{nm}"] = float(((y[:, k] - y[:, k].mean()) ** 2).mean())
        out[f"r_{nm}"] = (float(np.corrcoef(p[:, k], y[:, k])[0, 1])
                          if p[:, k].std() > 1e-12 else float("nan"))
        out[f"auc_{nm}"] = roc_auc(y[:, k] > tau, p[:, k])

    quad_pred = 2 * (p[:, 0] > tau_f).astype(int) + (p[:, 1] > tau_g).astype(int)
    out["quad_acc"] = float((quad_pred == quad_true).mean())
    for c in range(4):
        m = quad_true == c
        out[f"recall_q{c}"] = float((quad_pred[m] == c).mean()) if m.any() else float("nan")

    # DISCARD iff quad == 3, which is exactly FilterFrameForVLM's single skip quadrant
    discard_true = quad_true == 3
    out["auc_gate"] = roc_auc(discard_true, gate_scores(p, tau_f, tau_g))
    out.update({f"gate_{k}": v for k, v in
                gate_costs(~discard_true, ~(quad_pred == 3)).items() if k != "counts"})
    out["r2_mean"] = float(np.nanmean([out["r2_frame"], out["r2_gaze"]]))
    return out, quad_pred

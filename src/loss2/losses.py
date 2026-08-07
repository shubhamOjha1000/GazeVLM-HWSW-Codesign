"""Loss 2: regression of the two DINOv2 similarity scores, plus honest metrics.

The slide offers "KL divergence or MSE". MSE is used here, deliberately.

KL measures the distance between two probability DISTRIBUTIONS. [S_frame, S_gaze] is two
independent cosine similarities: it does not sum to 1 and is not a distribution.
Softmaxing it to force one would discard the absolute level and keep only the ratio --
and the absolute level is exactly what the gate thresholds on. [0.70, 0.60] and
[0.35, 0.30] would become identical while meaning completely different things to
FilterFrameForVLM.

If calibrated uncertainty is ever wanted, the principled KL route is to bin each score
into ~10 buckets and predict a categorical distribution per target. That keeps the
absolute level. It is more machinery, and only worth it if MSE plateaus.
"""

import numpy as np
import torch
import torch.nn.functional as F

TARGETS = ("frame_similarity", "gaze_patch_token_sim")


def similarity_loss(pred, target, weights=None):
    """MSE on STANDARDISED targets.

    Standardising matters: measured on real data the two targets have very different
    spreads (frame_similarity std ~0.13, gaze_patch_token_sim std ~0.22), so raw MSE
    would let the gaze term dominate the gradient roughly 3x purely because it varies
    more. `weights` allows further manual rebalancing.
    """
    per_target = F.mse_loss(pred, target, reduction="none").mean(dim=0)   # (n_targets,)
    if weights is not None:
        per_target = per_target * torch.as_tensor(weights, device=pred.device)
    return per_target.mean(), per_target


@torch.no_grad()
def regression_metrics(pred_raw, target_raw, train_mean=None, names=TARGETS):
    """Metrics in RAW similarity units, so they are interpretable.

    Reports, per target:
      mse      -- model error
      mse_base -- error of simply predicting the TRAINING mean for every row
      r2       -- 1 - SS_res/SS_tot against the split's own mean.
                  <= 0 means the model is no better than predicting the mean.
      pearson  -- correlation between prediction and truth

    `mse_base` is the number that matters. A model that cannot beat predict-the-mean on
    held-out videos has found no signal, and no amount of architecture will change that.
    """
    p = np.asarray(pred_raw, dtype=np.float64)
    y = np.asarray(target_raw, dtype=np.float64)
    out = {}
    for k, name in enumerate(names):
        pk, yk = p[:, k], y[:, k]
        ss_res = float(((yk - pk) ** 2).sum())
        ss_tot = float(((yk - yk.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        r = float(np.corrcoef(pk, yk)[0, 1]) if pk.std() > 1e-12 and yk.std() > 1e-12 \
            else float("nan")
        m = dict(mse=ss_res / len(yk), r2=r2, pearson=r,
                 pred_mean=float(pk.mean()), pred_std=float(pk.std()),
                 true_mean=float(yk.mean()), true_std=float(yk.std()))
        if train_mean is not None:
            m["mse_base"] = float(((yk - train_mean[k]) ** 2).mean())
            m["beats_baseline"] = bool(m["mse"] < m["mse_base"])
        out[name] = m

    out["r2_mean"] = float(np.mean([out[n]["r2"] for n in names]))
    out["pearson_mean"] = float(np.mean([out[n]["pearson"] for n in names]))
    return out


def format_metrics(m, names=TARGETS, indent="   "):
    lines = []
    for n in names:
        d = m[n]
        base = f"  base_mse {d['mse_base']:.5f}" if "mse_base" in d else ""
        flag = ""
        if "beats_baseline" in d:
            flag = "  [beats mean]" if d["beats_baseline"] else "  [NO BETTER THAN MEAN]"
        lines.append(f"{indent}{n:22s} mse {d['mse']:.5f}{base}  "
                     f"R2 {d['r2']:+.3f}  r {d['pearson']:+.3f}{flag}")
    return "\n".join(lines)

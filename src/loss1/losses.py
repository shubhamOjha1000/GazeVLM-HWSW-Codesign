"""InfoNCE for Loss 1, exactly as written in the Loss-1 slide / EgoDistill Eq. 7.

    L_NCE = sum_i -log [ sim(z_hat_M_i, z_M_i) / sum_j sim(z_hat_M_i, z_M_j) ]
    sim(q, k) = exp( (q.k / |q||k|) * 1/tau ),   tau = 0.1

Negatives are the other instances in the same mini-batch (j != i). With cosine
similarity and a temperature this is identical to cross-entropy over the scaled
similarity matrix, which is what the implementation below uses.
"""

import torch
import torch.nn.functional as F


def info_nce(z_hat, z_m, tau=0.1):
    """z_hat, z_m: (B, D). Returns (loss, logits).

    logits[i, j] is the scaled cosine similarity between the PREDICTED gaze embedding of
    row i and the ACTUAL gaze embedding of row j, so the correct match sits on the
    diagonal.
    """
    z_hat = F.normalize(z_hat, dim=-1)
    z_m = F.normalize(z_m, dim=-1)
    logits = z_hat @ z_m.t() / tau                       # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels), logits


@torch.no_grad()
def retrieval_metrics(logits):
    """How often does the right gaze signal rank first among the batch?

    This is the metric to watch: chance is 1/B, so anything well above it means gaze
    rates really are predictable from visual change -- the project's core premise,
    testable long before the gate exists.
    """
    b = logits.size(0)
    labels = torch.arange(b, device=logits.device)
    correct = logits.gather(1, labels[:, None])          # (B, 1)
    ranks = (logits > correct).sum(dim=1)                # 0 = best
    return dict(
        top1=(ranks == 0).float().mean().item(),
        top5=(ranks < 5).float().mean().item(),
        mean_rank=(ranks.float().mean().item() + 1.0),
        chance=1.0 / b,
        batch=b,
    )

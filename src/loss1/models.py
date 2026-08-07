"""Loss-1 architecture: the gaze-rate encoder f_M and the change predictor h.

Follows EgoDistill Sec 3.4 (self-supervised IMU feature learning) with the IMU stream
replaced by the gaze-rate signal:

    frame t-1 --> [DINOv2 CLS, FROZEN] --> z_I0 -.
                                                  >-- h --> z_hat_M
    frame t   --> [DINOv2 CLS, FROZEN] --> z_IT -'

    gaze rates (L, 3) --------------------------> f_M --> z_M

`z_hat_M` is a function of the two frame features ONLY, so pulling it towards `z_M`
forces f_M to encode exactly the part of eye motion that explains visual change.

The image encoder is deliberately absent from this file: features are precomputed and
stored on disk, so f_I is frozen by construction. EgoDistill footnote 2 reports that
finetuning it collapses training, so this is a requirement, not a shortcut.
"""

import torch
import torch.nn as nn


class GazeRateEncoder(nn.Module):
    """f_M -- 1D dilated CNN over the (L, 3) gaze-rate signal.

    Channels are [omega_yaw, omega_pitch, omega_mag] in rad/s. EgoDistill uses a 1D
    dilated CNN over a 422x6 IMU tensor; the gaze stream is far shorter (9 steps for a
    1 s window at 10 Hz), so dilations 1/2/4 with kernel 3 are enough -- they give a
    receptive field of 15, covering the whole window.

    Accepts a padding mask so variable-length windows pool correctly. Without it,
    zero-padded steps would drag the mean towards zero.
    """

    def __init__(self, in_ch=3, width=128, dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, width // 2, 3, padding=1, dilation=1),
            nn.BatchNorm1d(width // 2), nn.GELU(),
            nn.Conv1d(width // 2, width, 3, padding=2, dilation=2),
            nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width, 3, padding=4, dilation=4),
            nn.BatchNorm1d(width), nn.GELU(),
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width, dim)

    def forward(self, x, mask=None):
        """x: (B, 3, L) float. mask: (B, L) bool, True where the step is real."""
        h = self.net(x)                                    # (B, width, L)
        if mask is None:
            pooled = h.mean(dim=-1)
        else:
            m = mask.unsqueeze(1).to(h.dtype)              # (B, 1, L)
            pooled = (h * m).sum(-1) / m.sum(-1).clamp(min=1.0)
        return self.head(self.drop(pooled))                # (B, dim)


class ChangePredictor(nn.Module):
    """h -- predicts the gaze-rate embedding from the two frame embeddings alone.

    Takes [z_I0, z_IT, z_IT - z_I0]. The explicit difference is not in EgoDistill, but
    the task is literally "what changed between these frames", so handing the model the
    difference rather than making it learn subtraction is free.
    """

    def __init__(self, in_dim=384, hidden=1024, dim=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 3, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, z0, zT):
        """z0, zT: (B, in_dim) frozen frame features."""
        return self.mlp(torch.cat([z0, zT, zT - z0], dim=-1))   # (B, dim)


def build_models(feat_dim=384, dim=256, hidden=1024, width=128, dropout=0.1):
    """Both trainable modules. Nothing else is trained -- f_I stays on disk."""
    f_m = GazeRateEncoder(in_ch=3, width=width, dim=dim, dropout=dropout)
    h = ChangePredictor(in_dim=feat_dim, hidden=hidden, dim=dim, dropout=dropout)
    return f_m, h


def count_params(*modules):
    return sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)

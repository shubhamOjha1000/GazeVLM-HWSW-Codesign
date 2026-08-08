"""The PDF's architecture, assembled from the modules that already implement it.

    frame t-1 ──► [DINOv2 CLS, frozen] ──► z_I0 ─┐
                                                  ├─► h ──► ẑ_M ─┐
    frame t   ──► [DINOv2 CLS, frozen] ──► z_IT ─┘               │  L_NCE   (p2)
                                                                 │
    gaze rates (3, 9) ──► f_M ──► z_M ───────────────────────────┘
                                   │
                                   └──► SimilarityHead ──► [S_frame, S_gaze]   L_sim  (p3)

Nothing is reimplemented here. `GazeRateEncoder` and `ChangePredictor` come from
`src.loss1.models`, `SimilarityHead` from `src.loss2.models`; all three were written
against the slides and unit-tested. This file only wires them into one module so a single
optimiser sees every parameter and one checkpoint holds the whole thing.

The image encoder is absent by construction: features are precomputed on disk, so f_I is
frozen whether or not anyone remembers to freeze it. EgoDistill footnote 2 reports that
finetuning it collapses training.
"""

import torch
import torch.nn as nn

from src.loss1.models import ChangePredictor, GazeRateEncoder
from src.loss2.models import SimilarityHead


class GateModel(nn.Module):
    """f_M + h + the two-output similarity head, per PDF p2/p3/p9."""

    def __init__(self, feat_dim=384, dim=256, hidden=256, width=128, fusion=1024,
                 dropout=0.1, n_targets=2):
        super().__init__()
        self.f_m = GazeRateEncoder(in_ch=3, width=width, dim=dim, dropout=dropout)
        self.h = ChangePredictor(in_dim=feat_dim, hidden=fusion, dim=dim, dropout=dropout)
        self.head = SimilarityHead(dim=dim, hidden=hidden, n_targets=n_targets,
                                   dropout=dropout)
        self.dim, self.feat_dim = dim, feat_dim

    def forward(self, x, mask=None, z0=None, zT=None):
        """Returns (pred, z_M, ẑ_M). ẑ_M is None when frame features are not supplied,
        which is how --sim_only trains Loss 2 alone."""
        z_m = self.f_m(x, mask)
        pred = self.head(z_m)
        z_hat = self.h(z0, zT) if (z0 is not None and zT is not None) else None
        return pred, z_m, z_hat

    @torch.no_grad()
    def scores(self, x, mask=None):
        """[S_frame, S_gaze] in STANDARDISED units -- the caller un-standardises before
        thresholding, because tau lives in raw cosine units."""
        return self.head(self.f_m(x, mask))


def build(feat_dim=384, **kw):
    return GateModel(feat_dim=feat_dim, **kw)


def count_params(m):
    tot = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return dict(
        total=tot,
        f_m=sum(p.numel() for p in m.f_m.parameters()),
        h=sum(p.numel() for p in m.h.parameters()),
        head=sum(p.numel() for p in m.head.parameters()),
        # what actually ships: h exists only to create the Loss-1 target and is discarded
        # at inference, so the deployed cost is f_M + head
        deployed=sum(p.numel() for p in m.f_m.parameters())
        + sum(p.numel() for p in m.head.parameters()),
    )

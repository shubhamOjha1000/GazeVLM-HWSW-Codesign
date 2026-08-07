"""Loss-2 architecture: the similarity-prediction head.

The Loss-2 slide shows z_M passing through an "ML Module Encoder (Frame Similarity)" and
then a "Frame & Gaze Token Similarity Refinement" block carrying a loop marker, producing
a two-element vector compared against the DINOv2 teacher:

    z_M --> encoder --> refinement (loop) --> [S_frame, S_gaze]

The two targets are read off the inference slide, which puts exactly two heads on z_M --
one for a global frame similarity, one for a gaze-region similarity. Those are the
`frame_similarity` and `gaze_patch_token_sim` columns already in the feature CSV, so
Loss 2 needs no new data.

z_M comes from the SAME GazeRateEncoder that Loss 1 trains (src.loss1.models). That
sharing is what makes L = L_NCE + L_similarity a joint objective rather than two
independent models.
"""

import torch.nn as nn


class SimilarityHead(nn.Module):
    """z_M -> [S_frame, S_gaze].

    Two stages, matching the slide. The refinement stage is residual: it learns a
    correction to the encoded representation rather than replacing it, which is the
    natural reading of a refinement block and trains more stably than stacking plain
    layers.
    """

    def __init__(self, dim=256, hidden=256, n_targets=2, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.refine = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.out = nn.Linear(hidden, n_targets)

    def forward(self, z_m):
        h = self.encoder(z_m)
        h = h + self.refine(h)              # the slide's refinement loop
        return self.out(h)                  # (B, n_targets), in STANDARDISED units


def build_head(dim=256, hidden=256, n_targets=2, dropout=0.1):
    return SimilarityHead(dim, hidden, n_targets, dropout)

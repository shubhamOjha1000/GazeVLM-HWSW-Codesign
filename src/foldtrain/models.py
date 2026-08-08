"""The gate classifier: gaze rates -> quadrant (or SEND/DISCARD).

    gaze rates (B, 3, L) --> GazeRateEncoder (f_M) --> z_M (dim) --> head --> logits

`f_M` is the **same** `GazeRateEncoder` that Loss 1 and Loss 2 train, imported rather than
reimplemented. That is what makes a checkpoint from here comparable with those, and it
means an improvement to the encoder benefits all three objectives.

Note what is absent: no image encoder, no frame features, no DINOv2. The classifier sees
only the eye-velocity signal, which is exactly the constraint the deployed gate operates
under. A model that needed the pixels to decide whether to look at the pixels would be
useless.
"""

import torch
import torch.nn as nn

from src.loss1.models import GazeRateEncoder


class GazeClassifier(nn.Module):
    """f_M plus a small classification head.

    The head mirrors `src.loss2.models.SimilarityHead`: an encoder layer then a residual
    refinement block. Keeping the shape identical means a Loss-2 checkpoint can warm-start
    everything except the final projection.
    """

    def __init__(self, n_classes=2, in_ch=3, width=128, dim=256, hidden=256,
                 dropout=0.1):
        super().__init__()
        self.f_m = GazeRateEncoder(in_ch=in_ch, width=width, dim=dim, dropout=dropout)
        self.encoder = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.refine = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.out = nn.Linear(hidden, n_classes)
        self.n_classes = n_classes

    def forward(self, x, mask=None):
        h = self.encoder(self.f_m(x, mask))
        h = h + self.refine(h)
        return self.out(h)

    @torch.no_grad()
    def embed(self, x, mask=None):
        """z_M only -- useful for probing what the encoder learned."""
        return self.f_m(x, mask)


def build(n_classes=2, **kw):
    return GazeClassifier(n_classes=n_classes, **kw)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def load_encoder_from(model, ckpt_path, device="cpu"):
    """Warm-start f_M from a Loss-1 or Loss-2 checkpoint.

    Silently doing nothing when the key is missing would hide a typo'd path, so a missing
    `f_m` raises instead.
    """
    ck = torch.load(ckpt_path, map_location=device)
    if "f_m" not in ck:
        raise KeyError(f"{ckpt_path} has no 'f_m' state dict (keys: {list(ck)[:6]})")
    model.f_m.load_state_dict(ck["f_m"])
    return model

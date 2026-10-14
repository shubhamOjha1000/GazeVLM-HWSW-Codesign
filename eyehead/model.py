"""The eye + head gate: one small causal encoder per signal, fused per step, then pooled.

    eye_speed  (B,1,L) --> causal 1D CNN --> f_eye  (B,d,L) --+
    head_speed (B,1,L) --> causal 1D CNN --> f_head (B,d,L) --+-- concat on features
    cos        (B,1,L) --> causal 1D CNN --> f_cos  (B,d,L) --+   (B, 3d, L)
                                                                   |
                                    1x1 conv = the same MLP at every step (B, h, L)
                                                                   |
                       masked mean ++ masked max over real steps (B, 2h)
                                                                   |
                                                linear --> 4 class logits

Why per-step fusion and not pool-then-concat: "the eye spikes WHILE the head is still"
(REFIXATION) and "the eye spikes, then the head follows" (TRANSITION) differ only in timing
across streams. Pooling each stream first would throw that away. This is MARIO's
mid-level fusion (Sec. 3.3), which beat both raw early concatenation and cross-attention
in its Table 5.

Why mean AND max: a saccade is a spike in one step out of nine; a mean flattens it, a max
keeps it.

Causal: convolutions are padded on the left only, so step k never sees a later step and
the padded tail of a short window cannot leak into the real steps before the mask.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from eyehead.features import STREAM_CH


class CausalEnc(nn.Module):
    """(B, 1, L) -> (B, d, L): dilated causal convolutions, left padding only."""

    def __init__(self, d=16, k=3, dilations=(1, 2)):
        super().__init__()
        self.pads = [(k - 1) * dl for dl in dilations]
        self.convs = nn.ModuleList(
            nn.Conv1d(1 if i == 0 else d, d, k, dilation=dl)
            for i, dl in enumerate(dilations))

    def forward(self, x):
        for pad, conv in zip(self.pads, self.convs):
            x = F.relu(conv(F.pad(x, (pad, 0))))
        return x


class EyeHeadGate(nn.Module):
    def __init__(self, streams=("eye", "head", "cos"), d=16, h=32, k=3,
                 dilations=(1, 2), dropout=0.1, n_cls=4):
        super().__init__()
        bad = [s for s in streams if s not in STREAM_CH]
        if bad or not streams:
            raise ValueError(f"streams must be a non-empty subset of {list(STREAM_CH)}")
        self.streams = list(streams)
        self.ch = [STREAM_CH[s] for s in self.streams]
        self.enc = nn.ModuleList(CausalEnc(d, k, dilations) for _ in self.streams)
        self.fuse = nn.Sequential(nn.Conv1d(d * len(self.streams), h, 1), nn.ReLU())
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(2 * h, n_cls)

    def forward(self, X, mask):
        """X (B, 3, L) normalised [eye_speed, head_speed, cos]; mask (B, L) bool."""
        f = torch.cat([enc(X[:, c:c + 1]) for enc, c in zip(self.enc, self.ch)], dim=1)
        f = self.fuse(f)                                         # (B, h, L)
        m = mask[:, None, :].to(f.dtype)
        mean = (f * m).sum(-1) / m.sum(-1).clamp(min=1.0)
        mx = f.masked_fill(m == 0, -1e4).max(-1).values
        return self.cls(self.drop(torch.cat([mean, mx], dim=1)))


def count_params(model):
    return sum(p.numel() for p in model.parameters())

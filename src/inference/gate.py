"""Inference: the gating module.

Implements the Inference slide. At runtime **no frame is encoded and DINOv2 never runs** --
only the eye-velocity signal streams in from the tracker:

    gaze rates (L, 3) --> f_M --> z_M --> SimilarityHead --> [S_frame, S_gaze]
                                                                    |
                                                    FilterFrameForVLM(...) --> SEND / DISCARD

`f_M` and the head are exactly the modules trained by Loss 1 and Loss 2; nothing new is
learned here. This file is the deployment wrapper plus the decision rule.
"""

from dataclasses import dataclass

import numpy as np
import torch

from src.loss1.models import GazeRateEncoder
from src.loss2.models import build_head

SEND = "SEND"
DISCARD = "DISCARD"


def filter_frame_for_vlm(s_frame, threshold_frame, s_gaze, threshold_gaze):
    """The FilterFrameForVLM algorithm, transcribed from the Inference slide.

        IF S_frame > threshold_frame THEN
            IF S_gaze > threshold_gaze THEN   DISCARD I(T)
            ELSE                              SEND I(T) TO VLM
        ELSE                                  SEND I(T) TO VLM

    Returns (action, reason).
    """
    if s_frame > threshold_frame:
        if s_gaze > threshold_gaze:
            return DISCARD, "Frame not sent (High frame AND high gaze similarity)"
        return SEND, "Frame sent to VLM (High frame similarity, but low gaze similarity)"
    return SEND, "Frame sent to VLM (Low frame similarity)"


@dataclass
class Decision:
    action: str
    reason: str
    s_frame: float
    s_gaze: float

    @property
    def sent(self):
        return self.action == SEND


class GazeGate:
    """Runs the trained gaze encoder + similarity head, then applies the decision rule.

    Thresholds are NOT learned -- they trade compute against missed content, so they are
    set deliberately (see src/inference/calibrate.py) rather than fitted.
    """

    def __init__(self, f_m, head, rate_stats, target_stats,
                 threshold_frame=0.8, threshold_gaze=0.6, seq_len=9, device="cpu"):
        self.f_m = f_m.to(device).eval()
        self.head = head.to(device).eval()
        self.rate_mean, self.rate_std = (np.asarray(a, np.float32) for a in rate_stats)
        self.t_mean, self.t_std = (np.asarray(a, np.float32) for a in target_stats)
        self.threshold_frame = float(threshold_frame)
        self.threshold_gaze = float(threshold_gaze)
        self.seq_len = int(seq_len)
        self.device = device

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_checkpoint(cls, path, device="cpu", **kw):
        """Load a checkpoint written by `src.loss2.train` (it stores both modules)."""
        ck = torch.load(path, map_location=device)
        a = ck.get("args", {})
        f_m = GazeRateEncoder(3, a.get("width", 128), a.get("dim", 256), 0.0)
        head = build_head(a.get("dim", 256), a.get("hidden", 256), 2, 0.0)
        f_m.load_state_dict(ck["f_m"])
        head.load_state_dict(ck["head"])
        return cls(f_m, head, ck["rate_stats"], ck["target_stats"],
                   seq_len=a.get("seq_len", 9), device=device, **kw)

    # ------------------------------------------------------------------ scoring
    def _prepare(self, rates):
        """(L, 3) or (B, L, 3) -> standardised (B, 3, seq_len) plus a validity mask."""
        r = np.asarray(rates, dtype=np.float32)
        if r.ndim == 2:
            r = r[None]
        B, L = r.shape[0], self.seq_len
        buf = np.zeros((B, L, 3), np.float32)
        mask = np.zeros((B, L), bool)
        n = min(r.shape[1], L)
        if n:
            buf[:, :n] = (r[:, :n] - self.rate_mean) / self.rate_std
            mask[:, :n] = True
        return (torch.from_numpy(buf.transpose(0, 2, 1).copy()).to(self.device),
                torch.from_numpy(mask).to(self.device))

    @torch.no_grad()
    def scores(self, rates):
        """gaze rates -> (S_frame, S_gaze) in raw similarity units. No frame is touched."""
        x, m = self._prepare(rates)
        out = self.head(self.f_m(x, m)).cpu().numpy() * self.t_std + self.t_mean
        return out if np.asarray(rates).ndim == 3 else out[0]

    # ------------------------------------------------------------------ decision
    def decide(self, rates):
        s_frame, s_gaze = (float(v) for v in self.scores(rates))
        action, reason = filter_frame_for_vlm(
            s_frame, self.threshold_frame, s_gaze, self.threshold_gaze)
        return Decision(action, reason, s_frame, s_gaze)

    def decide_batch(self, rates):
        s = self.scores(np.asarray(rates))
        return [Decision(*filter_frame_for_vlm(float(a), self.threshold_frame,
                                               float(b), self.threshold_gaze),
                         float(a), float(b)) for a, b in s]

    # ------------------------------------------------------------------ cost
    def cost_report(self):
        """Parameters and multiply-accumulates per decision.

        Deliberately does NOT quote an encoder cost: the saving is simply that a skipped
        frame is never encoded, so the compute avoided equals whatever the image encoder
        or VLM would have cost. Reporting the gate's own cost makes the trade explicit.
        """
        p = sum(q.numel() for q in self.f_m.parameters()) + \
            sum(q.numel() for q in self.head.parameters())
        macs = 0
        for mod in self.f_m.net:
            if isinstance(mod, torch.nn.Conv1d):
                macs += mod.in_channels * mod.out_channels * mod.kernel_size[0] * self.seq_len
        for mod in list(self.head.encoder) + list(self.head.refine) + [self.head.out]:
            if isinstance(mod, torch.nn.Linear):
                macs += mod.in_features * mod.out_features
        macs += self.f_m.head.in_features * self.f_m.head.out_features
        return dict(params=p, macs_per_decision=macs)


class StreamingGate:
    """The runtime loop from the system-architecture slide.

    Gaze samples stream in continuously; frames arrive on a fixed interval. On each frame
    boundary the gate scores the gaze collected since the previous frame and decides.

        g = StreamingGate(gate)
        for t, yaw, pitch in eye_tracker:  g.push_gaze(t, yaw, pitch)
        d = g.on_frame(t_frame)            # SEND or DISCARD, no pixels involved
    """

    def __init__(self, gate, min_samples=2):
        self.gate = gate
        self.min_samples = int(min_samples)
        self._buf = []                 # (t_us, yaw, pitch) since the last frame
        self.n_seen = 0
        self.n_sent = 0

    def push_gaze(self, t_us, yaw, pitch):
        self._buf.append((float(t_us), float(yaw), float(pitch)))

    def rates_from_buffer(self):
        """Same computation as src.common.gaze_geometry.window_gaze_rates."""
        if len(self._buf) < self.min_samples:
            return np.zeros((0, 3), np.float32)
        a = np.asarray(self._buf, np.float64)
        ts, ya, pi = a[:, 0], a[:, 1], a[:, 2]
        dts = np.diff(ts) / 1e6
        dts = np.where(dts <= 0, 1e-6, dts)
        wy, wp = np.diff(ya) / dts, np.diff(pi) / dts
        return np.stack([wy, wp, np.hypot(wy, wp)], axis=1).astype(np.float32)

    def on_frame(self, t_us=None):
        """Decide for the frame closing the current window, then reset the buffer."""
        rates = self.rates_from_buffer()
        self._buf = self._buf[-1:]          # carry the last sample into the next window
        self.n_seen += 1
        if len(rates) == 0:                 # no gaze -> cannot judge, so do not discard
            d = Decision(SEND, "Frame sent to VLM (no gaze in window)", float("nan"),
                         float("nan"))
        else:
            d = self.gate.decide(rates)
        self.n_sent += int(d.sent)
        return d

    @property
    def keep_rate(self):
        return self.n_sent / self.n_seen if self.n_seen else float("nan")

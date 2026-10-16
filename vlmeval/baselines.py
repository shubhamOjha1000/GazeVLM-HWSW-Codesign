"""Content-based frame selectors on the 1 fps stream, and what each send/skip decision costs.

Every method sees the same candidate stream (one frame per second) and decides per frame
whether to send it to the VLM. They differ in what the decision looks at:

    Full (gaze + IMU)       policy D on the eye + head model: no pixels at all
    1 fps: all frames       sends everything (the ceiling; no decision)
    1 fps + pixel change    send when the frame differs from the LAST SENT frame by more than a
                            threshold in mean absolute grey-level difference (64 x 64 thumbnail)
    1 fps + DINOv2 change   send when the DINOv2 CLS cosine to the LAST SENT frame drops below
                            tau_frame: exactly the labels' own "frame changed" definition, but
                            it has to encode every frame to decide

Comparing against the last SENT frame (not the previous frame) is what a streaming selector
does: it asks "has the view moved on from what the VLM already has?", so slow drift is caught.
"""

import cv2
import numpy as np

from vlmeval.frames import frame_path

THUMB = 64


# ---------------------------------------------------------------------------- selectors
def _send_on_change(dist_to_last):
    """Greedy streaming rule: frame 0 is sent; frame t is sent when dist(t, last sent) says so.
    `dist_to_last(t, last)` returns True to send."""
    def run(n):
        sent, last = [0], 0
        for t in range(1, n):
            if dist_to_last(t, last):
                sent.append(t)
                last = t
        return sent
    return run


def thumbnails(meta, frame_nos):
    """Grey 64 x 64 thumbnails (float32, 0-255) of the given frames."""
    out = []
    for k in frame_nos:
        img = cv2.imread(frame_path(meta, int(k)), cv2.IMREAD_GRAYSCALE)
        out.append(cv2.resize(img, (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float32))
    return np.stack(out)


def consecutive_pixel_diffs(thumbs):
    return np.abs(np.diff(thumbs, axis=0)).mean(axis=(1, 2))


def pixel_select(fn, thumbs, thr):
    """Frame numbers sent by the pixel-change rule (thumbs aligned with fn)."""
    run = _send_on_change(lambda t, last: np.abs(thumbs[t] - thumbs[last]).mean() > thr)
    return [int(fn[i]) for i in run(len(fn))]


def dino_select(fn, cls, tau):
    """Frame numbers sent by the DINOv2-change rule (cls: L2-normalised, aligned with fn)."""
    run = _send_on_change(lambda t, last: float(cls[t] @ cls[last]) < tau)
    return [int(fn[i]) for i in run(len(fn))]


# ---------------------------------------------------------------------------- costs
def gate_decision_flops(d=16, h=32, k=3, dilations=(1, 2), streams=3, L=9, n_cls=4):
    """FLOPs of one EyeHeadGate decision (conv encoders + per-step fusion + classifier).
    The feature step (speeds, cos, a 2x3 rotation per IMU bin) adds a few hundred more."""
    macs = 0
    for s in range(streams):
        cin = 1
        for _ in dilations:
            macs += cin * d * k * L
            cin = d
    macs += streams * d * h * L          # 1x1 fusion at every step
    macs += 2 * h * n_cls                 # mean ++ max -> classes
    return 2 * macs


def vit_flops(img=98, patch=14, dim=384, depth=12, mlp=1536, chans=3):
    """FLOPs of one ViT forward (DINOv2 ViT-S/14 at the labels' 98 px input by default)."""
    n = (img // patch) ** 2 + 1
    per_layer = n * (4 * dim * dim + 2 * dim * mlp) + 2 * n * n * dim
    embed = (n - 1) * patch * patch * chans * dim
    return 2 * (depth * per_layer + embed)


def pixel_decision_flops(h=1408, w=1408):
    """Grey conversion + area resize to 64 x 64 + mean absolute difference, per frame."""
    return 3 * h * w + h * w + 3 * THUMB * THUMB


def vlm_costs(model, max_pixels, patch=14, merge=2):
    """Per sent frame: visual tokens the LLM sees, and approximate FLOPs of the vision
    encoder (2 x params x patches) plus the LLM prefill of those tokens (2 x params x
    tokens). Attention terms are ignored, so these are lower bounds."""
    side = int(round(max_pixels ** 0.5))
    patches = (side // patch) ** 2
    tokens = patches // (merge * merge)
    vis = getattr(model, "visual", None)
    if vis is None:                      # newer transformers nest it under model.model
        vis = getattr(getattr(model, "model", None), "visual", None)
    p_vis = sum(p.numel() for p in vis.parameters()) if vis is not None else 0
    p_all = sum(p.numel() for p in model.parameters())
    return dict(tokens_per_frame=tokens, vision_params=p_vis, llm_params=p_all - p_vis,
                vision_flops_per_frame=2 * p_vis * patches,
                llm_prefill_flops_per_frame=2 * (p_all - p_vis) * tokens)

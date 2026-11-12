"""Total FLOPs per question, for matching every row on compute instead of tokens (runbook section 14).

A question's FLOPs are what answering it costs from the decoded frames, as torch.utils.flop_counter counts them (matrix products,
convolutions and attention):

    gate        the eye + head gate on the whole video, the budget row only: 5 fold models x one decision's FLOPs x the video's
                decisions. Every video is counted as the 5-model ensemble, the most it can cost.
    vision      frames encoded x one encoded frame's FLOPs (a stock 384 px frame, or ours' k = 1 fovea), x 2 for the two-view
                GazeVLM, which encodes each frame and its gaze crop. A baseline encodes all its input frames, even those it then
                leaves out of the prompt.
    prefill     the LLM's one forward over the prompt, c0 + c1 n + c2 n^2 for n tokens. The accuracy tables read the first answer
                token's logits, so nothing is decoded.

The baselines' own work between the encoder and the LLM (similarities, clustering, saliency) is counted as zero, which favours them.
The calibrate job measures it on real questions and stops if it is more than VISION_TOLERANCE of their vision FLOPs.

Everything here is arithmetic on measured numbers, except the measure_* functions, which need the loaded model.
"""

import numpy as np
import torch

GATE_MODELS = 5                  # fold models in the test ensemble: every video's gate is counted at that
VISION_TOLERANCE = 0.02          # the baselines' own work, as a share of their vision FLOPs, before the calibrate job stops
EXACT_TOLERANCE = 1e-4           # the same encoder or the same LLM call as the units were measured on: must match this closely
NEAREST_ONLY_PTS = 0.5           # a row further than this from its target only reached the nearest frame count (as calibrate.py)
TWO_VIEWS = ("gazevlm2",)        # rows that encode 2 images per frame
FOVEA_ROWS = ("budget", "nogate")
GATE_ROWS = ("budget",)


# ---------------------------------------------------------------------------- counting
def count_flops(fn):
    """fn() under torch.utils.flop_counter -> (FLOPs, fn's result)."""
    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False)
    with fc:
        out = fn()
    return int(fc.get_total_flops()), out


def prefill(c, n):
    """The LLM's FLOPs on a prompt of n tokens (n may be an array)."""
    n = np.asarray(n, dtype=np.float64)
    return c[0] + c[1] * n + c[2] * n * n


def fit_prefill(points):
    """[(n tokens, FLOPs)]: the first 3 fix c0, c1, c2 exactly, and every point must then agree within EXACT_TOLERANCE, or the job
    stops (the LLM's cost wouldn't be a quadratic in the prompt's length, and nothing here would hold) -> [c0, c1, c2]."""
    if len(points) < 4 or len({n for n, _ in points}) != len(points):
        raise ValueError(f"need 4 or more distinct lengths, got {[n for n, _ in points]}")
    fit = points[:3]
    a = np.array([[1.0, float(n), float(n) * n] for n, _ in fit])
    c = [float(x) for x in np.linalg.solve(a, np.array([float(f) for _, f in fit]))]
    off = [(n, f, float(prefill(c, n))) for n, f in points if abs(float(prefill(c, n)) - f) > EXACT_TOLERANCE * f]
    if off:
        raise SystemExit(f"the LLM's FLOPs aren't a quadratic in the prompt length: (n, counted, fitted) {off}. Stop and tell the user.")
    return c


def fit_prompt(samples):
    """[(frames >= 1, tokens of the video prompt with no image token - tokens of the blind prompt)] -> dict(c0, o, max_residual):
    a video prompt is its blind prompt + c0 (the intro) + o per frame (its time label and image markers) + its image tokens."""
    f = np.array([s[0] for s in samples], dtype=np.float64)
    y = np.array([s[1] for s in samples], dtype=np.float64)
    if len(samples) < 2 or len(set(f)) < 2:
        raise ValueError("need samples at 2 or more frame counts")
    (c0, o), *_ = np.linalg.lstsq(np.stack([np.ones_like(f), f], 1), y, rcond=None)
    return dict(c0=float(c0), o=float(o), max_residual=float(np.abs(y - (c0 + o * f)).max()))


def prompt_tokens(blind, frames, image_tokens, prompt):
    """Prompt length per question from its blind prompt's length, its frames and its image tokens (arrays)."""
    blind, frames, image_tokens = (np.asarray(x, dtype=np.float64) for x in (blind, frames, image_tokens))
    return blind + np.where(frames > 0, prompt["c0"] + prompt["o"] * frames, 0.0) + image_tokens


def question_flops(units, frames, unit, n_tokens, decisions=0, views=1):
    """Total FLOPs per question (arrays): gate + vision + prefill. unit: 'v_stock' or 'v_fovea'."""
    gate = GATE_MODELS * units["gate_per_decision"] * np.asarray(decisions, dtype=np.float64)
    return gate + views * np.asarray(frames, dtype=np.float64) * units[unit] + prefill(units["prefill"], n_tokens)


def encoded(video_frames, k):
    """Frames a Uniform K input encodes on each video: common.uniform_indices gives min(K, frames)."""
    return np.minimum(int(k), np.maximum(np.asarray(video_frames, dtype=np.int64), 0))


# ---------------------------------------------------------------------------- frame counts on compute
def uniform_share(q, units, prompt, ref_total, k, tokens_per_frame, views=1):
    """FLOPs share (of ref_total) of Uniform K input with tokens_per_frame tokens per frame and views encodes per frame.
    q: blind (tokens of each question's blind prompt) and video_frames (frames of its video), one entry per question."""
    f = encoded(q["video_frames"], k)
    n = prompt_tokens(q["blind"], f, tokens_per_frame * f, prompt)
    return float(question_flops(units, f, "v_stock", n, views=views).sum() / ref_total)


def choose_k(share_of_k, target, k_max):
    """The frame count K in 1..k_max whose FLOPs share comes closest to target, the smaller on a tie -> (K, share)."""
    best = None
    for k in range(1, int(k_max) + 1):
        s = share_of_k(k)
        if best is None or abs(s - target) < abs(best[1] - target):
            best = (k, s)
    return best


# ---------------------------------------------------------------------------- measuring (needs the model)
def measure_gate(model, seq_len, channels=3):
    """FLOPs of one gate decision: one window of seq_len samples through one fold model."""
    x = torch.zeros(1, channels, int(seq_len))
    m = torch.ones(1, int(seq_len), dtype=torch.bool)
    with torch.no_grad():
        return count_flops(lambda: model(x, m))[0]


def measure_gate_ckpt(path):
    """measure_gate on a fold checkpoint, on the CPU (the count doesn't depend on the device)."""
    from eyehead.infer import load_ckpt, model_from_ckpt
    ck = load_ckpt(path, "cpu")
    return measure_gate(model_from_ckpt(ck, "cpu"), ck["seq_len"])


def measure_prefill_smolvlm(ad, n):
    """FLOPs of SmolVLM's answer forward (smolvlm_fovea.letter_probs, the call every row scores with) on a prompt of n tokens."""
    from vlmeval import smolvlm_fovea as sm
    w = ad.model.get_input_embeddings().weight
    x = torch.zeros(1, int(n), w.shape[1], dtype=w.dtype, device=w.device)
    inputs = dict(inputs_embeds=x, attention_mask=torch.ones((1, int(n)), dtype=torch.long, device=w.device))
    with torch.no_grad():
        return count_flops(lambda: sm.letter_probs(ad.model, inputs, ad.letter_ids))[0]


def prompt_len_smolvlm(ad, question, options, frames, image_tokens):
    """Tokens of the prompt SmolVLM's rows build for these frames with these image tokens per frame (smolvlm_fovea.run_questions)."""
    from accuracy_tables.models import build_messages
    from vlmeval import smolvlm_fovea as sm
    msgs = build_messages(question, options, [(fr["path"], fr["t"]) for fr in frames], sm.IMG * sm.IMG)
    return int(sm.prompt_ids(ad.proc, msgs, [int(t) for t in image_tokens], sm.TOKENIZE["add_special_tokens"]).shape[1])


MEASURE = {"smolvlm": dict(prefill=measure_prefill_smolvlm, prompt_len=prompt_len_smolvlm)}

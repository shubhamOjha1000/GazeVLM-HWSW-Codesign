"""Total FLOPs per question, for matching every row on compute instead of tokens (runbook sections 14 and 15).

A question's FLOPs are what answering it costs from the decoded frames, as torch.utils.flop_counter counts them (matrix products,
convolutions and attention):

    gate        the eye + head gate on the whole video, the budget row only: 5 fold models x one decision's FLOPs x the video's
                decisions. Every video is counted as the 5-model ensemble, the most it can cost.
    vision      frames encoded x one encoded frame's FLOPs (a stock frame, or ours' smallest fovea), x 2 for the two-view
                GazeVLM, which encodes each frame and its gaze crop. A baseline encodes all its input frames, even those it then
                leaves out of the prompt. On Qwen a foveated frame's cost depends on its layout, which follows the gaze (its cell
                count, and on Qwen2.5-VL how its patches fall into the 112 px attention windows), so ours' and the no-gate
                control's vision is the sum over each video's own frames of what its layout costs (fit_vision).
    prefill     the LLM's one forward over the prompt, c0 + c1 n + c2 n^2 for n tokens. The accuracy tables read the first answer
                token's logits, so nothing is decoded.

The baselines' own work between the encoder and the LLM (similarities, clustering, saliency) is counted as zero, which favours them.
The calibrate job measures it on real questions and stops if it is more than VISION_TOLERANCE of their vision FLOPs.

Everything here is arithmetic on measured numbers, except the measure_* functions, which need the loaded model.
"""

import contextlib

import numpy as np
import torch

GATE_MODELS = 5                  # fold models in the test ensemble: every video's gate is counted at that
VISION_TOLERANCE = 0.02          # the baselines' own work, as a share of their vision FLOPs, before the calibrate job stops
EXACT_TOLERANCE = 1e-4           # the same encoder or the same LLM call as the units were measured on: must match this closely
NEAREST_ONLY_PTS = 0.5           # a row further than this from its target only reached the nearest frame count (as calibrate.py)
TWO_VIEWS = ("gazevlm2",)        # rows that encode 2 images per frame
FOVEA_ROWS = ("budget", "nogate")
GATE_ROWS = ("budget",)
LLM_SHAPE_TOLERANCE = 0.01       # the fitted n^2 term against 4 x layers x heads x head_dim from the config, before the job stops
ATTENTION_WORDS = ("attention", "attn", "sdpa", "flash")
VISION_FIT_GAZES = ((0.5, 0.5), (0.12, 0.18), (0.88, 0.3), (0.35, 0.92), (0.7, 0.66), (0.03, 0.97), (0.95, 0.05))


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


def question_flops_vision(units, vision, n_tokens, decisions=0):
    """Total FLOPs per question (arrays) when its vision FLOPs are known per question (Qwen's foveated rows)."""
    gate = GATE_MODELS * units["gate_per_decision"] * np.asarray(decisions, dtype=np.float64)
    return gate + np.asarray(vision, dtype=np.float64) + prefill(units["prefill"], n_tokens)


def fit_vision(points):
    """[(layout features, counted FLOPs)] -> coefficients, so that one frame's vision FLOPs = features . coefficients. Least squares,
    which must use every feature and reproduce every counted point within EXACT_TOLERANCE, or the job stops."""
    x = np.array([f for f, _ in points], dtype=np.float64)
    y = np.array([v for _, v in points], dtype=np.float64)
    coef, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < x.shape[1]:
        raise SystemExit(f"the vision layouts don't tell the {x.shape[1]} terms apart (rank {rank}). Stop and tell the user.")
    off = [(list(f), v, float(np.dot(f, coef))) for f, v in points if abs(float(np.dot(f, coef)) - v) > EXACT_TOLERANCE * v]
    if off:
        raise SystemExit(f"the vision encoder's FLOPs aren't linear in its patches and attention sizes: (features, counted, fitted) "
                         f"{off}. Stop and tell the user.")
    return [float(c) for c in coef]


def llm_shape_check(c, shape):
    """The fitted prefill against the LLM's config -> dict(want_c, c_off, want_b, b_off): c should be 4 x layers x heads x head_dim
    (the counter's formula for attention over the whole prompt) and b about 2 x the non-embedding parameters."""
    want_c = 4 * shape["layers"] * shape["heads"] * shape["head_dim"]
    want_b = 2 * shape["non_embedding_params"]
    return dict(want_c=want_c, c_off=c[2] / want_c - 1, want_b=want_b, b_off=c[1] / want_b - 1 if want_b else float("nan"))


# ---------------------------------------------------------------------------- attention the counter can't see
class AttentionWatch:
    """with AttentionWatch() as w: <counting code> -> w.missed: attention ops that ran without a FLOP formula in
    torch.utils.flop_counter, so a FlopCounterMode around them left their FLOPs out. Ops that never reach torch's dispatcher can't
    be seen here: the prefill's n^2 term against the config (llm_shape_check) covers those on the LLM."""

    def __init__(self):
        self.missed = set()

    def __enter__(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils.flop_counter import flop_registry
        missed = self.missed

        class _Watch(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                packet = getattr(func, "_overloadpacket", func)
                if packet not in flop_registry and any(w in str(packet).lower() for w in ATTENTION_WORDS):
                    missed.add(str(packet))
                return func(*args, **(kwargs or {}))

        self._mode = _Watch()
        self._mode.__enter__()
        return self

    def __exit__(self, *exc):
        return self._mode.__exit__(*exc)


def module_configs(model):
    """Every config object the model's modules read: its own and its text and vision sub-configs."""
    out = {}
    for m in model.modules():
        c = getattr(m, "config", None)
        if c is not None and hasattr(c, "_attn_implementation"):
            out[id(c)] = c
    return list(out.values())


@contextlib.contextmanager
def attention_implementation(model, impl):
    """The model's attention implementation set to impl inside, then put back. None changes nothing. Only counting passes use it:
    eager attention gives the same FLOP formula as the fused kernels, as plain matrix products the counter sees."""
    if impl is None:
        yield
        return
    configs = module_configs(model)
    old = [c._attn_implementation for c in configs]
    try:
        for c in configs:
            c._attn_implementation = impl
        yield
    finally:
        for c, o in zip(configs, old):
            c._attn_implementation = o


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
        if best is None or abs(s - target) < abs(best[1] - target) - 1e-12:
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


def llm_shape(ad):
    """The LLM's layers, attention heads, head size and non-embedding parameters, read from the loaded model."""
    from system_perf import probe
    text = getattr(ad.model.config, "text_config", None) or ad.model.config
    heads = int(text.num_attention_heads)
    head_dim = int(getattr(text, "head_dim", None) or text.hidden_size // heads)
    decoder = probe.find_decoder(ad.model)
    params = sum(p.numel() for n, p in decoder.named_parameters() if "embed" not in n)
    return dict(layers=int(text.num_hidden_layers), heads=heads, head_dim=head_dim, non_embedding_params=int(params))


# ---------------------------------------------------------------------------- Qwen2.5-VL and Qwen3-VL
def _zero_prompt(ad, n):
    """inputs_embeds of n zero tokens with text M-RoPE positions (3, 1, n), as the scored rows pass them."""
    w = ad.model.get_input_embeddings().weight
    pos = torch.arange(int(n), dtype=torch.float32, device=w.device).view(1, 1, -1).expand(3, 1, -1)
    return dict(inputs_embeds=torch.zeros(1, int(n), w.shape[1], dtype=w.dtype, device=w.device), position_ids=pos,
                attention_mask=torch.ones((1, int(n)), dtype=torch.long, device=w.device))


def measure_prefill_qwen25(ad, n):
    """FLOPs of Qwen2.5-VL's answer forward (qwen_fovea.letter_probs, the call every row scores with) on n tokens."""
    from vlmeval import qwen_fovea as qf
    inputs = _zero_prompt(ad, n)
    with torch.no_grad():
        return count_flops(lambda: qf.letter_probs(ad.model, inputs, ad.letter_ids))[0]


def measure_prefill_qwen3(ad, n):
    """FLOPs of Qwen3-VL's answer forward (qwen3_fovea.letter_probs: its text model, DeepStack slots empty, and the head) on n tokens."""
    from vlmeval import qwen3_fovea as q3
    inputs = dict(_zero_prompt(ad, n), visual_pos_masks=None, deepstack_visual_embeds=None)
    with torch.no_grad():
        return count_flops(lambda: q3.letter_probs(ad.model, inputs, ad.letter_ids))[0]


def prompt_len_qwen(ad, question, options, frames, image_tokens):
    """Tokens of the prompt the Qwen rows build for these frames with these image tokens per frame (qwen_fovea / qwen3_fovea
    build_inputs, and the baselines' backends, all tokenize this way). Each frame is its own image: one image pad per token."""
    from accuracy_tables.models import build_messages
    from vlmeval import qwen_fovea as qf
    msgs = build_messages(question, options, [(fr["path"], fr["t"]) for fr in frames], qf.IMG * qf.IMG)
    text = ad.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    text = qf.expand_image_pads(text, [int(t) for t in image_tokens])
    return int(ad.proc.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1])


def features_qwen25(cells, img):
    """What a Qwen2.5-VL frame's encoder FLOPs can depend on: its patches P (4 per cell: linear layers, patch embedding, merger),
    the sum of squared patch counts of its 112 px attention windows (window-attention blocks) and P^2 (full-attention blocks)."""
    from vlmeval import qwen_fovea as qf
    cells = np.asarray(cells)
    _, cu = qf.window_groups(cells, int(img))
    windows = np.diff(np.asarray(cu, dtype=np.float64))
    p = 4.0 * len(cells)
    return [p, float((windows ** 2).sum()), p * p]


def features_qwen3(cells, img):
    """Qwen3-VL's encoder attends over the whole frame in every block: its patches P and P^2."""
    p = 4.0 * len(cells)
    return [p, p * p]


def vision_qwen25(ad, cells, img):
    """Counted FLOPs of Qwen2.5-VL's encoder (qwen_fovea.encode_cells, as every row runs it) on one frame of this layout."""
    from vlmeval import qwen_fovea as qf
    ip = ad.proc.image_processor
    x = torch.zeros(3, int(img), int(img))
    return count_flops(lambda: qf.encode_cells(qf.visual_of(ad.model), x, np.asarray(cells), int(img), ip.image_mean, ip.image_std))[0]


def vision_qwen3(ad, cells, img):
    """Counted FLOPs of Qwen3-VL's encoder with its DeepStack mergers (qwen3_fovea.encode_cells) on one frame of this layout."""
    from vlmeval import qwen3_fovea as q3
    ip = ad.proc.image_processor
    x = torch.zeros(3, int(img), int(img))
    return count_flops(lambda: q3.encode_cells(q3.parts(ad.model)[0], x, np.asarray(cells), int(img), ip.image_mean, ip.image_std))[0]


def layout_qwen25(conditions, cond, fr):
    from vlmeval import qwen_fovea as qf
    return qf.frame_layout(conditions, cond, fr)[:2]


def layout_qwen3(conditions, cond, fr):
    from vlmeval import qwen3_fovea as q3
    return q3.frame_layout(conditions, cond, fr)[:2]


def stock_qwen25():
    from vlmeval import qwen_fovea as qf
    from accuracy_tables.models import Qwen25
    return qf.uniform_cells(Qwen25.SIDE), Qwen25.SIDE


def stock_qwen3():
    from vlmeval import qwen3_fovea as q3
    return q3.uniform_cells(q3.IMG), q3.IMG


# Per model: prefill(ad, n) and prompt_len(...) always; llm_shape(ad) for the check against the config; and, where a foveated frame's
# cost follows its layout, layout(conditions, cond, frame) -> (cells, side), stock() -> the stock layout, features(cells, side) and
# vision(ad, cells, side) -> counted FLOPs.
MEASURE = {
    "smolvlm": dict(prefill=measure_prefill_smolvlm, prompt_len=prompt_len_smolvlm, llm_shape=llm_shape),
    "qwen2.5": dict(prefill=measure_prefill_qwen25, prompt_len=prompt_len_qwen, llm_shape=llm_shape, layout=layout_qwen25,
                    stock=stock_qwen25, features=features_qwen25, vision=vision_qwen25),
    "qwen3": dict(prefill=measure_prefill_qwen3, prompt_len=prompt_len_qwen, llm_shape=llm_shape, layout=layout_qwen3,
                  stock=stock_qwen3, features=features_qwen3, vision=vision_qwen3),
}

"""Instrumentation for one scored question, without changing any model or method code.

  Probe        forward hooks on the model's language-model stack (its decoder layers' module): the first call is
               the prefill, so everything before it is the vision side (preprocessing + vision encoder + the
               method's compression + prompt assembly) and the call itself is the prefill. Records the times, the
               vision-side peak memory, the FLOPs counted so far (under a FlopCounterMode) and the decoder's
               prefill inputs, which decode_benchmark reuses.
  decode_benchmark   prefill with a KV cache from those exact inputs, then greedy decoding (EOS ignored): TPS and
               the decoder's peak memory (activations + KV cache).
  expand_batch the prefill inputs repeated to a batch of B (memory vs batch size).
  Energy       GPU energy over a window from NVML's energy counter (mJ since driver load; Volta and newer), or
               sampled power where that counter is missing; None without NVML (e.g. Jetson: use tegrastats).
  FrameCache   decoded frames kept in memory, so timings exclude disk reads and JPEG decoding (the protocol:
               from decoded frames in memory): cv2.imread and PIL.Image.open are memoized for paths under root.
"""

import contextlib
import threading
import time

import torch


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mem_allocated():
    return torch.cuda.memory_allocated() if torch.cuda.is_available() else 0


def mem_peak():
    return torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def find_decoder(model):
    """The module that holds the language model's decoder layers (Qwen / InternVL / Gemma: model.language_model,
    SmolVLM: model.text_model, Phi-3 and plain causal LMs: model)."""
    for path in ("model.language_model", "model.text_model", "language_model", "model"):
        m = model
        try:
            for a in path.split("."):
                m = getattr(m, a)
        except AttributeError:
            continue
        if hasattr(m, "layers"):
            return m
    raise AttributeError(f"no decoder stack found in {type(model).__name__}")


def lm_head(model):
    head = getattr(model, "lm_head", None)
    return head if head is not None else model.get_output_embeddings()


class Probe:
    """Hooks on the decoder stack; reset() before each measured question."""

    def __init__(self, decoder, flop_counter=None):
        self.decoder, self.flop_counter, self.handles = decoder, flop_counter, []
        self.reset()

    def reset(self):
        self.calls, self.t_start, self.t_end, self.kwargs, self.ve_peak, self.flops_before = 0, None, None, None, None, None

    def _pre(self, module, args, kwargs):
        if self.calls == 0:
            sync()
            self.t_start = time.perf_counter()
            self.ve_peak = mem_peak()
            reset_peak()
            self.kwargs = dict(kwargs)
            if args:
                self.kwargs["_args"] = args
            if self.flop_counter is not None:
                self.flops_before = self.flop_counter.get_total_flops()
        self.calls += 1

    def _post(self, module, args, kwargs, output):
        if self.t_end is None:
            sync()
            self.t_end = time.perf_counter()

    def __enter__(self):
        self.handles = [self.decoder.register_forward_pre_hook(self._pre, with_kwargs=True),
                        self.decoder.register_forward_hook(self._post, with_kwargs=True)]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


def _hidden(out):
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


@torch.no_grad()
def decode_benchmark(model, decoder, kwargs, n_tokens=64):
    """Prefill (with a KV cache) from the decoder's captured prefill inputs, then greedy decoding of n_tokens - 1 more
    tokens, EOS ignored. Positions continue from the prompt's last: M-RoPE (3, B, L) and 1D (B, L) position ids
    advance by one per token; without position ids the cache position does it.
    -> dict(prefill_s, decode_s, tps (decoded tokens / s), tokens)."""
    kw = {k: v for k, v in kwargs.items() if k not in ("_args", "past_key_values")}
    kw["use_cache"] = True
    head, emb = lm_head(model), decoder.get_input_embeddings()
    sync()
    t0 = time.perf_counter()
    out = decoder(**kw)
    tok = head(_hidden(out)[:, -1:]).argmax(-1)
    sync()
    t1 = time.perf_counter()
    past, tokens = out.past_key_values, [int(tok[0, 0])]
    pos, cpos = kw.get("position_ids"), kw.get("cache_position")
    for i in range(n_tokens - 1):
        step = dict(inputs_embeds=emb(tok), past_key_values=past, use_cache=True)
        if pos is not None:
            step["position_ids"] = pos[..., -1:] + 1 + i
        if cpos is not None:
            step["cache_position"] = cpos[-1:] + 1 + i
        out = decoder(**step)
        past = out.past_key_values
        tok = head(_hidden(out)[:, -1:]).argmax(-1)
        tokens.append(int(tok[0, 0]))
    sync()
    t2 = time.perf_counter()
    n = n_tokens - 1
    return dict(prefill_s=t1 - t0, decode_s=t2 - t1, tps=n / (t2 - t1) if n else float("nan"), tokens=tokens)


def expand_batch(kwargs, B):
    """The decoder's prefill inputs repeated B times along the batch: inputs_embeds (1, L, D), position ids (3, 1, L)
    or (1, L), Qwen3-VL's visual_pos_masks (1, L) and its DeepStack features (flattened over the image tokens, so
    concatenated B times); the attention mask is dropped (all positions attend causally, as without padding)."""
    out = {}
    for k, v in kwargs.items():
        if k in ("_args", "past_key_values", "attention_mask"):
            continue
        if k == "inputs_embeds":
            v = v.expand(B, *v.shape[1:]).contiguous()
        elif k == "position_ids" and v is not None:
            v = v.expand(v.shape[0], B, v.shape[-1]).contiguous() if v.dim() == 3 else v.expand(B, v.shape[-1]).contiguous()
        elif k == "visual_pos_masks" and v is not None:
            v = v.expand(B, v.shape[-1]).contiguous()
        elif k == "deepstack_visual_embeds" and v is not None:
            v = [torch.cat([d] * B) for d in v]
        out[k] = v
    return out


class Energy:
    """with Energy() as e: ... -> e.joules, e.watts (None without NVML)."""

    def __init__(self, device=None):
        self.joules = self.watts = None
        self.h = self._mode = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(device if device is not None else torch.cuda.current_device())
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
                self._mode = "energy"
            except Exception:
                self._mode = "power"
        except Exception:
            self.nv = None

    def __enter__(self):
        sync()
        self.t0 = time.perf_counter()
        if self._mode == "energy":
            self.e0 = self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h)
        elif self._mode == "power":
            self.samples, self._stop = [], threading.Event()
            self._th = threading.Thread(target=self._sample, daemon=True)
            self._th.start()
        return self

    def _sample(self):
        while not self._stop.is_set():
            self.samples.append(self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0)
            time.sleep(0.02)

    def __exit__(self, *exc):
        sync()
        dt = time.perf_counter() - self.t0
        if self._mode == "energy":
            self.joules = (self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) - self.e0) / 1000.0
            self.watts = self.joules / dt if dt > 0 else None
        elif self._mode == "power":
            self._stop.set()
            self._th.join()
            if self.samples:
                self.watts = sum(self.samples) / len(self.samples)
                self.joules = self.watts * dt


class FrameCache:
    """Memoizes cv2.imread and PIL.Image.open for files under root (the decoded frames); other paths (e.g. the
    two-view GazeVLM's temporary crop) are read as usual. Returns copies, so callers can modify them."""

    def __init__(self, root):
        self.root, self.arrays, self.images = str(root), {}, {}

    def clear(self):
        self.arrays.clear()
        self.images.clear()

    def preload(self, paths):
        """Decode these frames once (both readers), inside the context, so that no timed run reads the disk."""
        from PIL import Image
        for p in paths:
            Image.open(p)
            if self._cv2 is not None:
                self._cv2.imread(p)

    def __enter__(self):
        from PIL import Image
        self._open = Image.open
        cache = self

        def open_(fp, *a, **k):
            if isinstance(fp, str) and fp.startswith(cache.root):
                if fp not in cache.images:
                    im = cache._open(fp, *a, **k)
                    im.load()
                    cache.images[fp] = im
                return cache.images[fp].copy()
            return cache._open(fp, *a, **k)

        Image.open = open_
        try:
            import cv2
            self._imread, self._cv2 = cv2.imread, cv2

            def imread(path, *a, **k):
                if isinstance(path, str) and path.startswith(cache.root):
                    key = (path, a, tuple(sorted(k.items())))
                    if key not in cache.arrays:
                        cache.arrays[key] = cache._imread(path, *a, **k)
                    v = cache.arrays[key]
                    return None if v is None else v.copy()
                return cache._imread(path, *a, **k)

            cv2.imread = imread
        except ImportError:
            self._cv2 = None
        return self

    def __exit__(self, *exc):
        from PIL import Image
        Image.open = self._open
        if self._cv2 is not None:
            self._cv2.imread = self._imread
        self.clear()


@contextlib.contextmanager
def flop_counter():
    """A FlopCounterMode that counts every aten op it knows (matmuls, convolutions, attention) without printing."""
    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False)
    with fc:
        yield fc

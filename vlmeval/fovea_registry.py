"""The foveation run of each VLM notebook: its result file and condition names (exactly as the
notebooks write them), so that any notebook can put the VLMs side by side."""

import os

import numpy as np
import pandas as pd

from vlmeval import egoev

# Ring settings (R, a) of the models whose layouts come from the 448 px layout (Qwen2.5-VL, Gemma-3,
# InternVL3.5, Qwen3-VL; Phi-3.5 on its 672 px view); label = tokens per frame on the 448 px grid.
RINGS = [dict(R=70, a=160, label=48), dict(R=180, a=40, label=96),
         dict(R=180, a=160, label=150), dict(R=240, a=160, label=205)]
# SmolVLM2: k of its 3 x 3 coarse cells at full resolution, 9 + 8 k tokens of 81.
SMOL_K = [dict(k=1, label=17), dict(k=3, label=33), dict(k=5, label=49), dict(k=7, label=65)]


def phi_fovea(p, src="gaze"):
    return f"D + Phi fovea {src}, rings ~{p['label']} (R={p['R']}, a={p['a']})"


def phi_uniform(p):
    return f"D + Phi uniform crops, same crop tokens as rings ~{p['label']}"


PHI_STOCK = "D + Phi stock num_crops=4 (757 tokens)"


def smol_fovea(p, src="gaze"):
    return f"D + SmolVLM fovea {src}, k={p['k']} (~{p['label']} tok)"


def smol_uniform(p):
    return f"D + SmolVLM uniform downscaled, same tokens ~{p['label']}"


SMOL_STOCK = "D + SmolVLM stock 384 (81 tokens)"


def runs(vlm_dir, tag):
    """{model: (result file, [fovea condition per budget level], [uniform ...], stock condition)};
    tag = the notebooks' POLICY_TAG (D, DP, DPall)."""
    path = lambda name: os.path.join(vlm_dir, f"{name}_{tag}_fovea.jsonl")
    iv = next((path(n) for n in ("InternVL3_5-4B-HF", "InternVL3_5-4B-hf") if os.path.exists(path(n))), "")
    R = RINGS
    return {
        "Qwen2.5-VL-3B": (path("Qwen2.5-VL-3B-Instruct"),
                          [f"D + fovea@448 gaze, ~{p['label']} tok (R={p['R']:g}, a={p['a']:g})" for p in R],
                          [f"D + uniform, same tokens as fovea@448 ~{p['label']}" for p in R],
                          "D + uniform 448 (256 tokens)"),
        "Gemma-3-4B": (path("gemma-3-4b-it"),
                       [f"D + Gemma fovea gaze, ~{p['label']} tok (R={p['R']}, a={p['a']})" for p in R],
                       [f"D + Gemma uniform pooled, same tokens ~{p['label']}" for p in R],
                       "D + Gemma stock (256 tokens)"),
        "InternVL3.5-4B": (iv,
                           [f"D + InternVL fovea gaze, ~{p['label']} tok (R={p['R']}, a={p['a']})" for p in R],
                           [f"D + InternVL uniform downscaled, same tokens ~{p['label']}" for p in R],
                           "D + InternVL stock 448 (256 tokens)"),
        "Qwen3-VL-4B": (path("Qwen3-VL-4B-Instruct"),
                        [f"D + Qwen3 fovea gaze, ~{p['label']} tok (R={p['R']}, a={p['a']})" for p in R],
                        [f"D + Qwen3 uniform downscaled, same tokens ~{p['label']}" for p in R],
                        "D + Qwen3 stock 512 (256 tokens)"),
        "Phi-3.5-vision": (path("Phi-3.5-vision-instruct"),
                           [phi_fovea(p) for p in R], [phi_uniform(p) for p in R], PHI_STOCK),
        "SmolVLM2-2.2B": (path("SmolVLM2-2.2B-Instruct"),
                          [smol_fovea(p) for p in SMOL_K], [smol_uniform(p) for p in SMOL_K], SMOL_STOCK),
    }


def side_by_side(vlm_dir, tag, videos):
    """(accuracy %, visual tokens per frame) of every model whose result file exists: rows stock,
    fovea 1-4, uniform 1-4 (budget levels, smallest first), one column per model."""
    acc, tok = {}, {}
    for name, (path, fov, uni, stock) in runs(vlm_dir, tag).items():
        if not path or not os.path.exists(path):
            continue
        r = egoev.load_results(path)
        r = r[r.video.isin(videos)]
        if "error" in r:
            r = r[r["error"].isna()]
        rows = ([("stock", stock)] + [(f"fovea {i + 1}", c) for i, c in enumerate(fov)]
                + [(f"uniform {i + 1}", c) for i, c in enumerate(uni)])
        a, t = {}, {}
        for lab, c in rows:
            s = r[r.condition == c]
            a[lab] = 100 * s.correct.astype(int).mean() if len(s) else np.nan
            t[lab] = (s.n_image_tokens / s.n_frames).mean() if len(s) else np.nan
        acc[name], tok[name] = a, t
    return pd.DataFrame(acc), pd.DataFrame(tok)

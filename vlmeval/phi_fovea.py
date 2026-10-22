"""Gaze-foveated tokenization for Phi-3.5-vision-instruct (no training).

Phi-3.5-vision (remote code on the hub) feeds an image to its LLM as an "HD" input: with the
processor's default num_crops = 4 a square frame becomes a 672 px view cut into 2 x 2 crops of
336 px, plus the whole frame at 336 px (the global view). Each 336 px view goes through CLIP
ViT-L/14-336 (24 x 24 patches of 14 px, learned position table, full attention; features of the
second-to-last layer, CLS dropped) and each 2 x 2 block of patch features is concatenated into one
4096-d token: one token = one 28 px cell. The crops' tokens form one 24 x 24 grid with a learned
row-break token (sub_GN) after each row, then a separator (glb_GN), then the global view's 12 x 12
grid with its row breaks; an MLP projects all of them into the LLM (Phi-3-mini, 1D positions):
600 + 1 + 156 = 757 tokens per frame.

Foveation thins out the high-resolution crop tokens and keeps the global view (Phi's own low-
resolution periphery), before the encoder as for Qwen2.5-VL and InternVL:

    fovea      28 px cells of the 672 px view, full resolution   (exactly Phi's crop patches)
    ring       56 px cells shrunk to 28 px                        (one token for 4x the area)
    periphery  112 px cells shrunk to 28 px                       (one token for 16x the area)

The layout is qwen_fovea.foveated_cells(img=672): the same rings as on the other models keep the
same share of high-resolution tokens. Cells never straddle a crop (336 = 3 x 112). Each cell
becomes 2 x 2 patches of 14 px at the centres of its quadrants in its crop, where CLIP's 24 x 24
position table is read (bilinear; exactly CLIP's own for 28 px cells); each crop's cells form one
variable-length sequence (CLS + patches) through CLIP, as the stock crops do. Tokens follow raster
order over the 672 px view with a row break after each token row (24 for the stock grid). Uniform
baseline at equal tokens: each crop downscaled and encoded by CLIP with its own interpolated
position table (Phi's crop structure kept).

Everything runs on native transformers pieces loaded from the checkpoint: CLIPVisionModel, the
row-break / separator tokens and projection MLP, and the LLM as Phi3ForCausalLM with SDPA attention
(the remote code only has eager or flash attention, too slow / unavailable for 30k-token
prompts). The HD layout and the prompt tokenization follow the remote code exactly; parity_report
checks them against it wherever this transformers version can still load it.
"""

import json
import os
import re
import time
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from vlmeval import qwen_fovea as qf

MODEL_ID = "microsoft/Phi-3.5-vision-instruct"
CROP, PATCH, BASE = 336, 14, 28
HD = 2 * CROP                                  # the 2 x 2 crop view (num_crops = 4, square frame)
GRID = HD // BASE                              # 24 crop tokens per side at stock
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_L336 = dict(attention_dropout=0.0, dropout=0.0, hidden_act="quick_gelu", hidden_size=1024, image_size=336,
                 initializer_factor=1.0, initializer_range=0.02, intermediate_size=4096, layer_norm_eps=1e-5,
                 num_attention_heads=16, num_channels=3, num_hidden_layers=24, patch_size=14, projection_dim=768)
PREFIX = "model.vision_embed_tokens."
IMAGE_TAG = re.compile(r"<\|image_\d+\|>")


# ---------------------------------------------------------------------------- model parts
class PhiImage(torch.nn.Module):
    """Phi-3.5-vision's image side: CLIP ViT-L/14-336, the row-break (sub_GN) and separator
    (glb_GN) tokens, and the projection MLP into the LLM (names as in the checkpoint)."""

    def __init__(self, hidden=3072, clip_cfg=None):
        super().__init__()
        from transformers import CLIPVisionConfig, CLIPVisionModel
        cfg = CLIPVisionConfig(**(clip_cfg or CLIP_L336))
        cfg._attn_implementation = "sdpa"
        self.img_processor = CLIPVisionModel(cfg)
        dim = 4 * cfg.hidden_size
        self.glb_GN = torch.nn.Parameter(torch.zeros(1, 1, dim))
        self.sub_GN = torch.nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.img_projection = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.GELU(),
                                                  torch.nn.Linear(hidden, hidden))

    @property
    def clip(self):
        """CLIP's transformer (embeddings, pre_layrnorm, encoder): nested as vision_model before
        transformers 5, flattened into CLIPVisionModel since."""
        return getattr(self.img_processor, "vision_model", self.img_processor)


def _checkpoint_tensors(model_id, prefix):
    """{name without prefix: tensor} for the checkpoint tensors whose name starts with prefix."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    index = json.load(open(hf_hub_download(model_id, "model.safetensors.index.json")))
    out = {}
    for f in sorted({f for k, f in index["weight_map"].items() if k.startswith(prefix)}):
        with safe_open(hf_hub_download(model_id, f), framework="pt") as fh:
            for k in fh.keys():
                if k.startswith(prefix):
                    out[k[len(prefix):]] = fh.get_tensor(k)
    return out


def load_image_weights(img, tensors):
    """Checkpoint tensors -> PhiImage, mapping CLIP's pre-5 names (img_processor.vision_model.*)
    onto this version's module; raises if anything is missing or left over."""
    nested = hasattr(img.img_processor, "vision_model")
    sd = {}
    for k, v in tensors.items():
        if not nested and k.startswith("img_processor.vision_model."):
            k = "img_processor." + k[len("img_processor.vision_model."):]
        sd[k] = v
    missing, unexpected = img.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("position_ids")]
    unexpected = [k for k in unexpected if not k.endswith("position_ids")]
    if missing or unexpected:
        raise RuntimeError(f"image weights: missing {missing[:5]} ... unexpected {unexpected[:5]} ...")
    return img


def letter_token_ids(tok, prefix, sep=""):
    """Token id of each answer letter as the model would write it right after `prefix` (the end of
    the generation prompt): the extra token when prefix + sep + letter is tokenized."""
    base = tok(prefix, add_special_tokens=False).input_ids
    ids = []
    for L in "ABCDE":
        full = tok(prefix + sep + L, add_special_tokens=False).input_ids
        if full[:len(base)] != base or len(full) != len(base) + 1:
            raise ValueError(f"letter {L!r} after {prefix!r} is not one extra token: {base} -> {full}")
        ids.append(full[-1])
    return ids


def llm_config(cfg):
    """transformers' Phi3Config for the LLM of a Phi-3.5-vision config.json dict: vision fields dropped
    (and sliding_window, 262144, longer than any prompt here, so a plain causal mask is the same).
    Its RoPE scaling is LongRoPE under the old name "su"; transformers 5 moves
    original_max_position_embeddings into the RoPE parameters only for a type already named
    longrope, so it is given as longrope with that length (the same frequencies and scaling)."""
    from transformers import Phi3Config
    drop = {"architectures", "auto_map", "embd_layer", "img_processor", "_name_or_path", "model_type",
            "transformers_version", "torch_dtype", "_attn_implementation", "sliding_window"}
    keep = {k: v for k, v in cfg.items() if k not in drop}
    rope = dict(keep.get("rope_scaling") or {})
    if rope:
        if rope.get("type", rope.get("rope_type")) == "su":
            rope["type"] = rope["rope_type"] = "longrope"
        rope.setdefault("original_max_position_embeddings", keep.get("original_max_position_embeddings", 4096))
        keep["rope_scaling"] = rope
    return Phi3Config.from_dict(keep)


def load_model(model_id=MODEL_ID):
    """(PhiImage, Phi3ForCausalLM, tokenizer, letter ids), all from the checkpoint, bf16 on the GPU."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, Phi3ForCausalLM
    cfg = json.load(open(hf_hub_download(model_id, "config.json")))
    llm_cfg = llm_config(cfg)
    kw = dict(config=llm_cfg, device_map="auto", attn_implementation="sdpa")
    try:
        llm = Phi3ForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, **kw)
    except TypeError:
        llm = Phi3ForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, **kw)
    img = load_image_weights(PhiImage(cfg["hidden_size"]), _checkpoint_tensors(model_id, PREFIX))
    img = img.to(llm.get_input_embeddings().weight.device, torch.bfloat16).eval()
    # config= spares AutoTokenizer its config lookup, which asks to run Phi's remote code; for phi3 it then
    # loads tokenizer.json as published, as Phi's own processor does (LlamaTokenizer rebuilds it in
    # transformers 5 and writes the letter after <|assistant|> as 'A', where Phi writes '▁A')
    tok = AutoTokenizer.from_pretrained(model_id, config=llm_cfg)
    gen = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True)
    return img, llm.eval(), tok, letter_token_ids(tok, gen)


# ---------------------------------------------------------------------------- views and layouts
def views(pil):
    """Phi's processor on a square frame (num_crops = 4): the 672 px view in [0, 1] (torchvision
    resize of the PIL image, as HD_transform does) and the 336 px global view, normalized (bicubic
    resize of the normalized 672 px view, as the processor does)."""
    import torchvision.transforms.functional as TF
    w, h = pil.size
    if w != h:
        raise ValueError(f"square frames only (got {w} x {h})")
    hd = TF.to_tensor(TF.resize(pil, [HD, HD]))
    glb = F.interpolate(normalize(hd)[None], size=(CROP, CROP), mode="bicubic")[0]
    return hd, glb


def normalize(x01):
    m = torch.tensor(MEAN, dtype=x01.dtype).view(3, 1, 1)
    s = torch.tensor(STD, dtype=x01.dtype).view(3, 1, 1)
    return (x01 - m) / s


def crops(view_norm, side):
    """(n * n, 3, side, side) crops of a normalized square view of n * side px, row-major."""
    n = view_norm.shape[-1] // side
    return view_norm.reshape(3, n, side, n, side).permute(1, 3, 0, 2, 4).reshape(n * n, 3, side, side)


def foveated_cells(gu, gv, a=40.0, R_deg=None, num_scales=3):
    """Cells (y, x, size) in px on the 672 px view for gaze (gu, gv) in [0, 1] (28/56/112 px)."""
    return qf.foveated_cells(gu, gv, a=a, img=HD, num_scales=num_scales, R_deg=R_deg)


# ---------------------------------------------------------------------------- vision
def merge_2x2(f):
    """(B, (2k)^2, C) patch features -> (B, k, k, 4C): each token its 2 x 2 patches concatenated
    row-major (Phi's reshape_hd_patches_2x2merge)."""
    B, L, C = f.shape
    H = int(round(L ** 0.5))
    return f.reshape(B, H // 2, 2, H // 2, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H // 2, H // 2, 4 * C)


def grid_rows(grid, sub_gn):
    """(h, w, D) token grid -> (h * (w + 1), D): each row followed by the row break (Phi's
    add_image_newline)."""
    h, w, D = grid.shape
    br = sub_gn.reshape(1, 1, D).to(grid.dtype).expand(h, 1, D)
    return torch.cat([grid, br], dim=1).reshape(-1, D)


def hd_rows(tokens, cells, sub_gn):
    """Crop tokens (n, D) of a cell layout -> raster order over the 672 px view with a row break
    after each 28 px token row that has tokens (the stock grid: 24 rows of 24 + 1)."""
    order = np.lexsort((cells[:, 1], cells[:, 0]))
    rows = cells[order, 0] // BASE
    br = sub_gn.reshape(1, -1).to(tokens.dtype)
    out = []
    for r in np.unique(rows):
        out += [tokens[torch.as_tensor(order[rows == r], device=tokens.device)], br]
    return torch.cat(out)


def _layer_minus2(clip, x):
    """CLIP encoder output after its second-to-last layer (Phi's feature layer, hidden_states[-2])."""
    got = {}
    hook = clip.encoder.layers[-2].register_forward_hook(
        lambda m, i, o: got.__setitem__("h", o[0] if isinstance(o, (tuple, list)) else o))
    try:
        clip.encoder(inputs_embeds=x)
    finally:
        hook.remove()
    return got["h"]


@torch.no_grad()
def clip_tokens(img, pixel_values, interpolate=False):
    """CLIP's own forward on B square views (normalized), features as Phi uses them (second-to-last
    layer, CLS dropped), merged 2 x 2: (B, k, k, 4096). interpolate: CLIP's own interpolated
    position table for views other than 336 px."""
    p = img.clip.embeddings.patch_embedding.weight                 # CLIP's own dtype (fp32 in parity checks)
    out = img.img_processor(pixel_values=pixel_values.to(p.device, p.dtype), output_hidden_states=True,
                            interpolate_pos_encoding=interpolate)
    return merge_2x2(out.hidden_states[-2][:, 1:])


def sample_positions(clip, h, w):
    """CLIP's learned patch position embeddings at patch coordinates (h, w) of its g x g grid,
    bilinear; exactly its own at integer coordinates."""
    pe = clip.embeddings.position_embedding.weight
    g = int(round((pe.shape[0] - 1) ** 0.5))
    table = pe[1:].view(g, g, -1).float()
    h, w = h.to(pe.device).float(), w.to(pe.device).float()
    h0, w0 = h.floor().clamp(0, g - 1), w.floor().clamp(0, g - 1)
    dh, dw = (h - h0)[:, None], (w - w0)[:, None]
    h0, w0 = h0.long(), w0.long()
    h1, w1 = (h0 + 1).clamp(max=g - 1), (w0 + 1).clamp(max=g - 1)
    out = (table[h0, w0] * (1 - dh) * (1 - dw) + table[h0, w1] * (1 - dh) * dw
           + table[h1, w0] * dh * (1 - dw) + table[h1, w1] * dh * dw)
    return out.to(pe.dtype)


@torch.no_grad()
def encode_cells(img, hd01, cells):
    """Crop tokens (n, 4096) before projection, one per cell in `cells` order, for any 28/56/112 px
    layout of the 672 px view: each crop's cells as one CLIP sequence (CLS + 2 x 2 patches per
    cell in raster order of their positions, position table read at the quadrant centres),
    second-to-last layer, CLS dropped, each cell's 4 patch features concatenated."""
    clip = img.clip
    emb = clip.embeddings
    p = emb.patch_embedding.weight                                 # CLIP's own dtype (fp32 in parity checks)
    feats = torch.empty(len(cells), 4 * emb.embed_dim, device=p.device, dtype=p.dtype)
    crop_id = (cells[:, 0] // CROP) * (HD // CROP) + cells[:, 1] // CROP
    cls = (emb.class_embedding + emb.position_embedding.weight[0])[None].to(p.dtype)
    for c in np.unique(crop_id):
        idx = np.nonzero(crop_id == c)[0]
        sub = cells[idx]
        local = sub.copy()
        local[:, :2] %= CROP
        n = len(sub)
        patches = qf.cells_to_patches(hd01, sub, MEAN, STD, temporal=1).view(4 * n, 3, PATCH, PATCH)
        x = emb.patch_embedding(patches.to(p.device, p.dtype)).flatten(1)                 # (4n, C)
        h, w = qf.patch_positions(local)
        x = x + sample_positions(clip, h, w).to(x.dtype)
        order = torch.argsort(h.double() * 4096 + w.double()).to(x.device)                 # raster order
        out = _layer_minus2(clip, clip.pre_layrnorm(torch.cat([cls, x[order]])[None]))[0, 1:]
        back = torch.empty_like(out)
        back[order] = out                                                                    # cell order
        feats[torch.as_tensor(idx, device=p.device)] = back.reshape(n, -1)
    return feats


@torch.no_grad()
def crop_grid(img, hd01, k=CROP // BASE):
    """Phi's crop tokens as a (2k, 2k, 4096) grid: the 672 px view resized so that each of its
    2 x 2 crops is 28 k px, each crop through CLIP's own forward (k = 12: Phi's stock crops; smaller
    k: the uniform baseline, CLIP's position table interpolated)."""
    side = 2 * k * BASE
    view = hd01 if side == HD else qf.resize01(hd01, side)
    return arrange_crops(clip_tokens(img, crops(normalize(view), side // 2), interpolate=side != HD))


def arrange_crops(t):
    """(4, k, k, D) tokens of the 2 x 2 crops (row-major) -> one (2k, 2k, D) grid over the view."""
    k = t.shape[1]
    return t.reshape(2, 2, k, k, -1).permute(0, 2, 1, 3, 4).reshape(2 * k, 2 * k, -1)


@torch.no_grad()
def global_rows(img, glb_norm):
    """The global view's 12 x 12 tokens with their row breaks (156 x 4096), before projection."""
    return grid_rows(clip_tokens(img, glb_norm[None])[0], img.sub_GN)


@torch.no_grad()
def image_embeds(img, hd_seq, glb_seq):
    """[crop tokens + row breaks, separator, global tokens + row breaks] -> LLM embeddings."""
    x = torch.cat([hd_seq, img.glb_GN.reshape(1, -1).to(hd_seq.dtype), glb_seq.to(hd_seq.dtype)])
    return img.img_projection(x)


# ---------------------------------------------------------------------------- conditions
def _fovea_layout(arg, fr):
    g = (0.5, 0.5) if arg["gaze"] == "centre" else fr.get("gaze")
    used = g
    if g is None:
        g = (0.5, 0.5)
    return foveated_cells(*g, a=arg["a"], R_deg=arg.get("R"), num_scales=arg.get("num_scales", 3)), used


def same_tokens_k(n):
    """Crop side (in 28 px tokens) whose 2 x 2 crops give the token count closest to n."""
    return max(1, int(round(np.sqrt(n / 4))))


@torch.no_grad()
def frame_tokens(img, conditions, cond, fr, hd01, glb_seq):
    """(LLM embeddings (L, 3072), cells or None, crop tokens, gaze used) for one frame.

    conditions: {name: (kind, arg)} with kind in
        'stock'          Phi's own input (num_crops = 4): 576 crop tokens + 24 + 1 + 156 = 757
        'fovea'          arg = dict(a=..., R=..., gaze='gaze' | 'centre', num_scales=3)
        'uniform_match'  arg = name of a 'fovea' condition: each crop downscaled so that the crop
                         tokens are closest to that condition's count on this frame
    The global view (and separator) is Phi's own in every condition. A missing gaze falls back to
    the image centre (counted in the results)."""
    kind, arg = conditions[cond]
    cells, used = None, None
    if kind == "stock":
        g = crop_grid(img, hd01)
        n, hd_seq = g.shape[0] * g.shape[1], grid_rows(g, img.sub_GN)
    elif kind == "uniform_match":
        k = same_tokens_k(len(_fovea_layout(conditions[arg][1], fr)[0]))
        g = crop_grid(img, hd01, k)
        n, hd_seq = g.shape[0] * g.shape[1], grid_rows(g, img.sub_GN)
    elif kind == "fovea":
        cells, used = _fovea_layout(arg, fr)
        n, hd_seq = len(cells), hd_rows(encode_cells(img, hd01, cells), cells, img.sub_GN)
    else:
        raise ValueError(f"unknown condition kind {kind!r}")
    return image_embeds(img, hd_seq, glb_seq), cells, n, used


# ---------------------------------------------------------------------------- LLM input
def phi_text(tok, messages):
    """Our chat messages (text and image parts) -> Phi's prompt: the parts joined by newlines, image
    k as <|image_k|>, in Phi's chat template."""
    parts, k = [], 0
    for c in messages[0]["content"]:
        if c.get("type") == "image":
            k += 1
            parts.append(f"<|image_{k}|>")
        else:
            parts.append(c["text"])
    return tok.apply_chat_template([{"role": "user", "content": "\n".join(parts)}], tokenize=False,
                                   add_generation_prompt=True)


def prompt_ids(tok, text, n_tokens):
    """Phi3VProcessor's tokenization: the text split at the image tags, each chunk tokenized on its
    own (with its BOS, as the processor does), image k inserted as n_tokens[k - 1] copies of -k."""
    chunks = [tok(c).input_ids for c in IMAGE_TAG.split(text)]
    tags = [int(t.split("|")[1].split("_")[-1]) for t in IMAGE_TAG.findall(text)]
    ids = []
    for i, ch in enumerate(chunks):
        ids += ch
        if i < len(tags):
            ids += [-tags[i]] * n_tokens[tags[i] - 1]
    return torch.tensor([ids])


@torch.no_grad()
def build_inputs(llm, tok, messages, feats_list):
    """inputs_embeds + attention_mask: text embeddings, image k's embeddings at its -k positions."""
    ids = prompt_ids(tok, phi_text(tok, messages), [len(f) for f in feats_list])
    emb_layer = llm.get_input_embeddings()
    ids = ids.to(emb_layer.weight.device)
    emb = emb_layer(ids.clamp(min=0))
    mask = ids < 0
    if int(mask.sum()) != sum(len(f) for f in feats_list):
        raise ValueError(f"{int(mask.sum())} image positions in the prompt, {sum(len(f) for f in feats_list)} embeddings")
    if feats_list:
        emb[mask] = torch.cat(feats_list).to(emb.device, emb.dtype)
    return dict(inputs_embeds=emb, attention_mask=torch.ones_like(ids)), ids


@torch.no_grad()
def letter_probs(llm, inputs, letter_ids, all_positions=False):
    """Softmax over A-E of the raw next-token logits (LM head over the last position, or over the
    whole sequence: the shape a stock forward computes)."""
    h = llm.model(**inputs, use_cache=False).last_hidden_state
    h = h if all_positions else h[:, -1:]
    logits = llm.lm_head(h)[0, -1, letter_ids].float()
    return torch.softmax(logits, -1).cpu().numpy()


# ---------------------------------------------------------------------------- checks on the real model
@contextmanager
def fp32_exact(*modules):
    """Inside: the modules' floating-point weights in fp32, TF32 off and SDPA's exact (math) kernel,
    so that two computations can be compared without bf16 rounding; the weights (the same tensors)
    and the flags are restored on exit."""
    saved = [(t, t.data) for m in modules for t in [*m.parameters(), *m.buffers()] if t.is_floating_point()]
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        exact_attention = sdpa_kernel(SDPBackend.MATH)
    except Exception:
        exact_attention = nullcontext()
    try:
        for t, d in saved:
            t.data = d.float()
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        with exact_attention:
            yield
    finally:
        for t, d in saved:
            t.data = d
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32


def load_remote(model_id=MODEL_ID):
    """Phi-3.5-vision's own remote-code model (eager attention) and processor (or, failing that,
    its image processor), for parity: (model or None, processor or None, notes)."""
    notes = {}
    model = ip = None
    try:
        from transformers import AutoProcessor
        ip = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)
    except Exception as ex:
        notes["remote_processor"] = f"not loaded ({type(ex).__name__}: {str(ex)[:120]})"
        try:
            from transformers import AutoImageProcessor
            ip = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)
        except Exception as ex2:
            notes["remote_image_processor"] = f"not loaded ({type(ex2).__name__}: {str(ex2)[:120]})"
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        cfg._attn_implementation = "eager"     # config.json names flash_attention_2: in transformers 5 that wins
                                               # over the keyword, and this remote class refuses it there
        kw = dict(config=cfg, trust_remote_code=True, attn_implementation="eager", device_map="auto")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, **kw)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, **kw)
        model.eval()
    except Exception as ex:
        notes["remote_model"] = f"not loaded ({type(ex).__name__}: {str(ex)[:120]})"
    return model, ip, notes


@torch.no_grad()
def parity_report(img, llm, tok, letter_ids, pil, question, options, build_messages, remote=None,
                  remote_proc=None):
    """Our path vs Phi's own computation on one real frame.

      vision  all-28 px layout through our cells path vs CLIP's own forward on the 4 crops (crop
              tokens before projection), in fp32 (TF32 off, exact attention) and bf16
      remote  where this transformers can load Phi's remote code: its processor's pixel values and
              token ids vs ours (views, prompt_ids); its model (eager attention) vs our path: image
              embeddings (Phi's own hd_feature_transform; in bf16, and in fp32 where only true
              differences remain), answer probabilities, and the LLMs alone (Phi's own embeddings
              through our native LLM)"""
    rep = {}
    hd01, glb = views(pil)
    cells = qf.uniform_cells(HD)
    stock16 = crop_grid(img, hd01).reshape(-1, img.glb_GN.shape[-1])
    ours16 = encode_cells(img, hd01, cells)
    with fp32_exact(img.img_processor):                      # vision parity in true fp32
        stock32 = crop_grid(img, hd01).reshape(-1, img.glb_GN.shape[-1]).float()
        ours32 = encode_cells(img, hd01, cells).float()
    cos = lambda a, b: F.cosine_similarity(a.float(), b.float(), dim=-1).min().item()
    mx = lambda a, b: (a.float() - b.float()).abs().max().item()
    rep.update(vision_fp32_cos_min=cos(ours32, stock32), vision_fp32_max_abs_diff=mx(ours32, stock32),
               vision_bf16_cos_min=cos(ours16, stock16), vision_bf16_max_abs_diff=mx(ours16, stock16))

    glb_seq = global_rows(img, glb)
    feats = image_embeds(img, grid_rows(stock16.view(GRID, GRID, -1), img.sub_GN), glb_seq)
    rep["tokens_per_frame"] = len(feats)
    msgs = build_messages(question, options, [("frame.jpg", 0.0)], CROP * CROP)
    inputs, ids = build_inputs(llm, tok, msgs, [feats])
    p_ours = letter_probs(llm, inputs, letter_ids, all_positions=True)
    rep["ours_answer"] = "ABCDE"[int(p_ours.argmax())]
    nxt = torch.softmax(llm.lm_head(llm.model(**inputs, use_cache=False).last_hidden_state[:, -1:])[0, -1].float(), -1)
    rep["top5_next_tokens"] = tok.convert_ids_to_tokens(nxt.topk(5).indices.tolist())   # is the letter what Phi writes?
    rep["letters_share_of_next_token_prob"] = round(float(nxt[letter_ids].sum()), 4)
    rep["next_token_is_a_letter"] = int(nxt.argmax()) in letter_ids

    ours_pv = torch.cat([glb[None], crops(normalize(hd01), CROP)])          # (global, 4 crops)
    if remote_proc is not None:
        try:
            if hasattr(remote_proc, "tokenizer"):
                rp = remote_proc(phi_text(tok, msgs), images=[pil], return_tensors="pt")
                rep["remote_token_ids_equal"] = bool(torch.equal(rp["input_ids"][0].cpu(), ids[0].cpu()))
            else:
                rp = remote_proc([pil], return_tensors="pt")
            rep["remote_pixel_max_abs_diff"] = (rp["pixel_values"][0].float() - ours_pv).abs().max().item()
        except Exception as ex:
            rep["remote_processor_check"] = f"not checked ({type(ex).__name__}: {str(ex)[:120]})"
    if remote is not None:
        try:
            emb_mod = remote.model.vision_embed_tokens
            w_ours = img.clip.embeddings.patch_embedding.weight
            w_rem = getattr(emb_mod.img_processor, "vision_model", emb_mod.img_processor).embeddings.patch_embedding.weight
            rep["remote_clip_weights_equal"] = bool(torch.equal(w_ours.cpu(), w_rem.cpu()))
            ip_ids = ids.clone()
            pv = ours_pv[None].to(w_rem.device, w_rem.dtype)
            sizes = torch.tensor([[HD, HD]], device=w_rem.device)
            rem_emb = emb_mod(ip_ids.to(w_rem.device), pixel_values=pv, image_sizes=sizes)
            mask = (ip_ids < 0).to(rem_emb.device)
            rep["remote_image_embeds_cos_min"] = cos(rem_emb[mask], feats)
            out = remote(input_ids=ip_ids.to(w_rem.device), attention_mask=torch.ones_like(ip_ids).to(w_rem.device),
                         pixel_values=pv, image_sizes=sizes, use_cache=False)
            p_rem = torch.softmax(out.logits[0, -1, letter_ids].float(), -1).cpu().numpy()
            rep["remote_probs_max_abs_diff"] = float(np.abs(p_rem - p_ours).max())
            rep["remote_same_answer"] = bool(p_rem.argmax() == p_ours.argmax())
            # the LLMs alone: Phi's own embeddings through our native LLM
            p_llm = letter_probs(llm, dict(inputs_embeds=rem_emb.to(inputs["inputs_embeds"].device),
                                           attention_mask=inputs["attention_mask"]), letter_ids, all_positions=True)
            rep["llm_native_vs_remote_probs_max_abs_diff"] = float(np.abs(p_llm - p_rem).max())
            rep["embeds_ours_vs_remote_probs_max_abs_diff"] = float(np.abs(p_llm - p_ours).max())
            with fp32_exact(img, emb_mod):                   # the image embeddings again, without bf16 rounding
                rem32 = emb_mod(ip_ids.to(w_rem.device), pixel_values=ours_pv[None].to(w_rem.device),
                                image_sizes=sizes)
                feats32 = image_embeds(img, grid_rows(crop_grid(img, hd01), img.sub_GN), global_rows(img, glb))
            rep["remote_image_embeds_fp32_cos_min"] = cos(rem32[mask], feats32)
        except Exception as ex:
            rep["remote_model_check"] = f"not checked ({type(ex).__name__}: {str(ex)[:120]})"
    return rep


@torch.no_grad()
def fovea_feature_drift(img, hd01, cells):
    """Cosine of each foveated crop token with the mean of Phi's own crop tokens over the same area."""
    import pandas as pd
    full = crop_grid(img, hd01).float()
    fov = encode_cells(img, hd01, cells).float()
    rows = []
    for (y, x, s), f in zip(cells, fov):
        y0, x0, k = y // BASE, x // BASE, s // BASE
        ref = full[y0:y0 + k, x0:x0 + k].reshape(-1, full.shape[-1]).mean(0)
        rows.append((int(s), F.cosine_similarity(f, ref, dim=0).item()))
    return pd.DataFrame(rows, columns=["cell_px", "cos_to_stock"]).groupby("cell_px").agg(
        cells=("cos_to_stock", "size"), mean_cos=("cos_to_stock", "mean"), min_cos=("cos_to_stock", "min"))


# ---------------------------------------------------------------------------- evaluation
def load_pil(path):
    from PIL import Image
    return Image.open(path).convert("RGB")


@torch.no_grad()
def run_questions(img, llm, tok, letter_ids, items, conditions, out_path, build_messages, log_every=25):
    """Score questions on their frames under each condition; resumable. Work runs video by video
    and, within a video, condition by condition; each frame's views and global tokens are computed
    once per video, its condition tokens once per (video, condition)."""
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                done.add((r["condition"], r["qid"]))
    todo = [(it, c) for c in conditions for it in items if (c, it["qid"]) not in done]
    todo.sort(key=lambda x: x[0]["video"])                   # stable: video, condition, question
    print(f"{len(done)} already scored, {len(todo)} to go -> {out_path}")

    toks, base, block, video, t_start, n_new = {}, {}, None, None, time.time(), 0
    with open(out_path, "a") as fh:
        for it, c in todo:
            if it["video"] != video:
                base.clear()
                video = it["video"]
            if (video, c) != block:
                toks.clear()
                torch.cuda.empty_cache()
                block = (video, c)
            t0 = time.time()
            per = []
            for fr in it["frames"]:
                if fr["key"] not in toks:
                    if fr["key"] not in base:
                        hd01, glb = views(load_pil(fr["path"]))
                        base[fr["key"]] = (hd01, global_rows(img, glb))
                    e, cells, n_crop, used = frame_tokens(img, conditions, c, fr, *base[fr["key"]])
                    toks[fr["key"]] = (e, cells, n_crop, conditions[c][0] == "fovea" and used is None)
                per.append(toks[fr["key"]])
            try:
                msgs = build_messages(it["question"], it["options"],
                                      [(fr["path"], fr["t"]) for fr in it["frames"]], CROP * CROP)
                inputs, ids = build_inputs(llm, tok, msgs, [q[0] for q in per])
                p = letter_probs(llm, inputs, letter_ids)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                fh.write(json.dumps(dict(condition=c, qid=int(it["qid"]), video=it["video"],
                                         error="oom", n_frames=len(per))) + "\n")
                continue
            err = None if np.isfinite(p).all() else "nan"
            pred = "ABCDE"[int(np.nanargmax(p))] if np.isfinite(p).any() else "?"
            sizes = np.concatenate([q[1][:, 2] for q in per if q[1] is not None] or [np.zeros(0, int)])
            fh.write(json.dumps(dict(
                condition=c, qid=int(it["qid"]), video=it["video"], answer=it["answer"],
                pred=pred, correct=int(pred == it["answer"]), p=[round(float(v), 4) for v in p],
                n_frames=len(per), n_image_tokens=int(sum(len(q[0]) for q in per)),
                n_crop_tokens=int(sum(q[2] for q in per)),
                cells_by_size={str(int(s)): int((sizes == s).sum()) for s in np.unique(sizes)},
                n_gaze_missing=int(sum(q[3] for q in per)),
                n_tokens=int(ids.shape[1]), sec=round(time.time() - t0, 3), error=err)) + "\n")
            fh.flush()
            n_new += 1
            if n_new % log_every == 0:
                rate = (time.time() - t_start) / n_new
                print(f"   {n_new}/{len(todo)}  {rate:.2f} s each  ~{rate * (len(todo) - n_new) / 60:.0f} min left"
                      f"  ({c}, {it['video']})", flush=True)
    print(f"done: {n_new} new in {(time.time() - t_start) / 60:.1f} min")

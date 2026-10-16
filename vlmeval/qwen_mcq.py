"""Five-option multiple choice with Qwen2.5-VL, scored by the next-token logits of A-E.

Frames go in as separate images, each preceded by its time ("Frame at 1:37"), because the
conditions send unevenly spaced frames and the questions are about time and memory. Every
condition uses the same per-image resolution (`max_pixels`), so only WHICH frames are sent
differs. The answer is the letter with the highest next-token probability: no free-text
parsing, and the full distribution is kept.

Results are appended to a JSONL file one question at a time, so a disconnected Colab
session resumes where it stopped.
"""

import json
import os
import time

import numpy as np
import torch

from vlmeval.egoev import LETTERS
from vlmeval.frames import frame_path

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
INTRO = ("These frames come from a video recorded through the eyes of a person wearing AR "
         "glasses, in time order. Answer the question about what this person saw or did.")
TAIL = "Answer with the option's letter from the given choices directly."


def load_model(model_id=MODEL_ID):
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as Model
    except ImportError:
        from transformers import AutoModelForImageTextToText as Model
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if bf16 else torch.float16       # T4 has no native bf16
    kw = dict(device_map="auto", attn_implementation="sdpa")
    try:
        model = Model.from_pretrained(model_id, dtype=dtype, **kw)
    except TypeError:                                        # older transformers
        model = Model.from_pretrained(model_id, torch_dtype=dtype, **kw)
    proc = AutoProcessor.from_pretrained(model_id)
    ids = []
    for L in LETTERS:
        t = proc.tokenizer.encode(L, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"letter {L!r} is not a single token: {t}")
        ids.append(t[0])
    return model.eval(), proc, ids, dtype


def fmt_time(s):
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


def build_messages(question, options, frames, max_pixels):
    """frames: [(image path, seconds)]."""
    content = []
    if frames:
        content.append({"type": "text", "text": INTRO})
        for path, t in frames:
            content.append({"type": "text", "text": f"Frame at {fmt_time(t)}:"})
            content.append({"type": "image", "image": path, "max_pixels": max_pixels})
    opts = "\n".join(f"{L}. {o}" for L, o in zip(LETTERS, options))
    content.append({"type": "text", "text": f"Question: {question}\nOptions:\n{opts}\n{TAIL}"})
    return [{"role": "user", "content": content}]


@torch.no_grad()
def score(model, proc, letter_ids, messages):
    """-> (probabilities over A-E, prompt length in tokens)."""
    from qwen_vl_utils import process_vision_info
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)[:2]
    inputs = proc(text=[text], images=images or None, videos=videos or None,
                  return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                         output_scores=True, return_dict_in_generate=True)
    logits = out.scores[0][0, letter_ids].float()
    return torch.softmax(logits, -1).cpu().numpy(), int(inputs["input_ids"].shape[1])


def run(model, proc, letter_ids, qs, sel, metas, out_path, conditions, max_pixels,
        tag="", log_every=25):
    """Score every (condition, question) not yet in out_path."""
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                r = json.loads(line)
                done.add((r["condition"], r["qid"]))
    todo = [(c, q) for c in conditions for q in qs.itertuples() if (c, q.qid) not in done]
    print(f"{len(done)} already scored, {len(todo)} to go -> {out_path}")
    t_start, n_new = time.time(), 0
    with open(out_path, "a") as fh:
        for c, q in todo:
            meta = metas[q.video]
            frames = [(frame_path(meta, k), meta["ts"][k]) for k in sel[c].get(q.video, [])
                      if k < meta["n"]]
            px = max_pixels
            for _ in range(3):                       # halve the resolution on OOM
                try:
                    t0 = time.time()
                    p, ntok = score(model, proc, letter_ids,
                                    build_messages(q.question, q.options, frames, px))
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    px //= 2
            else:
                fh.write(json.dumps(dict(condition=c, qid=int(q.qid), video=q.video,
                                         error="oom", n_frames=len(frames))) + "\n")
                continue
            err = None if np.isfinite(p).all() else "nan"
            pred = LETTERS[int(np.nanargmax(p))] if np.isfinite(p).any() else "?"
            fh.write(json.dumps(dict(
                condition=c, qid=int(q.qid), video=q.video, answer=q.answer, pred=pred,
                correct=int(pred == q.answer), p=[round(float(v), 4) for v in p],
                n_frames=len(frames), n_tokens=ntok, max_pixels=px,
                sec=round(time.time() - t0, 2), tag=tag, error=err)) + "\n")
            fh.flush()
            n_new += 1
            if n_new % log_every == 0:
                rate = (time.time() - t_start) / n_new
                print(f"   {n_new}/{len(todo)}  {rate:.1f} s/question  "
                      f"~{rate*(len(todo)-n_new)/60:.0f} min left  ({c})", flush=True)
    print(f"done: {n_new} new in {(time.time()-t_start)/60:.1f} min")

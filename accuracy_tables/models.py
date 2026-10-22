"""One adapter per VLM over its vlmeval module, so run.py treats the six models alike.

Each adapter knows how to load its model, check parity on one frame, name the stock and the
foveated conditions in its module's vocabulary, score items through the module's
run_questions (Uniform K, ours), tokenize a text-only prompt (Blind), and hand its loaded model
to its GazeVLM and StreamingTOM backends (vlmeval/gazevlm.py, vlmeval/streamingtom.py). The model
code itself lives in vlmeval/ and is not duplicated here.

    key        module                      stock input per frame
    qwen2.5    vlmeval/qwen_fovea.py       256 tokens (448 px)
    qwen3      vlmeval/qwen3_fovea.py      256 tokens (512 px)
    internvl   vlmeval/internvl_fovea.py   256 tokens (448 px tile)
    gemma      vlmeval/gemma_fovea.py      256 tokens (896 px, 4 x 4 pooling)
    phi        vlmeval/phi_fovea.py        757 tokens (672 px HD view + 336 px global view)
    smolvlm    vlmeval/smolvlm_fovea.py     81 tokens (384 px)
"""

import gc

import torch

from vlmeval import qwen_mcq

build_messages = qwen_mcq.build_messages        # the prompt every model and row shares


def _num(rep, k):
    return isinstance(rep.get(k), float)


class Adapter:
    key = ""
    stock_tokens = 0
    default_ids = ()

    def __init__(self, model_id=None, cfg=None):
        ids = model_id if model_id else self.default_ids
        self.model_ids = [ids] if isinstance(ids, str) else list(ids)
        self.cfg = cfg or {}
        self.model_id = None

    # -- to implement per model
    def _load(self, model_id):
        raise NotImplementedError

    def parity(self, frame, question, options):
        """-> (ok, report dict) on one real frame."""
        raise NotImplementedError

    def stock_condition(self):
        raise NotImplementedError

    def fovea_condition(self, setting):
        return ("fovea", dict(a=setting["a"], R=setting["R"], gaze="gaze"))

    def fovea_settings(self, ours_cfg):
        return list(ours_cfg["rings"])

    def setting_label(self, s):
        return f"R={s['R']} a={s['a']}"

    def score(self, items, conditions, out_path):
        raise NotImplementedError

    def blind_ids(self, messages):
        raise NotImplementedError

    def blind_logits(self, ids):
        """Next-token logits of the stock model on a text-only prompt."""
        out = self.model(input_ids=ids.to(self.model.device), attention_mask=torch.ones_like(ids).to(self.model.device),
                         use_cache=False)
        return out.logits[0, -1]

    def gazevlm_backend(self):
        """This model's GazeVLM backend in vlmeval/gazevlm.py (same class name as the adapter)."""
        from vlmeval import gazevlm as gz
        return getattr(gz, type(self).__name__)(self.model, self.proc, self.letter_ids)

    def streamingtom_backend(self):
        """This model's StreamingTOM backend in vlmeval/streamingtom.py (same class name)."""
        from vlmeval import streamingtom as stm
        return getattr(stm, type(self).__name__)(self.model, self.proc, self.letter_ids)

    # -- shared
    def load(self):
        errors = []
        for mid in self.model_ids:
            try:
                self._load(mid)
                self.model_id = mid
                return self
            except (OSError, ValueError) as ex:
                errors.append(f"{mid}: {type(ex).__name__}: {str(ex)[:200]}")
        raise RuntimeError(f"{self.key}: no model id loaded:\n  " + "\n  ".join(errors))

    def free(self):
        for k in list(vars(self)):
            if k not in ("model_ids", "cfg", "model_id"):
                delattr(self, k)
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------- Qwen-style text prompt
def _qwen_text_ids(proc, messages):
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return proc.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids


class Qwen25(Adapter):
    key, stock_tokens, default_ids = "qwen2.5", 256, ("Qwen/Qwen2.5-VL-3B-Instruct",)
    SIDE = 448                                       # stock frame: 16 x 16 tokens of 28 px

    def _load(self, model_id):
        self.model, self.proc, self.letter_ids, _ = qwen_mcq.load_model(model_id)

    def parity(self, frame, question, options):
        from vlmeval import qwen_fovea as qf
        rep = qf.parity_report(self.model, self.proc, self.letter_ids, qf.load_frame01(frame["path"]),
                               question, options, build_messages, size=self.SIDE)
        ok = (rep["pixel_max_abs_diff"] < 1e-2 and rep["vision_cos_min"] > 0.999 and rep["tokens_ids_equal"]
              and rep["probs_max_abs_diff"] < 0.02 and rep["same_answer"]
              and (not _num(rep, "positions_max_abs_diff") or rep["positions_max_abs_diff"] == 0))
        return ok, rep

    def stock_condition(self):
        return ("uniform", self.SIDE)

    def fovea_condition(self, setting):
        return ("fovea", dict(a=setting["a"], R=setting["R"], img=self.SIDE, gaze="gaze"))

    def score(self, items, conditions, out_path):
        from vlmeval import qwen_fovea as qf
        qf.run_questions(self.model, self.proc, self.letter_ids, items, conditions, out_path, build_messages)

    def blind_ids(self, messages):
        return _qwen_text_ids(self.proc, messages)

    def blind_logits(self, ids):
        """Through inputs_embeds with explicit M-RoPE positions, as the scored rows are."""
        from vlmeval import qwen_fovea as qf
        emb_layer = self.model.get_input_embeddings()
        ids = ids.to(emb_layer.weight.device)
        img_id = self.proc.tokenizer.convert_tokens_to_ids(qf.IMAGE_PAD)
        pos = qf.mrope_positions(ids.cpu(), img_id, [], []).to(ids.device)
        out = self.model(inputs_embeds=emb_layer(ids), position_ids=pos, attention_mask=torch.ones_like(ids),
                         use_cache=False)
        return out.logits[0, -1]


class Qwen3(Adapter):
    key, stock_tokens, default_ids = "qwen3", 256, ("Qwen/Qwen3-VL-4B-Instruct",)

    def _load(self, model_id):
        from vlmeval import qwen3_fovea as q3
        self.model, self.proc, self.letter_ids = q3.load_model(model_id)

    def parity(self, frame, question, options):
        from vlmeval import qwen_fovea as qf, qwen3_fovea as q3
        rep = q3.parity_report(self.model, self.proc, self.letter_ids, qf.load_frame01(frame["path"]),
                               question, options, build_messages)
        ok = (rep["pixel_max_abs_diff"] < 2e-2 and rep["vision_fp32_cos_min"] > 0.9999
              and rep["deepstack_fp32_cos_min"] > 0.9999 and rep["tokens_ids_equal"]
              and rep["llm_probs_max_abs_diff"] < 1e-3 and rep["llm_same_answer"])
        return ok, rep

    def stock_condition(self):
        from vlmeval import qwen3_fovea as q3
        return ("uniform", q3.IMG)

    def score(self, items, conditions, out_path):
        from vlmeval import qwen3_fovea as q3
        q3.run_questions(self.model, self.proc, self.letter_ids, items, conditions, out_path, build_messages)

    def blind_ids(self, messages):
        return _qwen_text_ids(self.proc, messages)

    def blind_logits(self, ids):
        """Through the module's own text path (explicit M-RoPE positions), as the scored rows are."""
        from vlmeval import qwen3_fovea as q3
        return q3.next_token_logits(self.model, q3.inputs_from_ids(self.model, ids, [], [], [], []))


class InternVL(Adapter):
    key, stock_tokens = "internvl", 256
    default_ids = ("OpenGVLab/InternVL3_5-4B-HF", "OpenGVLab/InternVL3_5-4B-hf")

    def _load(self, model_id):
        from vlmeval import internvl_fovea as iv
        self.model, self.proc, self.letter_ids = iv.load_model(model_id)

    def parity(self, frame, question, options):
        from vlmeval import qwen_fovea as qf, internvl_fovea as iv
        rep = iv.parity_report(self.model, self.proc, self.letter_ids, qf.load_frame01(frame["path"]),
                               question, options, build_messages)
        ok = (rep["tiles_in_stock"] == 1 and rep["pixel_max_abs_diff"] < 2e-2
              and rep["uniform_vs_get_image_features"] < 1e-3 and rep["vision_fp32_cos_min"] > 0.9999
              and rep["tokens_ids_equal"] and rep["llm_probs_max_abs_diff"] < 1e-3 and rep["llm_same_answer"])
        return ok, rep

    def stock_condition(self):
        return ("uniform", 448)

    def score(self, items, conditions, out_path):
        from vlmeval import internvl_fovea as iv
        iv.run_questions(self.model, self.proc, self.letter_ids, items, conditions, out_path, build_messages)

    def blind_ids(self, messages):
        from vlmeval import internvl_fovea as iv
        return iv.prompt_ids(self.proc, messages, [], iv.TOKENIZE["add_special_tokens"])


class Gemma(Adapter):
    key, stock_tokens, default_ids = "gemma", 256, ("google/gemma-3-4b-it",)

    def _load(self, model_id):
        from vlmeval import gemma_fovea as gf
        self.model, self.proc, self.letter_ids = gf.load_model(model_id)

    def parity(self, frame, question, options):
        from vlmeval import qwen_fovea as qf, gemma_fovea as gf
        rep = gf.parity_report(self.model, self.proc, self.letter_ids, qf.load_frame01(frame["path"]),
                               question, options, build_messages)
        ok = (rep["pixel_max_abs_diff"] < 2e-2 and rep["vision_cos_min"] > 0.999 and rep["tokens_ids_equal"]
              and rep["probs_max_abs_diff"] < 0.02 and rep["same_answer"])
        return ok, rep

    def stock_condition(self):
        return ("stock", None)

    def score(self, items, conditions, out_path):
        from vlmeval import gemma_fovea as gf
        gf.run_questions(self.model, self.proc, self.letter_ids, items, conditions, out_path, build_messages)

    def blind_ids(self, messages):
        from vlmeval import gemma_fovea as gf
        return gf.prompt_ids(self.proc, messages, [], gf.TOKENIZE["strip_bos"], gf.TOKENIZE["add_special_tokens"])

    def blind_logits(self, ids):
        from vlmeval import gemma_fovea as gf
        dev = self.model.device
        ids = ids.to(dev)
        out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                         **{gf._type_ids_key(self.model): torch.zeros_like(ids)}, use_cache=False)
        return out.logits[0, -1]


class Phi(Adapter):
    key, stock_tokens, default_ids = "phi", 757, ("microsoft/Phi-3.5-vision-instruct",)

    def _load(self, model_id):
        from vlmeval import phi_fovea as ph
        self.img, self.llm, self.tok, self.letter_ids = ph.load_model(model_id)
        self.model = self.llm

    def parity(self, frame, question, options):
        from vlmeval import phi_fovea as ph
        remote = remote_proc = None
        notes = {}
        if self.cfg.get("run", {}).get("phi_remote_parity"):
            remote, remote_proc, notes = ph.load_remote(self.model_id)
        rep = ph.parity_report(self.img, self.llm, self.tok, self.letter_ids, ph.load_pil(frame["path"]),
                               question, options, build_messages, remote, remote_proc)
        del remote
        torch.cuda.empty_cache()
        ok = rep["vision_fp32_cos_min"] > 0.9999 and rep["tokens_per_frame"] == 757 and rep["next_token_is_a_letter"]
        if _num(rep, "remote_pixel_max_abs_diff"):
            ok &= rep["remote_pixel_max_abs_diff"] < 1e-3
        if "remote_token_ids_equal" in rep:
            ok &= rep["remote_token_ids_equal"]
        if rep.get("remote_clip_weights_equal") and _num(rep, "remote_probs_max_abs_diff"):
            ok &= rep.get("remote_image_embeds_fp32_cos_min", 0) > 0.9999 and rep["remote_same_answer"]
        return ok, {**notes, **rep}

    def stock_condition(self):
        return ("stock", None)

    def score(self, items, conditions, out_path):
        from vlmeval import phi_fovea as ph
        ph.run_questions(self.img, self.llm, self.tok, self.letter_ids, items, conditions, out_path, build_messages)

    def blind_ids(self, messages):
        from vlmeval import phi_fovea as ph
        return ph.prompt_ids(self.tok, ph.phi_text(self.tok, messages), [])

    def gazevlm_backend(self):
        from vlmeval import gazevlm as gz
        return gz.Phi(self.img, self.llm, self.tok, self.letter_ids)

    def streamingtom_backend(self):
        from vlmeval import streamingtom as stm
        return stm.Phi(self.img, self.llm, self.tok, self.letter_ids)


class SmolVLM(Adapter):
    key, stock_tokens, default_ids = "smolvlm", 81, ("HuggingFaceTB/SmolVLM2-2.2B-Instruct",)

    def _load(self, model_id):
        from vlmeval import smolvlm_fovea as sm
        self.model, self.proc, self.letter_ids = sm.load_model(model_id)

    def parity(self, frame, question, options):
        from PIL import Image
        from vlmeval import smolvlm_fovea as sm
        rep = sm.parity_report(self.model, self.proc, self.letter_ids, Image.open(frame["path"]).convert("RGB"),
                               question, options, build_messages)
        ok = (rep["images_in_stock"] == 1 and rep["pixel_max_abs_diff"] < 1e-3
              and rep["own_path_vs_get_image_features"] < 1e-3 and rep["vision_fp32_cos_min"] > 0.9999
              and rep["tokens_ids_equal"] and rep["llm_probs_max_abs_diff"] < 1e-3 and rep["llm_same_answer"])
        return ok, rep

    def stock_condition(self):
        from vlmeval import smolvlm_fovea as sm
        return ("uniform", sm.IMG)

    def fovea_condition(self, setting):
        return ("fovea", dict(k=setting["k"], gaze="gaze"))

    def fovea_settings(self, ours_cfg):
        return [dict(k=int(k)) for k in ours_cfg["smolvlm_k"]]

    def setting_label(self, s):
        return f"k={s['k']}"

    def score(self, items, conditions, out_path):
        from vlmeval import smolvlm_fovea as sm
        sm.run_questions(self.model, self.proc, self.letter_ids, items, conditions, out_path, build_messages)

    def blind_ids(self, messages):
        from vlmeval import smolvlm_fovea as sm
        return sm.prompt_ids(self.proc, messages, [], sm.TOKENIZE["add_special_tokens"])


ADAPTERS = {a.key: a for a in (Qwen25, Qwen3, InternVL, Gemma, Phi, SmolVLM)}

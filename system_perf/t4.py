"""Run one step of the T4 notebook (system_perf/colab_system_perf_t4.ipynb) with Qwen3-VL in fp16, because a T4 has no bf16.

    python -m system_perf.t4 accuracy_tables.run --config ... --model qwen3 --rows uniform,ours --videos test --out-dir ...
    python -m system_perf.t4 accuracy_tables.calibrate --config ... --model qwen3 --points 2 --videos test --out-dir ... --write ...
    python -m system_perf.t4 system_perf.measure --config ... --model qwen3 --videos ... --dtype auto --out-dir ...

vlmeval/qwen3_fovea.load_model always loads bf16. On a GPU without bf16 (a T4), this wraps it, inside this process only, so the loaded
model's bf16 weights and buffers are converted to fp16. Its float32 buffers (the rotary position frequencies) stay float32, as loading
in fp16 keeps them: model.to(torch.float16) would round them to fp16 too, and at ~25,000 positions that scrambles the rotary angles.
Qwen2.5-VL already loads fp16 on such a GPU (qwen_mcq.load_model), and SmolVLM2 follows VLMEVAL_DTYPE, which system_perf.measure sets,
so they change nothing here. On a GPU with bf16 nothing is wrapped.

No file in vlmeval/ or accuracy_tables/ changes, so the cluster's jobs, and the code fingerprint their shards check (it covers
accuracy_tables, vlmeval and eyehead), stay as they are. fp16 has a smaller range than bf16, and Qwen's activations can overflow it:
parity at the start of each step and the records' nan errors show that. Never use --no-parity.
"""
import importlib
import itertools
import sys

import torch

STEPS = ("accuracy_tables.run", "accuracy_tables.calibrate", "accuracy_tables.summarize", "system_perf.measure", "system_perf.report")


def has_bf16():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def bf16_to_fp16(model):
    """bf16 parameters and buffers to fp16, in place; every other dtype (the float32 rotary frequencies, integer ids) is kept."""
    for t in itertools.chain(model.parameters(), model.buffers()):
        if t.dtype == torch.bfloat16:
            t.data = t.data.to(torch.float16)
    return model


def fp16_qwen3():
    """qwen3_fovea.load_model, in this process, returning the model in fp16. Wrapping twice changes nothing."""
    from vlmeval import qwen3_fovea as q3
    if getattr(q3.load_model, "fp16", False):
        return
    own = q3.load_model

    def load_model(model_id):
        model, proc, letter_ids = own(model_id)
        return bf16_to_fp16(model), proc, letter_ids

    load_model.fp16 = True
    q3.load_model = load_model


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in STEPS:
        sys.exit(f"usage: python -m system_perf.t4 <{' | '.join(STEPS)}> <that step's arguments>")
    step, rest = argv[0], argv[1:]
    if has_bf16():
        print("this GPU has bf16: nothing is changed, every model loads as on the cluster", flush=True)
    else:
        fp16_qwen3()
        print("no bf16 on this GPU (T4): Qwen3-VL is converted to fp16 in this process; Qwen2.5-VL and SmolVLM2 pick fp16 "
              "themselves", flush=True)
    importlib.import_module(step).main(rest)


if __name__ == "__main__":
    main()

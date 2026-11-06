"""Stops a job on a GPU the models can't run on: hardware bf16 and at least 40 GB.

    python -u -m accuracy_tables.hpc.check_gpu [--expect B200] [--record <file>]

--expect also stops a job on any other GPU type, because answers can differ between types and every row of a model must be comparable.
--record adds the GPU's name to a file, so the report can tell whether a results folder holds one GPU type.
"""
import argparse
import os
import sys


def wrong_type(name, expect):
    """Is this GPU, by its full name, something other than the type asked for?"""
    return bool(expect) and expect.lower() not in name.lower()


def record(path, name):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(name + "\n")


def main():
    import torch
    import transformers

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expect", help="stop unless the GPU's name contains this, such as B200")
    ap.add_argument("--record", help="add the GPU's name to this file")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("no GPU visible")
    g = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"{g.name}, {g.total_memory / 1e9:.0f} GB, CUDA capability {capability}; python {sys.version.split()[0]}, "
          f"torch {torch.__version__}, transformers {transformers.__version__}", flush=True)
    if capability[0] < 8 or g.total_memory <= 35e9:
        sys.exit("this GPU has no hardware bf16 or under 40 GB: resubmit with a constraint for a larger one")
    if wrong_type(g.name, a.expect):
        sys.exit(f"this job must run on a {a.expect}, and this is a {g.name}: stop and tell the user")
    if a.record:
        record(a.record, g.name)


if __name__ == "__main__":
    main()

"""Stops a job on a GPU the models can't run on: hardware bf16 and at least 40 GB, as on Colab's A100."""
import sys

import torch
import transformers

if not torch.cuda.is_available():
    sys.exit("no GPU visible")
g = torch.cuda.get_device_properties(0)
capability = torch.cuda.get_device_capability(0)
print(f"{g.name}, {g.total_memory / 1e9:.0f} GB, CUDA capability {capability}; python {sys.version.split()[0]}, "
      f"torch {torch.__version__}, transformers {transformers.__version__}", flush=True)
if capability[0] < 8 or g.total_memory <= 35e9:
    sys.exit("this GPU has no hardware bf16 or under 40 GB: resubmit with a constraint for a larger one")

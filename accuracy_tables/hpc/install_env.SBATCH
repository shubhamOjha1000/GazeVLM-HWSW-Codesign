#!/bin/bash
#SBATCH --job-name=acc-install
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Installs what the accuracy runs need into the overlay's base environment, keeping its torch. The overlay is mounted
# read-write, so no other job may have it open. Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
singularity exec --fakeroot --overlay "$OVERLAY:rw" "$IMAGE" /bin/bash -c "
    set -e
    source /ext3/env.sh; unset PYTHONPATH; export PYTHONNOUSERSITE=True
    pip install 'transformers==5.20.1' accelerate safetensors huggingface_hub num2words sentencepiece protobuf pillow \
        scipy scikit-image pyyaml pandas opencv-python-headless pytest
    python -c 'import torchvision' 2>/dev/null || pip install torchvision --index-url https://download.pytorch.org/whl/cu132
    python -c 'import sys, torch, torchvision, transformers, accelerate, cv2, num2words, yaml, pandas; print(\"ok: python\", sys.version.split()[0], \"torch\", torch.__version__, \"transformers\", transformers.__version__)'
"

#!/bin/bash
#SBATCH --job-name=smolvlm256-download
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Runbook section 16, step 1: downloads SmolVLM2-256M-Video-Instruct at its pinned revision into $HF_HOME (as your shell sets it; this
# job doesn't change it), without the ONNX exports: about 0.5 GB. Needs internet on the node, as download_models.SBATCH. Every other
# job runs offline. Rerunning skips what is already there. Submit from the repository root.
set -euo pipefail
export HF_HUB_OFFLINE=0
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
run_in_env "python -u -m accuracy_tables.hpc.download_models smolvlm256"

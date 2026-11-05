#!/bin/bash
#SBATCH --job-name=acc-models
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Downloads the six models. Needs internet on the node, and HF_TOKEN exported in the shell that runs sbatch (for Gemma).
# Rerunning skips what is already there. Submit from the repository root.
set -euo pipefail
export HF_HUB_OFFLINE=0
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
run_in_env "python -u -m accuracy_tables.hpc.download_models"

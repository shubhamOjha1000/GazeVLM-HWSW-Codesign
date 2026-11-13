#!/bin/bash
#SBATCH --job-name=qwen-flops-tests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Runbook section 15, step 1: every test in tests/, on the CPU. No dataset, no model. Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
run_in_env "python -m pytest tests -q"

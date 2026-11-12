#!/bin/bash
#SBATCH --job-name=smolvlm-flops-summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# GPU use and the FLOPs-matched table for SmolVLM (runbook section 14), without holding a GPU. Safe to rerun any time. Submit from the
# repository root:
#   sbatch accuracy_tables/hpc/smolvlm_flops_summary.SBATCH
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
A=accuracy_tables.hpc.all350
OUT=$GV_ROOT/results/smolvlm_flops
run_in_env "python -u -m $A.gpu_summary $GV_ROOT/logs/gpu_smolvlm-flops-full_*.csv \
    && python -u -m $A.flops_report --out-dir $OUT --model smolvlm"

#!/bin/bash
#SBATCH --job-name=new-gate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Check 3 (runbook section 9): gate predictions on the 13 other AEA videos and all 325 Nymeria videos, then the comparison with the
# 12 known AEA videos. It stops with a list if any video has no rows. Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
M=accuracy_tables.hpc.newvideos
run_in_env "python -u -m $M.score_gate --group aea13 \
    && python -u -m $M.score_gate --group nymeria \
    && python -u -m $M.check_gate --groups aea12,aea13,nymeria"

#!/bin/bash
#SBATCH --job-name=all350-prepare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Runbook section 10, step 1: the CPU tests, the joined data folder, its checks, the pilot list and the token count. It only reads
# everything outside newvideos/all350/, and can be rerun. Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
M=accuracy_tables.hpc.all350
run_in_env "python -m pytest tests/test_all350.py tests/test_newvideos.py tests/test_budget_rule.py tests/test_accuracy_tables.py \
    tests/test_hpc.py -q \
    && python -u -m $M.prepare \
    && python -u -m $M.measure_tokens"

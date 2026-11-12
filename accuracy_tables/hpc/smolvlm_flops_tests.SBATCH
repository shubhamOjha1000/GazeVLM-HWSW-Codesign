#!/bin/bash
#SBATCH --job-name=smolvlm-flops-tests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Runbook section 14, step 1: the CPU tests of the FLOPs-matched run, and the section 11 tests it builds on. No dataset, no model.
# Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
run_in_env "python -m pytest tests/test_smolvlm_flops.py tests/test_smolvlm350.py tests/test_calibrate.py tests/test_budget_rule.py \
    tests/test_accuracy_tables.py -q"

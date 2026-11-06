#!/bin/bash
#SBATCH --job-name=qwen3350-tests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Runbook section 12, step 1: the CPU tests of what this task changed, and of section 11's path. No dataset, no model. Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
run_in_env "python -m pytest tests/test_qwen3350.py tests/test_smolvlm350.py tests/test_all350.py tests/test_newvideos.py tests/test_calibrate.py \
    tests/test_budget_rule.py tests/test_accuracy_tables.py tests/test_hpc.py tests/test_gazevlm.py tests/test_gazevlm2.py tests/test_avgpool.py \
    tests/test_streamingtom.py tests/test_dycoke.py tests/test_fastvid.py tests/test_prunevid.py tests/test_dtd.py -q"

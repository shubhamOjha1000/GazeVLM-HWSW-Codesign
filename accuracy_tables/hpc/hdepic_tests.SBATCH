#!/bin/bash
#SBATCH --job-name=hdepic-tests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Runbook section 17, step 5a, on the CPU: every test in tests/. No dataset, no model. Submit from the repository root:
#   sbatch accuracy_tables/hpc/hdepic_tests.SBATCH
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
run_in_env "python -m pytest tests -q"

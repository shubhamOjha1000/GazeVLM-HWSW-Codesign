#!/bin/bash
#SBATCH --job-name=new-rebuild
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Check 1 (runbook section 9): writes the video lists, runs the tests, rebuilds the 12 known videos with the new builder and
# compares them with data/folds/ and oof_preds_full.csv, which it only reads. The log must end with REBUILD MATCHES.
# Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
M=accuracy_tables.hpc.newvideos
run_in_env "python -u -m $M.plan \
    && python -m pytest tests/test_newvideos.py tests/test_budget_rule.py tests/test_accuracy_tables.py tests/test_hpc.py -q"
run_in_aria "python -u -m $M.project_gaze --list aea12"
run_in_env "python -u -m $M.build_rows --list aea12 && python -u -m $M.compare_rows"

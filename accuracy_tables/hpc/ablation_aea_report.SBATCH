#!/bin/bash
#SBATCH --job-name=ablation-aea-report
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# The temporal / spatial ablation's table (accuracy_tables/hpc/ablation_aea.py report), CPU only: per model and averaged over the models
# that have run, accuracy with its interval and TFLOPs per question for the 5 rows. Reads $GV_ROOT/results/ablation_aea/ and writes
# summary_ablation.csv there. No winner is declared. Submit from the repository root:
#   sbatch accuracy_tables/hpc/ablation_aea_report.SBATCH
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
run_in_env "python -u -m accuracy_tables.hpc.ablation_aea report --root $GV_ROOT/results/ablation_aea"
echo "ABLATION REPORT DONE"

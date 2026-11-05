#!/bin/bash
#SBATCH --job-name=acc-summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Writes results/hpc_<videos>/summary.csv. Safe to rerun any time.
#   sbatch accuracy_tables/hpc/summarize.SBATCH test
# Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
V=${1:?usage: summarize.SBATCH <test|all>}
OUT=$GV_ROOT/results/hpc_$V
run_in_env "python -u -m accuracy_tables.summarize --config $CONFIG --videos $V --out-dir $OUT"

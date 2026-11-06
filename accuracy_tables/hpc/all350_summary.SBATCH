#!/bin/bash
#SBATCH --job-name=all350-summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# GPU use and the results table for a run, without holding a GPU. Safe to rerun any time. Submit from the repository root:
#   sbatch accuracy_tables/hpc/all350_summary.SBATCH pilot
#   sbatch accuracy_tables/hpc/all350_summary.SBATCH full
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
M=accuracy_tables.hpc.all350
MODE=${1:?usage: all350_summary.SBATCH <pilot|full>}
case "$MODE" in
    pilot) OUT=$GV_ROOT/results/hpc350_pilot; ONLY="--list all350_pilot" ;;
    full) OUT=$GV_ROOT/results/hpc_350; ONLY="" ;;
    *) echo "usage: all350_summary.SBATCH <pilot|full>" >&2; exit 1 ;;
esac
run_in_env "python -u -m $M.gpu_summary $GV_ROOT/logs/gpu_all350-$MODE-*.csv \
    && python -u -m $M.report --out-dir $OUT $ONLY"

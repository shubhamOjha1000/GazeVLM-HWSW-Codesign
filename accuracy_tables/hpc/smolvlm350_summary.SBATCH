#!/bin/bash
#SBATCH --job-name=smolvlm350-summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# GPU use and the results table for SmolVLM's run (runbook section 11), without holding a GPU. Safe to rerun any time. Submit from the repository root:
#   sbatch accuracy_tables/hpc/smolvlm350_summary.SBATCH pilot
#   sbatch accuracy_tables/hpc/smolvlm350_summary.SBATCH full
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
A=accuracy_tables.hpc.all350
MODE=${1:?usage: smolvlm350_summary.SBATCH <pilot|full>}
case "$MODE" in
    pilot) OUT=$GV_ROOT/results/smolvlm_350_pilot; ONLY="--list all350_pilot" ;;
    full) OUT=$GV_ROOT/results/smolvlm_350; ONLY="" ;;
    *) echo "usage: smolvlm350_summary.SBATCH <pilot|full>" >&2; exit 1 ;;
esac
run_in_env "python -u -m $A.gpu_summary $GV_ROOT/logs/gpu_smolvlm350-${MODE}_*.csv \
    && python -u -m $A.report --out-dir $OUT $ONLY --models smolvlm --targets $GV_ROOT/results/smolvlm_350/smolvlm/targets.csv"

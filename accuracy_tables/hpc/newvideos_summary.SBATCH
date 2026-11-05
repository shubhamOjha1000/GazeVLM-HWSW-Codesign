#!/bin/bash
#SBATCH --job-name=new-summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# summary.csv and budget_compare.csv for a Nymeria run, without holding a GPU. Safe to rerun any time. Submit from the repository root:
#   sbatch accuracy_tables/hpc/newvideos_summary.SBATCH pilot
#   sbatch accuracy_tables/hpc/newvideos_summary.SBATCH full
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
CONFIG=accuracy_tables/hpc/config.newvideos.yaml
case "${1:?usage: newvideos_summary.SBATCH <pilot|full>}" in
    pilot) OUT=$GV_ROOT/results/nymeria_pilot ;;
    full) OUT=$GV_ROOT/results/nymeria_smolvlm ;;
    *) echo "usage: newvideos_summary.SBATCH <pilot|full>" >&2; exit 1 ;;
esac
run_in_env "python -u -m accuracy_tables.summarize --config $CONFIG --out-dir $OUT \
    && python -u -m accuracy_tables.compare_budget --model smolvlm --out-dir $OUT"

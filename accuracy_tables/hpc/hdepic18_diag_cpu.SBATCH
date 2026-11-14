#!/bin/bash
#SBATCH --job-name=hdepic18-diag-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Runbook section 18, diagnostic C1 or C2, on the CPU (diagnostic only: no setting changes; nothing feeds a results table). Every results
# folder is only read; writes only $GV_ROOT/data/hdepic/diagnose18/. The log holds HD-EPIC question text: never commit it.
#   sbatch --job-name=hdepic18-diag-c1 accuracy_tables/hpc/hdepic18_diag_cpu.SBATCH c1    frame provenance and prompts
#   sbatch --job-name=hdepic18-diag-c2 accuracy_tables/hpc/hdepic18_diag_cpu.SBATCH c2    answer agreement from the saved answers
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
export HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}
MODE=${1:?usage: hdepic18_diag_cpu.SBATCH <c1|c2>}
case "$MODE" in
    c1|c2) ;;
    *) echo "usage: hdepic18_diag_cpu.SBATCH <c1|c2>" >&2; exit 1 ;;
esac
run_in_env "python -u -m accuracy_tables.hpc.hdepic.diag18 $MODE"

#!/bin/bash
#SBATCH --job-name=hdepic18-diag-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:b200:1
#SBATCH --time=03:00:00
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --dependency=singleton
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Runbook section 18, diagnostic C3 or C4, on one B200 (section 14's GPU type; diagnostic only: no setting changes, nothing feeds a results
# table). Both start with the parity check. Every results folder is only read; writes only $GV_ROOT/data/hdepic/diagnose18/. The two jobs
# have their own names, so they run side by side; a requeue continues from the answers already saved.
#   sbatch --job-name=hdepic18-diag-c3 accuracy_tables/hpc/hdepic18_diag_gpu.SBATCH c3    the foveated builder at full resolution vs stock
#   sbatch --job-name=hdepic18-diag-c4 accuracy_tables/hpc/hdepic18_diag_gpu.SBATCH c4    the ablation ladder on 940 questions, K' = 9
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
export HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}
GPU=B200
MODE=${1:?usage: hdepic18_diag_gpu.SBATCH <c3|c4>}
case "$MODE" in
    c3|c4) WANT=hdepic18-diag-$MODE ;;
    *) echo "usage: hdepic18_diag_gpu.SBATCH <c3|c4>" >&2; exit 1 ;;
esac
if [ "$SLURM_JOB_NAME" != "$WANT" ]; then
    echo "this job must be named $WANT: submit it with --job-name=$WANT" >&2
    exit 1
fi
DIAG=$GV_ROOT/data/hdepic/diagnose18/$MODE
export OMP_NUM_THREADS=1
NV=--nv
gpu_log
mkdir -p "$DIAG"
run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU --record $DIAG/gpu_types.txt \
    && python -u -m accuracy_tables.hpc.hdepic.diag18 $MODE \
    && python -u -m accuracy_tables.hpc.all350.gpu_summary $GV_ROOT/logs/gpu_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.csv"
echo "DIAG DONE: $MODE"

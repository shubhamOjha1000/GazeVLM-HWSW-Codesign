#!/bin/bash
#SBATCH --job-name=hdepic-v2-stage1
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
# Ours v2, Stage 1 on one B200 (diagnostic dev: nothing feeds a results table). The 940 dev questions, 9 Uniform frames each, no gate, no
# budget: stock 81, blind, avgpool 4x4, L1, L2, L3 at the gaze, L2 at the centre, the periphery alone at L2's tokens. Starts with the
# parity check. Writes only $GV_ROOT/data/hdepic/v2/stage1/; a requeue continues from the answers already saved. Submit from the root:
#   sbatch accuracy_tables/hpc/hdepic_v2_gpu.SBATCH
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
export HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}
GPU=B200
OUT=$GV_ROOT/data/hdepic/v2/stage1
export OMP_NUM_THREADS=1
NV=--nv
gpu_log
mkdir -p "$OUT"
run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU --record $OUT/gpu_types.txt \
    && python -u -m accuracy_tables.hpc.hdepic.v2 stage1 \
    && python -u -m accuracy_tables.hpc.all350.gpu_summary $GV_ROOT/logs/gpu_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.csv"
echo "V2 DONE: stage1"

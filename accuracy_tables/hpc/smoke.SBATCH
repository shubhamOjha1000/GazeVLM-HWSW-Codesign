#!/bin/bash
#SBATCH --job-name=acc-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Every method on every model on one question; the log must end with "=== 0 failed of 66". Uses default budgets, in its
# own folder, so it never mixes with real results. Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
NV=--nv
gpu_log
rm -rf "$GV_ROOT/results/smoke"
run_in_env "python -u -m accuracy_tables.hpc.check_gpu \
    && python -u -m accuracy_tables.run --config $CONFIG --model all --videos loc2_script1_seq2_rec1 --max-questions 1 \
        --out-dir $GV_ROOT/results/smoke --keep-going"

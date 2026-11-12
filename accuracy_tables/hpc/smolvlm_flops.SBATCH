#!/bin/bash
#SBATCH --job-name=smolvlm-flops
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:b200:1
#SBATCH --time=08:00:00
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --dependency=singleton
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# SmolVLM on all 350 videos with every row matched on total FLOPs per question (runbook section 14), all on one GPU type:
#   sbatch --job-name=smolvlm-flops-calibrate accuracy_tables/hpc/smolvlm_flops.SBATCH calibrate   the plan: targets, K', knobs, checked FLOPs
#   sbatch --job-name=smolvlm-flops-full      accuracy_tables/hpc/smolvlm_flops.SBATCH full        every planned condition on all 350 videos
# Results are saved after every question, so rerunning (or a requeue after preemption) continues where it stopped. singleton: jobs with the
# same name never run together, because the repair step touches the whole results folder. Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
CONFIG=accuracy_tables/hpc/config.all350.yaml
M=smolvlm
GPU=B200
MODE=${1:?usage: smolvlm_flops.SBATCH <calibrate|full>}
W=8
OUT=$GV_ROOT/results/smolvlm_flops
PLAN=$OUT/$M/flops_plan.json
export OMP_NUM_THREADS=1      # 8 loading threads each starting 8 OpenMP threads would fight over the 8 cores
NV=--nv
gpu_log
mkdir -p "$OUT/$M"
case "$MODE" in
    calibrate)
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU --record $OUT/$M/gpu_types.txt \
            && python -u -m accuracy_tables.hpc.repair_results $OUT/$M \
            && python -u -m accuracy_tables.flops_calibrate --config $CONFIG --model $M --videos all --prefetch-workers $W --out-dir $OUT"
        ;;
    full)
        if [ ! -f "$PLAN" ]; then
            echo "no $PLAN: run the calibrate job first" >&2
            exit 1
        fi
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU --same-as $OUT/$M/gpu_types.txt --record $OUT/$M/gpu_types.txt \
            && python -u -m accuracy_tables.hpc.repair_results $OUT/$M \
            && python -u -m accuracy_tables.flops_run --config $CONFIG --model $M --plan $PLAN --videos all --prefetch-workers $W \
                --out-dir $OUT --keep-going"
        ;;
    *)
        echo "usage: smolvlm_flops.SBATCH <calibrate|full>" >&2
        exit 1
        ;;
esac

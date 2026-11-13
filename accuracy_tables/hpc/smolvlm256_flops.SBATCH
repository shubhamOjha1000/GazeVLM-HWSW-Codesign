#!/bin/bash
#SBATCH --job-name=smolvlm256-flops
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:b200:1
#SBATCH --time=08:00:00
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --dependency=singleton
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# SmolVLM2-256M on all 350 videos with every row matched on total FLOPs per question (runbook section 16): section 14's design on one B200.
#   sbatch --job-name=smolvlm256-flops-calibrate accuracy_tables/hpc/smolvlm256_flops.SBATCH calibrate
#       the plan (targets, K', knobs, checked FLOPs), then a timing run on every 25th video, the projected full run and this job's GPU use
#   sbatch --job-name=smolvlm256-flops-full      accuracy_tables/hpc/smolvlm256_flops.SBATCH full
#       every planned condition on all 350 videos
# Results are saved after every question, so rerunning (or a requeue after preemption) continues where it stopped, skipping every finished
# record. singleton, and each mode's own job name, keep two runs from writing the same files. 16 loading threads on 16 cores keep the GPU
# fed; step 2's check job showed they change no answer. Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
CONFIG=accuracy_tables/hpc/config.smolvlm256.yaml
M=smolvlm256
GPU=B200
MODE=${1:?usage: smolvlm256_flops.SBATCH <calibrate|full>}
W=16
C=accuracy_tables.hpc.all350.smolvlm256_checks
OUT=$GV_ROOT/results/smolvlm256_flops
PLAN=$OUT/$M/flops_plan.json
TIMING=$OUT/timing
export OMP_NUM_THREADS=1      # 16 loading threads each starting 16 OpenMP threads would fight over the 16 cores
case "$MODE" in
    calibrate|full) WANT=smolvlm256-flops-$MODE ;;
    *) echo "usage: smolvlm256_flops.SBATCH <calibrate|full>" >&2; exit 1 ;;
esac
if [ "$SLURM_JOB_NAME" != "$WANT" ]; then
    echo "this job must be named $WANT, or singleton won't protect its results: submit it with --job-name=$WANT" >&2
    exit 1
fi
NV=--nv
gpu_log
mkdir -p "$OUT/$M"
case "$MODE" in
    calibrate)
        if [ -f "$PLAN" ]; then
            echo "$PLAN exists, and the full run reads it. Calibrating again needs the user's OK: stop and tell them." >&2
            exit 1
        fi
        SAME=""
        if [ -s "$OUT/$M/gpu_types.txt" ]; then    # a rerun after a time limit or a preemption continues on the same GPU name
            SAME="--same-as $OUT/$M/gpu_types.txt"
        fi
        rm -rf "$TIMING"
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU $SAME --record $OUT/$M/gpu_types.txt \
            && python -u -m accuracy_tables.hpc.repair_results $OUT/$M \
            && python -u -m accuracy_tables.flops_calibrate --config $CONFIG --model $M --videos all --prefetch-workers $W --out-dir $OUT \
            && V=\$(python -m $C pick-videos --every 25) \
            && python -u -m accuracy_tables.flops_run --config $CONFIG --model $M --plan $PLAN --videos \$V --prefetch-workers $W \
                --out-dir $TIMING --keep-going \
            && python -u -m $C estimate --timing $TIMING/$M \
            && python -u -m accuracy_tables.hpc.all350.gpu_summary $GV_ROOT/logs/gpu_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.csv"
        chmod a-w "$PLAN" "$OUT/$M/flops_targets.csv"
        echo "CALIBRATE DONE: $M"
        ;;
    full)
        if [ ! -f "$PLAN" ]; then
            echo "no $PLAN: run the calibrate job first" >&2
            exit 1
        fi
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU --same-as $OUT/$M/gpu_types.txt --record $OUT/$M/gpu_types.txt \
            && python -u -m accuracy_tables.hpc.repair_results $OUT/$M \
            && python -u -m accuracy_tables.flops_run --config $CONFIG --model $M --plan $PLAN --videos all --prefetch-workers $W \
                --out-dir $OUT --keep-going \
            && python -u -m accuracy_tables.hpc.all350.gpu_summary $GV_ROOT/logs/gpu_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.csv"
        echo "RUN DONE: full $M (one job, so this stands for section 15's SHARD DONE)"
        ;;
esac

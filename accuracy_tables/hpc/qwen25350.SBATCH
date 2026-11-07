#!/bin/bash
#SBATCH --job-name=qwen25350
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --dependency=singleton
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Qwen2.5-VL-3B on all 350 videos (runbook section 13): section 12's rows and Python on this model, with no check job and no pilot.
#   sbatch --job-name=qwen25350-calibrate accuracy_tables/hpc/qwen25350.SBATCH calibrate     once, before any shard: the assignment and the knobs
#   sbatch --job-name=qwen25350-full-0    accuracy_tables/hpc/qwen25350.SBATCH full 0        all 350 videos, one shard (also 1 and 2)
# Each shard has its own job name, so singleton only stops that shard running twice, and the job refuses any other name. Results are saved after
# every question, so rerunning a shard (or a requeue after preemption) continues from its own files. Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
CONFIG=accuracy_tables/hpc/config.all350.yaml
M=qwen2.5
GPU=A100
MODE=${1:?usage: qwen25350.SBATCH <calibrate|full> [shard]}
SHARD=${2:-}
A=accuracy_tables.hpc.all350
SHARDS=3
W=8
OUT=$GV_ROOT/results/qwen25_350
CAL=$OUT/calibrate
BASE=$CAL/config_baselines.yaml
TARGETS=$CAL/$M/targets.csv
CAL_GPU=$CAL/$M/gpu_types.txt
ROWS=uniform,blind,budget,avgpool,gazevlm,gazevlm2,streamingtom,dycoke,fastvid,prunevid,dtd
export OMP_NUM_THREADS=1      # 8 loading threads each starting 8 OpenMP threads would fight over the 8 cores

case "$MODE" in
    calibrate) WANT=qwen25350-calibrate ;;
    full)
        case "$SHARD" in
            0|1|2) WANT=qwen25350-full-$SHARD ;;
            *) echo "full needs a shard: 0, 1 or 2" >&2; exit 1 ;;
        esac
        ;;
    *) echo "usage: qwen25350.SBATCH <calibrate|full> [shard]" >&2; exit 1 ;;
esac
if [ "$SLURM_JOB_NAME" != "$WANT" ]; then
    echo "this job must be named $WANT, or singleton won't protect its results: submit it with --job-name=$WANT" >&2
    exit 1
fi
NV=--nv
if [ "$MODE" = calibrate ]; then
    if [ -f "$TARGETS" ]; then
        echo "$TARGETS exists, and the shards read it. Calibrating again needs the user's OK: stop and tell them." >&2
        exit 1
    fi
    gpu_log
    mkdir -p "$CAL/$M"
    SAME=""
    if [ -s "$CAL_GPU" ]; then    # a rerun after a time limit or a preemption continues on the same GPU name
        SAME="--same-as $CAL_GPU"
    fi
    run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU $SAME --record $CAL_GPU \
        && python -u -m accuracy_tables.hpc.repair_results $CAL/$M \
        && python -u -m $A.shards write --root $OUT --shards $SHARDS --all \
        && python -u -m accuracy_tables.calibrate --config $CONFIG --model $M --fractions --videos all \
            --prefetch-workers $W --counts $CAL/$M/kept_counts.jsonl --out-dir $CAL --write $BASE"
    chmod a-w "$TARGETS" "$BASE"
else
    if [ ! -f "$TARGETS" ] || [ ! -f "$BASE" ]; then
        echo "no $TARGETS or $BASE: run the calibrate job first" >&2
        exit 1
    fi
    gpu_log
    S=$OUT/shards/$SHARD
    VIDEOS=$(paste -sd, "$OUT/manifest/shard_$SHARD.txt")
    mkdir -p "$S/$M"
    run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU --same-as $CAL_GPU --record $S/$M/gpu_types.txt \
        && python -u -m accuracy_tables.hpc.repair_results $S/$M \
        && python -u -m $A.shards start --root $OUT --shard $SHARD --model $M --config $BASE --targets $TARGETS --rows $ROWS --workers $W \
        && python -u -m accuracy_tables.run --config $BASE --model $M --rows $ROWS --videos $VIDEOS \
            --prefetch-workers $W --out-dir $S --keep-going"
    echo "SHARD DONE: full $SHARD"
fi

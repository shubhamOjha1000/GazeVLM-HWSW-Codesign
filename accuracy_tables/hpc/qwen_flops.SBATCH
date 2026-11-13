#!/bin/bash
#SBATCH --job-name=qwen-flops
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
# Qwen2.5-VL-3B or Qwen3-VL-4B on all 350 videos with every row matched on total FLOPs (runbook section 15): section 14's design, split
# across 3 one-GPU A100 jobs per model on the split sections 12 and 13 ran on.
#   sbatch --job-name=qwen25-flops-calibrate accuracy_tables/hpc/qwen_flops.SBATCH qwen2.5 calibrate   once: the split and the sealed plan
#   sbatch --job-name=qwen25-flops-full-0    accuracy_tables/hpc/qwen_flops.SBATCH qwen2.5 full 0      one shard (also 1 and 2)
#   (qwen3: job names qwen3-flops-calibrate and qwen3-flops-full-<n>)
# Each job has its own name, so singleton only stops that job running twice, and the job refuses any other name. Results are saved after
# every question, so rerunning a shard (or a requeue after preemption) continues from its own files. Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
USAGE="usage: qwen_flops.SBATCH <qwen2.5|qwen3> <calibrate|full> [shard]"
CONFIG=accuracy_tables/hpc/config.all350.yaml
M=${1:?$USAGE}
MODE=${2:?$USAGE}
SHARD=${3:-}
case "$M" in
    qwen2.5) TAG=qwen25; SECTION=qwen25_350 ;;
    qwen3) TAG=qwen3; SECTION=qwen3_350 ;;
    *) echo "$USAGE" >&2; exit 1 ;;
esac
GPU=A100
A=accuracy_tables.hpc.all350
W=8
OUT=$GV_ROOT/results/${TAG}_flops
CAL=$OUT/calibrate
PLAN=$CAL/$M/flops_plan.json
TARGETS=$CAL/$M/flops_targets.csv
SEAL=$CAL/$M/flops_plan.sha256
CAL_GPU=$CAL/$M/gpu_types.txt
ROWS=uniform,blind,budget,nogate,uniform_flops,gazevlm,streamingtom,dycoke,fastvid,prunevid,dtd,avgpool,gazevlm2
export OMP_NUM_THREADS=1      # 8 loading threads each starting 8 OpenMP threads would fight over the 8 cores

case "$MODE" in
    calibrate) WANT=$TAG-flops-calibrate ;;
    full)
        case "$SHARD" in
            0|1|2) WANT=$TAG-flops-full-$SHARD ;;
            *) echo "full needs a shard: 0, 1 or 2" >&2; exit 1 ;;
        esac
        ;;
    *) echo "$USAGE" >&2; exit 1 ;;
esac
if [ "$SLURM_JOB_NAME" != "$WANT" ]; then
    echo "this job must be named $WANT, or singleton won't protect its results: submit it with --job-name=$WANT" >&2
    exit 1
fi
NV=--nv
if [ "$MODE" = calibrate ]; then
    if [ -f "$PLAN" ] || [ -f "$SEAL" ]; then
        echo "$PLAN exists, and the shards read it. Calibrating again needs the user's OK: stop and tell them." >&2
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
        && python -u -m $A.flops_shards split --source $GV_ROOT/results/$SECTION --root $OUT \
        && python -u -m accuracy_tables.flops_calibrate --config $CONFIG --model $M --videos all --prefetch-workers $W --out-dir $CAL \
        && python -u -m $A.flops_shards seal --root $OUT --model $M"
    chmod a-w "$PLAN" "$TARGETS" "$SEAL"
    echo "CALIBRATE DONE: $M"
else
    if [ ! -f "$PLAN" ] || [ ! -f "$SEAL" ]; then
        echo "no $PLAN or $SEAL: run the calibrate job first" >&2
        exit 1
    fi
    gpu_log
    S=$OUT/shards/$SHARD
    VIDEOS=$(paste -sd, "$OUT/manifest/shard_$SHARD.txt")
    mkdir -p "$S/$M"
    run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU --same-as $CAL_GPU --record $S/$M/gpu_types.txt \
        && python -u -m accuracy_tables.hpc.repair_results $S/$M \
        && python -u -m $A.flops_shards start --root $OUT --shard $SHARD --model $M --config $CONFIG --rows $ROWS --workers $W \
        && python -u -m accuracy_tables.flops_run --config $CONFIG --model $M --plan $PLAN --videos $VIDEOS --prefetch-workers $W \
            --out-dir $S --keep-going"
    echo "SHARD DONE: full $SHARD"
fi

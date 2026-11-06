#!/bin/bash
#SBATCH --job-name=qwen3350
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
# Qwen3-VL-4B on all 350 videos (runbook section 12): the rows of section 11, split across 3 separate one-GPU jobs, all on one GPU type.
#   sbatch --job-name=qwen3350-calibrate    accuracy_tables/hpc/qwen3350.SBATCH calibrate      once, before any shard: the assignments and the knobs
#   sbatch --job-name=qwen3350-check        accuracy_tables/hpc/qwen3350.SBATCH check          parity, threads, full budget, the small set on one A100
#   sbatch --job-name=qwen3350-checkshard-0 accuracy_tables/hpc/qwen3350.SBATCH checkshard 0   the small set, one shard (also 1 and 2)
#   sbatch --job-name=qwen3350-pilot-0      accuracy_tables/hpc/qwen3350.SBATCH pilot 0        the 7 pilot videos, one shard (also 1 and 2)
#   sbatch --job-name=qwen3350-full-0       accuracy_tables/hpc/qwen3350.SBATCH full 0         all 350 videos, one shard (also 1 and 2)
# Each shard has its own job name, so singleton only stops that shard running twice, and the job refuses any other name. Results are saved after
# every question, so rerunning a shard (or a requeue after preemption) continues from its own files. Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
CONFIG=accuracy_tables/hpc/config.all350.yaml
M=qwen3
GPU=A100
MODE=${1:?usage: qwen3350.SBATCH <calibrate|check|checkshard|pilot|full> [shard]}
SHARD=${2:-}
A=accuracy_tables.hpc.all350
SHARDS=3
W=8
N=30
OUT=$GV_ROOT/results/qwen3_350
CK=$GV_ROOT/results/qwen3_350_check
PILOT=$GV_ROOT/results/qwen3_350_pilot
CAL=$OUT/calibrate
BASE=$CAL/config_baselines.yaml
TARGETS=$CAL/$M/targets.csv
CAL_GPU=$CAL/$M/gpu_types.txt
BASELINES=avgpool,gazevlm,gazevlm2,streamingtom,dycoke,fastvid,prunevid,dtd
ROWS=uniform,blind,budget,$BASELINES
LIST=$GV_ROOT/newvideos/lists/all350_pilot.txt
export OMP_NUM_THREADS=1      # 8 loading threads each starting 8 OpenMP threads would fight over the 8 cores

case "$MODE" in
    calibrate|check) WANT=qwen3350-$MODE ;;
    checkshard|pilot|full)
        case "$SHARD" in
            0|1|2) WANT=qwen3350-$MODE-$SHARD ;;
            *) echo "$MODE needs a shard: 0, 1 or 2" >&2; exit 1 ;;
        esac
        ;;
    *) echo "usage: qwen3350.SBATCH <calibrate|check|checkshard|pilot|full> [shard]" >&2; exit 1 ;;
esac
if [ "$SLURM_JOB_NAME" != "$WANT" ]; then
    echo "this job must be named $WANT, or singleton won't protect its results: submit it with --job-name=$WANT" >&2
    exit 1
fi
if [ "$MODE" != calibrate ] && { [ ! -f "$TARGETS" ] || [ ! -f "$BASE" ]; }; then
    echo "no $TARGETS or $BASE: run the calibrate job first" >&2
    exit 1
fi
NV=--nv
gpu_log

run_shard() {                 # one shard of the results folder $1: it writes only to its own folder, and a rerun continues from its own files
    local ROOT=$1
    local S=$ROOT/shards/$SHARD
    local VIDEOS
    VIDEOS=$(paste -sd, "$ROOT/manifest/shard_$SHARD.txt")
    mkdir -p "$S/$M"
    run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU --same-as $CAL_GPU --record $S/$M/gpu_types.txt \
        && python -u -m accuracy_tables.hpc.repair_results $S/$M \
        && python -u -m $A.shards start --root $ROOT --shard $SHARD --model $M --config $BASE --targets $TARGETS --rows $ROWS --workers $W \
        && python -u -m accuracy_tables.run --config $BASE --model $M --rows $ROWS --videos $VIDEOS \
            --prefetch-workers $W --out-dir $S --keep-going"
    echo "SHARD DONE: $MODE $SHARD"
}

case "$MODE" in
    calibrate)
        if [ -f "$TARGETS" ]; then
            echo "$TARGETS exists, and the shards read it. Calibrating again needs the user's OK: stop and tell them." >&2
            exit 1
        fi
        mkdir -p "$CAL/$M"
        SAME=""
        if [ -s "$CAL_GPU" ]; then    # a rerun after a time limit or a preemption continues on the same GPU name
            SAME="--same-as $CAL_GPU"
        fi
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU $SAME --record $CAL_GPU \
            && python -u -m accuracy_tables.hpc.repair_results $CAL/$M \
            && python -u -m $A.shards write --root $OUT --shards $SHARDS --all \
            && python -u -m $A.shards write --root $PILOT --shards $SHARDS --list all350_pilot \
            && python -u -m $A.shards write --root $CK --shards $SHARDS --smallest \
            && python -u -m accuracy_tables.calibrate --config $CONFIG --model $M --fractions --videos all \
                --prefetch-workers $W --counts $CAL/$M/kept_counts.jsonl --out-dir $CAL --write $BASE"
        chmod a-w "$TARGETS" "$BASE"
        ;;
    check)
        NYMERIA=$(sed -n 1p "$LIST")
        SMALL=$(paste -sd, "$CK/manifest/all_videos.txt")
        rm -rf "$CK/off" "$CK/on" "$CK/identity" "$CK/whole" "$CK/shards" "$CK/$M"
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU --same-as $CAL_GPU --record $CK/whole/$M/gpu_types.txt \
            && python -u -m accuracy_tables.run --config $BASE --model $M --rows $ROWS --videos $NYMERIA --max-questions $N \
                --prefetch-workers 0 --out-dir $CK/off --keep-going \
            && python -u -m accuracy_tables.run --config $BASE --model $M --rows $ROWS --videos $NYMERIA --max-questions $N \
                --prefetch-workers $W --out-dir $CK/on --keep-going \
            && python -u -m accuracy_tables.hpc.newvideos.compare_records --new $CK/on/$M --ref $CK/off/$M --rows $ROWS \
            && python -u -m $A.check_identity --write $CK/identity.yaml --config $CONFIG \
            && python -u -m accuracy_tables.run --config $CK/identity.yaml --model $M --rows uniform,avgpool,dtd,dycoke,streamingtom \
                --videos $NYMERIA --max-questions $N --prefetch-workers $W --out-dir $CK/identity --keep-going \
            && python -u -m $A.check_identity --compare $CK/identity/$M \
            && python -u -m accuracy_tables.run --config $BASE --model $M --rows $ROWS --videos $SMALL \
                --prefetch-workers $W --out-dir $CK/whole --keep-going"
        echo "CHECK DONE: $M"
        ;;
    checkshard) run_shard "$CK" ;;
    pilot) run_shard "$PILOT" ;;
    full) run_shard "$OUT" ;;
esac

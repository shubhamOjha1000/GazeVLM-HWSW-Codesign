#!/bin/bash
#SBATCH --job-name=hdepic18-flops
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
# SmolVLM2-2.2B on HD-EPIC's 6 groups with every row matched on total FLOPs (runbook section 18, steps 6 and 7), one B200 a job:
#   sbatch --job-name=hdepic18-flops-calibrate accuracy_tables/hpc/hdepic18_flops.SBATCH calibrate
#       the plan on every question (targets, K', knobs, checked FLOPs), a timing run on every 25th question, the projected full run
#   sbatch --job-name=hdepic18-flops-full      accuracy_tables/hpc/hdepic18_flops.SBATCH full          one job for every question, or,
#   sbatch --job-name=hdepic18-flops-full-0    accuracy_tables/hpc/hdepic18_flops.SBATCH full 0        only on the user's OK when the
#       projection is over 6 hours: 3 shards (also 1 and 2), each every 3rd question, so they hold the same count to within 1
# Section 14's settings and code (flops_calibrate.py, flops_run.py); only the data differ. Results are saved after every question, so a
# rerun (or a requeue after preemption) continues, skipping finished records. Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
export HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}
CONFIG=accuracy_tables/hpc/config.hdepic18.yaml
M=smolvlm
GPU=B200
MODE=${1:?usage: hdepic18_flops.SBATCH <calibrate|full> [shard]}
SHARD=${2:-}
W=16
C=accuracy_tables.hpc.all350.smolvlm256_checks
H=accuracy_tables.hpc.hdepic.checks18
OUT=$GV_ROOT/results/hdepic_all_smolvlm_flops
PLAN=$OUT/$M/flops_plan.json
TIMING=$OUT/timing
export OMP_NUM_THREADS=1
case "$MODE" in
    calibrate) WANT=hdepic18-flops-calibrate ;;
    full)
        case "$SHARD" in
            "") WANT=hdepic18-flops-full ;;
            0|1|2) WANT=hdepic18-flops-full-$SHARD ;;
            *) echo "a shard is 0, 1 or 2" >&2; exit 1 ;;
        esac
        ;;
    *) echo "usage: hdepic18_flops.SBATCH <calibrate|full> [shard]" >&2; exit 1 ;;
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
        if [ -s "$OUT/$M/gpu_types.txt" ]; then
            SAME="--same-as $OUT/$M/gpu_types.txt"
        fi
        rm -rf "$TIMING"
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU $SAME --record $OUT/$M/gpu_types.txt \
            && python -u -m accuracy_tables.hpc.repair_results $OUT/$M \
            && python -u -m accuracy_tables.flops_calibrate --config $CONFIG --model $M --videos all --prefetch-workers $W --out-dir $OUT \
            && V=\$(python -m $H pick --every 25) \
            && python -u -m accuracy_tables.flops_run --config $CONFIG --model $M --plan $PLAN --videos \$V --prefetch-workers $W \
                --out-dir $TIMING --keep-going \
            && T=\$(python -m $H totals) \
            && python -u -m $C estimate --timing $TIMING/$M --questions \${T% *} --videos \${T#* } \
            && python -u -m accuracy_tables.hpc.all350.gpu_summary $GV_ROOT/logs/gpu_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.csv"
        chmod a-w "$PLAN" "$OUT/$M/flops_targets.csv"
        echo "CALIBRATE DONE: hdepic18"
        ;;
    full)
        if [ ! -f "$PLAN" ]; then
            echo "no $PLAN: run the calibrate job first" >&2
            exit 1
        fi
        if [ -z "$SHARD" ]; then
            DEST=$OUT
            V=all
        else
            DEST=$OUT/shard$SHARD
            V='$(python -m '$H' pick --shard '$SHARD' --of 3)'
        fi
        mkdir -p "$DEST/$M"
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU --same-as $OUT/$M/gpu_types.txt --record $DEST/$M/gpu_types.txt \
            && python -u -m accuracy_tables.hpc.repair_results $DEST/$M \
            && python -u -m accuracy_tables.flops_run --config $CONFIG --model $M --plan $PLAN --videos $V --prefetch-workers $W \
                --out-dir $DEST --keep-going \
            && python -u -m accuracy_tables.hpc.all350.gpu_summary $GV_ROOT/logs/gpu_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.csv"
        echo "RUN DONE: full hdepic18${SHARD:+ shard $SHARD}"
        ;;
esac

#!/bin/bash
#SBATCH --job-name=hdepic-flops
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
# SmolVLM2-2.2B on HD-EPIC's gaze questions with every row matched on total FLOPs (runbook section 17, steps 6 and 7), on one B200:
#   sbatch --job-name=hdepic-flops-calibrate accuracy_tables/hpc/hdepic_flops.SBATCH calibrate
#       the plan (targets, K', knobs, checked FLOPs), a timing run on every 25th question, the projected full run and this job's GPU use
#   sbatch --job-name=hdepic-flops-full      accuracy_tables/hpc/hdepic_flops.SBATCH full
#       every planned condition on every question
# Section 14's settings and code (flops_calibrate.py, flops_run.py); only the data differ. Results are saved after every question, so a
# rerun (or a requeue after preemption) continues, skipping finished records. Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
CONFIG=accuracy_tables/hpc/config.hdepic.yaml
M=smolvlm
GPU=B200
MODE=${1:?usage: hdepic_flops.SBATCH <calibrate|full>}
W=16
C=accuracy_tables.hpc.all350.smolvlm256_checks
H=accuracy_tables.hpc.hdepic.checks
OUT=$GV_ROOT/results/hdepic_gaze_smolvlm_flops
PLAN=$OUT/$M/flops_plan.json
TIMING=$OUT/timing
export OMP_NUM_THREADS=1
case "$MODE" in
    calibrate|full) WANT=hdepic-flops-$MODE ;;
    *) echo "usage: hdepic_flops.SBATCH <calibrate|full>" >&2; exit 1 ;;
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
        echo "CALIBRATE DONE: hdepic"
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
        echo "RUN DONE: full hdepic (one job, so this stands for SHARD DONE)"
        ;;
esac

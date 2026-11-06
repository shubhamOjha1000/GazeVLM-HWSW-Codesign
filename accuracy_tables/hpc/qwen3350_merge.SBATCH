#!/bin/bash
#SBATCH --job-name=qwen3350-merge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --dependency=singleton
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Merges Qwen3's 3 shards, then summarizes them (runbook section 12), without holding a GPU. Submit by hand once all 3 shards have finished, from
# the repository root. It never depends on the shard jobs, because a shard that hits the time limit gets resubmitted. Safe to rerun any time:
#   sbatch accuracy_tables/hpc/qwen3350_merge.SBATCH check     then compares the merged files with the unsharded run
#   sbatch accuracy_tables/hpc/qwen3350_merge.SBATCH pilot     then compares with section 10's Qwen3 pilot, estimates the full run, summarizes
#   sbatch accuracy_tables/hpc/qwen3350_merge.SBATCH full      then summarizes
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
M=qwen3
A=accuracy_tables.hpc.all350
MODE=${1:?usage: qwen3350_merge.SBATCH <check|pilot|full>}
CAL=$GV_ROOT/results/qwen3_350/calibrate
TARGETS=$CAL/$M/targets.csv
ROWS=uniform,blind,budget,avgpool,gazevlm,gazevlm2,streamingtom,dycoke,fastvid,prunevid,dtd
case "$MODE" in
    check) ROOT=$GV_ROOT/results/qwen3_350_check ;;
    pilot) ROOT=$GV_ROOT/results/qwen3_350_pilot ;;
    full) ROOT=$GV_ROOT/results/qwen3_350 ;;
    *) echo "usage: qwen3350_merge.SBATCH <check|pilot|full>" >&2; exit 1 ;;
esac
MERGE="python -u -m $A.merge_shards --root $ROOT --model $M --targets $TARGETS --config $CAL/config_baselines.yaml --rows $ROWS --calibration-gpu $CAL/$M/gpu_types.txt"
REPORT="python -u -m $A.report --out-dir $ROOT --models $M --targets $TARGETS"
case "$MODE" in
    check)
        run_in_env "$MERGE \
            && python -u -m accuracy_tables.hpc.newvideos.compare_records --new $ROOT/$M --ref $ROOT/whole/$M --rows $ROWS --exact"
        ;;
    pilot)
        run_in_env "$MERGE \
            && python -u -m accuracy_tables.hpc.newvideos.compare_records --new $ROOT/$M --ref $GV_ROOT/results/hpc350_pilot/$M --rows uniform,blind,budget --report \
            && python -u -m $A.estimate_hours --root $ROOT --model $M \
            && python -u -m $A.gpu_summary $GV_ROOT/logs/gpu_qwen3350-pilot-*_*.csv \
            && $REPORT --list all350_pilot"
        ;;
    full)
        run_in_env "$MERGE \
            && python -u -m $A.gpu_summary $GV_ROOT/logs/gpu_qwen3350-full-*_*.csv \
            && $REPORT"
        ;;
esac

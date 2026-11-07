#!/bin/bash
#SBATCH --job-name=qwen25350-merge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --dependency=singleton
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Merges Qwen2.5's 3 shards, summarizes them, then compares with the earlier pilot (runbook section 13), without holding a GPU. Submit by hand once all
# 3 logs show SHARD DONE, from the repository root. It never depends on the shard jobs, because a shard that hits the time limit gets resubmitted.
# Safe to rerun any time.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
M=qwen2.5
A=accuracy_tables.hpc.all350
ROOT=$GV_ROOT/results/qwen25_350
CAL=$ROOT/calibrate
TARGETS=$CAL/$M/targets.csv
ROWS=uniform,blind,budget,avgpool,gazevlm,gazevlm2,streamingtom,dycoke,fastvid,prunevid,dtd
run_in_env "python -u -m $A.merge_shards --root $ROOT --model $M --targets $TARGETS --config $CAL/config_baselines.yaml --rows $ROWS --calibration-gpu $CAL/$M/gpu_types.txt \
    && python -u -m $A.gpu_summary $GV_ROOT/logs/gpu_qwen25350-full-*_*.csv \
    && python -u -m $A.report --out-dir $ROOT --models $M --targets $TARGETS"
# Only the merged results above are the run's. The pilot ran on an L40S, so a difference here is reported and never fails the job. The pilot's folder is
# only read: it's --new because only its records are the overlap.
if ! run_in_env "python -u -m accuracy_tables.hpc.newvideos.compare_records --new $GV_ROOT/results/hpc350_pilot/$M --ref $ROOT/$M --rows uniform,blind,budget --report"; then
    echo "The comparison with the L40S pilot didn't end with TOKENS MATCH. Report the lines above to the user. The merged results above still stand."
fi

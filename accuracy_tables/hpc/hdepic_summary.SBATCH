#!/bin/bash
#SBATCH --job-name=hdepic-summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Runbook section 17, step 8, without holding a GPU. Safe to rerun. Submit from the repository root:
#   sbatch accuracy_tables/hpc/hdepic_summary.SBATCH
# The one job's records (oom / nan, the GPU per record, every planned answer), the full job's GPU use, then the FLOPs tables for both
# types, each type and their mean, chance and Uniform minus Blind per type, and the side table against section 14 (only read).
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
A=accuracy_tables.hpc.all350
M=smolvlm
OUT=$GV_ROOT/results/hdepic_gaze_smolvlm_flops
ROWS=uniform,blind,budget,nogate,uniform_flops,gazevlm,streamingtom,dycoke,fastvid,prunevid,dtd,avgpool,gazevlm2
run_in_env "T=\$(python -m accuracy_tables.hpc.hdepic.checks totals) \
    && python -u -m $A.smolvlm256_checks records --folder $OUT/$M --plan $OUT/$M/flops_plan.json --rows $ROWS --questions \${T% *} \
    && python -u -m $A.gpu_summary $GV_ROOT/logs/gpu_hdepic-flops-full_*.csv \
    && python -u -m accuracy_tables.hpc.hdepic.report --out-dir $OUT --model $M --reference $GV_ROOT/results/smolvlm_flops/summary_flops_by_group.csv"
echo "written $OUT/summary_flops_by_group.csv, summary_chance_uniform_blind.csv and summary_vs_egoeverything.csv"

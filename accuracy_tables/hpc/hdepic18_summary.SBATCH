#!/bin/bash
#SBATCH --job-name=hdepic18-summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48GB
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Runbook section 18, step 8, without holding a GPU. Safe to rerun. Submit from the repository root:
#   sbatch accuracy_tables/hpc/hdepic18_summary.SBATCH          after the one full job
#   sbatch accuracy_tables/hpc/hdepic18_summary.SBATCH merge    after 3 shards: merge them first (once; it never overwrites)
# The records (oom / nan, the GPU per record, every planned answer), the full run's GPU use, then the FLOPs tables for all questions, each
# group, each type and the mean of the 6 groups, chance and Uniform minus Blind per group, and the side table against section 17 (only read).
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
A=accuracy_tables.hpc.all350
H=accuracy_tables.hpc.hdepic
M=smolvlm
OUT=$GV_ROOT/results/hdepic_all_smolvlm_flops
ROWS=uniform,blind,budget,nogate,uniform_flops,gazevlm,streamingtom,dycoke,fastvid,prunevid,dtd,avgpool,gazevlm2
MERGED_FROM=""
if [ -d "$OUT/shard0/$M" ]; then
    MERGED_FROM="--merged-from 3"
fi
if [ "${1:-}" = "merge" ]; then
    run_in_env "python -u -m $H.checks18 merge --out-dir $OUT --rows $ROWS --model $M"
fi
run_in_env "T=\$(python -m $H.checks18 totals) \
    && python -u -m $A.smolvlm256_checks records --folder $OUT/$M --plan $OUT/$M/flops_plan.json --rows $ROWS --questions \${T% *} $MERGED_FROM \
    && python -u -m $A.gpu_summary $GV_ROOT/logs/gpu_hdepic18-flops-full*.csv \
    && python -u -m $H.report18 --out-dir $OUT --model $M --reference $GV_ROOT/results/hdepic_gaze_smolvlm_flops/summary_flops_by_group.csv"
echo "written $OUT/summary_flops_by_group.csv, summary_chance_uniform_blind.csv and summary_vs_section17.csv"

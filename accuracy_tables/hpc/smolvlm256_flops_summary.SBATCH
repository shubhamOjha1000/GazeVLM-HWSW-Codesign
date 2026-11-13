#!/bin/bash
#SBATCH --job-name=smolvlm256-flops-summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# SmolVLM2-256M's FLOPs-matched table (runbook section 16, step 5), without holding a GPU. Safe to rerun any time. Submit from the
# repository root:
#   sbatch accuracy_tables/hpc/smolvlm256_flops_summary.SBATCH
# Checks the one job's records (oom / nan, the GPU per record, every planned answer), the full job's GPU use, then the table per group
# and the side table against SmolVLM2-2.2B's section 14 table (results/smolvlm_flops/, only read).
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
A=accuracy_tables.hpc.all350
M=smolvlm256
OUT=$GV_ROOT/results/smolvlm256_flops
ROWS=uniform,blind,budget,nogate,uniform_flops,gazevlm,streamingtom,dycoke,fastvid,prunevid,dtd,avgpool,gazevlm2
run_in_env "python -u -m $A.smolvlm256_checks records --folder $OUT/$M --plan $OUT/$M/flops_plan.json --rows $ROWS \
    && python -u -m $A.gpu_summary $GV_ROOT/logs/gpu_smolvlm256-flops-full_*.csv \
    && python -u -m $A.flops_report --out-dir $OUT --model $M \
    && python -u -m $A.smolvlm256_checks vs2p2b --ours $OUT/summary_flops_by_group.csv \
        --other $GV_ROOT/results/smolvlm_flops/summary_flops_by_group.csv --out $OUT/summary_vs_2p2b.csv"
echo "TOKENS MATCH: not applicable, different model (64 tokens per frame here, 81 on SmolVLM2-2.2B, and another LLM)"
echo "written $OUT/summary_flops_by_group.csv and $OUT/summary_vs_2p2b.csv"

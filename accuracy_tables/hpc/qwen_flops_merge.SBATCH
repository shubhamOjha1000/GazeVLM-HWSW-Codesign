#!/bin/bash
#SBATCH --job-name=qwen-flops-merge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --dependency=singleton
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Merges one model's 3 FLOPs-matched shards, then writes its FLOPs table (runbook section 15), without holding a GPU. Submit by hand once
# all 3 of that model's logs show SHARD DONE, from the repository root. It never depends on the shard jobs, because a shard that hits the
# time limit gets resubmitted. singleton: the two models' merges run one after the other. Safe to rerun any time:
#   sbatch accuracy_tables/hpc/qwen_flops_merge.SBATCH qwen2.5
#   sbatch accuracy_tables/hpc/qwen_flops_merge.SBATCH qwen3
# Then it compares uniform, blind and budget with section 12's or 13's merged records, which also ran on A100s. Those folders are only
# read: --new is this run's. The comparison reports and never fails the job.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
USAGE="usage: qwen_flops_merge.SBATCH <qwen2.5|qwen3>"
M=${1:?$USAGE}
case "$M" in
    qwen2.5) TAG=qwen25; SECTION=qwen25_350 ;;
    qwen3) TAG=qwen3; SECTION=qwen3_350 ;;
    *) echo "$USAGE" >&2; exit 1 ;;
esac
CONFIG=accuracy_tables/hpc/config.all350.yaml
A=accuracy_tables.hpc.all350
OUT=$GV_ROOT/results/${TAG}_flops
PLAN=$OUT/calibrate/$M/flops_plan.json
ROWS=uniform,blind,budget,nogate,uniform_flops,gazevlm,streamingtom,dycoke,fastvid,prunevid,dtd,avgpool,gazevlm2
run_in_env "python -u -m $A.flops_shards merge --root $OUT --model $M --config $CONFIG --rows $ROWS \
    && python -u -m $A.gpu_summary $GV_ROOT/logs/gpu_${TAG}-flops-full-*_*.csv \
    && python -u -m $A.flops_report --out-dir $OUT --model $M --plan $PLAN"
if ! run_in_env "python -u -m accuracy_tables.hpc.newvideos.compare_records --new $OUT/$M --ref $GV_ROOT/results/$SECTION/$M --rows uniform,blind,budget --report"; then
    echo "The comparison with results/$SECTION didn't end with TOKENS MATCH. Report the lines above to the user. The merged results above still stand."
fi
echo "MERGE JOB DONE: $M"

#!/bin/bash
#SBATCH --job-name=acc-budget
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --dependency=singleton
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# The frame budget on the 12 videos: the budget rule and the no-gate control at 5%, 10% and 20% (runbook section 8).
# With a question count it is a quick check in its own folder; without one it is the real run, in results/hpc_budget/.
#   sbatch --job-name=acc-budget-check-smolvlm accuracy_tables/hpc/run_budget.SBATCH smolvlm 6
#   sbatch --job-name=acc-budget-smolvlm accuracy_tables/hpc/run_budget.SBATCH smolvlm
# The real run copies the model's Uniform K, Blind and Ours records from results/hpc_all/, which it only reads. Results are
# saved after every question, so rerunning (or a requeue after preemption) continues where it stopped.
# Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
M=${1:?usage: run_budget.SBATCH <model> [questions for a quick check]}
N=${2:-}
NV=--nv
gpu_log

if [ -n "$N" ]; then
    OUT=$GV_ROOT/results/smoke_budget
    rm -rf "$OUT"
    run_in_env "python -m pytest tests/test_budget_rule.py tests/test_policy_pursuit.py tests/test_accuracy_tables.py -q \
        && python -u -m accuracy_tables.hpc.check_gpu \
        && python -u -m accuracy_tables.run --config $CONFIG --model $M --rows budget,nogate --videos all \
            --max-questions $N --out-dir $OUT --keep-going"
    exit 0
fi

OUT=$GV_ROOT/results/hpc_budget
REF=$GV_ROOT/results/hpc_all/$M
mkdir -p "$OUT/$M"
for ROW in uniform blind ours; do
    [ -e "$OUT/$M/$ROW.jsonl" ] || cp "$REF/$ROW.jsonl" "$OUT/$M/$ROW.jsonl"
done
run_in_env "python -u -m accuracy_tables.hpc.check_gpu \
    && python -u -m accuracy_tables.hpc.repair_results $OUT/$M \
    && python -u -m accuracy_tables.run --config $CONFIG --model $M --rows budget,nogate --videos all --out-dir $OUT --keep-going \
    && python -u -m accuracy_tables.summarize --config $CONFIG --videos all --out-dir $OUT \
    && python -u -m accuracy_tables.compare_budget --model $M --out-dir $OUT"

#!/bin/bash
#SBATCH --job-name=ablation-aea
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --dependency=singleton
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# The temporal / spatial ablation on EgoEverything's AEA videos (accuracy_tables/hpc/ablation_aea.py): one model on one A100, every row
# scored fresh. The A100 is set here; never override --gres. Submit from the repository root (TAG: qwen25, qwen3, smolvlm22, smolvlm256):
#   sbatch --job-name=ablation-aea-<TAG> accuracy_tables/hpc/ablation_aea.SBATCH <qwen2.5|qwen3|smolvlm|smolvlm256>
# In order, in one job:
#   plan      the FLOPs units on the real model and every row checked on 2 questions under the counter (skipped once plan.json exists)
#   prepare   half-written lines repaired, out-of-memory records dropped (their questions are scored again), the 25 videos split over
#             PROCS processes (answers/assignment.json, fixed at the first start)
#   run       PROCS processes on the one GPU, each scoring all 5 rows on its own videos, frames loading on threads
#   status    RUN DONE, or RUN INCOMPLETE
# PROCS per model: Qwen2.5 and Qwen3 2, SmolVLM2-2.2B 3, SmolVLM2-256M 4. Another count, only before the first run starts:
# sbatch --export=ALL,PROCS=<n> ... Under 50% GPU use an A100 job is cancelled, so the run prints the GPU's mean over its first 10 minutes of
# scoring: under 55% it stops the job at once (GPU EARLY LOW) and tells the user. Saved after every answer: a requeue (preemption, or the
# one this job asks for 15 minutes before its time limit) continues and skips every finished record.
# Writes only $GV_ROOT/results/ablation_aea/<model>/; every other results folder is only read. Never commit what it writes.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
USAGE="usage: ablation_aea.SBATCH <qwen2.5|qwen3|smolvlm|smolvlm256>"
M=${1:?$USAGE}
case "$M" in
    qwen2.5) TAG=qwen25; CFG=accuracy_tables/hpc/config.all350.yaml; P0=2 ;;
    qwen3) TAG=qwen3; CFG=accuracy_tables/hpc/config.all350.yaml; P0=2 ;;
    smolvlm) TAG=smolvlm22; CFG=accuracy_tables/hpc/config.all350.yaml; P0=3 ;;
    smolvlm256) TAG=smolvlm256; CFG=accuracy_tables/hpc/config.smolvlm256.yaml; P0=4 ;;
    *) echo "$USAGE" >&2; exit 1 ;;
esac
WANT=ablation-aea-$TAG
if [ "$SLURM_JOB_NAME" != "$WANT" ]; then
    echo "this job must be named $WANT, or singleton won't protect its files: submit it with --job-name=$WANT" >&2
    exit 1
fi
GPU=A100
PROCS=${PROCS:-$P0}
CPUS=${SLURM_CPUS_PER_TASK:-24}
WP=$(( CPUS / PROCS > 2 ? CPUS / PROCS : 2 ))         # loading threads of each process
ROOT=$GV_ROOT/results/ablation_aea
D=$ROOT/$M
A="python -u -m accuracy_tables.hpc.ablation_aea"
LOG=$GV_ROOT/logs/gpu_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.csv
export OMP_NUM_THREADS=1
NV=--nv
mkdir -p "$D/answers"
gpu_log
NVSMI=$!
gpu_mean() {
    awk -F',' '{split($2, a, " "); if (a[1] ~ /^[0-9]+$/) {n++; s += a[1]; u[n] = a[1]}}
        END {if (!n) {print "GPU MEAN: no samples yet"; exit}
             k = (n < 30) ? n : 30; m = 0; for (i = n - k + 1; i <= n; i++) m += u[i]; m /= k
             printf "GPU MEAN: last %d min %.1f%%, whole job %.1f%% over %d min%s\n", k, m, s / n, n, (m < 50 ? "   GPU LOW: under 50%" : "")}' "$1"
}
early_check() {     # the GPU's mean over the first 10 minutes of scoring, once all PROCS processes have started; under 55%: stop
    local waited=0
    while [ "$(ls "$D"/answers/.scoring_w* 2>/dev/null | wc -l)" -lt "$PROCS" ]; do
        sleep 20
        waited=$((waited + 20))
        if [ "$waited" -ge 2400 ]; then
            echo "GPU FIRST 10 MIN: the processes hadn't all started scoring after 40 minutes; no early reading"
            return
        fi
    done
    local n0
    n0=$(wc -l < "$LOG")
    sleep 600
    local m
    m=$(tail -n +$((n0 + 1)) "$LOG" | awk -F',' '{split($2, a, " "); if (a[1] ~ /^[0-9]+$/) {n++; s += a[1]}} END {if (n) printf "%.1f", s / n; else print "nan"}')
    echo "GPU FIRST 10 MIN: mean $m% over the first 10 minutes of scoring, $PROCS processes ($WP loading threads each)"
    if [ "$m" = nan ] || awk -v m="$m" 'BEGIN {exit !(m < 55)}'; then
        echo "GPU EARLY LOW: $m% is under 55%, so the job stops now, before a 30-minute window can trigger a cancellation. Every answer so far is saved. Tell the user."
        scancel "$SLURM_JOB_ID"
    fi
}
( while sleep 1800; do gpu_mean "$LOG"; done ) &
WATCH=$!
trap 'kill $NVSMI $WATCH 2>/dev/null || true' EXIT
REQUEUED=0
trap 'REQUEUED=1; echo "TIME LIMIT NEAR: requeued; the next start skips every finished record"; scontrol requeue $SLURM_JOB_ID' USR1
SAME=""
if [ -s "$D/gpu_types.txt" ]; then
    SAME="--same-as $D/gpu_types.txt"
fi
parallel() {        # parallel <tag>... : run each "$CMD_<i>" in its own container process at once; fails if any fails
    local pids=() fail=0
    for i in "$@"; do
        local cmd_var="CMD_$i"
        run_in_env "set -o pipefail; ${!cmd_var} 2>&1 | sed -u 's/^/[$i] /'" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do
        wait "$p" || fail=1
    done
    return $fail
}
rm -f "$D"/answers/.scoring_w*
run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU $SAME --record $D/gpu_types.txt \
    && $A plan --model $M --config $CFG --root $ROOT --workers 16 \
    && $A prepare --model $M --root $ROOT --procs $PROCS" &
wait $!
TAGS=()
for i in $(seq 0 $((PROCS - 1))); do
    declare "CMD_w$i=$A run --model $M --config $CFG --root $ROOT --worker $i --of $PROCS --workers $WP"
    TAGS+=("w$i")
done
early_check &
EARLY=$!
FAIL=0
parallel "${TAGS[@]}" || FAIL=1
kill "$EARLY" 2>/dev/null || true
if [ "$REQUEUED" = 1 ]; then
    exit 0
fi
if [ "$FAIL" = 1 ]; then
    echo "RUN FAILED: a process stopped (its [w<i>] lines above say why). The answers saved stay; stop and tell the user." >&2
    exit 1
fi
run_in_env "$A status --model $M --root $ROOT && python -u -m accuracy_tables.hpc.all350.gpu_summary $LOG" &
wait $!
gpu_mean "$LOG"
echo "ABLATION GPU DONE: $M"

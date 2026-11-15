#!/bin/bash
#SBATCH --job-name=sweep10-a100
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
# The 10% FLOPs sweep on one A100 (diagnostic dev work: nothing feeds a results table). Every model runs here (the user's decision,
# 2026-11-14): Qwen2.5-VL-3B, Qwen3-VL-4B, SmolVLM2-2.2B and SmolVLM2-256M. For the two SmolVLMs that is a deviation, since their earlier
# sections ran on B200s. The A100 is set here; never override --gres. Submit from the repository root with the job name the model and
# mode need (TAG: qwen25, qwen3, smolvlm22-a100, smolvlm256-a100):
#   sbatch --job-name=sweep10-<TAG>-<mode> accuracy_tables/hpc/hdepic_sweep10_a100.SBATCH <qwen2.5|qwen3|smolvlm|smolvlm256> <mode>
# Modes:
#   checks      S1: the regression against the model's earlier section, only read, then sweep10_gpu checks, in one process.
#               - Qwen: SAME ANSWERS against section 15's A100 records.
#               - SmolVLM: no earlier SmolVLM run was on an A100, so its regression and its Blind recheck against v2 Stage 1 are
#                 cross-GPU. Frames and tokens must match; the answers that differ are counted.
#   plan        S2: sweep10_gpu plan, in one process; the plan is then read-only.
#   conc-check  20 questions x 3 conditions (uniform, ours_res_0.50, fastvid_0.50): one process, then 3 processes at once on the GPU.
#               SAME ANSWERS at max_prob_diff 0.0, with each process's peak GPU memory.
#   run         S3: the plan's conditions shared out over PROCS processes on the one GPU, each condition scored whole by one process
#               (answers/assignment.json). Frames load on threads in each. Before they start:
#               - half-written lines are repaired;
#               - out-of-memory records are dropped, so their questions are scored again (every answer is kept);
#               - conditions with answers saved resume from them.
# PROCS per model: Qwen2.5 and Qwen3 3, SmolVLM2-2.2B 4, SmolVLM2-256M 6. A single Qwen process kept the GPU at 34-40%, and the
# lighter models need more. Another count: sbatch --export=ALL,PROCS=<n> ... (this changes no answer: each process scores its own
# conditions exactly as one process would, which conc-check shows).
# Under 50% GPU use an A100 job is cancelled, so the run prints the GPU's mean over its first 10 minutes of scoring (all processes
# started): under 55% it stops the job at once (GPU EARLY LOW) and tells the user, before a 30-minute window can trigger the
# cancellation. Every 30 minutes the log also gets the last 30 minutes' mean (GPU LOW under 50%).
# Saved after every answer: a requeue after preemption, or the requeue this job asks for 15 minutes before its time limit, continues and
# skips every finished record. Writes only $GV_ROOT/data/hdepic/sweep10/<folder>/ (qwen2.5, qwen3, smolvlm_a100, smolvlm256_a100);
# every results folder is only read. Never commit what it writes.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
export HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}
USAGE="usage: hdepic_sweep10_a100.SBATCH <qwen2.5|qwen3|smolvlm|smolvlm256> <checks|plan|conc-check|run>"
M=${1:?$USAGE}
MODE=${2:?$USAGE}
case "$M" in
    qwen2.5) TAG=qwen25; RUN=$M; RCONFIG=accuracy_tables/hpc/config.all350.yaml; REF=$GV_ROOT/results/qwen25_flops/qwen2.5; RQ=10; RGPU=same; P0=3 ;;
    qwen3) TAG=qwen3; RUN=$M; RCONFIG=accuracy_tables/hpc/config.all350.yaml; REF=$GV_ROOT/results/qwen3_flops/qwen3; RQ=10; RGPU=same; P0=3 ;;
    smolvlm) TAG=smolvlm22-a100; RUN=smolvlm_a100; RCONFIG=accuracy_tables/hpc/config.all350.yaml; REF=$GV_ROOT/results/smolvlm_flops/smolvlm; RQ=20; RGPU=other; P0=4 ;;
    smolvlm256) TAG=smolvlm256-a100; RUN=smolvlm256_a100; RCONFIG=accuracy_tables/hpc/config.smolvlm256.yaml; REF=$GV_ROOT/results/smolvlm256_flops/smolvlm256; RQ=10; RGPU=other; P0=6 ;;
    *) echo "$USAGE" >&2; exit 1 ;;
esac
case "$MODE" in
    checks|plan|conc-check|run) ;;
    *) echo "$USAGE" >&2; exit 1 ;;
esac
WANT=sweep10-$TAG-$MODE
if [ "$SLURM_JOB_NAME" != "$WANT" ]; then
    echo "this job must be named $WANT, or singleton won't protect its files: submit it with --job-name=$WANT" >&2
    exit 1
fi
GPU=A100
PROCS=${PROCS:-$P0}
CPUS=${SLURM_CPUS_PER_TASK:-24}
W=16                                                  # loading threads of a one-process mode
WP=$(( CPUS / PROCS > 2 ? CPUS / PROCS : 2 ))         # loading threads of each of PROCS processes
D=$GV_ROOT/data/hdepic/sweep10/$RUN
export SWEEP10_RUN=$RUN       # sweep10_gpu writes to this folder
G=accuracy_tables.hpc.hdepic.sweep10_gpu
C=accuracy_tables.hpc.all350.smolvlm256_checks
CC=uniform,ours_res_0.50,fastvid_0.50
if [ "$RGPU" = same ]; then    # the reference ran on an A100 too: every answer must match
    REGRESS="python -u -m $C same --new $D/checks/regression/$M --ref $REF --rows uniform,blind,budget"
else                           # the reference ran on a B200: frames and tokens must match, differing answers are counted
    REGRESS="python -u -m accuracy_tables.hpc.newvideos.compare_records --new $D/checks/regression/$M --ref $REF --rows uniform,blind,budget --report"
fi
LOG=$GV_ROOT/logs/gpu_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.csv
export OMP_NUM_THREADS=1      # loading threads each starting their own OpenMP threads would fight over the cores
NV=--nv
mkdir -p "$D"
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
need_plan() {
    if [ ! -f "$D/plan.json" ]; then
        echo "no $D/plan.json: run the plan job first" >&2
        exit 1
    fi
}
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
case "$MODE" in
    checks)
        rm -rf "$D/checks"
        mkdir -p "$D/checks"
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU $SAME --record $D/gpu_types.txt \
            && V=\$(python -m $C pick-videos --questions $RQ) \
            && python -u -m accuracy_tables.run --config $RCONFIG --model $M --rows uniform,blind,budget --videos \$V --max-questions $RQ \
                --prefetch-workers 8 --out-dir $D/checks/regression --keep-going \
            && $REGRESS \
            && python -u -m $G checks --model $M --workers $W \
            && python -u -m accuracy_tables.hpc.all350.gpu_summary $LOG" &
        wait $!
        ;;
    plan)
        if [ -f "$D/plan.json" ]; then
            echo "$D/plan.json exists, and the run reads it. Planning again needs the user's OK: stop and tell them." >&2
            exit 1
        fi
        if [ ! -f "$D/units.json" ]; then
            echo "no $D/units.json: run the checks job first" >&2
            exit 1
        fi
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU --same-as $D/gpu_types.txt --record $D/gpu_types.txt \
            && python -u -m $G plan --model $M --workers $W \
            && python -u -m accuracy_tables.hpc.all350.gpu_summary $LOG" &
        wait $!
        chmod a-w "$D/plan.json" "$D/plan.csv"
        ;;
    conc-check)
        need_plan
        rm -rf "$D/conc"
        mkdir -p "$D/conc"
        WC=$(( CPUS / 3 > 2 ? CPUS / 3 : 2 ))
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU --same-as $D/gpu_types.txt --record $D/gpu_types.txt \
            && python -u -m $G run --model $M --workers $WC --only $CC --max-questions 20 --out-dir $D/conc/single" &
        wait $!
        TAGS=()
        for name in ${CC//,/ }; do
            T=${name//./_}
            declare "CMD_$T=python -u -m $G run --model $M --workers $WC --only $name --max-questions 20 --out-dir $D/conc/multi"
            TAGS+=("$T")
        done
        parallel "${TAGS[@]}"
        run_in_env "python -u -m $G same-dirs --a $D/conc/single --b $D/conc/multi --names $CC \
            && python -u -m accuracy_tables.hpc.all350.gpu_summary $LOG" &
        wait $!
        ;;
    run)
        need_plan
        mkdir -p "$D/answers"
        rm -f "$D"/answers/.scoring_w*
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU --same-as $D/gpu_types.txt --record $D/gpu_types.txt \
            && python -u -m accuracy_tables.hpc.repair_results $D/answers \
            && python -u -m $G drop-oom --model $M \
            && python -u -m $G split --model $M --procs $PROCS" &
        wait $!
        TAGS=()
        for i in $(seq 0 $((PROCS - 1))); do
            declare "CMD_w$i=python -u -m $G run --model $M --workers $WP --worker $i --of $PROCS"
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
        run_in_env "python -u -m $G status --model $M && python -u -m accuracy_tables.hpc.all350.gpu_summary $LOG" &
        wait $!
        ;;
esac
gpu_mean "$LOG"
echo "SWEEP10 GPU DONE: $M $MODE"

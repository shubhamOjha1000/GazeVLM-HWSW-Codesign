#!/bin/bash
#SBATCH --job-name=sweep10-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --dependency=singleton
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# The 10% FLOPs sweep on one A100 (diagnostic dev work: nothing feeds a results table): Qwen2.5-VL-3B, Qwen3-VL-4B, or SmolVLM2-2.2B or
# -256M on an A100. The A100 is set here; never override --gres. Submit from the repository root, with the job name its model and mode need:
#   sbatch --job-name=sweep10-qwen25-checks accuracy_tables/hpc/hdepic_sweep10_a100.SBATCH qwen2.5 checks
#   sbatch --job-name=sweep10-qwen25-plan accuracy_tables/hpc/hdepic_sweep10_a100.SBATCH qwen2.5 plan
#   sbatch --job-name=sweep10-qwen25-run accuracy_tables/hpc/hdepic_sweep10_a100.SBATCH qwen2.5 run
#   (Qwen3-VL-4B: model qwen3, job names sweep10-qwen3-<mode>; SmolVLM2-2.2B: model smolvlm, sweep10-smolvlm22-a100-<mode>;
#    SmolVLM2-256M: model smolvlm256, sweep10-smolvlm256-a100-<mode>)
# A SmolVLM run here writes $GV_ROOT/data/hdepic/sweep10/<model>_a100/, apart from its B200 run in <model>/: answers differ between GPU
# types, so the two are never mixed, and every step (checks, plan, run) of one folder runs on one GPU type.
# checks  S1: the model's own earlier regression, only read, then sweep10_gpu checks (parity, tokens, identity, threads, the FLOPs units,
#         exact FLOPs). Qwen: 50 of section 15's records, also on an A100: SAME ANSWERS. SmolVLM: no earlier SmolVLM run was on an A100,
#         so the A100 regression is the cross-GPU one (compare_records --report) against section 14's 100 records (2.2B) or section 16's 50
#         (256M), both on a B200: frames and tokens must match exactly (TOKENS MATCH, else the job stops), and the answers that differ
#         are counted for the user (GPU rounding on near-ties).
# A100 nodes cancel jobs under 50% GPU use: frames load on 8 threads ahead of the GPU (the next questions' too), and the 30-minute means
# below show it. A warning or a cancellation: stop and tell the user.
# plan    S2: sweep10_gpu plan (K' per condition, the knobs, FLOPs checked, timing, the projected hours); the plan is then read-only
# run     S3: every planned condition on the 940 dev questions. Saved after every answer: a requeue after preemption, or the requeue this
#         job asks for 15 minutes before its time limit, continues and skips every finished record
# The GPU's mean over the last 30 minutes goes into this log every 30 minutes (GPU LOW under 50%). Writes only
# $GV_ROOT/data/hdepic/sweep10/<model>/; every results folder is only read. Never commit what it writes.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
export HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}
USAGE="usage: hdepic_sweep10_a100.SBATCH <qwen2.5|qwen3|smolvlm|smolvlm256> <checks|plan|run>"
M=${1:?$USAGE}
MODE=${2:?$USAGE}
case "$M" in
    qwen2.5) TAG=qwen25; RUN=$M; RCONFIG=accuracy_tables/hpc/config.all350.yaml; REF=$GV_ROOT/results/qwen25_flops/qwen2.5; RQ=10; RGPU=same ;;
    qwen3) TAG=qwen3; RUN=$M; RCONFIG=accuracy_tables/hpc/config.all350.yaml; REF=$GV_ROOT/results/qwen3_flops/qwen3; RQ=10; RGPU=same ;;
    smolvlm) TAG=smolvlm22-a100; RUN=smolvlm_a100; RCONFIG=accuracy_tables/hpc/config.all350.yaml; REF=$GV_ROOT/results/smolvlm_flops/smolvlm; RQ=20; RGPU=other ;;
    smolvlm256) TAG=smolvlm256-a100; RUN=smolvlm256_a100; RCONFIG=accuracy_tables/hpc/config.smolvlm256.yaml; REF=$GV_ROOT/results/smolvlm256_flops/smolvlm256; RQ=10; RGPU=other ;;
    *) echo "$USAGE" >&2; exit 1 ;;
esac
case "$MODE" in
    checks|plan|run) ;;
    *) echo "$USAGE" >&2; exit 1 ;;
esac
WANT=sweep10-$TAG-$MODE
if [ "$SLURM_JOB_NAME" != "$WANT" ]; then
    echo "this job must be named $WANT, or singleton won't protect its files: submit it with --job-name=$WANT" >&2
    exit 1
fi
GPU=A100
W=8
D=$GV_ROOT/data/hdepic/sweep10/$RUN
export SWEEP10_RUN=$RUN       # sweep10_gpu writes to this folder
G=accuracy_tables.hpc.hdepic.sweep10_gpu
C=accuracy_tables.hpc.all350.smolvlm256_checks
if [ "$RGPU" = same ]; then    # the reference ran on an A100 too: every answer must match
    REGRESS="python -u -m $C same --new $D/checks/regression/$M --ref $REF --rows uniform,blind,budget"
else                           # the reference ran on a B200: frames and tokens must match, differing answers are counted
    REGRESS="python -u -m accuracy_tables.hpc.newvideos.compare_records --new $D/checks/regression/$M --ref $REF --rows uniform,blind,budget --report"
fi
LOG=$GV_ROOT/logs/gpu_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.csv
export OMP_NUM_THREADS=1      # 8 loading threads each starting 8 OpenMP threads would fight over the 8 cores
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
( while sleep 1800; do gpu_mean "$LOG"; done ) &
WATCH=$!
trap 'kill $NVSMI $WATCH 2>/dev/null || true' EXIT
trap 'echo "TIME LIMIT NEAR: requeued; the next start skips every finished record"; scontrol requeue $SLURM_JOB_ID' USR1
SAME=""
if [ -s "$D/gpu_types.txt" ]; then
    SAME="--same-as $D/gpu_types.txt"
fi
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
    run)
        if [ ! -f "$D/plan.json" ]; then
            echo "no $D/plan.json: run the plan job first" >&2
            exit 1
        fi
        mkdir -p "$D/answers"
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU --same-as $D/gpu_types.txt --record $D/gpu_types.txt \
            && python -u -m accuracy_tables.hpc.repair_results $D/answers \
            && python -u -m $G run --model $M --workers $W \
            && python -u -m accuracy_tables.hpc.all350.gpu_summary $LOG" &
        wait $!
        ;;
esac
gpu_mean "$LOG"
echo "SWEEP10 GPU DONE: $M $MODE"

#!/bin/bash
#SBATCH --job-name=smolvlm350
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:b200:1
#SBATCH --time=08:00:00
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --dependency=singleton
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# SmolVLM on all 350 videos (runbook section 11): Uniform K, Blind, the budget rule and the 8 baselines at 5, 10 and 20%, all on one GPU type:
#   sbatch --job-name=smolvlm350-calibrate accuracy_tables/hpc/smolvlm350.SBATCH calibrate     the baselines' knobs, from token counts only
#   sbatch --job-name=smolvlm350-check     accuracy_tables/hpc/smolvlm350.SBATCH check         parity, threads on against off, full budget against Uniform K
#   sbatch --job-name=smolvlm350-pilot     accuracy_tables/hpc/smolvlm350.SBATCH pilot         the 7 pilot videos
#   sbatch --job-name=smolvlm350-full      accuracy_tables/hpc/smolvlm350.SBATCH full          all 350 videos
# Results are saved after every question, so rerunning (or a requeue after preemption) continues where it stopped. singleton: jobs with the
# same name never run together, because the repair step touches the whole results folder. Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
CONFIG=accuracy_tables/hpc/config.all350.yaml
M=smolvlm
GPU=B200
MODE=${1:?usage: smolvlm350.SBATCH <calibrate|check|pilot|full>}
W=8
N=30
OUT=$GV_ROOT/results/smolvlm_350
BASE=$OUT/config_baselines.yaml
BASELINES=avgpool,gazevlm,gazevlm2,streamingtom,dycoke,fastvid,prunevid,dtd
ROWS=uniform,blind,budget,$BASELINES
LIST=$GV_ROOT/newvideos/lists/all350_pilot.txt
export OMP_NUM_THREADS=1      # 8 loading threads each starting 8 OpenMP threads would fight over the 8 cores
NV=--nv
gpu_log
if [ "$MODE" != calibrate ] && [ ! -f "$BASE" ]; then
    echo "no $BASE: run the calibrate job first" >&2
    exit 1
fi
case "$MODE" in
    calibrate)
        mkdir -p "$OUT/$M"
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU --record $OUT/$M/gpu_types.txt \
            && python -u -m accuracy_tables.calibrate --config $CONFIG --model $M --fractions --videos all \
                --prefetch-workers $W --out-dir $OUT --write $BASE"
        ;;
    check)
        CK=$GV_ROOT/results/smolvlm_350_check
        NYMERIA=$(sed -n 1p "$LIST")
        rm -rf "$CK"
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU \
            && python -u -m accuracy_tables.run --config $BASE --model $M --rows $ROWS --videos $NYMERIA --max-questions $N \
                --prefetch-workers 0 --out-dir $CK/off --keep-going \
            && python -u -m accuracy_tables.run --config $BASE --model $M --rows $ROWS --videos $NYMERIA --max-questions $N \
                --prefetch-workers $W --out-dir $CK/on --keep-going \
            && python -u -m accuracy_tables.hpc.newvideos.compare_records --new $CK/on/$M --ref $CK/off/$M --rows $ROWS \
            && python -u -m accuracy_tables.hpc.all350.check_identity --write $CK/identity.yaml --config $CONFIG \
            && python -u -m accuracy_tables.run --config $CK/identity.yaml --model $M --rows uniform,avgpool,dtd,dycoke,streamingtom \
                --videos $NYMERIA --max-questions $N --prefetch-workers $W --out-dir $CK/identity --keep-going \
            && python -u -m accuracy_tables.hpc.all350.check_identity --compare $CK/identity/$M"
        echo "CHECK DONE: $M"
        ;;
    pilot)
        P=$GV_ROOT/results/smolvlm_350_pilot
        VIDEOS=$(paste -sd, "$LIST")
        mkdir -p "$P/$M"
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU \
            && python -u -m accuracy_tables.hpc.repair_results $P/$M \
            && python -u -m accuracy_tables.run --config $BASE --model $M --rows $ROWS --videos $VIDEOS \
                --prefetch-workers $W --out-dir $P --keep-going \
            && python -u -m accuracy_tables.hpc.newvideos.compare_records --new $P/$M --ref $GV_ROOT/results/nymeria_smolvlm/$M \
                --rows uniform,blind,budget --report"
        ;;
    full)
        mkdir -p "$OUT/$M"
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU --record $OUT/$M/gpu_types.txt \
            && python -u -m accuracy_tables.hpc.repair_results $OUT/$M \
            && python -u -m accuracy_tables.run --config $BASE --model $M --rows $ROWS --videos all \
                --prefetch-workers $W --out-dir $OUT --keep-going"
        ;;
    *)
        echo "usage: smolvlm350.SBATCH <calibrate|check|pilot|full>" >&2
        exit 1
        ;;
esac

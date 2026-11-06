#!/bin/bash
#SBATCH --job-name=all350
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
# One model on the joined data (runbook section 10), rows uniform, blind and budget, with frames loaded on 8 threads:
#   sbatch --job-name=all350-check-phi accuracy_tables/hpc/all350_vlm.SBATCH phi check     threads on against off, and answers against hpc_all
#   sbatch --job-name=all350-pilot-phi accuracy_tables/hpc/all350_vlm.SBATCH phi pilot     the 7 pilot videos
#   sbatch --job-name=all350-full-phi  accuracy_tables/hpc/all350_vlm.SBATCH phi full      all 350 videos, after copying what can be reused
# Results are saved after every question, so rerunning (or a requeue after preemption) continues where it stopped. singleton: jobs with
# the same name never run together, because the repair step touches the whole results folder. Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
CONFIG=accuracy_tables/hpc/config.all350.yaml
M=${1:?usage: all350_vlm.SBATCH <model> <check|pilot|full>}
MODE=${2:?usage: all350_vlm.SBATCH <model> <check|pilot|full>}
W=8
N=30
ROWS=uniform,blind,budget
KNOWN=loc4_script1_seq1_rec1
LIST=$GV_ROOT/newvideos/lists/all350_pilot.txt
export OMP_NUM_THREADS=1      # 8 loading threads each starting 8 OpenMP threads would fight over the 8 cores
NV=--nv
gpu_log
case "$MODE" in
    check)
        OUT=$GV_ROOT/results/hpc350_check/$M
        NYMERIA=$(sed -n 1p "$LIST")
        rm -rf "$OUT"
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu \
            && python -u -m accuracy_tables.run --config $CONFIG --model $M --rows $ROWS --videos $NYMERIA --max-questions $N \
                --prefetch-workers 0 --out-dir $OUT/off --keep-going \
            && python -u -m accuracy_tables.run --config $CONFIG --model $M --rows $ROWS --videos $NYMERIA --max-questions $N \
                --prefetch-workers $W --out-dir $OUT/on --keep-going \
            && python -u -m accuracy_tables.hpc.newvideos.compare_records --new $OUT/on/$M --ref $OUT/off/$M --rows $ROWS \
            && python -u -m accuracy_tables.run --config $CONFIG --model $M --rows uniform,blind --videos $KNOWN --max-questions $N \
                --prefetch-workers $W --out-dir $OUT/known --keep-going \
            && python -u -m accuracy_tables.hpc.newvideos.compare_records --new $OUT/known/$M --ref $GV_ROOT/results/hpc_all/$M \
                --rows uniform,blind --report"
        echo "CHECK DONE: $M"
        ;;
    pilot)
        OUT=$GV_ROOT/results/hpc350_pilot
        VIDEOS=$(paste -sd, "$LIST")
        mkdir -p "$OUT/$M"
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu \
            && python -u -m accuracy_tables.hpc.repair_results $OUT/$M \
            && python -u -m accuracy_tables.run --config $CONFIG --model $M --rows $ROWS --videos $VIDEOS \
                --prefetch-workers $W --out-dir $OUT --keep-going"
        if [ "$M" = smolvlm ]; then           # it has scored these Nymeria videos before, so it can be compared
            run_in_env "python -u -m accuracy_tables.hpc.newvideos.compare_records --new $OUT/$M \
                --ref $GV_ROOT/results/nymeria_smolvlm/$M --rows $ROWS --report"
        fi
        ;;
    full)
        OUT=$GV_ROOT/results/hpc_350
        mkdir -p "$OUT/$M"
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu \
            && python -u -m accuracy_tables.hpc.all350.seed_results --model $M --out $OUT \
            && python -u -m accuracy_tables.hpc.repair_results $OUT/$M \
            && python -u -m accuracy_tables.run --config $CONFIG --model $M --rows $ROWS --videos all \
                --prefetch-workers $W --out-dir $OUT --keep-going"
        ;;
    *)
        echo "usage: all350_vlm.SBATCH <model> <check|pilot|full>" >&2
        exit 1
        ;;
esac

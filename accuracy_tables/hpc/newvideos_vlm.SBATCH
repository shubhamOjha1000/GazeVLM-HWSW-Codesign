#!/bin/bash
#SBATCH --job-name=new-vlm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --dependency=singleton
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# SmolVLM2 on Nymeria (runbook section 9), rows uniform, blind, budget and nogate, with frames loaded on 8 threads:
#   sbatch --job-name=new-vlm-check accuracy_tables/hpc/newvideos_vlm.SBATCH check     the thread check, then 6 Nymeria questions
#   sbatch --job-name=new-vlm-pilot accuracy_tables/hpc/newvideos_vlm.SBATCH pilot     the 5 pilot videos
#   sbatch --job-name=new-vlm-full  accuracy_tables/hpc/newvideos_vlm.SBATCH full      all 325 videos
# Results are saved after every question, so rerunning (or a requeue after preemption) continues where it stopped. Summaries run
# in a CPU job, newvideos_summary.SBATCH. Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
AEA_CONFIG=$CONFIG
CONFIG=accuracy_tables/hpc/config.newvideos.yaml
MODE=${1:?usage: newvideos_vlm.SBATCH <check|pilot|full>}
W=8
ROWS=uniform,blind,budget,nogate
NV=--nv
gpu_log
case "$MODE" in
    check)
        CHECK=$GV_ROOT/results/prefetch_check
        SMOKE=$GV_ROOT/results/smoke_nymeria
        rm -rf "$CHECK" "$SMOKE"
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu \
            && python -u -m accuracy_tables.run --config $AEA_CONFIG --model smolvlm --rows budget,nogate --videos all \
                --max-questions 30 --prefetch-workers $W --out-dir $CHECK --keep-going \
            && python -u -m accuracy_tables.hpc.newvideos.compare_records --new $CHECK/smolvlm \
                --ref $GV_ROOT/results/hpc_budget/smolvlm --rows budget,nogate \
            && python -u -m accuracy_tables.run --config $CONFIG --model smolvlm --rows $ROWS --videos all \
                --max-questions 6 --prefetch-workers $W --out-dir $SMOKE --keep-going"
        ;;
    pilot)
        OUT=$GV_ROOT/results/nymeria_pilot
        VIDEOS=$(paste -sd, "$GV_ROOT/newvideos/lists/pilot.txt")
        mkdir -p "$OUT/smolvlm"
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu \
            && python -u -m accuracy_tables.hpc.repair_results $OUT/smolvlm \
            && python -u -m accuracy_tables.run --config $CONFIG --model smolvlm --rows $ROWS --videos $VIDEOS \
                --prefetch-workers $W --out-dir $OUT --keep-going"
        ;;
    full)
        OUT=$GV_ROOT/results/nymeria_smolvlm
        mkdir -p "$OUT/smolvlm"
        run_in_env "python -u -m accuracy_tables.hpc.check_gpu \
            && python -u -m accuracy_tables.hpc.repair_results $OUT/smolvlm \
            && python -u -m accuracy_tables.run --config $CONFIG --model smolvlm --rows $ROWS --videos all \
                --prefetch-workers $W --out-dir $OUT --keep-going"
        ;;
    *)
        echo "usage: newvideos_vlm.SBATCH <check|pilot|full>" >&2
        exit 1
        ;;
esac

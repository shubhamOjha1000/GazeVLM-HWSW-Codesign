#!/bin/bash
#SBATCH --job-name=acc-run
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --dependency=singleton
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# One model through the budget-matched protocol: Uniform K, Blind and Ours, then calibration, then the 8 baselines.
# Results are saved after every question, so rerunning (or a requeue after preemption) continues where it stopped.
# singleton: jobs with the same name never run together, so two jobs can't write the same result files.
#   sbatch --job-name=acc-test-phi accuracy_tables/hpc/run_model.SBATCH phi test
# Submit from the repository root; submit_all.sh does this for every model.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
M=${1:?usage: run_model.SBATCH <model> <test|all>}
V=${2:?usage: run_model.SBATCH <model> <test|all>}
OUT=$GV_ROOT/results/hpc_$V
MATCHED=$OUT/config_matched_$M.yaml
mkdir -p "$OUT"
NV=--nv
gpu_log
run_in_env "python -u -m accuracy_tables.hpc.check_gpu \
    && python -u -m accuracy_tables.hpc.repair_results $OUT/$M \
    && python -u -m accuracy_tables.run --config $CONFIG --model $M --rows uniform,blind,ours --videos $V --out-dir $OUT \
    && python -u -m accuracy_tables.calibrate --config $CONFIG --model $M --points 2 --videos $V --out-dir $OUT --write $MATCHED \
    && python -u -m accuracy_tables.run --config $MATCHED --model $M \
        --rows gazevlm,gazevlm2,avgpool,streamingtom,dycoke,fastvid,prunevid,dtd --videos $V --out-dir $OUT --keep-going"

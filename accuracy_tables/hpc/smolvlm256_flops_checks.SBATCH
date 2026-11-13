#!/bin/bash
#SBATCH --job-name=smolvlm256-flops-checks
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:b200:1
#SBATCH --time=03:00:00
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --dependency=singleton
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Runbook section 16, step 2b, on one B200 (section 14's GPU type), each on a small set of videos so the GPU waits little:
#   1. the 2.2B's regression: SmolVLM2-2.2B, 20 questions x uniform, blind and budget (all 3 budgets), against section 14's records
#      in results/smolvlm_flops/ (only read): SAME ANSWERS at max_prob_diff 0.0
#   2. parity OK on the 256M (every run below starts with it)
#   3. full budget: avgpool, dtd, dycoke and streamingtom keeping every token give Uniform K's answers on the 256M: SAME AS UNIFORM K
#   4. threads: all 12 rows on 50 questions with 16 loading threads against none: SAME ANSWERS at max_prob_diff 0.0
# Writes only to results/smolvlm256_flops/checks/. Rerunning starts the checks again. Submit from the repository root:
#   sbatch accuracy_tables/hpc/smolvlm256_flops_checks.SBATCH
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
CONFIG22=accuracy_tables/hpc/config.all350.yaml
CONFIG=accuracy_tables/hpc/config.smolvlm256.yaml
GPU=B200
M=smolvlm256
W=16
C=accuracy_tables.hpc.all350.smolvlm256_checks
CHK=$GV_ROOT/results/smolvlm256_flops/checks
REF=$GV_ROOT/results/smolvlm_flops/smolvlm
ROWS=uniform,blind,budget,nogate,gazevlm,streamingtom,dycoke,fastvid,prunevid,dtd,avgpool,gazevlm2
export OMP_NUM_THREADS=1      # 16 loading threads each starting 16 OpenMP threads would fight over the 16 cores
NV=--nv
gpu_log
rm -rf "$CHK/regression" "$CHK/identity" "$CHK/identity.yaml" "$CHK/off" "$CHK/on"
mkdir -p "$CHK"
run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU --record $CHK/gpu_types.txt \
    && V=\$(python -m $C pick-videos --questions 20) \
    && python -u -m accuracy_tables.run --config $CONFIG22 --model smolvlm --rows uniform,blind,budget --videos \$V --max-questions 20 \
        --prefetch-workers 8 --out-dir $CHK/regression --keep-going \
    && python -u -m $C same --new $CHK/regression/smolvlm --ref $REF --rows uniform,blind,budget \
    && python -u -m accuracy_tables.hpc.all350.check_identity --write $CHK/identity.yaml --config $CONFIG \
    && V=\$(python -m $C pick-videos --questions 30) \
    && python -u -m accuracy_tables.run --config $CHK/identity.yaml --model $M --rows uniform,avgpool,dtd,dycoke,streamingtom --videos \$V \
        --max-questions 30 --prefetch-workers $W --out-dir $CHK/identity --keep-going \
    && python -u -m $C identity --folder $CHK/identity/$M \
    && V=\$(python -m $C pick-videos --questions 50) \
    && python -u -m accuracy_tables.run --config $CONFIG --model $M --rows $ROWS --videos \$V --max-questions 50 \
        --prefetch-workers 0 --out-dir $CHK/off --keep-going \
    && python -u -m accuracy_tables.run --config $CONFIG --model $M --rows $ROWS --videos \$V --max-questions 50 \
        --prefetch-workers $W --out-dir $CHK/on --keep-going \
    && python -u -m $C same --new $CHK/on/$M --ref $CHK/off/$M --rows $ROWS --all \
    && python -u -m accuracy_tables.hpc.all350.gpu_summary $GV_ROOT/logs/gpu_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.csv"
echo "CHECKS DONE: $M"

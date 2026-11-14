#!/bin/bash
#SBATCH --job-name=hdepic18-checks
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu:b200:1
#SBATCH --time=04:00:00
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --dependency=singleton
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Runbook section 18, step 5b, on one B200 (section 14's GPU type), each on a small set so the GPU waits little:
#   1. section 14's regression: SmolVLM2-2.2B on 20 EgoEverything questions x uniform, blind and budget at 5, 10 and 20% (100 records)
#      against results/smolvlm_flops/ (only read): SAME ANSWERS: 0 of 100 records differ, max_prob_diff 0.0
#   2. parity OK on HD-EPIC's inputs (every HD-EPIC run below starts with it)
#   3. full budget on 30 HD-EPIC questions: avgpool, dtd, dycoke and streamingtom keeping every token give Uniform K's answers
#   4. threads: all 12 rows on 60 HD-EPIC questions (10 a group) with 16 loading threads against none: SAME ANSWERS, max_prob_diff 0.0
#   5. 17 foveated tokens on every kept frame of those budget and no-gate records
# Writes only to results/hdepic_all_smolvlm_flops/checks/. Rerunning starts the checks again. Submit from the repository root:
#   sbatch accuracy_tables/hpc/hdepic18_checks.SBATCH
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
export HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}
CONFIG14=accuracy_tables/hpc/config.all350.yaml
CONFIG=accuracy_tables/hpc/config.hdepic18.yaml
GPU=B200
M=smolvlm
W=16
C=accuracy_tables.hpc.all350.smolvlm256_checks
H=accuracy_tables.hpc.hdepic.checks18
CHK=$GV_ROOT/results/hdepic_all_smolvlm_flops/checks
REF=$GV_ROOT/results/smolvlm_flops/smolvlm
ROWS=uniform,blind,budget,nogate,gazevlm,streamingtom,dycoke,fastvid,prunevid,dtd,avgpool,gazevlm2
export OMP_NUM_THREADS=1      # 16 loading threads each starting 16 OpenMP threads would fight over the 16 cores
NV=--nv
gpu_log
rm -rf "$CHK/regression" "$CHK/identity" "$CHK/identity.yaml" "$CHK/off" "$CHK/on"
mkdir -p "$CHK"
run_in_env "python -u -m accuracy_tables.hpc.check_gpu --expect $GPU --record $CHK/gpu_types.txt \
    && V=\$(python -m $C pick-videos --questions 20) \
    && python -u -m accuracy_tables.run --config $CONFIG14 --model $M --rows uniform,blind,budget --videos \$V --max-questions 20 \
        --prefetch-workers 8 --out-dir $CHK/regression --keep-going \
    && python -u -m $C same --new $CHK/regression/$M --ref $REF --rows uniform,blind,budget \
    && python -u -m accuracy_tables.hpc.all350.check_identity --write $CHK/identity.yaml --config $CONFIG \
    && V=\$(python -m $H pick --questions 30) \
    && python -u -m accuracy_tables.run --config $CHK/identity.yaml --model $M --rows uniform,avgpool,dtd,dycoke,streamingtom --videos \$V \
        --max-questions 30 --prefetch-workers $W --out-dir $CHK/identity --keep-going \
    && python -u -m $C identity --folder $CHK/identity/$M \
    && V=\$(python -m $H pick --questions 60) \
    && python -u -m accuracy_tables.run --config $CONFIG --model $M --rows $ROWS --videos \$V --max-questions 60 \
        --prefetch-workers 0 --out-dir $CHK/off --keep-going \
    && python -u -m accuracy_tables.run --config $CONFIG --model $M --rows $ROWS --videos \$V --max-questions 60 \
        --prefetch-workers $W --out-dir $CHK/on --keep-going \
    && python -u -m $C same --new $CHK/on/$M --ref $CHK/off/$M --rows $ROWS --all \
    && python -u -m $H fovea --folder $CHK/on/$M \
    && python -u -m accuracy_tables.hpc.all350.gpu_summary $GV_ROOT/logs/gpu_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.csv"
echo "CHECKS DONE: hdepic18"

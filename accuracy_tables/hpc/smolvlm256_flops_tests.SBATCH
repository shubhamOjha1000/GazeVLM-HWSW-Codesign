#!/bin/bash
#SBATCH --job-name=smolvlm256-flops-tests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Runbook section 16, step 2a, on the CPU: every test in tests/, then the 256M's tokens per frame on every frame of all 350 videos,
# SAME FRAMES AS 2.2B against results/smolvlm_flops/ (only read), and the PROPOSED frame limits, which wait for the user's OK.
# The GPU half of step 2 is smolvlm256_flops_checks.SBATCH. Submit from the repository root:
#   sbatch accuracy_tables/hpc/smolvlm256_flops_tests.SBATCH
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
CONFIG=accuracy_tables/hpc/config.smolvlm256.yaml
REF=$GV_ROOT/results/smolvlm_flops/smolvlm
run_in_env "python -m pytest tests -q \
    && python -u -m accuracy_tables.hpc.all350.smolvlm256_checks tokens --config $CONFIG --reference $REF"

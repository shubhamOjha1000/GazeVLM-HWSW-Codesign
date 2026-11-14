#!/bin/bash
#SBATCH --job-name=hdepic18-diagnose
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Runbook section 18, step 0, on the CPU: why section 17's foveated rows scored below Blind on "where looking". It changes no setting and
# holds no GPU. It reads $HDEPIC_ROOT, section 17's derived data and results/hdepic_gaze_smolvlm_flops/ only, and writes only
# $GV_ROOT/data/hdepic/diagnose/. Its log and sheets hold HD-EPIC text and images: never commit them. Submit from the repository root:
#   sbatch accuracy_tables/hpc/hdepic18_diagnose.SBATCH
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
export HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}
H=accuracy_tables.hpc.hdepic
echo "HDEPIC_ROOT=$HDEPIC_ROOT"
run_in_env "python -u -m $H.diagnose files && python -u -m $H.diagnose answers"
run_in_aria "python -u -m $H.diagnose_aria"
run_in_env "python -u -m $H.diagnose motion && python -u -m $H.diagnose render"
echo "--- the findings"
grep -h "^FINDING" "$GV_ROOT/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" || true
echo "STOP: step 0's findings go to the user. Nothing past step 0 runs until they decide."

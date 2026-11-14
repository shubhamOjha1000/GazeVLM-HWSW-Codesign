#!/bin/bash
#SBATCH --job-name=hdepic-frames
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%A_%a.log
#
# Runbook section 17, step 3: one array task per planned video decodes it once and writes exactly its questions' planned frames.
# A question whose meta.json exists is skipped, so resubmitting the same line only does what's left. Submit from the repository root,
# with the array covering the plan's videos (N of them, from plan.json):
#   sbatch --array=0-$((N-1)) accuracy_tables/hpc/hdepic_frames.SBATCH
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
export HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}
: "${SLURM_ARRAY_TASK_ID:?submit this as an array: sbatch --array=0-\$((N-1)) accuracy_tables/hpc/hdepic_frames.SBATCH}"
run_in_env "python -u -m accuracy_tables.hpc.hdepic.frames extract --index $SLURM_ARRAY_TASK_ID"

#!/bin/bash
#SBATCH --job-name=hdepic18-frames
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%A_%a.log
#
# Runbook section 18, step 3: one planned video's candidate frames per array task (plan order), section 17's frames hard-linked where they
# match. Submit from the repository root with one task per planned video:
#   N=$(python3 -c "import json; print(len(json.load(open('/scratch/$USER/GazeVLM/data/hdepic/s18/plan.json'))['videos']))")
#   sbatch --array=0-$((N-1))%40 accuracy_tables/hpc/hdepic18_frames.SBATCH
# A rerun skips every question part already written.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
export HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}
run_in_env "python -u -m accuracy_tables.hpc.hdepic.frames18 extract --index $SLURM_ARRAY_TASK_ID"

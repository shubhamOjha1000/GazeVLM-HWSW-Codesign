#!/bin/bash
#SBATCH --job-name=hdepic-v2-frames
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%A_%a.log
#
# Ours v2, Stage 1: one dev video's 9-frame questions per array task (dev plan order), frames sections 17 and 18 already extracted
# hard-linked. Writes only $GV_ROOT/data/hdepic/v2/frames/. Submit from the repository root with one task per dev video:
#   N=$(python3 -c "import json; print(len(json.load(open('/scratch/$USER/GazeVLM/data/hdepic/v2/dev_plan.json'))['videos']))")
#   sbatch --array=0-$((N-1))%40 accuracy_tables/hpc/hdepic_v2_frames.SBATCH
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
export HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}
run_in_env "python -u -m accuracy_tables.hpc.hdepic.v2 frames --index $SLURM_ARRAY_TASK_ID"

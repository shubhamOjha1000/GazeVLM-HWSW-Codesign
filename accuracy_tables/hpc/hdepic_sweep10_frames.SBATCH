#!/bin/bash
#SBATCH --job-name=sweep10-frames
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%A_%a.log
#
# The 10% FLOPs sweep, S0: one planned video's candidate frames per array task (plan order); frames sections 17 and 18 and v2 Stage 1
# already extracted are hard-linked, the rest decoded by MP4 frame number. Writes only $GV_ROOT/data/hdepic/sweep10/frames/. A rerun skips
# every question part already written. Submit from the repository root with one task per planned video:
#   N=$(python3 -c "import json; print(len(json.load(open('/scratch/$USER/GazeVLM/data/hdepic/sweep10/plan.json'))['videos']))")
#   sbatch --array=0-$((N-1))%40 accuracy_tables/hpc/hdepic_sweep10_frames.SBATCH
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
export HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}
run_in_env "python -u -m accuracy_tables.hpc.hdepic.sweep10 frames --index $SLURM_ARRAY_TASK_ID"

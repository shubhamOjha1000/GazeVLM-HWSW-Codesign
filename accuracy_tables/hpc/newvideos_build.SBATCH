#!/bin/bash
#SBATCH --job-name=new-build
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%A_%a.log
#
# One task per video in lists/rest.txt: its frames, gaze and rows. plan.py prints the --array range. A finished step is skipped,
# so a task can be rerun safely. Submit from the repository root:
#   sbatch --array=0-332%20 accuracy_tables/hpc/newvideos_build.SBATCH
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
M=accuracy_tables.hpc.newvideos
V=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$GV_ROOT/newvideos/lists/rest.txt")
if [ -z "$V" ]; then
    echo "no video at line $((SLURM_ARRAY_TASK_ID + 1)) of lists/rest.txt" >&2
    exit 1
fi
run_in_env "python -u -m $M.extract_frames --video $V"
run_in_aria "python -u -m $M.project_gaze --video $V"
run_in_env "python -u -m $M.build_rows --video $V"

#!/bin/bash
#SBATCH --job-name=new-pilot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Check 2 and an early check 3 (runbook section 9): frames, gaze and rows for the 5 pilot Nymeria videos, their gate predictions,
# the comparison with the 12 AEA videos, and the gaze sheets to look at. Reruns skip finished videos. Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
M=accuracy_tables.hpc.newvideos
run_in_env "python -u -m $M.extract_frames --list pilot"
run_in_aria "python -u -m $M.project_gaze --list pilot"
run_in_env "python -u -m $M.build_rows --list pilot \
    && python -u -m $M.score_gate --group pilot \
    && python -u -m $M.check_gate --groups aea12,pilot \
    && python -u -m $M.draw_gaze --lists pilot,control"

#!/bin/bash
#SBATCH --job-name=hdepic-explore
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Runbook section 17, step 0, on the CPU: what HD-EPIC's gaze questions (2 types) and the videos they use hold, for decisions D1 to D5.
# Reads $HDEPIC_ROOT only (never writes there), and writes ids, times and counts to $GV_ROOT/data/hdepic/explore/. The log shows 2 example
# questions per type: never commit it. Submit from the repository root:
#   sbatch accuracy_tables/hpc/hdepic_explore.SBATCH
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
export HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}
echo "HDEPIC_ROOT=$HDEPIC_ROOT"
echo "--- myquota"
myquota || echo "myquota isn't on this node: run it on a login node and add its lines to the report"
run_in_env "python -u -m accuracy_tables.hpc.hdepic.explore"

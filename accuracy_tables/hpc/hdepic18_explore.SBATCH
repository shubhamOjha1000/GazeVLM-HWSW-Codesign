#!/bin/bash
#SBATCH --job-name=hdepic18-explore
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Runbook section 18, the explore step, on the CPU: what HD-EPIC's 6 non-gaze groups and their videos hold, for decisions D1 to D5, and the
# fine_grained sample (2,000, seed 0), written once to $GV_ROOT/data/hdepic/sample_fine_grained_2000.json and checked on every later run.
# Reads $HDEPIC_ROOT only; writes ids, times and counts to $GV_ROOT/data/hdepic/explore18/. The log and the box sheets hold HD-EPIC text and
# images: never commit them. Submit from the repository root, only after the user has read step 0's findings:
#   sbatch accuracy_tables/hpc/hdepic18_explore.SBATCH
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
export HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}
echo "HDEPIC_ROOT=$HDEPIC_ROOT"
echo "--- myquota"
myquota || echo "myquota isn't on this node: run it on a login node and add its lines to the report"
run_in_env "python -u -m accuracy_tables.hpc.hdepic.explore18"

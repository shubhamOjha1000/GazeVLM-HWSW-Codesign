#!/bin/bash
#SBATCH --job-name=hdepic-v2-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Ours v2, design and Stage 1's preparation, on the CPU (diagnostic dev work: nothing feeds a results table, v1 doesn't change). Every
# results folder is only read; writes only $GV_ROOT/data/hdepic/v2/ (and the new dev videos' gaze in data/hdepic/gaze/). The pictures
# and logs hold HD-EPIC images and text: never commit them. Submit from the repository root:
#   sbatch --job-name=hdepic-v2-layouts accuracy_tables/hpc/hdepic_v2_cpu.SBATCH layouts        the layouts' numbers and pictures
#   sbatch --job-name=hdepic-v2-dev accuracy_tables/hpc/hdepic_v2_cpu.SBATCH dev                 the 940 dev questions, their frames' plan,
#                                                                                                 the gaze of any video not yet projected
#   sbatch --job-name=hdepic-v2-framescheck accuracy_tables/hpc/hdepic_v2_cpu.SBATCH frames-check
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
export HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}
H=accuracy_tables.hpc.hdepic
MODE=${1:?usage: hdepic_v2_cpu.SBATCH <layouts|dev|frames-check>}
case "$MODE" in
    layouts) run_in_env "python -u -m $H.v2 layouts" ;;
    dev)
        run_in_env "python -u -m $H.v2 dev"
        run_in_aria "python -u -m $H.project --plan $GV_ROOT/data/hdepic/v2/dev_plan.json"
        ;;
    frames-check) run_in_env "python -u -m $H.v2 frames-check" ;;
    *) echo "usage: hdepic_v2_cpu.SBATCH <layouts|dev|frames-check>" >&2; exit 1 ;;
esac
echo "V2 DONE: $MODE"

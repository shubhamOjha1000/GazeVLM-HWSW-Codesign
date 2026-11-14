#!/bin/bash
#SBATCH --job-name=sweep10-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# The 10% FLOPs sweep's CPU steps (diagnostic dev work: nothing feeds a results table). Every results folder and HD-EPIC are only read;
# writes only $GV_ROOT/data/hdepic/sweep10/ (and the gaze of any planned video not yet projected, in data/hdepic/gaze/). The logs hold
# HD-EPIC text: never commit them. Submit from the repository root:
#   sbatch --job-name=sweep10-blind accuracy_tables/hpc/hdepic_sweep10_cpu.SBATCH blind-source   S0: v2 Stage 1's Blind ran on the 940?
#   sbatch --job-name=sweep10-plan accuracy_tables/hpc/hdepic_sweep10_cpu.SBATCH plan            S0: every dev question's candidates
#   sbatch --job-name=sweep10-gate accuracy_tables/hpc/hdepic_sweep10_cpu.SBATCH gate            S0: the gate's rows, scores, statistics
#   sbatch --job-name=sweep10-framescheck accuracy_tables/hpc/hdepic_sweep10_cpu.SBATCH frames-check
#   sbatch --job-name=sweep10-data accuracy_tables/hpc/hdepic_sweep10_cpu.SBATCH data            S0: the manifest and the data check
#   sbatch --job-name=sweep10-summary accuracy_tables/hpc/hdepic_sweep10_cpu.SBATCH summary      S4: the tables (dev only, no winner)
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
export HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}
H=accuracy_tables.hpc.hdepic
S=$H.sweep10
USAGE="usage: hdepic_sweep10_cpu.SBATCH <blind-source|plan|gate|frames-check|data|summary>"
MODE=${1:?$USAGE}
case "$MODE" in
    blind-source|gate|frames-check|summary) run_in_env "python -u -m $S $MODE" ;;
    plan)
        run_in_env "python -u -m $S plan && python -u -m $S sizes"
        run_in_aria "python -u -m $H.project --plan $GV_ROOT/data/hdepic/sweep10/plan.json"
        ;;
    data) run_in_env "python -u -m $S data --config accuracy_tables/hpc/config.sweep10.yaml" ;;
    *) echo "$USAGE" >&2; exit 1 ;;
esac
echo "SWEEP10 DONE: $MODE"

#!/bin/bash
#SBATCH --job-name=acc-summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Writes results/hpc_<videos>/summary.csv, and on the test videos compares it with the Colab runs. Safe to rerun any time.
#   sbatch accuracy_tables/hpc/summarize.SBATCH test
# Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
V=${1:?usage: summarize.SBATCH <test|all>}
OUT=$GV_ROOT/results/hpc_$V
REF=$GV_ROOT/data/colab_reference
run_in_env "python -u -m accuracy_tables.summarize --config $CONFIG --videos $V --out-dir $OUT"
if [ "$V" = test ] && [ -d "$REF" ]; then
    run_in_env "python -u -m accuracy_tables.hpc.compare_reference $OUT/summary.csv $REF"
fi

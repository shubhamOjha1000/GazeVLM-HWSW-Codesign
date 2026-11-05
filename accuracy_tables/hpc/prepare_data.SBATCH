#!/bin/bash
#SBATCH --job-name=acc-prepare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Unit tests, the frames of the 12 videos, and the data check, without holding a GPU. Submit from the repository root.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
run_in_env "python -m pytest tests/test_hpc.py tests/test_accuracy_tables.py tests/test_calibrate.py tests/test_gazevlm.py \
    tests/test_streamingtom.py tests/test_dycoke.py tests/test_fastvid.py tests/test_prunevid.py tests/test_dtd.py \
    tests/test_avgpool.py tests/test_gazevlm2.py -q \
    && python -u -m accuracy_tables.hpc.prepare_data --config $CONFIG"

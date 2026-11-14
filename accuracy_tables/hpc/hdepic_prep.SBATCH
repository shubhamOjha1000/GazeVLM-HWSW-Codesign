#!/bin/bash
#SBATCH --job-name=hdepic-prep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Runbook section 17, steps 1, 2 and 4 on the CPU (step 3's frames are hdepic_frames.SBATCH). Submit from the repository root:
#   sbatch --job-name=hdepic-step1 accuracy_tables/hpc/hdepic_prep.SBATCH step1     the plan, the gaze projection, the overlay sheets,
#                                                                                    the IMU against AEA, D4's 30 Hz proof (REBUILD MATCHES)
#   sbatch --job-name=hdepic-step2 accuracy_tables/hpc/hdepic_prep.SBATCH step2     the gate's rows, its 5 models' scores, its statistics
#   sbatch --job-name=hdepic-step3check accuracy_tables/hpc/hdepic_prep.SBATCH frames-check
#   sbatch --job-name=hdepic-step4 accuracy_tables/hpc/hdepic_prep.SBATCH step4     the manifest and HD-EPIC's data check
# Reads $HDEPIC_ROOT only; writes only $GV_ROOT/data/hdepic/ (outside the repository; never commit it).
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
export HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}
H=accuracy_tables.hpc.hdepic
CONFIG=accuracy_tables/hpc/config.hdepic.yaml
MODE=${1:?usage: hdepic_prep.SBATCH <step1|step2|frames-check|step4>}
echo "HDEPIC_ROOT=$HDEPIC_ROOT"
case "$MODE" in
    step1)
        run_in_env "python -u -m $H.plan"
        run_in_aria "python -u -m $H.project"
        run_in_env "python -u -m $H.sheets && python -u -m $H.gate imu && python -u -m $H.gate resample-check"
        ;;
    step2) run_in_env "python -u -m $H.gate rows && python -u -m $H.gate score && python -u -m $H.gate stats" ;;
    frames-check) run_in_env "python -u -m $H.frames check" ;;
    step4) run_in_env "python -u -m $H.manifest --config $CONFIG" ;;
    *) echo "usage: hdepic_prep.SBATCH <step1|step2|frames-check|step4>" >&2; exit 1 ;;
esac
echo "PREP DONE: $MODE"

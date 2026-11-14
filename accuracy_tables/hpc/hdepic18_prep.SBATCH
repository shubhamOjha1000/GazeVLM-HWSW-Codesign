#!/bin/bash
#SBATCH --job-name=hdepic18-prep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/GazeVLM/logs/%x_%j.log
#
# Runbook section 18, steps 1, 2 and 4 on the CPU (step 3's frames are hdepic18_frames.SBATCH). Submit from the repository root:
#   sbatch --job-name=hdepic18-step1 accuracy_tables/hpc/hdepic18_prep.SBATCH step1   the plan (D1, D2, D5), the gaze projection of every
#                                                                                      new video, the sheets, the IMU, D4's 30 Hz proof
#   sbatch --job-name=hdepic18-step2 accuracy_tables/hpc/hdepic18_prep.SBATCH step2   the gate's rows, its 5 models' scores, its statistics,
#                                                                                      and D2's exception against the plain rule
#   sbatch --job-name=hdepic18-step3check accuracy_tables/hpc/hdepic18_prep.SBATCH frames-check
#   sbatch --job-name=hdepic18-step4 accuracy_tables/hpc/hdepic18_prep.SBATCH step4   the manifest, the data check, the stamps' tokens
# Reads $HDEPIC_ROOT and section 17's derived data only; writes only $GV_ROOT/data/hdepic/s18/ and the new videos' gaze in
# $GV_ROOT/data/hdepic/gaze/ (outside the repository; never commit them).
set -euo pipefail
source "$SLURM_SUBMIT_DIR/accuracy_tables/hpc/common.sh"
export HDEPIC_ROOT=${HDEPIC_ROOT:-/scratch/$USER/hdepic}
H=accuracy_tables.hpc.hdepic
CONFIG=accuracy_tables/hpc/config.hdepic18.yaml
PLAN=$GV_ROOT/data/hdepic/s18/plan.json
MODE=${1:?usage: hdepic18_prep.SBATCH <step1|step2|frames-check|step4>}
echo "HDEPIC_ROOT=$HDEPIC_ROOT"
case "$MODE" in
    step1)
        run_in_env "python -u -m $H.plan18"
        run_in_aria "python -u -m $H.project --plan $PLAN"
        run_in_env "python -u -m $H.sheets18 && python -u -m $H.gate18 imu && python -u -m $H.gate resample-check"
        ;;
    step2) run_in_env "python -u -m $H.gate18 rows && python -u -m $H.gate18 score && python -u -m $H.gate18 stats && python -u -m $H.gate18 compare" ;;
    frames-check) run_in_env "python -u -m $H.frames18 check" ;;
    step4) run_in_env "python -u -m $H.manifest18 --config $CONFIG" ;;
    *) echo "usage: hdepic18_prep.SBATCH <step1|step2|frames-check|step4>" >&2; exit 1 ;;
esac
echo "PREP DONE: $MODE"

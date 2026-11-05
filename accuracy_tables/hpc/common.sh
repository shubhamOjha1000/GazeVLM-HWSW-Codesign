# Sourced by every job in this folder: where things live on NYU Torch, and how to run inside the container.
# Submit jobs from the repository root.
export EGO_ROOT=${EGO_ROOT:-/scratch/$USER/egoeverything}       # the dataset, read-only
export GV_ROOT=${GV_ROOT:-/scratch/$USER/GazeVLM}               # this project's data, frames, results and logs
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}                      # models are downloaded once, by download_models.SBATCH
OVERLAY=${OVERLAY:-/scratch/$USER/pytorch-example/my_pytorch.ext3}
IMAGE=${IMAGE:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}
CONFIG=accuracy_tables/hpc/config.torch.yaml

# run_in_env "<command line>": runs it from the repository root in the container's base environment. GPU jobs set NV=--nv.
run_in_env() {
    singularity exec ${NV:-} --overlay "$OVERLAY:ro" "$IMAGE" /bin/bash -c \
        "source /ext3/env.sh; unset PYTHONPATH; export PYTHONNOUSERSITE=True; cd '$SLURM_SUBMIT_DIR'; $1"
}

# gpu_log: GPU use once a minute in logs/, because the cluster cancels jobs with low GPU use.
gpu_log() {
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 60 \
        >> "$GV_ROOT/logs/gpu_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.csv" &
    trap "kill $! 2>/dev/null || true" EXIT
}

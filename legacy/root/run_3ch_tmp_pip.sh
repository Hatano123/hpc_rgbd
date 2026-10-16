#!/bin/bash
#PBS -q Eduq
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -l walltime=24:00:00

set -euo pipefail

cd "$PBS_O_WORKDIR"

LOGDIR="$PBS_O_WORKDIR/logs"
mkdir -p "$LOGDIR"

echo "Job started at $(date)" > "$LOGDIR/job_start.log"

# Use system python to create an isolated venv in TMPDIR to avoid user env glibc issues
PYTHON_CMD=$(command -v python3 || command -v python)
if [ -z "$PYTHON_CMD" ]; then
    echo "python3 not found on node" >&2
    exit 1
fi

VENV_DIR="$TMPDIR/yolo_venv"
rm -rf "$VENV_DIR"
${PYTHON_CMD} -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

export PIP_NO_CACHE_DIR=1
python -m pip install --upgrade pip setuptools wheel

# Minimal required packages for training; install ultralytics + common deps
pip install ultralytics pyyaml tifffile

# Try installing torch; this may install CPU-only wheel if GPU-specific wheel unavailable.
# If your cluster requires a specific extra-index-url (ROCm/CUDA), edit this script accordingly.
pip install torch torchvision || true

echo "Installed packages:" > "$LOGDIR/pip_list.log"
pip freeze >> "$LOGDIR/pip_list.log"

DATA_YAML="${DATA_YAML:-$PBS_O_WORKDIR/sunrgbd_v2/data.yaml}"
if [ ! -f "$DATA_YAML" ]; then
    echo "data.yaml not found: $DATA_YAML" >&2
    exit 1
fi

NAME="${NAME:-hpc_sunrgbd_rgb_only_3ch_e100}"
EPOCHS="${EPOCHS:-100}"
IMG_SZ="${IMG_SZ:-640}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-4}"

echo "Running training: data=$DATA_YAML epochs=$EPOCHS name=$NAME" >> "$LOGDIR/job_start.log"

# Run training and tee logs
python -u train_hpc_3ch.py --data "$DATA_YAML" --epochs "$EPOCHS" --imgsz "$IMG_SZ" --batch "$BATCH" --workers "$WORKERS" --name "$NAME" 2>&1 | tee "$LOGDIR/${NAME}.log"

echo "Job finished at $(date)" >> "$LOGDIR/job_start.log"

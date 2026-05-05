#!/bin/bash
#PBS -q Eduq
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -l walltime=02:00:00

set -euo pipefail

cd "$PBS_O_WORKDIR"

if [ ! -f yolo_env/bin/activate ]; then
    echo "ERROR: yolo_env/bin/activate not found" >&2
    exit 1
fi
source yolo_env/bin/activate

export PYTHONUNBUFFERED=1

DATA_YAML="${DATA_YAML:-$PBS_O_WORKDIR/sunrgbd_top5_rgb/data.yaml}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
IMG_SIZE="${IMG_SIZE:-640}"
WORKERS="${WORKERS:-4}"
RUN_NAME="${RUN_NAME:-top5_rgb_simple}"
DEVICE="${DEVICE:-0}"
AUGMENT="${AUGMENT:-0}"

echo "=========================================="
echo "[Simple RGB Debug Run]"
echo "Dataset:   $DATA_YAML"
echo "Epochs:    $EPOCHS"
echo "Batch:     $BATCH_SIZE"
echo "ImgSize:   $IMG_SIZE"
echo "Workers:   $WORKERS"
echo "Device:    $DEVICE"
echo "Run Name:  $RUN_NAME"
echo "=========================================="

echo "[GPU Preflight]"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo "nvidia-smi: not found"
ls -l /dev/nvidia* 2>/dev/null || echo "/dev/nvidia*: not found"

python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
PY

if ! python - <<'PY'
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY
then
    echo "ERROR: CUDA is not available to torch in this job environment." >&2
    exit 2
fi

AUGMENT_FLAG=""
if [ "$AUGMENT" = "1" ] || [ "$AUGMENT" = "true" ]; then
    AUGMENT_FLAG="--augment"
fi

python -u train_hpc_top5_rgb.py \
    --data "$DATA_YAML" \
    --epochs "$EPOCHS" \
    --batch "$BATCH_SIZE" \
    --imgsz "$IMG_SIZE" \
    --workers "$WORKERS" \
    --name "$RUN_NAME" \
    --device "$DEVICE" \
    $AUGMENT_FLAG

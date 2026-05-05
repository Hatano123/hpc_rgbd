#!/bin/bash
# Local/development version for RGB-only training on sunrgbd_top5_rgb
# This is a simpler version without PBS job scheduler directives

set -euo pipefail

cd "$(dirname "$0")"

# ============================================================
# Virtual Environment
# ============================================================
if [ ! -f yolo_env/bin/activate ]; then
    echo "ERROR: yolo_env/bin/activate not found. Please setup virtual environment first." >&2
    exit 1
fi
source yolo_env/bin/activate

export PYTHONUNBUFFERED=1

# ============================================================
# Runtime Dependencies (Optional - install to TMPDIR if available)
# ============================================================
if command -v python &> /dev/null; then
    if ! python -c "import torch, ultralytics" >/dev/null 2>&1; then
        echo "[INFO] Installing runtime dependencies..."
        if [ -n "${TMPDIR:-}" ]; then
            RUNTIME_PY_DIR="$TMPDIR/yolo_pydeps_local_$$"
            mkdir -p "$RUNTIME_PY_DIR"
            export PYTHONPATH="$RUNTIME_PY_DIR:${PYTHONPATH:-}"
            python -m pip install --no-cache-dir --target "$RUNTIME_PY_DIR" torch ultralytics pyyaml
        else
            python -m pip install torch ultralytics pyyaml
        fi
    fi
fi

# ============================================================
# Dataset Preparation
# ============================================================
if [ ! -d sunrgbd_top5_rgb/images ]; then
    if [ -f sunrgbd_top5_rgb.zip ]; then
        echo "[INFO] Extracting sunrgbd_top5_rgb.zip..."
        unzip -q sunrgbd_top5_rgb.zip
    else
        echo "ERROR: sunrgbd_top5_rgb.zip not found" >&2
        exit 1
    fi
fi

# ============================================================
# Training Configuration
# ============================================================
DATA_YAML="${DATA_YAML:-$PWD/sunrgbd_top5_rgb/data.yaml}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMG_SIZE="${IMG_SIZE:-640}"
WORKERS="${WORKERS:-4}"
RUN_NAME="${RUN_NAME:-local_sunrgbd_top5_rgb_e${EPOCHS}}"
DEVICE="${DEVICE:-0}"
AUGMENT="${AUGMENT:-0}"

echo "=========================================="
echo "[Local RGB-Only Training]"
echo "Dataset:   $DATA_YAML"
echo "Epochs:    $EPOCHS"
echo "Batch:     $BATCH_SIZE"
echo "ImgSize:   $IMG_SIZE"
echo "Workers:   $WORKERS"
echo "Device:    $DEVICE"
echo "Run Name:  $RUN_NAME"
echo "=========================================="

# ============================================================
# Execute Training
# ============================================================
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

echo "=========================================="
echo "[Training Complete]"
echo "Results: runs/$RUN_NAME"
echo "=========================================="

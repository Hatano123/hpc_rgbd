#!/bin/bash
#PBS -q Eduq
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -l walltime=08:00:00

set -euo pipefail

# 1) Base settings and robust workdir selection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [ -n "${PBS_O_WORKDIR:-}" ] && [ -d "$PBS_O_WORKDIR" ] && [ -f "$PBS_O_WORKDIR/run_top5_rgb.sh" ]; then
    cd "$PBS_O_WORKDIR"
else
    cd "$SCRIPT_DIR"
fi

LOG_FILE="train_runtime_top5_rgb.log"
PROJECT_ROOT="$(pwd -P)"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUNBUFFERED=1

# Optional: load CUDA module when cluster environment requires module-based CUDA setup.
if [ -n "${CUDA_MODULE:-}" ]; then
    if type module >/dev/null 2>&1; then
        module load "$CUDA_MODULE" || echo "[WARN] Failed to load CUDA module: $CUDA_MODULE"
    else
        echo "[WARN] 'module' command is not available; skipping CUDA module load"
    fi
fi

# Keep PBS stdout/stderr small and write runtime details into one file.
exec >>"$LOG_FILE" 2>&1

log_err() {
    echo "$*" | tee -a "$LOG_FILE" >&2
}

echo "=== Start: $(date) ==="
echo "PROJECT_ROOT=$PROJECT_ROOT"

if [ ! -x ./yolo_env/bin/python ]; then
    log_err "[ERROR] Missing virtualenv python: $PROJECT_ROOT/yolo_env/bin/python"
    log_err "[ERROR] Recreate with: ./setup_pyenv_yolo_env.sh"
    exit 1
fi
source ./yolo_env/bin/activate

if ! ./yolo_env/bin/python -c "import ssl" >/dev/null 2>&1; then
    log_err "[ERROR] ssl module is unavailable in yolo_env python. pip cannot use HTTPS indexes."
    log_err "[ERROR] Recreate with: ./setup_pyenv_yolo_env.sh"
    exit 1
fi

# 2) Prepare temporary work and lib directories
TMP_WORK="${TMPDIR:-/tmp}/top5_rgb_work_${PBS_JOBID:-local}_$$"
TMP_LIB_ROOT="${TMPDIR:-/tmp}/top5_rgb_lib_${PBS_JOBID:-local}"
LIB_DIR="$TMP_LIB_ROOT/lib"
PIP_CACHE_DIR="$TMP_LIB_ROOT/pip-cache"
mkdir -p "$TMP_WORK" "$LIB_DIR" "$PIP_CACHE_DIR"

export LIB_DIR
export PIP_CACHE_DIR
export PYTHONPATH="$LIB_DIR:${PYTHONPATH:-}"

trap 'rm -rf "$TMP_WORK"; echo "=== TMP cleanup done ==="' EXIT

echo "TMP_WORK=$TMP_WORK"
echo "LIB_DIR=$LIB_DIR"

# 3) Dataset handling: prefer zip extraction into TMPDIR for fast local IO
DATASET_ZIP="${DATASET_ZIP:-$PROJECT_ROOT/sunrgbd_top5_rgb.zip}"
if [ -f "$DATASET_ZIP" ]; then
    echo "[INFO] Extracting $(basename "$DATASET_ZIP") into TMP_WORK"
    ./yolo_env/bin/python - <<PY
import zipfile
zipfile.ZipFile("$DATASET_ZIP", "r").extractall("$TMP_WORK")
PY
    CANDIDATE_DATA="$TMP_WORK/sunrgbd_top5_rgb/data.yaml"
else
    CANDIDATE_DATA="$PROJECT_ROOT/sunrgbd_top5_rgb/data.yaml"
fi

DATA_YAML="${DATA_YAML:-$CANDIDATE_DATA}"
if [ ! -f "$DATA_YAML" ]; then
    log_err "[ERROR] data.yaml not found: $DATA_YAML"
    find "$TMP_WORK" "$PROJECT_ROOT" -type f \( -name "data.yaml" -o -name "*.yml" -o -name "*.yaml" \) 2>/dev/null | head -n 20 | tee -a "$LOG_FILE"
    exit 1
fi
echo "DATA_YAML=$DATA_YAML"

# 4) Runtime dependency install to TMPDIR only (torch must not persist on home)
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.5.1}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-torchvision==0.20.1}"

if ! ./yolo_env/bin/python - <<'PY' >/dev/null 2>&1
import os
import sys

lib_dir = os.environ["LIB_DIR"]
sys.path.insert(0, lib_dir)

try:
    import torch
except Exception:
    sys.exit(1)

torch_file = os.path.realpath(getattr(torch, "__file__", ""))
lib_dir_real = os.path.realpath(lib_dir)
sys.exit(0 if torch_file.startswith(lib_dir_real + os.sep) else 1)
PY
then
    echo "[INFO] Installing torch runtime to TMPDIR: $LIB_DIR"
    ./yolo_env/bin/python -m pip install --target="$LIB_DIR" --upgrade --no-warn-script-location \
        --quiet --progress-bar off \
        --index-url "$TORCH_INDEX_URL" \
        "$TORCH_SPEC" "$TORCHVISION_SPEC"
fi

if ! ./yolo_env/bin/python - <<'PY' >/dev/null 2>&1
import os
import sys

lib_dir = os.environ["LIB_DIR"]
sys.path.insert(0, lib_dir)

try:
    import ultralytics
except Exception:
    sys.exit(1)

pkg_file = os.path.realpath(getattr(ultralytics, "__file__", ""))
lib_dir_real = os.path.realpath(lib_dir)
sys.exit(0 if pkg_file.startswith(lib_dir_real + os.sep) else 1)
PY
then
    echo "[INFO] Installing ultralytics runtime to TMPDIR: $LIB_DIR"
    ./yolo_env/bin/python -m pip install --target="$LIB_DIR" --upgrade --no-warn-script-location \
        --quiet --progress-bar off --no-deps ultralytics
fi

if ! ./yolo_env/bin/python -c "import importlib.util, os, sys; sys.path.insert(0, os.environ['LIB_DIR']); import matplotlib, cv2, yaml, requests, scipy, psutil, PIL, numpy, polars, tifffile" >/dev/null 2>&1; then
    echo "[INFO] Installing non-torch runtime deps to TMPDIR: $LIB_DIR"
    ./yolo_env/bin/python -m pip install --target="$LIB_DIR" --upgrade --no-warn-script-location \
        --quiet --progress-bar off \
        matplotlib opencv-python-headless pyyaml requests scipy psutil pillow numpy polars tifffile
fi

# 5) Normalize data.yaml path for extracted runtime location
echo "[INFO] Rewriting data.yaml path field for runtime"
DATA_YAML="$DATA_YAML" ./yolo_env/bin/python - <<'PY'
import os
from pathlib import Path
import yaml

yaml_path = Path(os.environ["DATA_YAML"]).resolve()
cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
cfg["path"] = str(yaml_path.parent.resolve())
yaml_path.write_text(yaml.safe_dump(cfg, allow_unicode=False, sort_keys=False), encoding="utf-8")
print(f"[INFO] Updated data.yaml path={cfg.get('path')}")
PY

# 6) GPU preflight
echo "[INFO] GPU preflight"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
else
    echo "[WARN] nvidia-smi not found"
fi
ls -l /dev/nvidia* 2>/dev/null || echo "[WARN] /dev/nvidia* not found"

if ! ./yolo_env/bin/python - <<'PY'
import os
import sys
import torch

print(f"[INFO] torch={torch.__version__}")
print(f"[INFO] torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"[INFO] torch.cuda.device_count()={torch.cuda.device_count()}")
print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

if not torch.cuda.is_available():
    sys.exit(1)
PY
then
    log_err "[ERROR] CUDA is not available in this PBS job environment."
    log_err "[ERROR] This is usually a node/container GPU mapping issue (missing /dev/nvidia* or driver tools)."
    exit 2
fi

# 7) Training config and run
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
IMG_SIZE="${IMG_SIZE:-640}"
WORKERS="${WORKERS:-4}"
RUN_NAME="${RUN_NAME:-top5_rgb}"
DEVICE="${DEVICE:-0}"
AUGMENT="${AUGMENT:-0}"

echo "=========================================="
echo "[Training Config]"
echo "Dataset:   $DATA_YAML"
echo "Epochs:    $EPOCHS"
echo "Batch:     $BATCH_SIZE"
echo "ImgSize:   $IMG_SIZE"
echo "Workers:   $WORKERS"
echo "Device:    $DEVICE"
echo "Augment:   $AUGMENT"
echo "Run Name:  $RUN_NAME"
echo "=========================================="

AUGMENT_FLAG=""
if [ "$AUGMENT" = "1" ] || [ "$AUGMENT" = "true" ]; then
    AUGMENT_FLAG="--augment"
fi

./yolo_env/bin/python -u train_hpc_top5_rgb.py \
    --data "$DATA_YAML" \
    --epochs "$EPOCHS" \
    --batch "$BATCH_SIZE" \
    --imgsz "$IMG_SIZE" \
    --workers "$WORKERS" \
    --name "$RUN_NAME" \
    --device "$DEVICE" \
    $AUGMENT_FLAG

echo "=== End: $(date) ==="

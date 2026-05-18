#!/bin/bash
#PBS -q Eduq
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -l walltime=48:00:00
#PBS -N yolo_top5_full

set -euo pipefail

# 1. 基本設定
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKDIR="${PBS_O_WORKDIR:-$SCRIPT_DIR}"
if [ ! -d "$WORKDIR" ] || [ ! -f "$WORKDIR/train_hpc_top5.py" ]; then
    WORKDIR="$SCRIPT_DIR"
fi
cd "$WORKDIR" || { echo "Failed to cd to $WORKDIR" >&2; exit 1; }

PROJECT_ROOT="$(pwd -P)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Create log file in TMPDIR
TMPLOG_DIR="${TMPDIR:-/tmp}"
LOG_FILE="$TMPLOG_DIR/train_runtime_top5_full_$$.log"

exec >>"$LOG_FILE" 2>&1

echo "=== PRODUCTION RUN START: $(date) ===" | tee -a $LOG_FILE
echo "Logging to: $LOG_FILE" >&2

# 2. 一時領域の準備
TMP_LIB_ROOT="${TMPDIR:-/tmp}/y263357_full"
LIB_DIR="$TMP_LIB_ROOT/lib"
PIP_CACHE_DIR="$TMP_LIB_ROOT/pip-cache"
mkdir -p "$LIB_DIR" "$PIP_CACHE_DIR"
export LIB_DIR
export PIP_CACHE_DIR
export PYTHONPATH="$LIB_DIR:$PYTHONPATH"

trap 'echo "=== PRODUCTION RUN END: $(date) ===" | tee -a "$LOG_FILE"' EXIT

# 3. 仮想環境確認
if [ ! -x ./yolo_env/bin/python ]; then
    echo "[ERROR] Virtual environment missing: $(pwd)/yolo_env/bin/python" | tee -a "$LOG_FILE"
    echo "[ERROR] Recreate with: ./setup_pyenv_yolo_env.sh" | tee -a "$LOG_FILE"
    exit 1
fi

source ./yolo_env/bin/activate

# 4. 必須ライブラリのインストール
echo "[INFO] Installing runtime dependencies..." | tee -a "$LOG_FILE"

if ! ./yolo_env/bin/python -c "import yaml" >/dev/null 2>&1; then
    ./yolo_env/bin/python -m pip install --target="$LIB_DIR" --upgrade --no-warn-script-location \
        --quiet --progress-bar off pyyaml
fi

if ! ./yolo_env/bin/python -c "import matplotlib, cv2, yaml, requests, scipy, psutil, PIL, numpy, polars" >/dev/null 2>&1; then
    ./yolo_env/bin/python -m pip install --upgrade --no-warn-script-location \
        --quiet --progress-bar off \
        matplotlib opencv-python-headless pyyaml requests scipy psutil pillow numpy polars
fi

# 5. torch/torchvision/ultralyticsをTMPDIRにインストール
echo "[INFO] Installing torch and ultralytics to TMPDIR: $LIB_DIR" | tee -a "$LOG_FILE"
rm -rf "$PIP_CACHE_DIR"/* || true

./yolo_env/bin/python -m pip install --target="$LIB_DIR" --upgrade --no-warn-script-location \
    --quiet --progress-bar off --no-cache-dir \
    --index-url https://download.pytorch.org/whl/rocm6.0 \
    torch torchvision

./yolo_env/bin/python -m pip install --target="$LIB_DIR" --upgrade --no-warn-script-location \
    --no-deps --quiet --progress-bar off ultralytics

if [ -s requirements_hpc.txt ]; then
    ./yolo_env/bin/python -m pip install --target="$LIB_DIR" --upgrade --no-warn-script-location \
        --quiet --progress-bar off -r requirements_hpc.txt
fi

# 6. 本番実行（300エポック）
echo "=== START PRODUCTION TRAINING (300 epochs) ===" | tee -a $LOG_FILE
export PYTHONUNBUFFERED=1

DATA_YAML="/home/0/y263357/YOLORGBD/sunrgbd_top5_rgb/data.yaml"
echo "[INFO] Using data.yaml: $DATA_YAML" | tee -a $LOG_FILE
echo "[INFO] torch version: $(./yolo_env/bin/python -c 'import torch; print(torch.__version__)')" | tee -a $LOG_FILE
echo "[INFO] cuda available: $(./yolo_env/bin/python -c 'import torch; print(torch.cuda.is_available())')" | tee -a $LOG_FILE

# Production: 300 epochs (default)
./yolo_env/bin/python -u train_hpc_top5.py --data "$DATA_YAML"

echo "=== PRODUCTION RUN COMPLETED: $(date) ===" | tee -a $LOG_FILE

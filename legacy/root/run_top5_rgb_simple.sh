#!/bin/bash
#PBS -q Eduq
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -l walltime=02:00:00

set -euo pipefail

cd "$PBS_O_WORKDIR"

#!/bin/bash
# Simple CLI-only training wrapper for quick tests
# PBS-friendly: set EPOCHS, BATCH_SIZE, DEVICE via -v when qsubbing

set -euo pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}"

# Defaults (short test-friendly)
DATA_YAML="${DATA_YAML:-$PWD/sunrgbd_top5_rgb/data.yaml}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
IMG_SIZE="${IMG_SIZE:-640}"
DEVICE="${DEVICE:-0}"
RUN_NAME="${RUN_NAME:-top5_rgb_cli}"

LIB_DIR="${LIB_DIR:-${TMPDIR:-/tmp}/yolo_pydeps_${PBS_JOBID:-local}/lib}"
mkdir -p "$LIB_DIR"
export PYTHONPATH="$LIB_DIR:${PYTHONPATH:-}"

if [ ! -f yolo_env/bin/activate ]; then
    echo "ERROR: yolo_env not found. Create virtualenv first." >&2
    exit 1
fi
source yolo_env/bin/activate

# Ensure torch exists in runtime path; install to LIB_DIR if missing
if ! python - <<'PY' >/dev/null 2>&1
import os,sys
sys.path.insert(0, os.environ.get('LIB_DIR',''))
try:
    import torch
    print('FOUND', torch.__version__)
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
then
    echo "[INFO] Installing torch/torchvision to $LIB_DIR"
    python -m pip install --target="$LIB_DIR" --upgrade --no-warn-script-location \
        --index-url https://download.pytorch.org/whl/cu121 torch torchvision --no-deps
fi

echo "Running YOLO train (CLI)"
echo "data=$DATA_YAML model=yolo11n.pt epochs=$EPOCHS batch=$BATCH_SIZE imgsz=$IMG_SIZE device=$DEVICE project=runs name=$RUN_NAME"

if command -v yolo >/dev/null 2>&1; then
    yolo train data="$DATA_YAML" model=yolo11n.pt epochs="$EPOCHS" batch="$BATCH_SIZE" imgsz="$IMG_SIZE" device="$DEVICE" project=runs name="$RUN_NAME"
else
    python -m ultralytics train data="$DATA_YAML" model=yolo11n.pt epochs="$EPOCHS" batch="$BATCH_SIZE" imgsz="$IMG_SIZE" device="$DEVICE" project=runs name="$RUN_NAME"
fi

echo "Completed"
fi

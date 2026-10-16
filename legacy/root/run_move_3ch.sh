#!/bin/bash
#PBS -q Eduq
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -l walltime=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [ -n "${PBS_O_WORKDIR:-}" ] && [ -d "$PBS_O_WORKDIR" ] && [ -f "$PBS_O_WORKDIR/run_move_3ch.sh" ]; then
    cd "$PBS_O_WORKDIR"
else
    cd "$SCRIPT_DIR"
fi

LOG_FILE="train_3ch_runtime.log"
PROJECT_ROOT="$(pwd -P)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "=== run_move_3ch start: $(date) ===" | tee -a $LOG_FILE

# TMP work
TMP_WORK="${TMPDIR:-/tmp}/yolo_work_$$"
mkdir -p "$TMP_WORK"
echo "展開先: $TMP_WORK" | tee -a $LOG_FILE
trap 'rm -rf "$TMP_WORK"; echo "=== TMP cleanup done ===" | tee -a $LOG_FILE' EXIT

# If zip exists, extract; otherwise assume dataset already present in repo
if [ -f sunrgbd_v2.zip ]; then
    echo "データ展開 (sunrgbd_v2.zip)" | tee -a $LOG_FILE
    python3 -c "import zipfile; zipfile.ZipFile('sunrgbd_v2.zip','r').extractall('$TMP_WORK')"
fi

DATA_YAML="${DATA_YAML:-$PWD/sunrgbd_v2/data.yaml}"
if [ ! -f "$DATA_YAML" ]; then
    echo "ERROR: data.yaml not found: $DATA_YAML" | tee -a $LOG_FILE
    exit 1
fi

echo "データ設定: $DATA_YAML" | tee -a $LOG_FILE

export PYTHONUNBUFFERED=1

# Prepare TMP lib root
TMP_LIB_ROOT="${TMPDIR:-/tmp}/y263357"
LIB_DIR="$TMP_LIB_ROOT/lib"
PIP_CACHE_DIR="$TMP_LIB_ROOT/pip-cache"
mkdir -p "$LIB_DIR" "$PIP_CACHE_DIR"
export LIB_DIR
export PIP_CACHE_DIR
export PYTHONPATH="$LIB_DIR:$PYTHONPATH"

# Use system python (on compute node) to run pip installs and training
PYTHON_CMD=$(command -v python3 || command -v python)
if [ -z "$PYTHON_CMD" ]; then
    echo "python not found on node" >&2
    exit 1
fi

echo "Installing runtime deps to persistent venv if missing" | tee -a $LOG_FILE
$PYTHON_CMD -m pip install --user --upgrade pip pyyaml tifffile || true

echo "Installing torch/ultralytics into TMP lib: $LIB_DIR" | tee -a $LOG_FILE
$PYTHON_CMD -m pip install --target="$LIB_DIR" --upgrade --no-warn-script-location ultralytics || true
$PYTHON_CMD -m pip install --target="$LIB_DIR" --upgrade --no-warn-script-location torch torchvision || true

echo "PYTHON used: $PYTHON_CMD" | tee -a $LOG_FILE
$PYTHON_CMD -c "import sys, site; print('sys.executable', sys.executable); print('site-packages', site.getsitepackages() if hasattr(site,'getsitepackages') else 'n/a')" >> $LOG_FILE 2>&1

NAME="${NAME:-hpc_sunrgbd_rgb_only_3ch_e100}"
EPOCHS="${EPOCHS:-100}"

echo "Running training: $PYTHON_CMD train_hpc_3ch.py --data $DATA_YAML --epochs $EPOCHS --name $NAME" | tee -a $LOG_FILE

# Run training using system python but with TMP lib path first
PYTHONPATH="$LIB_DIR:$PYTHONPATH" $PYTHON_CMD -u train_hpc_3ch.py --data "$DATA_YAML" --epochs "$EPOCHS" --name "$NAME" 2>&1 | tee -a "$LOG_FILE"

echo "=== run_move_3ch end: $(date) ===" | tee -a $LOG_FILE

#!/bin/bash
#PBS -q Eduq
#PBS -l select=1:ncpus=4:mem=64gb
#PBS -l walltime=08:00:00

set -euo pipefail

cd "$PBS_O_WORKDIR"

if [ ! -f yolo_env/bin/activate ]; then
    echo "yolo_env/bin/activate が見つかりません。先に仮想環境を作成してください。" >&2
    exit 1
fi
source yolo_env/bin/activate

export PYTHONUNBUFFERED=1

# Keep heavy deps out of home storage: install runtime packages under TMPDIR per job.
RUNTIME_PY_DIR="${TMPDIR:-/tmp}/yolo_pydeps_${PBS_JOBID:-local}"
mkdir -p "$RUNTIME_PY_DIR"
export PYTHONPATH="$RUNTIME_PY_DIR:${PYTHONPATH:-}"

if ! python -c "import torch, ultralytics" >/dev/null 2>&1; then
    echo "[INFO] Installing runtime deps into TMPDIR: $RUNTIME_PY_DIR"
    python -m pip install --no-cache-dir --target "$RUNTIME_PY_DIR" torch ultralytics pyyaml tifffile
fi

# 使い方:
#   qsub run_3ch.sh
#   DATA_YAML=/path/to/data.yaml qsub run_3ch.sh
DATA_YAML="${DATA_YAML:-$PBS_O_WORKDIR/sunrgbd_v2/data.yaml}"
if [ -z "$DATA_YAML" ]; then
    echo "DATA_YAML が未設定です。例: DATA_YAML=/path/to/data.yaml qsub run_3ch.sh" >&2
    exit 1
fi

EPOCHS="${EPOCHS:-100}"
RUN_NAME="${RUN_NAME:-hpc_sunrgbd_rgb_only_3ch_e100}"

python -u train_hpc_3ch.py --data "$DATA_YAML" --epochs "$EPOCHS" --name "$RUN_NAME"
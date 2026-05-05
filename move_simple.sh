#!/bin/bash
#PBS -q Eduq
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -l walltime=02:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -n "${PBS_O_WORKDIR:-}" && -d "$PBS_O_WORKDIR" && -f "$PBS_O_WORKDIR/move.sh" ]]; then
  cd "$PBS_O_WORKDIR"
else
  cd "$SCRIPT_DIR"
fi

LOG_FILE="train_runtime.log"
PROJECT_ROOT="$(pwd -P)"
VENV_PY="$PROJECT_ROOT/yolo_env/bin/python"
ZIP_NAME="sunrgbd_v2.zip"

export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

log() {
  echo "$*" | tee -a "$LOG_FILE"
}

die() {
  echo "[ERROR] $*" | tee -a "$LOG_FILE" >&2
  exit 1
}

log "=== 処理開始: $(date) ==="

[[ -x "$VENV_PY" ]] || die "Virtual environment is missing: $VENV_PY (run ./setup_pyenv_yolo_env.sh)"
"$VENV_PY" -c "import ssl" >/dev/null 2>&1 || die "Python ssl module is unavailable (run ./setup_pyenv_yolo_env.sh)"
[[ -f "$ZIP_NAME" ]] || die "Dataset zip not found: $ZIP_NAME"

TMP_BASE="${TMPDIR:-/tmp}"
TMP_WORK="$TMP_BASE/yolo_work_${PBS_JOBID:-$$}"
TMP_LIB_ROOT="$TMP_BASE/y263357"
LIB_DIR="$TMP_LIB_ROOT/lib"
PIP_CACHE_DIR="$TMP_LIB_ROOT/pip-cache"

mkdir -p "$TMP_WORK" "$LIB_DIR" "$PIP_CACHE_DIR"
trap 'rm -rf "$TMP_WORK"; echo "=== 一時フォルダ削除完了 ===" | tee -a "$LOG_FILE"' EXIT

export LIB_DIR
export PIP_CACHE_DIR
export PYTHONPATH="$LIB_DIR:$PYTHONPATH"

log "展開先: $TMP_WORK"
log "データ展開中 ($ZIP_NAME)..."
"$VENV_PY" - <<'PY' "$ZIP_NAME" "$TMP_WORK"
import sys
import zipfile

zip_path, out_dir = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(out_dir)
print(f"[INFO] extracted: {zip_path} -> {out_dir}")
PY

DATA_YAML=$(find "$TMP_WORK" -type f \( -name "*.yaml" -o -name "*.yml" \) | head -n 1)
[[ -n "${DATA_YAML:-}" ]] || die "展開先に yaml ファイルが見つかりません"
log "設定ファイル: $DATA_YAML"

log "[INFO] Rewriting extracted data.yaml path for HPC runtime"
DATA_YAML="$DATA_YAML" "$VENV_PY" - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import os
from pathlib import Path
import yaml

yaml_path = Path(os.environ["DATA_YAML"]).resolve()
dataset_root = yaml_path.parent.resolve()

with yaml_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

cfg["path"] = str(dataset_root)

with yaml_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)

print(f"[INFO] Updated data.yaml path={cfg.get('path')}")
PY

if ! "$VENV_PY" -c "import cv2, yaml, numpy" >/dev/null 2>&1; then
  log "[INFO] Installing minimal persistent deps to yolo_env"
  "$VENV_PY" -m pip install --upgrade --no-warn-script-location opencv-python-headless pyyaml numpy
fi

# torch/torchvision/ultralytics は永続領域に入れず TMPDIR 側のみへ配置
if ! "$VENV_PY" -c "import importlib.util, os, sys; sys.path.insert(0, os.environ['LIB_DIR']); s=importlib.util.find_spec('torch'); sys.exit(0 if (s and s.origin and os.environ['LIB_DIR'] in s.origin) else 1)"; then
  log "[INFO] Installing torch/torchvision to TMPDIR: $LIB_DIR"
  "$VENV_PY" -m pip install --target="$LIB_DIR" --upgrade --no-warn-script-location \
    --index-url https://download.pytorch.org/whl/rocm6.0 \
    torch torchvision
else
  log "[INFO] torch already exists in TMPDIR lib: $LIB_DIR"
fi

if ! "$VENV_PY" -c "import importlib.util, os, sys; sys.path.insert(0, os.environ['LIB_DIR']); s=importlib.util.find_spec('ultralytics'); sys.exit(0 if (s and s.origin and os.environ['LIB_DIR'] in s.origin) else 1)"; then
  log "[INFO] Installing ultralytics to TMPDIR: $LIB_DIR"
  "$VENV_PY" -m pip install --target="$LIB_DIR" --upgrade --no-warn-script-location --no-deps ultralytics
  if [[ -s requirements_hpc.txt ]]; then
    "$VENV_PY" -m pip install --target="$LIB_DIR" --upgrade --no-warn-script-location -r requirements_hpc.txt
  fi
else
  log "[INFO] ultralytics already exists in TMPDIR lib: $LIB_DIR"
fi

log "=== 学習開始 ==="
"$VENV_PY" -c "import torch; print(f'[INFO] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}')" | tee -a "$LOG_FILE"
"$VENV_PY" -u train_hpc.py --data "$DATA_YAML" 2>&1 | tee -a "$LOG_FILE"

log "=== 処理終了: $(date) ==="

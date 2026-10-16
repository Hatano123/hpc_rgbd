#!/bin/bash
#PBS -q Eduq
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -l walltime=04:00:00

set -euo pipefail

# 1. 基本設定
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKDIR="${PBS_O_WORKDIR:-$SCRIPT_DIR}"
if [ ! -d "$WORKDIR" ] || [ ! -f "$WORKDIR/train_hpc_top5_rgb.sh" ]; then
    WORKDIR="$SCRIPT_DIR"
fi
cd "$WORKDIR" || { echo "Failed to cd to $WORKDIR" >&2; exit 1; }

PROJECT_ROOT="$(pwd -P)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Create log file in TMPDIR to avoid permission issues
TMPLOG_DIR="${TMPDIR:-/tmp}"
LOG_FILE="$TMPLOG_DIR/train_runtime_top5_rgb_$$.log"
echo "Logging to: $LOG_FILE" >&2

# Keep PBS stdout/stderr quiet; write runtime details only to the log file.
exec >>"$LOG_FILE" 2>&1

log_err() {
    # Write to runtime log and stderr so PBS .e file contains debug details.
    echo "$*" | tee -a "$LOG_FILE" >&2
}

echo "=== 処理開始: $(date) ===" | tee -a $LOG_FILE

# 2. 高速な一時領域(TMPDIR)の準備
TMP_WORK="${TMPDIR:-/tmp}/yolo_work_$$"
mkdir -p "$TMP_WORK"
echo "展開先: $TMP_WORK" | tee -a $LOG_FILE

# スクリプト終了時に必ず一時フォルダを削除する設定
trap 'rm -rf "$TMP_WORK"; echo "=== 一時フォルダ削除完了 ===" | tee -a "$LOG_FILE"' EXIT

# 3. データセットの展開
ZIP_PATH="/home/0/y263357/YOLORGBD/sunrgbd_top5_rgb.zip"
echo "データ展開中 (${ZIP_PATH})..." | tee -a $LOG_FILE
./yolo_env/bin/python3 -c "import zipfile; zipfile.ZipFile('${ZIP_PATH}', 'r').extractall('$TMP_WORK')"

# 展開先のyamlファイルを動的に取得
DATA_YAML=$(find "$TMP_WORK" -type f \( -name "data.yaml" -o -name "*.yaml" \) | head -n 1)

if [ -z "$DATA_YAML" ] || [ ! -f "$DATA_YAML" ]; then
    echo "エラー: 展開先にyamlファイルが見つかりません" | tee -a "$LOG_FILE"
    echo "展開先($TMP_WORK)の構成:" | tee -a "$LOG_FILE"
    ls -R "$TMP_WORK" | tee -a "$LOG_FILE"
    exit 1
fi

echo "設定ファイル: $DATA_YAML" | tee -a "$LOG_FILE"

# 4. ライブラリ用の一時領域を作成して環境変数に追加
TMP_LIB_ROOT="${TMPDIR:-/tmp}/y263357"
LIB_DIR="$TMP_LIB_ROOT/lib"
PIP_CACHE_DIR="$TMP_LIB_ROOT/pip-cache"
mkdir -p "$LIB_DIR" "$PIP_CACHE_DIR"
export LIB_DIR
export PIP_CACHE_DIR
export PYTHONPATH="$LIB_DIR:$PYTHONPATH"

# 仮想環境のPythonを使って学習スクリプトを実行
if [ ! -x ./yolo_env/bin/python ]; then
    log_err "[ERROR] Virtual environment is missing: $(pwd)/yolo_env/bin/python"
    log_err "[ERROR] Recreate it with: ./setup_pyenv_yolo_env.sh"
    exit 1
fi

source ./yolo_env/bin/activate

if ! ./yolo_env/bin/python -c "import ssl" >/dev/null 2>&1; then
    log_err "[ERROR] Current Python has no ssl module, so pip cannot access HTTPS indexes."
    exit 1
fi

if ! ./yolo_env/bin/python -c "import yaml" >/dev/null 2>&1; then
    echo "[INFO] Installing PyYAML to TMPDIR: $LIB_DIR"
    ./yolo_env/bin/python -m pip install --quiet --progress-bar off --target="$LIB_DIR" --upgrade --no-warn-script-location pyyaml
fi

# Keep torch in TMPDIR, but install non-torch runtime deps in local venv once.
if ! ./yolo_env/bin/python -c "import matplotlib, cv2, yaml, requests, scipy, psutil, PIL, numpy, polars" >/dev/null 2>&1; then
    echo "[INFO] Installing non-torch runtime deps to local venv (persistent)"
    ./yolo_env/bin/python -m pip install --upgrade --no-warn-script-location \
        --quiet --progress-bar off \
        matplotlib opencv-python-headless pyyaml requests scipy psutil pillow numpy polars
fi

echo "[INFO] Rewriting extracted data.yaml for HPC runtime: $DATA_YAML" | tee -a "$LOG_FILE"
DATA_YAML="$DATA_YAML" ./yolo_env/bin/python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import os
from pathlib import Path
import yaml

yaml_path = Path(os.environ["DATA_YAML"]).resolve()
dataset_root = yaml_path.parent.resolve()

with yaml_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

# Keep all dataset fields as-is, only rewrite root path for HPC runtime.
cfg["path"] = str(dataset_root)

with yaml_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)

print(f"[INFO] Updated data.yaml path={cfg.get('path')}")
PY

# Torch/TorchVision/Ultralytics は永続領域に入れず、TMPDIR 側にのみ配置する。
if ! ./yolo_env/bin/python -c "import importlib.util, os, sys; sys.path.insert(0, os.environ['LIB_DIR']); s=importlib.util.find_spec('torch'); sys.exit(0 if (s and s.origin and os.environ['LIB_DIR'] in s.origin) else 1)"; then
    echo "[INFO] Installing torch/torchvision to TMPDIR: $LIB_DIR"
    # Clear pip cache to avoid stale cuda-toolkit deps being pulled
    rm -rf "$PIP_CACHE_DIR"/* || true
    ./yolo_env/bin/python -m pip install --target="$LIB_DIR" --upgrade --no-warn-script-location \
        --quiet --progress-bar off \
        --no-cache-dir \
        --index-url https://download.pytorch.org/whl/rocm6.0 \
        torch torchvision
else
    echo "[INFO] torch already exists in TMPDIR lib: $LIB_DIR"
fi

if ! ./yolo_env/bin/python -c "import importlib.util, os, sys; sys.path.insert(0, os.environ['LIB_DIR']); s=importlib.util.find_spec('ultralytics'); sys.exit(0 if (s and s.origin and os.environ['LIB_DIR'] in s.origin) else 1)"; then
    echo "[INFO] Installing ultralytics to TMPDIR: $LIB_DIR"
    ./yolo_env/bin/python -m pip install --target="$LIB_DIR" --upgrade --no-warn-script-location --no-deps --quiet --progress-bar off ultralytics
    if [ -s requirements_hpc.txt ]; then
        ./yolo_env/bin/python -m pip install --target="$LIB_DIR" --upgrade --no-warn-script-location --quiet --progress-bar off -r requirements_hpc.txt
    fi
else
    echo "[INFO] ultralytics already exists in TMPDIR lib: $LIB_DIR"
fi

# 5. 学習の実行
echo "=== 学習開始 ===" | tee -a $LOG_FILE
export PYTHONUNBUFFERED=1

./yolo_env/bin/python -c "import torch; print(f'[INFO] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}')"

# スクリプト名と引数を調整（--no-rgb-dropout を削除）
./yolo_env/bin/python -u train_hpc_top5_rgb.py --data "$DATA_YAML"

echo "=== 処理終了: $(date) ==="
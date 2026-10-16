#!/bin/bash
#PBS -q Eduq
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -l walltime=7:00:00

set -euo pipefail

# 1. 基本設定
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [ -n "${PBS_O_WORKDIR:-}" ] && [ -d "$PBS_O_WORKDIR" ] && [ -f "$PBS_O_WORKDIR/move_top5.sh" ]; then
    cd "$PBS_O_WORKDIR"
else
    cd "$SCRIPT_DIR"
fi
LOG_FILE="train_runtime_top5.log"
PROJECT_ROOT="$(pwd -P)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

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
echo "データ展開中 (sunrgbd_top5.zip)..." | tee -a $LOG_FILE
./yolo_env/bin/python3 -c "import zipfile; zipfile.ZipFile('sunrgbd_top5.zip', 'r').extractall('$TMP_WORK')"

# 展開先の想定パスを固定して参照
DATA_YAML="$TMP_WORK/sunrgbd_top5/data.yaml"

if [ ! -f "$DATA_YAML" ]; then
    echo "エラー: 指定したdata.yamlが見つかりません: $DATA_YAML" | tee -a "$LOG_FILE"
    echo "展開済みyaml候補:" | tee -a "$LOG_FILE"
    find "$TMP_WORK" -type f \( -name "*.yaml" -o -name "*.yml" \) | tee -a "$LOG_FILE"
    exit 1
fi

echo "設定ファイル: $DATA_YAML" | tee -a "$LOG_FILE"

# 5. 学習の実行
echo "=== 学習開始 ===" | tee -a $LOG_FILE

# ログをリアルタイムで出力するための設定
export PYTHONUNBUFFERED=1

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
    log_err "[DEBUG] Python executable: $(./yolo_env/bin/python -c 'import sys; print(sys.executable)' 2>/dev/null || echo unknown)"
    log_err "[DEBUG] Python version: $(./yolo_env/bin/python -V 2>&1 || echo unknown)"
    log_err "[DEBUG] OpenSSL probe follows:"
    ./yolo_env/bin/python - <<'PY' 2>&1 | tee -a "$LOG_FILE" >&2
import traceback
import importlib.util
import subprocess
try:
    import ssl
    print("ssl OK:", ssl.OPENSSL_VERSION)
except Exception:
    print("ssl import failed")
    traceback.print_exc()
try:
    import _ssl
    print("_ssl OK:", _ssl.__file__)
except Exception:
    print("_ssl import failed")
    traceback.print_exc()
spec = importlib.util.find_spec("_ssl")
print("_ssl module path:", spec.origin if spec else "not found")
if spec and spec.origin:
    out = subprocess.run(["ldd", spec.origin], text=True, capture_output=True, check=False)
    print("ldd(_ssl) first lines:")
    print("\n".join(out.stdout.splitlines()[:12]))
PY
    log_err "[ERROR] Rebuild yolo_env with SSL-enabled Python:"
    log_err "        ./setup_pyenv_yolo_env.sh"
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
        matplotlib \
        opencv-contrib-python-headless \
        pyyaml \
        requests \
        scipy \
        psutil \
        pillow \
    numpy \
    polars
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
    ./yolo_env/bin/python -m pip install --target="$LIB_DIR" --upgrade --no-warn-script-location \
        --quiet --progress-bar off \
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

./yolo_env/bin/python -c "import torch; print(f'[INFO] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}')"
./yolo_env/bin/python -u train_hpc_top5.py --data "$DATA_YAML" --weights "" --log-file "$(pwd)/train_hpc_top5_debug.log"

echo "=== 処理終了: $(date) ==="
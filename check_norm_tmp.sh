#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${PBS_O_WORKDIR:-$BASE_DIR}"

DATA_YAML="${1:-$BASE_DIR/code/data.yaml}"
SAMPLES="${2:-32}"
DATA_ZIP="${3:-}"

if [[ ! -f "$DATA_YAML" ]]; then
  echo "[ERROR] data yaml not found: $DATA_YAML" >&2
  echo "Usage: ./check_norm_tmp.sh [data_yaml] [samples] [optional_data_zip]" >&2
  exit 1
fi

if [[ -f "$BASE_DIR/yolo_env/bin/activate" ]]; then
  # Reuse project venv when available.
  # shellcheck disable=SC1091
  source "$BASE_DIR/yolo_env/bin/activate"
fi

if [[ -z "${TMPDIR:-}" ]]; then
  TMPDIR="/tmp/$USER"
fi

WORK_TMP="$TMPDIR/rgbd_norm_check_${PBS_JOBID:-$$}"
OUT_TMP="$WORK_TMP/out"
mkdir -p "$WORK_TMP"

echo "[INFO] TMP workspace: $WORK_TMP"

EFFECTIVE_YAML="$DATA_YAML"

if [[ -n "$DATA_ZIP" ]]; then
  if [[ ! -f "$DATA_ZIP" ]]; then
    echo "[ERROR] zip not found: $DATA_ZIP" >&2
    exit 1
  fi

  echo "[INFO] extracting zip to TMP: $DATA_ZIP"
  EXTRACT_DIR="$WORK_TMP/extracted"
  mkdir -p "$EXTRACT_DIR"
  unzip -q "$DATA_ZIP" -d "$EXTRACT_DIR"

  TMP_YAML="$WORK_TMP/data_tmp.yaml"
  python - <<'PY' "$DATA_YAML" "$EXTRACT_DIR" "$TMP_YAML"
import sys
from pathlib import Path
import yaml

src_yaml = Path(sys.argv[1]).resolve()
extract_dir = Path(sys.argv[2]).resolve()
dst_yaml = Path(sys.argv[3]).resolve()

with src_yaml.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

orig_path = Path(cfg.get("path", ""))
candidate = extract_dir / orig_path.name if orig_path.name else extract_dir
cfg["path"] = str(candidate if candidate.exists() else extract_dir)

with dst_yaml.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

print(f"[INFO] rewritten yaml: {dst_yaml}")
print(f"[INFO] dataset path in tmp yaml: {cfg['path']}")
PY

  EFFECTIVE_YAML="$TMP_YAML"
fi

python "$BASE_DIR/verify_normalization_tmp.py" \
  --data "$EFFECTIVE_YAML" \
  --samples "$SAMPLES" \
  --tmp-out "$OUT_TMP"

echo "[DONE] normalization check finished"
echo "[DONE] inspect artifacts under: $OUT_TMP"

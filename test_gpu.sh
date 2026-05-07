#!/bin/bash
#PBS -q Eduq
#PBS -l select=1:ncpus=1:ngpus=1:mem=4gb
#PBS -l walltime=00:10:00

set -euo pipefail

cd "${PBS_O_WORKDIR:-$PWD}"

if type module >/dev/null 2>&1; then
    module load "${CUDA_MODULE:-cuda/12.1}" || true
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo "nvidia-smi not found"
ls -l /dev/nvidia* 2>/dev/null || echo "/dev/nvidia* not found"

if [ ! -x ./yolo_env/bin/python ]; then
    echo "ERROR: ./yolo_env/bin/python not found" >&2
    exit 1
fi

./yolo_env/bin/python - <<'PY'
import torch
print(f"torch location: {torch.__file__}")
print(f"torch version: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"cuda device count: {torch.cuda.device_count()}")
PY
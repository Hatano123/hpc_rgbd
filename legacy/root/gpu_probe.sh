#!/bin/bash
#PBS -q Eduq
#PBS -l select=1:ncpus=1:ngpus=1:mem=4gb
#PBS -l walltime=00:10:00

set -euo pipefail
cd "$PBS_O_WORKDIR"

echo "[GPU Probe]"
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "PATH=$PATH"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[nvidia-smi -L]"
  nvidia-smi -L || true
else
  echo "nvidia-smi: not found"
fi

echo "[/dev/nvidia*]"
ls -l /dev/nvidia* 2>/dev/null || echo "not found"

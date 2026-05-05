#!/bin/bash
#PBS -q Eduq
#PBS -l select=1:ncpus=1:ngpus=1:mem=8gb
#PBS -l walltime=00:05:00

cd $PBS_O_WORKDIR
echo "--- GPU PROBE ---"
echo "Hostname: $(hostname)"
echo "Uname: $(uname -a)"
echo "--- PATHS ---"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "--- TOOLS ---"
command -v nvidia-smi && nvidia-smi || echo "nvidia-smi not found"
command -v rocm-smi && rocm-smi || echo "rocm-smi not found"
ls /dev/nvidia* || echo "no nvidia devs"
ls /dev/kfd || echo "no kfd dev"
echo "--- PYTHON ---"
./yolo_env/bin/python -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

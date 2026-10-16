#!/bin/bash
#PBS -q Eduq
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -l walltime=7:00:00
set -euo pipefail
cd "${PBS_O_WORKDIR:-$(dirname "$0")}"
export PBS_O_WORKDIR="$(pwd -P)"
exec bash scripts/hpc/train.pbs "$@"

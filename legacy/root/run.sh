#!/bin/bash
#PBS -q Eduq
#PBS -l select=1:ncpus=4:ngpus=1:mem=64gb
#PBS -l walltime=02:00:00

cd $PBS_O_WORKDIR

# 環境起動 (pyenv/conda ではなくプロジェクトローカル venv)
if [ ! -f yolo_env/bin/activate ]; then
	echo "yolo_env/bin/activate が見つかりません。先に仮想環境を作成してください。" >&2
	exit 1
fi
source yolo_env/bin/activate

# ログを強制的に即出し
export PYTHONUNBUFFERED=1

# 実行
python -u train_hpc.py
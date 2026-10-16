#!/usr/bin/env bash
# Run RGBD experiment: sets YOLO_CH=4, conservative augmentations (modality dropout active)
# Edit DATA_YAML to point to your dataset yaml
DATA_YAML="data.yaml"
NAME="rgbd_experiment"
PYTHON=${PYTHON:-python3}
export YOLO_CH=4
$PYTHON train_hpc.py --data ${DATA_YAML} --mode rgbd --imgsz 640 --batch 16 --epochs 100 --name ${NAME}

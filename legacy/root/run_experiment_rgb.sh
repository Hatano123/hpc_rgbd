#!/usr/bin/env bash
# Run RGB experiment: sets YOLO_CH=3 and enables standard augmentations
# Edit DATA_YAML to point to your dataset yaml
DATA_YAML="data.yaml"
NAME="rgb_experiment"
PYTHON=${PYTHON:-python3}
export YOLO_CH=3
$PYTHON train_hpc.py --data ${DATA_YAML} --mode rgb --augment --mosaic 0.5 --mixup 0.5 --hsv_h 0.015 --hsv_s 0.7 --hsv_v 0.4 --degrees 10 --fliplr 0.5 --imgsz 640 --batch 16 --epochs 100 --name ${NAME}

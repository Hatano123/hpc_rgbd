import os
import cv2
import torch
import random
import numpy as np
import argparse
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn import tasks

# ==========================================
# 引数の受け取り設定
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--name', type=str, default='sunrgbd_4ch_v5_300ep_real')
args = parser.parse_args()

TOTAL_EPOCHS = args.epochs
RUN_NAME = args.name
TOTAL_IMAGES_ESTIMATE = 200 * TOTAL_EPOCHS

# (・・・中略：先ほどのコードの続きをすべて貼る・・・)

if __name__ == "__main__":
    # (中略)
    model.train(
        data=DATA_YAML,
        epochs=TOTAL_EPOCHS,
        imgsz=640,
        batch=16 if torch.cuda.is_available() else 4,
        project=DRIVE_RUNS_DIR,
        name=RUN_NAME,
        exist_ok=True,
        device=0 if torch.cuda.is_available() else "cpu",
        cache=False,
        workers=4 if IS_COLAB else 0,
        lr0=0.01,
        mosaic=0.0,
        augment=False,
        save_period=10,
        patience=0   # 早期終了無効化
    )

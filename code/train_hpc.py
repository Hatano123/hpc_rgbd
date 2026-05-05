import os
import cv2
import torch
import random
import numpy as np
import warnings
import logging
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn import tasks

# --- 1. HPC対策：警告ログの物理遮断 ---
warnings.filterwarnings('ignore')
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# --- 2. 環境設定 ---
BASE_DIR = "/home/0/y263357/YOLORGBD"
DATA_YAML = os.path.join(BASE_DIR, "datasets/sunrgbd/sunrgbd_tiff/data.yaml")
RUNS_DIR = os.path.join(BASE_DIR, "runs")

# キャッシュを「コード内からも」削除（念押し）
CACHE_PATH = DATA_YAML.replace("data.yaml", "labels.cache")
if os.path.exists(CACHE_PATH):
    os.remove(CACHE_PATH)

# --- 3. パラメータ ---
MAX_RGB_VAL = 65535.0
MAX_DEPTH_VAL = 5000.0
TOTAL_EPOCHS = 300 
AUGMENT_PROB_RGB_DROP = 0.5 

# --- 4. モンキーパッチ ---
def custom_load_image(self, i):
    im_path = self.im_files[i].replace('\\', '/')
    im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    if im is None: raise FileNotFoundError(f"Image Not Found: {im_path}")
    im = im.astype(np.float32)
    im[:, :, :3] = (im[:, :, :3] / MAX_RGB_VAL) * 255.0
    depth = im[:, :, 3]
    depth = np.nan_to_num(depth, nan=0.0)
    depth = np.where(depth > 0, np.log1p(depth) / np.log1p(MAX_DEPTH_VAL), 0)
    im[:, :, 3] = np.clip(depth, 0, 1.0) * 255.0
    
    current_drop_prob = 0.0
    if 'train' in im_path.lower():
        try:
            with open('/tmp/yolo_current_epoch.txt', 'r') as f:
                curr_epoch = int(f.read().strip())
                current_drop_prob = AUGMENT_PROB_RGB_DROP * min(1.0, curr_epoch / TOTAL_EPOCHS)
        except: pass
        if random.random() < current_drop_prob:
            im[:, :, :3] = 0.0 
    return im, im.shape[:2], im.shape[:2]

YOLODataset.load_image = custom_load_image

original_init = tasks.DetectionModel.__init__
def patched_init(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):
    original_init(self, cfg=cfg, ch=4, nc=nc, verbose=verbose)
tasks.DetectionModel.__init__ = patched_init

# --- 5. 学習実行 (強制設定上書き版) ---
if __name__ == "__main__":
    with open('/tmp/yolo_current_epoch.txt', 'w') as f: f.write("0")

    model = YOLO("yolov8n.yaml")
    # --- 🌟 ここが重要！ Mosaicをモデル内部の設定レベルで強制固定する 🌟 ---
    model.overrides['mosaic'] = 0.0
    model.overrides['mixup'] = 0.0
    model.overrides['close_mosaic'] = 0
    # ------------------------------------------------------------------

    pretrained_weights = YOLO("yolov8n.pt").model.state_dict()
    new_model = model.model
    new_dict = new_model.state_dict()
    for k, v in pretrained_weights.items():
        if k in new_dict and v.shape == new_dict[k].shape:
            new_dict[k] = v
        elif k == "model.0.conv.weight":
            new_v = torch.zeros(new_dict[k].shape)
            new_v[:, :3, :, :] = v
            new_dict[k] = new_v
    new_model.load_state_dict(new_dict)

    def on_train_epoch_start(trainer):
        with open('/tmp/yolo_current_epoch.txt', 'w') as f:
            f.write(str(trainer.epoch))
    model.add_callback("on_train_epoch_start", on_train_epoch_start)

    print(f"\n🚀 HPC TRAINING START (FORCE OVERRIDE): {TOTAL_EPOCHS} Epochs")
    
    model.train(
        data=DATA_YAML,
        epochs=TOTAL_EPOCHS,
        imgsz=640,
        batch=8,           # バッチを少し下げて確実に通す
        workers=0,         # 🚨 ここを 0 にして数分待つ
        device=0,
        project=RUNS_DIR,
        name="sunrgbd_4ch_final_test",
        exist_ok=True,
        cache=False,
        mosaic=0.0,
        amp=False,         # 🚨 MI210で止まる原因の9割はこれです
        save_period=1,
        verbose=True
    )
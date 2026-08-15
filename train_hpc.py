import os
import sys
import cv2
import torch
import random
import numpy as np
import argparse
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn import tasks

# ==========================================
# 1. パラメータ設定 (HPC Research Parameters)
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MAX_RGB_VAL = 65535.0          # 16-bit RGB max value
MAX_DEPTH_VAL = 5000.0         # Depth clip at 5000mm (5 meters)
AUGMENT_PROB_RGB_DROP = 0.3    # Curriculum Modality Dropout (RGB->0)
TOTAL_EPOCHS = 300

# HPC環境でのI/O最適化（TMPDIRへの過剰なアクセスを避ける）
EPOCH_TRACKER_FILE = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'yolo_current_epoch.txt')

# ==========================================
# 2. モンキーパッチ (Monkey Patching)
# ==========================================
_cached_epoch = 0

def custom_load_image(self, i):
    global _cached_epoch
    im_path = self.im_files[i]
    
    # 16ビット画像のまま読み込み
    im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"Image Not Found: {im_path}")

    # チャンネル次元の保護 (2ch -> 4ch, 3ch -> 4ch)
    if im.ndim == 2:
        im = np.stack([im, im, im, np.zeros_like(im)], axis=-1)
    elif im.ndim == 3 and im.shape[2] < 4:
        pad = np.zeros((im.shape[0], im.shape[1], 4 - im.shape[2]), dtype=im.dtype)
        im = np.concatenate([im, pad], axis=2)
    elif im.ndim == 3 and im.shape[2] > 4:
        im = im[:, :, :4]
    
    im = im.astype(np.float32)

    # ---------------------------------------------------------
    # 1. High-Precision Normalization (マッピング先: [0, 255.0])
    # ---------------------------------------------------------
    im[:, :, :3] = (im[:, :, :3] / MAX_RGB_VAL) * 255.0

    depth = im[:, :, 3]
    depth = np.nan_to_num(depth, nan=0.0)
    
    # 実距離(mm)に戻してからクリップする
    depth_mm = depth / 8.0
    depth_mm = np.clip(depth_mm, 0, MAX_DEPTH_VAL)
    
    # 5000mmスケールを 0.0〜255.0 にマッピング
    im[:, :, 3] = (depth_mm / MAX_DEPTH_VAL) * 255.0

    # ---------------------------------------------------------
    # 2. Curriculum Modality Dropout (I/O最適化済み)
    # ---------------------------------------------------------
    current_drop_prob = 0.0
    if "train" in im_path:
        # HPC環境ではファイルI/O負荷を極力下げる (5%の確率で同期)
        if random.random() < 0.05: 
            try:
                with open(EPOCH_TRACKER_FILE, 'r') as f:
                    _cached_epoch = int(f.read().strip())
            except Exception:
                pass
                
        curr_epoch = _cached_epoch
        # Curriculum: Epoch 20以降で徐々にDropout率を上げる
        if curr_epoch >= 20:
            ramp = (curr_epoch - 20) / 30.0
            current_drop_prob = min(AUGMENT_PROB_RGB_DROP, ramp * AUGMENT_PROB_RGB_DROP)
            
        if random.random() < current_drop_prob:
            im[:, :, :3] = 0.0  # RGBのみをブラックアウトし、Depthへの依存を促す

    im = np.nan_to_num(im, nan=0.0, posinf=255.0, neginf=0.0)
    im = np.clip(im, 0.0, 255.0)  # 余計な変形(.astype)を削除しました

    h0, w0 = im.shape[:2]
    
    if not hasattr(self, "_patch_verified"):
        self._patch_verified = True
        print("\n" + "="*50)
        print(f"🔍 [PATCH VERIFY] Custom load_image is ACTIVE in HPC!")
        print(f"🔍 Tracker File: {EPOCH_TRACKER_FILE}")
        print(f"🔍 Max Val (Scaled): {np.max(im[:, :, 3]):.3f}")
        print("="*50 + "\n")

    return im, (h0, w0), im.shape[:2]

YOLODataset.load_image = custom_load_image
print("✅ DEBUG: Monkey-patched YOLODataset for 16-bit & HPC Dropout")

# --- モデル構築時のチャンネル数を強制的に4chにするパッチ ---
original_init = tasks.DetectionModel.__init__
def patched_init(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):
    # 環境変数 YOLO_CH によりチャネル数を制御する（デフォルト:4）
    try:
        ch_env = int(os.environ.get('YOLO_CH', '4'))
    except Exception:
        ch_env = 4
    original_init(self, cfg=cfg, ch=ch_env, nc=nc, verbose=verbose)
tasks.DetectionModel.__init__ = patched_init
print("✅ DEBUG: Monkey-patched DetectionModel.__init__ (YOLO_CH env aware).")

# ==========================================
# 3. 学習実行 (HPC向け)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.yaml (Provided by HPC bash script)")
    parser.add_argument("--mode", choices=["rgb","rgbd"], default="rgbd", help="Run mode: rgb or rgbd (default: rgbd)")
    parser.add_argument("--mosaic", type=float, default=0.0, help="Mosaic probability/strength")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp probability/strength")
    parser.add_argument("--augment", action='store_true', help="Enable standard augmentations (RGB-only safe)")
    parser.add_argument("--hsv_h", type=float, default=0.0, help="HSV hue (for RGB only)")
    parser.add_argument("--hsv_s", type=float, default=0.0, help="HSV sat (for RGB only)")
    parser.add_argument("--hsv_v", type=float, default=0.0, help="HSV val (for RGB only)")
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation degrees")
    parser.add_argument("--fliplr", type=float, default=0.0, help="Horizontal flip probability")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS, help="Total epochs")
    parser.add_argument("--name", type=str, default="hpc_sunrgbd_v1", help="run name")
    args = parser.parse_args()

    # Set YOLO_CH early so patched_init uses correct channels
    if args.mode == 'rgb':
        os.environ['YOLO_CH'] = '3'
    else:
        os.environ['YOLO_CH'] = '4'

    cache_path = args.data.replace("data.yaml", "labels.cache")
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            print(f"✅ DEBUG: Deleted old labels.cache: {cache_path}")
        except Exception as e:
            print(f"⚠️ DEBUG: Failed to delete cache: {e}")

    # 外部のYAMLファイルを読まず、Ultralytics内蔵の構造を呼び出す
    model = YOLO("yolo11n.yaml")
    
    # Custom Weight Transfer (0-initialization for depth channel)
    pretrained_model = YOLO("yolo11n.pt").model
    new_model = model.model
    new_dict = new_model.state_dict()
    old_dict = pretrained_model.state_dict()
    
    update_dict = {}
    for k, v in old_dict.items():
        if k in new_dict:
            if v.shape == new_dict[k].shape:
                update_dict[k] = v
            elif k == "model.0.conv.weight":
                print(f"✅ DEBUG: Initializing {k} with 0-weight depth channel")
                new_v = torch.zeros(new_dict[k].shape)
                new_v[:, :3, :, :] = v
                new_v[:, 3, :, :] = 0.0 
                update_dict[k] = new_v
    
    new_dict.update(update_dict)
    new_model.load_state_dict(new_dict)
    
    # Epoch Tracker
    def on_train_epoch_start(trainer):
        try:
            with open(EPOCH_TRACKER_FILE, 'w') as f:
                f.write(str(trainer.epoch))
        except Exception as e:
            pass
            
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    
    try:
        with open(EPOCH_TRACKER_FILE, 'w') as f:
            f.write("0")
    except Exception as e:
        print(f"⚠️ DEBUG: Could not initialize epoch tracker file: {e}")
    
    print(f"🚀 HPC Training Start: {args.epochs} Epochs (mode={args.mode})")
    
    runs_dir = os.path.join(SCRIPT_DIR, "runs")

    # For RGBD mode, be conservative with augmentations to avoid 4ch bugs.
    if args.mode == 'rgbd':
        aug = False
        mosaic = float(0.0)
        mixup = float(0.0)
        hsv_h = 0.0
        hsv_s = 0.0
        hsv_v = 0.0
        degrees = float(args.degrees)
        fliplr = float(args.fliplr)
    else:
        aug = bool(args.augment)
        mosaic = float(args.mosaic)
        mixup = float(args.mixup)
        hsv_h = float(args.hsv_h)
        hsv_s = float(args.hsv_s)
        hsv_v = float(args.hsv_v)
        degrees = float(args.degrees)
        fliplr = float(args.fliplr)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=runs_dir,
        name=args.name,
        exist_ok=True,
        device=0 if torch.cuda.is_available() else "cpu",
        cache=False,
        workers=4,
        augment=aug,
        mosaic=mosaic,
        mixup=mixup,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        fliplr=fliplr,
        save_period=-1,
        save=True,
        verbose=True,
        val=True,
        amp=False
    )
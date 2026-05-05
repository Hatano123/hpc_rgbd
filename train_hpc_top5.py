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

# Runtime toggles (can be set via CLI args)
RGB_DROPOUT_ENABLED = True
FORCE_8BIT = False

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
    # 1. High-Precision Preservation (16-bitレンジを保持)
    #    RGBは元の16bitスケール(0..MAX_RGB_VAL)を残す
    # ---------------------------------------------------------
    # 何もしない：imのRGBは既に16-bit相当の値 (float32) になっている

    depth = im[:, :, 3]
    depth = np.nan_to_num(depth, nan=0.0)
    
    # 実距離(mm)に戻してからクリップする
    depth_mm = depth / 8.0
    depth_mm = np.clip(depth_mm, 0, MAX_DEPTH_VAL)
    
    # --- 【追加】深度マップの欠損補完 (Dilation) ---
    # 深度0のピクセルをマスク化
    mask = (depth_mm == 0).astype(np.uint8)
    # 3x3のカーネルで膨張させ、黒い穴を周辺の深度値で埋める
    kernel = np.ones((3, 3), np.uint8)
    depth_dilated = cv2.dilate(depth_mm, kernel, iterations=1)
    # 元が0だった箇所だけ、補完した値に置き換え
    depth_mm = np.where(mask == 1, depth_dilated, depth_mm)
    # -----------------------------------------------

    # 5000mmスケールを 0.0〜MAX_RGB_VAL (16bit相当) にマッピング
    im[:, :, 3] = (depth_mm / MAX_DEPTH_VAL) * MAX_RGB_VAL

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
            
        # Apply RGB dropout only when enabled at runtime
        if RGB_DROPOUT_ENABLED and random.random() < current_drop_prob:
            im[:, :, :3] = 0.0  # RGBのみをブラックアウトし、Depthへの依存を促す

    im = np.nan_to_num(im, nan=0.0, posinf=65535.0, neginf=0.0)
    # Clip to valid 16-bit range
    im = np.clip(im, 0.0, MAX_RGB_VAL)

    # Optional: force conversion to 8-bit for comparison experiments
    if FORCE_8BIT:
        scale = MAX_RGB_VAL / 255.0 if MAX_RGB_VAL > 255.0 else 1.0
        im[:, :, :3] = (im[:, :, :3] / scale)
        im = np.clip(im, 0.0, 255.0).astype(np.uint8)
    
    # By default return float32 (16-bit preserved as float32)

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
def patched_init(self, cfg="yolo8n.yaml", ch=3, nc=None, verbose=True):
    # cfgが何であれ、常にch=4として初期化する
    original_init(self, cfg=cfg, ch=4, nc=nc, verbose=verbose)
tasks.DetectionModel.__init__ = patched_init
print("✅ DEBUG: Monkey-patched DetectionModel.__init__ for ch=4 stability.")

# ==========================================
# 3. 学習実行 (HPC向け)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.yaml (Provided by HPC bash script)")
    parser.add_argument("--force8bit", action="store_true", help="Force convert images to 8-bit in loader for comparison")
    parser.add_argument("--no-rgb-dropout", action="store_true", help="Disable RGB modality dropout during training")
    args = parser.parse_args()

    # Apply runtime toggles (module scope; no 'global' needed at top level)
    RGB_DROPOUT_ENABLED = not bool(args.no_rgb_dropout)
    FORCE_8BIT = bool(args.force8bit)
    print(f"🔧 Runtime toggles - RGB_DROPOUT_ENABLED={RGB_DROPOUT_ENABLED}, FORCE_8BIT={FORCE_8BIT}")

    cache_path = args.data.replace("data.yaml", "labels.cache")
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            print(f"✅ DEBUG: Deleted old labels.cache: {cache_path}")
        except Exception as e:
            print(f"⚠️ DEBUG: Failed to delete cache: {e}")

    # 外部のYAMLファイルを読まず、Ultralytics内蔵の構造を呼び出す
    model = YOLO("yolov8n")
    # Mosaic を利用したいが、内部バッファが空だと DataLoader が落ちる問題がある。
    # そこで ultralytics の augment 側の get_indexes をラップして IndexError を回避する。
    try:
        import ultralytics.data.augment as ua
        import inspect

        def make_safe_get_indexes(orig):
            def safe_get_indexes(self, *args, **kwargs):
                try:
                    return orig(self, *args, **kwargs)
                except IndexError:
                    # バッファが小さい場合は Mosaic を使わずに単純なインデックスを返す
                    try:
                        n = getattr(self, 'n', None) or kwargs.get('n', None)
                        k = (n - 1) if n else kwargs.get('k', 1)
                        ds = getattr(self, 'dataset', None)
                        if ds is None:
                            return []
                        ds_len = len(getattr(ds, 'im_files', getattr(ds, 'images', [])))
                        if ds_len <= 0:
                            return []
                        return [random.randrange(ds_len) for _ in range(max(1, int(k)))]
                    except Exception:
                        return []
            return safe_get_indexes

        for name, obj in inspect.getmembers(ua, inspect.isclass):
            if hasattr(obj, 'get_indexes'):
                orig = getattr(obj, 'get_indexes')
                try:
                    setattr(obj, 'get_indexes', make_safe_get_indexes(orig))
                except Exception:
                    pass

        # モデル側は mosaic を有効にしておく（mixup は任意で無効化）
        try:
            model.overrides['mosaic'] = 1.0
            model.overrides['mixup'] = 0.0
            model.overrides['close_mosaic'] = 0
        except Exception:
            pass

        print("✅ DEBUG: Applied augment get_indexes guard; mosaic enabled (safe mode)")
    except Exception as e:
        print("⚠️ DEBUG: Could not apply augment guard:", e)
    
    # Custom Weight Transfer (RGBの平均をDepthの初期値に)
    pretrained_model = YOLO("yolov8n.pt").model
    new_model = model.model
    new_dict = new_model.state_dict()
    old_dict = pretrained_model.state_dict()
    
    update_dict = {}
    for k, v in old_dict.items():
        if k in new_dict:
            if v.shape == new_dict[k].shape:
                update_dict[k] = v
            elif k == "model.0.conv.weight":
                print(f"✅ DEBUG: Initializing {k} with mean-weight depth channel")
                new_v = torch.zeros(new_dict[k].shape)
                new_v[:, :3, :, :] = v
                # 【修正】RGB3チャンネルの平均を4チャンネル目の初期値とする
                new_v[:, 3, :, :] = v.mean(dim=1) 
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
    
    print(f"🚀 HPC Training Start: {TOTAL_EPOCHS} Epochs")
    
    runs_dir = os.path.join(SCRIPT_DIR, "runs_top5")
    os.makedirs(runs_dir, exist_ok=True)
    
    # 【修正】拡張を有効にする（ただし Mosaic/MixUp は内部で無効化）
    model.train(
        data=args.data,
        epochs=TOTAL_EPOCHS,
        imgsz=640,
        batch=16,
        project=runs_dir,
        name="hpc_sunrgbd_v1_top5",
        exist_ok=True,
        device=0 if torch.cuda.is_available() else "cpu",
        cache=False,
        workers=0,  # 短期対策: バッファ挙動観察のためワーカを0に固定

        # --- Data Augmentation Parameters ---
        augment=True,
        mosaic=0.5,    # 内部でのMosaicは無効化（安全策）
        mixup=0.0,     # MixUpも無効化
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        fliplr=0.5,
        # ------------------------------------

        save_period=-1,
        save=True,
        verbose=True,
        val=True,
        amp=False
    )
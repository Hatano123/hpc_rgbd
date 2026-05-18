import os
import sys
import cv2
import torch
import random
import numpy as np
import argparse
import traceback
from datetime import datetime
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn import tasks

# ==========================================
# 1. パラメータ設定 (HPC Phase 3: Final Strike)
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MAX_DEPTH_VAL = 8000.0         # 8m (SUN-RGBD 室内最適値)
TOTAL_EPOCHS = 150             # 再点火後の追加エポック数
MODALITY_DROPOUT_PROB = 0.35   # 【劇薬1】RGBを消してDepthを鍛える確率(35%)

# 【変更】last.ptのパスをプログラム内に直接指定
# ※前回のジョブのディレクトリに合わせています。必要に応じて修正してください。
LAST_PT_PATH = os.path.join(SCRIPT_DIR, "runs_marathon", "hpc_4ch_16bit_boost", "weights", "last.pt")


def resolve_device(device_arg=None):
    if device_arg is not None:
        return device_arg
    return 0 if torch.cuda.is_available() else "cpu"


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_tee_logging(log_file=None):
    if log_file is None:
        log_dir = os.path.join(SCRIPT_DIR, "runs_guided", "debug_logs")
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"train_hpc_top5_{stamp}.log")

    log_fp = open(log_file, "a", buffering=1, encoding="utf-8")
    sys.stdout = TeeStream(sys.__stdout__, log_fp)
    sys.stderr = TeeStream(sys.__stderr__, log_fp)
    print(f"📝 Log file: {log_file}", flush=True)
    return log_file


def complete_depth_map(depth_mm, guide_bgr):
    hole_mask = (~np.isfinite(depth_mm)) | (depth_mm <= 0)
    valid_values = depth_mm[~hole_mask]
    fill_value = float(valid_values.max()) if valid_values.size > 0 else MAX_DEPTH_VAL
    fill_value = max(fill_value, 0.0)

    depth_seed = depth_mm.copy()
    depth_seed[hole_mask] = fill_value

    try:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
            guide = guide_bgr[:, :, :3].astype(np.float32) / 65535.0
            guide_gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
            src = np.clip(depth_seed / MAX_DEPTH_VAL, 0.0, 1.0).astype(np.float32)
            filtered = cv2.ximgproc.guidedFilter(
                guide=guide_gray.astype(np.float32),
                src=src,
                radius=8,
                eps=1e-3,
            )
            depth_completed = np.clip(filtered * MAX_DEPTH_VAL, 0.0, MAX_DEPTH_VAL)
            return np.where(hole_mask, depth_completed, depth_mm), "guided_filter"
    except Exception:
        pass

    return np.where(hole_mask, fill_value, depth_mm), "max_fill"

# ==========================================
# 2. モンキーパッチ (Monkey Patching)
# ==========================================

def custom_load_image(self, i):
    im_path = self.im_files[i]
    
    # 16ビット画像の読み込み
    im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"Image Not Found: {im_path}")

    # チャンネル次元の保護 (4ch化)
    if im.ndim == 2:
        im = np.stack([im, im, im, np.zeros_like(im)], axis=-1)
    elif im.ndim == 3 and im.shape[2] < 4:
        pad = np.zeros((im.shape[0], im.shape[1], 4 - im.shape[2]), dtype=im.dtype)
        im = np.concatenate([im, pad], axis=2)
    elif im.ndim == 3 and im.shape[2] > 4:
        im = im[:, :, :4]
    
    im = im.astype(np.float32)

    # Depthチャンネルの処理
    depth = im[:, :, 3]
    depth = np.nan_to_num(depth, nan=0.0)
    depth_mm = depth / 8.0
    depth_mm = np.clip(depth_mm, 0, MAX_DEPTH_VAL)
    
    # 欠損補完: RGB ガイドの Guided Filter を優先し、失敗時は奥側値で埋める
    depth_mm, depth_fill_mode = complete_depth_map(depth_mm, im)

    # ---------------------------------------------------------
    # 🌟 根本改善: Modality Dropout (RGB消去)
    # ---------------------------------------------------------
    # 欠損補完の後に、学習時(train)かつ35%の確率でRGBを0にする。
    if "train" in im_path and random.random() < MODALITY_DROPOUT_PROB:
        im[:, :, :3] = 0.0

    # 【劇薬3】Depthの非線形スケーリング（平方根で手前の形状を強調）
    depth_norm = depth_mm / MAX_DEPTH_VAL
    im[:, :, 3] = np.sqrt(depth_norm) * 255.0
    im[:, :, :3] = (im[:, :, :3] / 65535.0) * 255.0

    im = np.nan_to_num(im, nan=0.0, posinf=255.0, neginf=0.0)
    im = np.clip(im, 0.0, 255.0).astype(np.float32)

    h0, w0 = im.shape[:2]
    
    if not hasattr(self, "_patch_verified"):
        self._patch_verified = True
        print("\n" + "="*50)
        print(f"🔥 [RE-IGNITION MODE] Modality Dropout ACTIVE ({MODALITY_DROPOUT_PROB*100}%)")
        print(f"🚀 Depth Completion: {depth_fill_mode}")
        print(f"🚀 Depth Scaling: Non-linear (Square Root)")
        print(f"🚀 Precision Preserved Range: 0-255 (float32)")
        print("="*50 + "\n")

    return im, (h0, w0), im.shape[:2]

YOLODataset.load_image = custom_load_image

# チャンネル数強制パッチ (yolov8n.yaml固定)
original_init = tasks.DetectionModel.__init__
def patched_init(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):
    original_init(self, cfg=cfg, ch=4, nc=nc, verbose=verbose)
tasks.DetectionModel.__init__ = patched_init

# ==========================================
# 3. 学習実行 (HPC Phase 3)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--device", default=None, help="Device id or string (default: auto)")
    parser.add_argument("--log-file", default=None, help="Path to debug log file (default: auto)")
    args = parser.parse_args()

    setup_tee_logging(args.log_file)

    print("=" * 60, flush=True)
    print("🚀 train_hpc_top5.py starting", flush=True)
    print(f"data={args.data}", flush=True)
    print(f"LAST_PT_PATH={LAST_PT_PATH}", flush=True)
    print(f"MODALITY_DROPOUT_PROB={MODALITY_DROPOUT_PROB}", flush=True)
    print("=" * 60, flush=True)

    try:
        # 既存の重みを読み込む（引数ではなく直接指定）
        if not os.path.exists(LAST_PT_PATH):
            print(f"❌ Error: last.ptが見つかりません。パスを確認してください:\n{LAST_PT_PATH}", flush=True)
            sys.exit(1)

        device = resolve_device(args.device)
        print(f"✅ Loading weights from: {LAST_PT_PATH}", flush=True)
        print(f"✅ Resolved device: {device}", flush=True)
        print(f"✅ torch.cuda.is_available(): {torch.cuda.is_available()}", flush=True)
        print(f"✅ torch.cuda.device_count(): {torch.cuda.device_count()}", flush=True)
        model = YOLO(LAST_PT_PATH)

        # ---------------------------------------------------------
        # Mosaicバグ回避 & カリキュラム設定
        # ---------------------------------------------------------
        try:
            import ultralytics.data.augment as ua
            import inspect

            def make_safe_get_indexes(orig):
                def safe_get_indexes(self, *args, **kwargs):
                    try:
                        return orig(self, *args, **kwargs)
                    except IndexError:
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
                    setattr(obj, 'get_indexes', make_safe_get_indexes(getattr(obj, 'get_indexes')))

            # 強制的にパラメータを上書き
            model.overrides['mosaic'] = 1.0
            model.overrides['close_mosaic'] = 50  # 最後の50エポックで幾何学を仕上げる
            print("✅ Augment patch applied", flush=True)
        except Exception as e:
            print(f"⚠️ Augment Patch Failed: {e}", flush=True)
            traceback.print_exc()

        runs_dir = os.path.join(SCRIPT_DIR, "runs_guided")

        # 学習開始
        model.train(
            data=args.data,
            epochs=TOTAL_EPOCHS,
            imgsz=640,
            batch=16,
            project=runs_dir,
            name="hpc_4ch_modality_dropout_final",
            exist_ok=True,
            device=device,
            cache=False,
            workers=4,

            # --- 根本改善のための戦略的パラメータ ---
            lr0=0.002,            # 学習率を一度戻して「ほぐす」
            lrf=0.01,             # 最終学習率を下げすぎず探索させる
            warmup_epochs=5,      # 最初の5エポックで新しい勾配に慣らす
            weight_decay=0.002,   # 正則化をさらに強めて過学習を防ぐ
            box=10.0,             # 幾何学(D)の寄与度を最大化 (default 7.5)
            cls=0.5,              # クラス分類への依存を減らす (default 0.5)
            dropout=0.3,          # 【劇薬2】ネットワーク自体の怠けを防止
            # --------------------------------------

            augment=True,
            mosaic=1.0,
            mixup=0.1,            # わずかなMixupで色の確信度をボカす
            fliplr=0.5,
            amp=False,            # 16bit保持のため絶対OFF
            save=True,
            val=True,
            verbose=True,
        )
        print("✅ Training finished", flush=True)
    except Exception:
        print("❌ Fatal error in train_hpc_top5.py", flush=True)
        traceback.print_exc()
        sys.exit(1)
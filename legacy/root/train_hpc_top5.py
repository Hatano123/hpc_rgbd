import argparse
import os
import random
import sys
import traceback
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn import tasks

# ==========================================
# 1. Parameters
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MAX_RGB_VAL = 65535.0
MAX_DEPTH_VAL = 8000.0
DEPTH_SCALE_MODE = "log"
DEPTH_LOG_REF_MM = 1000.0
DEPTH_LOG_EPS_MM = 1.0
TOTAL_EPOCHS = 150
MODALITY_DROPOUT_PROB = 0.35
MODALITY_DROPOUT_MODE = "rgb"  # rgb / depth / either
MODALITY_DROPOUT_WARMUP_EPOCH = 20
MODALITY_DROPOUT_RAMP_EPOCHS = 30
MODEL_CFG = "yolo11n.yaml"

LAST_PT_PATH = os.path.join(SCRIPT_DIR, "runs_marathon", "hpc_4ch_16bit_boost", "weights", "last.pt")
EPOCH_TRACKER_FILE = os.path.join(os.environ.get("TMPDIR", "/tmp"), "yolo_current_epoch_top5.txt")
RAW_BIT_DEPTH_DEBUG_LIMIT = 5

_cached_epoch = 0
_guided_filter_available = bool(hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"))


def resolve_device(device_arg=None):
    if device_arg is not None:
        return device_arg
    return 0 if torch.cuda.is_available() else "cpu"


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


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


def warn_guided_filter_status():
    if _guided_filter_available:
        print("✅ Guided Filter available via cv2.ximgproc", flush=True)
    else:
        print("⚠️ Guided Filter unavailable: cv2.ximgproc.guidedFilter not found; depth completion will fall back to max_fill.", flush=True)


def log_raw_bit_depth_stats(im_raw, im_path, sample_index):
    if im_raw is None or sample_index >= RAW_BIT_DEPTH_DEBUG_LIMIT:
        return

    print(f"DEBUG [raw #{sample_index + 1}] path: {im_path}", flush=True)
    print(f"DEBUG [raw #{sample_index + 1}] shape: {im_raw.shape}, dtype: {im_raw.dtype}", flush=True)

    if im_raw.ndim == 2:
        print(
            f"DEBUG [Depth-only raw #{sample_index + 1}] min: {im_raw.min()}, max: {im_raw.max()}, dtype: {im_raw.dtype}",
            flush=True,
        )
        return

    rgb = im_raw[:, :, :3]
    depth = im_raw[:, :, 3] if im_raw.shape[2] > 3 else None
    print(
        f"DEBUG [RGB channels #{sample_index + 1}] min: {rgb.min()}, max: {rgb.max()}, dtype: {im_raw.dtype}",
        flush=True,
    )
    if depth is not None:
        print(
            f"DEBUG [Depth channel #{sample_index + 1}] min: {depth.min()}, max: {depth.max()}, dtype: {im_raw.dtype}",
            flush=True,
        )


def read_current_epoch():
    global _cached_epoch
    if random.random() < 0.05:
        try:
            with open(EPOCH_TRACKER_FILE, "r", encoding="utf-8") as fp:
                _cached_epoch = int(fp.read().strip())
        except Exception:
            pass
    return _cached_epoch


def get_curriculum_drop_prob(epoch):
    if epoch < MODALITY_DROPOUT_WARMUP_EPOCH:
        return 0.0
    ramp = min(1.0, (epoch - MODALITY_DROPOUT_WARMUP_EPOCH) / max(1, MODALITY_DROPOUT_RAMP_EPOCHS))
    return MODALITY_DROPOUT_PROB * ramp


def scale_depth_channel(depth_mm):
    depth_mm = np.clip(depth_mm, 0.0, MAX_DEPTH_VAL)

    if DEPTH_SCALE_MODE == "linear":
        return depth_mm / MAX_DEPTH_VAL

    if DEPTH_SCALE_MODE == "sqrt":
        return np.sqrt(depth_mm / MAX_DEPTH_VAL)

    clamped = np.clip(depth_mm, DEPTH_LOG_EPS_MM, MAX_DEPTH_VAL)
    numerator = np.log1p(clamped / DEPTH_LOG_REF_MM)
    denominator = np.log1p(MAX_DEPTH_VAL / DEPTH_LOG_REF_MM)
    if denominator <= 0:
        return depth_mm / MAX_DEPTH_VAL
    return np.clip(numerator / denominator, 0.0, 1.0)


def maybe_apply_modality_dropout(im, im_path):
    if "train" not in im_path:
        return im, 0.0, None

    current_epoch = read_current_epoch()
    current_drop_prob = get_curriculum_drop_prob(current_epoch)
    if current_drop_prob <= 0.0 or random.random() >= current_drop_prob:
        return im, current_drop_prob, None

    if MODALITY_DROPOUT_MODE == "depth":
        im[:, :, 3] = 0.0
        return im, current_drop_prob, "depth"

    if MODALITY_DROPOUT_MODE == "either":
        if random.random() < 0.5:
            im[:, :, :3] = 0.0
            return im, current_drop_prob, "rgb"
        im[:, :, 3] = 0.0
        return im, current_drop_prob, "depth"

    im[:, :, :3] = 0.0
    return im, current_drop_prob, "rgb"


def complete_depth_map(depth_mm, guide_bgr):
    hole_mask = (~np.isfinite(depth_mm)) | (depth_mm <= 0)
    valid_values = depth_mm[~hole_mask]
    fill_value = float(valid_values.max()) if valid_values.size > 0 else MAX_DEPTH_VAL
    fill_value = max(fill_value, 0.0)

    depth_seed = depth_mm.copy()
    depth_seed[hole_mask] = fill_value

    try:
        if _guided_filter_available:
            guide = guide_bgr[:, :, :3].astype(np.float32) / MAX_RGB_VAL
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


def build_model_with_transfer(cfg_path, weights_path):
    model = YOLO(cfg_path)
    if not weights_path:
        return model

    print(f"✅ Initializing from: {weights_path}", flush=True)
    pretrained_model = YOLO(weights_path).model
    new_model = model.model
    new_dict = new_model.state_dict()
    old_dict = pretrained_model.state_dict()

    transferred = 0
    for key, value in old_dict.items():
        if key in new_dict and value.shape == new_dict[key].shape:
            new_dict[key] = value
            transferred += 1
            continue

        if key == "model.0.conv.weight" and key in new_dict and value.ndim == 4:
            target = new_dict[key]
            adapted = torch.zeros_like(target)
            copy_channels = min(value.shape[1], target.shape[1], 3)
            adapted[:, :copy_channels, :, :] = value[:, :copy_channels, :, :]
            if target.shape[1] > copy_channels:
                rgb_mean = value[:, :copy_channels, :, :].mean(dim=1, keepdim=True)
                adapted[:, copy_channels:, :, :] = rgb_mean.repeat(1, target.shape[1] - copy_channels, 1, 1)
            new_dict[key] = adapted
            transferred += 1

    missing, unexpected = new_model.load_state_dict(new_dict, strict=False)
    print(f"✅ Weight transfer complete: transferred={transferred}, missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"⚠️ Missing keys sample: {missing[:5]}", flush=True)
    if unexpected:
        print(f"⚠️ Unexpected keys sample: {unexpected[:5]}", flush=True)
    return model


def setup_epoch_tracker(model):
    try:
        with open(EPOCH_TRACKER_FILE, "w", encoding="utf-8") as fp:
            fp.write("0")
    except Exception as exc:
        print(f"⚠️ Could not initialize epoch tracker file: {exc}", flush=True)

    def on_train_epoch_start(trainer):
        try:
            with open(EPOCH_TRACKER_FILE, "w", encoding="utf-8") as fp:
                fp.write(str(trainer.epoch))
        except Exception:
            pass

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    print(f"✅ Epoch tracker file: {EPOCH_TRACKER_FILE}", flush=True)


def patch_augmentation_safety(model):
    try:
        import inspect
        import ultralytics.data.augment as ua

        def make_safe_get_indexes(orig):
            def safe_get_indexes(self, *args, **kwargs):
                try:
                    return orig(self, *args, **kwargs)
                except IndexError:
                    try:
                        n = getattr(self, "n", None) or kwargs.get("n", None)
                        k = (n - 1) if n else kwargs.get("k", 1)
                        ds = getattr(self, "dataset", None)
                        if ds is None:
                            return []
                        ds_len = len(getattr(ds, "im_files", getattr(ds, "images", [])))
                        if ds_len <= 0:
                            return []
                        return [random.randrange(ds_len) for _ in range(max(1, int(k)))]
                    except Exception:
                        return []

            return safe_get_indexes

        for _, obj in inspect.getmembers(ua, inspect.isclass):
            if hasattr(obj, "get_indexes"):
                setattr(obj, "get_indexes", make_safe_get_indexes(getattr(obj, "get_indexes")))

        model.overrides["mosaic"] = 0.5
        model.overrides["close_mosaic"] = 50
        print("✅ Augment patch applied", flush=True)
    except Exception as exc:
        print(f"⚠️ Augment Patch Failed: {exc}", flush=True)
        traceback.print_exc()


# ==========================================
# 2. Monkey Patching
# ==========================================
def custom_load_image(self, i):
    im_path = self.im_files[i]
    sample_index = getattr(self, "_raw_bit_depth_debug_count", 0)

    im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"Image Not Found: {im_path}")

    log_raw_bit_depth_stats(im, im_path, sample_index)
    if sample_index < RAW_BIT_DEPTH_DEBUG_LIMIT:
        self._raw_bit_depth_debug_count = sample_index + 1

    if im.ndim == 2:
        im = np.stack([im, im, im, np.zeros_like(im)], axis=-1)
    elif im.ndim == 3 and im.shape[2] < 4:
        pad = np.zeros((im.shape[0], im.shape[1], 4 - im.shape[2]), dtype=im.dtype)
        im = np.concatenate([im, pad], axis=2)
    elif im.ndim == 3 and im.shape[2] > 4:
        im = im[:, :, :4]

    im = im.astype(np.float32)

    depth = np.nan_to_num(im[:, :, 3], nan=0.0)
    depth_mm = np.clip(depth / 8.0, 0.0, MAX_DEPTH_VAL)
    depth_mm, depth_fill_mode = complete_depth_map(depth_mm, im)
    im, current_drop_prob, dropped_modality = maybe_apply_modality_dropout(im, im_path)

    # Depth (pure uint16) -> log-scaled [0, 1] -> [0, 255]
    depth_norm = scale_depth_channel(depth_mm)
    im[:, :, 3] = depth_norm * 255.0

    # RGB is stored as uint16 with x256 scale, so restore 8-bit range directly.
    im[:, :, :3] = im[:, :, :3] / 256.0

    im = np.nan_to_num(im, nan=0.0, posinf=255.0, neginf=0.0)
    im = np.clip(im, 0.0, 255.0).astype(np.float32)

    h0, w0 = im.shape[:2]
    if not hasattr(self, "_patch_verified"):
        self._patch_verified = True
        print("\n" + "=" * 50)
        print(f"🔥 [RE-IGNITION MODE] Modality Dropout ACTIVE (max={MODALITY_DROPOUT_PROB * 100:.1f}%)")
        print(f"🚀 Modality Dropout Mode: {MODALITY_DROPOUT_MODE}")
        print(f"🚀 Curriculum Drop Prob Now: {current_drop_prob * 100:.1f}%")
        print(f"🚀 Dropped Modality: {dropped_modality}")
        print(f"🚀 Depth Completion: {depth_fill_mode}")
        print(f"🚀 Depth Scaling: {DEPTH_SCALE_MODE}")
        print("🚀 Precision Preserved Range: 0-255 (float32)")
        print("=" * 50 + "\n")

    return im, (h0, w0), im.shape[:2]


YOLODataset.load_image = custom_load_image


# チャンネル数強制パッチ (YOLO11固定)
original_init = tasks.DetectionModel.__init__


def patched_init(self, cfg=MODEL_CFG, ch=3, nc=None, verbose=True):
    original_init(self, cfg=cfg, ch=4, nc=nc, verbose=verbose)


tasks.DetectionModel.__init__ = patched_init


# ==========================================
# 3. 学習実行
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--cfg", default=MODEL_CFG, help="YOLO model cfg (default: yolo11n.yaml)")
    parser.add_argument("--weights", default=None, help="Checkpoint or pretrained weights (default: last.pt if present, else yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--project", default=None, help="Output project directory (default: ./runs_guided)")
    parser.add_argument("--name", default="hpc_4ch_modality_dropout_final", help="Run name")
    parser.add_argument("--device", default=None, help="Device id or string (default: auto)")
    parser.add_argument("--log-file", default=None, help="Path to debug log file (default: auto)")
    parser.add_argument("--depth-scale", choices=["linear", "sqrt", "log"], default=DEPTH_SCALE_MODE, help="Depth scaling mode")
    parser.add_argument("--modality-drop-prob", type=float, default=MODALITY_DROPOUT_PROB, help="Maximum modality dropout probability")
    parser.add_argument("--modality-drop-mode", choices=["rgb", "depth", "either"], default=MODALITY_DROPOUT_MODE, help="Which modality to drop")
    parser.add_argument("--drop-warmup-epoch", type=int, default=MODALITY_DROPOUT_WARMUP_EPOCH, help="Epoch before dropout starts")
    parser.add_argument("--drop-ramp-epochs", type=int, default=MODALITY_DROPOUT_RAMP_EPOCHS, help="Epochs to ramp dropout")
    args = parser.parse_args()

    globals()["DEPTH_SCALE_MODE"] = args.depth_scale
    globals()["MODALITY_DROPOUT_PROB"] = args.modality_drop_prob
    globals()["MODALITY_DROPOUT_MODE"] = args.modality_drop_mode
    globals()["MODALITY_DROPOUT_WARMUP_EPOCH"] = args.drop_warmup_epoch
    globals()["MODALITY_DROPOUT_RAMP_EPOCHS"] = args.drop_ramp_epochs

    setup_tee_logging(args.log_file)
    warn_guided_filter_status()

    print("=" * 60, flush=True)
    print("🚀 train_hpc_top5.py starting", flush=True)
    print(f"data={args.data}", flush=True)
    print(f"cfg={args.cfg}", flush=True)
    print(f"weights={args.weights}", flush=True)
    print(f"LAST_PT_PATH={LAST_PT_PATH}", flush=True)
    print(f"MODALITY_DROPOUT_PROB={MODALITY_DROPOUT_PROB}", flush=True)
    print(f"MODALITY_DROPOUT_MODE={MODALITY_DROPOUT_MODE}", flush=True)
    print(f"DEPTH_SCALE_MODE={DEPTH_SCALE_MODE}", flush=True)
    print("=" * 60, flush=True)

    try:
        weights_path = args.weights
        if weights_path is None:
            weights_path = LAST_PT_PATH if os.path.exists(LAST_PT_PATH) else "yolo11n.pt"

        device = resolve_device(args.device)
        print(f"✅ Loading weights from: {weights_path}", flush=True)
        print(f"✅ Resolved device: {device}", flush=True)
        print(f"✅ torch.cuda.is_available(): {torch.cuda.is_available()}", flush=True)
        print(f"✅ torch.cuda.device_count(): {torch.cuda.device_count()}", flush=True)

        model = build_model_with_transfer(args.cfg, weights_path)
        setup_epoch_tracker(model)
        patch_augmentation_safety(model)

        runs_dir = args.project or os.path.join(SCRIPT_DIR, "runs_guided")

        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=runs_dir,
            name=args.name,
            exist_ok=True,
            device=device,
            cache=False,
            workers=args.workers,
            lr0=0.002,
            lrf=0.01,
            warmup_epochs=5,
            weight_decay=0.002,
            box=10.0,
            cls=0.5,
            dropout=0.3,
            augment=True,
            mosaic=0.5,
            mixup=0.05,
            fliplr=0.5,
            amp=False,
            save=True,
            val=True,
            verbose=True,
        )
        print("✅ Training finished", flush=True)
    except Exception:
        print("❌ Fatal error in train_hpc_top5.py", flush=True)
        traceback.print_exc()
        sys.exit(1)
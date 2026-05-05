"""Compare mAP between a baseline model and a 4-channel model (Fixed Collate Function & Warmup)."""

import argparse
import json
import os
import cv2
import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from copy import deepcopy

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from torch.utils.data import Dataset, DataLoader
from ultralytics.utils import DEFAULT_CFG

# --- 設定 ---
MAX_RGB_VAL = 65535.0
MAX_DEPTH_VAL = 5000.0
NUM_WORKERS = 0
SUPPORTED_EXTS = (".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp")

# ===================================================================
# 1. 共通: バッチ処理関数
# ===================================================================
def custom_collate_fn(batch):
    im_files = [item['im_file'] for item in batch]
    ori_shapes = [item['ori_shape'] for item in batch]
    resized_shapes = [item['resized_shape'] for item in batch]
    
    imgs = [item['img'] for item in batch]
    stacked_imgs = torch.stack(imgs, 0)
    
    batch_idx = []
    cls_list = []
    bboxes_list = []
    
    for i, item in enumerate(batch):
        c = item['cls']
        b = item['bboxes']
        n = c.shape[0]
        if n > 0:
            batch_idx.append(torch.full((n,), i, dtype=torch.float32))
            cls_list.append(c)
            bboxes_list.append(b)
            
    if batch_idx:
        batch_idx = torch.cat(batch_idx, 0)
        cls = torch.cat(cls_list, 0)
        bboxes = torch.cat(bboxes_list, 0)
    else:
        batch_idx = torch.empty((0,), dtype=torch.float32)
        cls = torch.empty((0, 1), dtype=torch.float32)
        bboxes = torch.empty((0, 4), dtype=torch.float32)
        
    return {
        'img': stacked_imgs,
        'batch_idx': batch_idx,
        'cls': cls,
        'bboxes': bboxes,
        'im_file': im_files,
        'ori_shape': ori_shapes,
        'resized_shape': resized_shapes,
        'ratio_pad': [(1.0, 1.0)] * len(batch)
    }

# ===================================================================
# 2. データセットクラス
# ===================================================================
class Base_TIFF_Dataset(Dataset):
    def __init__(self, img_dir, label_dir, zero_rgb: bool = False):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.image_files = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
        self.labels = self._load_labels()
        self.zero_rgb = zero_rgb

    def __len__(self):
        return len(self.image_files)

    def _load_labels(self):
        labels_data = []
        for img_path in self.image_files:
            label_path = self.label_dir / (img_path.stem + '.txt')
            cls_data = np.empty((0, 1), dtype=np.float32)
            bboxes_data = np.empty((0, 4), dtype=np.float32)
            
            if label_path.exists():
                try:
                    raw = np.loadtxt(str(label_path), ndmin=2).astype(np.float32)
                    if raw.shape[1] == 5:
                        cls_data = raw[:, 0:1]
                        bboxes_data = raw[:, 1:]
                except Exception:
                    pass
            
            labels_data.append({
                'cls': cls_data,
                'bboxes': bboxes_data,
                'im_file': str(img_path)
            })
        return labels_data

    def _load_tiff(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None: raise IOError(f"Could not read: {img_path}")
        if img.ndim == 2: img = np.expand_dims(img, axis=2)
        return img.astype(np.float32), img_path

class Dataset4Ch(Base_TIFF_Dataset):
    def __getitem__(self, idx):
        img, path = self._load_tiff(idx)
        # Ensure at least 4 channels (pad depth if missing)
        if img.shape[2] < 4:
            h, w, c = img.shape
            pad = np.zeros((h, w, 4 - c), dtype=img.dtype)
            img = np.concatenate([img, pad], axis=2)

        # dtype-aware RGB scaling
        if img.dtype == np.uint8:
            img[:, :, :3] /= 255.0
        else:
            img[:, :, :3] /= MAX_RGB_VAL

        depth = np.clip(img[:, :, 3], 0, MAX_DEPTH_VAL) / MAX_DEPTH_VAL
        img[:, :, 3] = depth
        if self.zero_rgb:
            img[:, :, :3] = 0.0
        
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        lbl = self.labels[idx]
        return {
            "img": img_tensor,
            "cls": torch.from_numpy(lbl['cls']),
            "bboxes": torch.from_numpy(lbl['bboxes']),
            "im_file": str(path),
            "ori_shape": (img.shape[0], img.shape[1]), 
            "resized_shape": (img.shape[0], img.shape[1])
        }

class Dataset3Ch(Base_TIFF_Dataset):
    def __getitem__(self, idx):
        img, path = self._load_tiff(idx)
        # Drop extra channels; ensure at least 3 by padding zeros
        if img.shape[2] > 3:
            img = img[:, :, :3]
        if img.shape[2] < 3:
            h, w, c = img.shape
            pad = np.zeros((h, w, 3 - c), dtype=img.dtype)
            img = np.concatenate([img, pad], axis=2)

        # dtype-aware RGB scaling
        if img.dtype == np.uint8:
            img_rgb = img / 255.0
        else:
            img_rgb = img / MAX_RGB_VAL
        img_rgb = np.ascontiguousarray(img_rgb)
        if self.zero_rgb:
            img_rgb[:] = 0.0

        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        lbl = self.labels[idx]
        return {
            "img": img_tensor,
            "cls": torch.from_numpy(lbl['cls']),
            "bboxes": torch.from_numpy(lbl['bboxes']),
            "im_file": str(path),
            "ori_shape": (img_rgb.shape[0], img_rgb.shape[1]), 
            "resized_shape": (img_rgb.shape[0], img_rgb.shape[1])
        }

# ===================================================================
# 3. Validator クラス
# ===================================================================
class Validator4Ch(DetectionValidator):
    def __init__(self, *args, zero_rgb=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.zero_rgb = zero_rgb

    def get_dataloader(self, dataset_path, batch_size):
        label_path = str(Path(dataset_path)).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
        dataset = Dataset4Ch(img_dir=dataset_path, label_dir=label_path, zero_rgb=self.zero_rgb)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, collate_fn=custom_collate_fn)

class Validator3Ch(DetectionValidator):
    def __init__(self, *args, zero_rgb=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.zero_rgb = zero_rgb

    def get_dataloader(self, dataset_path, batch_size):
        label_path = str(Path(dataset_path)).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
        dataset = Dataset3Ch(img_dir=dataset_path, label_dir=label_path, zero_rgb=self.zero_rgb)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, collate_fn=custom_collate_fn)

# ===================================================================
# 4. メイン処理 (修正版)
# ===================================================================
def load_data_config(data_yaml: Union[str, Path]) -> Dict[str, Any]:
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        alt = Path(__file__).parent / data_yaml
        if alt.exists():
            data_yaml = alt
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_yaml.resolve()}")

    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f) or {}
    # Resolve dataset root to an absolute path to avoid Ultralytics default download dir.
    # If YAML path is relative, prefer the directory containing the YAML itself.
    if 'path' in data_cfg:
        root = Path(data_cfg['path'])
        if not root.is_absolute():
            root = data_yaml.parent.resolve()
        data_cfg['path'] = str(root)
    else:
        data_cfg['path'] = str(data_yaml.parent.resolve())
    return data_cfg

def run_direct_validation(model_path, data_yaml, args_dict, mode='3ch', zero_rgb=False):
    print(f"\n🚀 Running validation for {mode} model: {model_path}")
    
    # ターゲットチャンネル数
    target_channels = 4 if mode == '4ch' else 3
    
    # YAML読み込み
    data_cfg = load_data_config(data_yaml)
    
    # メモリ上の設定を書き換え
    data_cfg['channels'] = target_channels
    
    # ★重要: Ultralyticsはファイルを再読み込みするため、
    # チャンネル数を修正した一時的なYAMLファイルを作成して渡す
    temp_yaml_path = f"temp_val_{mode}.yaml"
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(data_cfg, f)

    try:
        # 設定の準備
        args = deepcopy(DEFAULT_CFG)
        args.model = str(model_path)
        args.data = temp_yaml_path  # ★一時ファイルを使用
        args.imgsz = args_dict['imgsz']
        args.batch = args_dict['batch']
        args.conf = args_dict['conf']
        args.iou = args_dict['iou']
        args.device = args_dict['device']
        args.split = args_dict['split']
        args.mode = 'val'
        args.workers = NUM_WORKERS

        if mode == '4ch':
            validator = Validator4Ch(args=args, zero_rgb=zero_rgb)
        else:
            validator = Validator3Ch(args=args, zero_rgb=zero_rgb)

        # 実行
        validator(model=str(model_path))
        
        metrics = validator.metrics
        speed = validator.speed

    finally:
        # 一時ファイルの削除
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)

    return {
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "speed_inference": float(speed.get('inference', 0.0)),
    }

def format_table(baseline, rgbd):
    headers = ["Metric", "Baseline(3ch)", "Proposed(4ch)", "Diff"]
    rows = []
    for key, label in [
        ("map50_95", "mAP@50-95"), ("map50", "mAP@50"),
        ("precision", "Precision"), ("recall", "Recall"),
        ("speed_inference", "Inference(ms)"),
    ]:
        b = baseline.get(key, 0.0)
        r = rgbd.get(key, 0.0)
        diff = r - b
        rows.append((label, f"{b:.4f}", f"{r:.4f}", f"{diff:+.4f}"))
    
    lines = ["-"*55, f"{headers[0]:<15} | {headers[1]:>12} | {headers[2]:>12} | {headers[3]:>10}", "-"*55]
    for r in rows: lines.append(f"{r[0]:<15} | {r[1]:>12} | {r[2]:>12} | {r[3]:>10}")
    lines.append("-"*55)
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-model", default="yolov8n.pt")
    parser.add_argument("--rgbd-model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--baseline-data", help="Optional dataset YAML dedicated to the baseline model")
    parser.add_argument("--rgbd-data", help="Optional dataset YAML dedicated to the RGBD model")
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--baseline-rgb-zero", action="store_true")
    parser.add_argument("--rgbd-rgb-zero", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    val_args = {
        'split': args.split, 'imgsz': args.imgsz, 'batch': args.batch,
        'device': args.device, 'conf': args.conf, 'iou': args.iou
    }

    baseline_data = args.baseline_data or args.data
    rgbd_data = args.rgbd_data or args.data

    # 1. Baseline
    baseline_metrics = run_direct_validation(
        args.baseline_model, baseline_data, val_args, mode='3ch', zero_rgb=args.baseline_rgb_zero
    )

    # 2. Proposed
    rgbd_metrics = run_direct_validation(
        args.rgbd_model, rgbd_data, val_args, mode='4ch', zero_rgb=args.rgbd_rgb_zero
    )

    print("\n" + format_table(baseline_metrics, rgbd_metrics))
    
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"baseline": baseline_metrics, "rgbd_4ch": rgbd_metrics}, f, indent=2)

if __name__ == "__main__":
    main()
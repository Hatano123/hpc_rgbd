import os
import cv2
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from torch.utils.data import Dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Union

print("✅ Setup Complete!")

# --- 設定 ---

AUGMENT_PROB_RGB_DROP = 0.3   # ★変更: 暗闇化の確率を0.3 (30%) に設定
AUGMENT_PROB_DEPTH_DROP = 0.0 # 深度欠損は無効化
AUGMENT_PROB_MOSAIC = 0.5     # ★追加: モザイク拡張を行う確率 (50%)
NUM_WORKERS = 0               # Windows用
ENABLE_HFLIP = True           # 反転拡張を有効化

# 16bit画像 (0-65535) なので、65535.0 で割って 0.0-1.0 に正規化する
MAX_RGB_VAL = 65535.0          

# 単位が mm なので、5メートルは 5000.0 に設定する
MAX_DEPTH_VAL = 5000.0

# Mosaic用の入力サイズ (YOLOのimgszと合わせる)
IMG_SIZE = 640

# ===================================================================
# 1. 4チャンネル対応のカスタムデータセットクラス
# ===================================================================
class RGBD_Dataset(Dataset):
    """
    4ch Float32 TIFF を読み込み、以下の処理を行う
    1. RGB/Depthの正規化
    2. Mosaic拡張 (RGB+Depth 同時処理) ★追加
    3. 水平反転 (Flip)
    4. RGB欠損 (Darkness)
    """
    def __init__(self, img_dir, label_dir, augment_mode='train'):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.image_files = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() in ('.tiff', '.tif')])
        self.augment_mode = augment_mode
        
        self.prob_rgb_drop = AUGMENT_PROB_RGB_DROP
        self.prob_mosaic = AUGMENT_PROB_MOSAIC if augment_mode == 'train' else 0.0
        
        self.labels = self._load_labels()
        self.samples = self._build_samples()

    def __len__(self):
        return len(self.samples)

    def _build_samples(self):
        samples = []
        for idx, _ in enumerate(self.image_files):
            samples.append((idx, False))
            if self.augment_mode == 'train' and ENABLE_HFLIP:
                samples.append((idx, True))
        return samples

    def _load_labels(self):
        labels_data = []
        print(f"Loading labels from {self.label_dir}...")
        nf = len(self.image_files)
        nm = 0
        for i, img_path in enumerate(self.image_files):
            label_path = self.label_dir / (img_path.stem + '.txt')
            labels = np.empty((0, 5), dtype=np.float32)
            if label_path.exists():
                try:
                    labels = np.loadtxt(str(label_path), ndmin=2).astype(np.float32)
                    if labels.shape[1] != 5:
                        labels = np.empty((0, 5), dtype=np.float32)
                except Exception:
                    labels = np.empty((0, 5), dtype=np.float32)
            else:
                 nm += 1

            labels_dict = {
                 'cls': labels[:, 0:1] if labels.shape[0] > 0 else np.empty((0, 1), dtype=np.float32),
                 'bboxes': labels[:, 1:] if labels.shape[0] > 0 else np.empty((0, 4), dtype=np.float32),
                 'im_file': str(img_path)
            }
            labels_data.append(labels_dict)

        if nm > 0:
             print(f"WARNING: {nm} out of {nf} label files were missing.")
        print(f"Labels loaded: {len(labels_data)} items.")
        return labels_data

    # --- ★追加: 画像読み込み処理をメソッド化 (Mosaicで再利用するため) ---
    def load_aligned_item(self, idx):
        """指定インデックスの画像(4ch)とラベル(xywh)を読み込んで返す"""
        img_path = self.image_files[idx]
        
        # 1. 画像読み込み
        rgbd_image_float = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if rgbd_image_float is None:
            raise IOError(f"Could not read image: {img_path}")
        
        if rgbd_image_float.ndim == 2:
             rgbd_image_float = np.expand_dims(rgbd_image_float, axis=2)
        
        # チャンネル不足時のパディング
        if rgbd_image_float.shape[2] < 4:
             h, w, c = rgbd_image_float.shape
             dummy_depth = np.zeros((h, w, 1), dtype=np.float32)
             rgbd_image_float = np.concatenate([rgbd_image_float, dummy_depth], axis=2)

        if rgbd_image_float.dtype != np.float32:
            rgbd_image_float = rgbd_image_float.astype(np.float32)

        # 2. 正規化
        rgbd_image_float[:, :, :3] /= MAX_RGB_VAL # RGB
        
        depth_ch = rgbd_image_float[:, :, 3]
        depth_ch = np.nan_to_num(depth_ch, nan=MAX_DEPTH_VAL)
        depth_ch = np.clip(depth_ch, 0, MAX_DEPTH_VAL)
        depth_ch /= MAX_DEPTH_VAL
        rgbd_image_float[:, :, 3] = depth_ch

        # ラベル
        label_info = self.labels[idx]
        if label_info['cls'].shape[0] > 0:
            labels_np = np.hstack((label_info['cls'], label_info['bboxes'])).copy()
        else:
            labels_np = np.empty((0, 5), dtype=np.float32)
            
        return rgbd_image_float, labels_np, str(img_path)

    # --- ★追加: Mosaic拡張ロジック ---
    def load_mosaic(self, idx):
        """
        4枚の画像を組み合わせて1枚の画像を作る (4ch対応)
        """
        labels4 = []
        s = IMG_SIZE
        # モザイクの中心点 (ランダム)
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in [-s // 2]] 
        
        # 中心画像以外の3枚をランダムに選ぶ
        indices = [idx] + [random.randint(0, len(self.image_files) - 1) for _ in range(3)]
        
        # 出力画像バッファ (背景グレー初期化)
        img4 = np.full((s * 2, s * 2, 4), 0.5, dtype=np.float32)

        for i, index in enumerate(indices):
            # 画像読み込み
            img, labels, _ = self.load_aligned_item(index)
            h, w = img.shape[:2]

            # 配置位置の計算
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

            # 画像をキャンバスに貼り付け (RGB+Depthまとめて)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # ラベルの座標変換
            padw = x1a - x1b
            padh = y1a - y1b
            
            if labels.size > 0:
                # 正規化座標(xywh) -> ピクセル座標(x1y1x2y2)
                labels[:, 1] = labels[:, 1] * w + padw # x_center
                labels[:, 2] = labels[:, 2] * h + padh # y_center
                labels[:, 3] = labels[:, 3] * w # w
                labels[:, 4] = labels[:, 4] * h # h
                
                # xywh -> x1y1x2y2
                boxes = labels[:, 1:].copy()
                x1 = boxes[:, 0] - boxes[:, 2] / 2
                y1 = boxes[:, 1] - boxes[:, 3] / 2
                x2 = boxes[:, 0] + boxes[:, 2] / 2
                y2 = boxes[:, 1] + boxes[:, 3] / 2
                
                # キャンバス内にクリップ
                # np.clipを使うと変なボックスが残るので、論理チェックを行うのが理想ですが、
                # 簡易的にクリップして後で正規化します。
                labels[:, 1] = (x1 + x2) / 2
                labels[:, 2] = (y1 + y2) / 2
                labels4.append(labels)

        # ラベルの結合
        if len(labels4) > 0:
            labels4 = np.concatenate(labels4, 0)
            # クリップ処理 (画像外へはみ出したボックスの修正)
            # xywh形式のまま、まずはx,yをクリップ
            np.clip(labels4[:, 1], 0, 2 * s, out=labels4[:, 1])
            np.clip(labels4[:, 2], 0, 2 * s, out=labels4[:, 2])
            
            # 再度正規化 (0-1) に戻す (キャンバスサイズ 2*s に対して)
            labels4[:, 1:] /= (2 * s) 
            
        # 最後に指定サイズ (IMG_SIZE) にリサイズ
        # cv2.resize はチャンネル数が増えても対応可能
        img4 = cv2.resize(img4, (s, s))
        
        return img4, labels4, f"mosaic_{idx}"


    def __getitem__(self, idx) -> Dict[str, Union[np.ndarray, str, torch.Tensor]]:
        base_idx, is_flip_copy = self.samples[idx]
        
        # ===========================================================
        # ★★★ データ拡張セクション ★★★
        # ===========================================================

        # 【拡張1】 Mosaic (確率的に実行)
        # Flip用コピーでないとき、かつランダム確率でMosaic実行
        use_mosaic = (self.augment_mode == 'train' and np.random.rand() < self.prob_mosaic)
        
        if use_mosaic:
            rgbd_image_float, labels_np, im_id = self.load_mosaic(base_idx)
        else:
            # 通常読み込み
            rgbd_image_float, labels_np, im_id = self.load_aligned_item(base_idx)

        # 【拡張2】 水平反転
        # Mosaic後であっても、Flipコピーとしての呼び出しならさらに反転させる
        if is_flip_copy:
            rgbd_image_float = cv2.flip(rgbd_image_float, 1)
            if labels_np.shape[0] > 0:
                labels_np[:, 1] = 1.0 - labels_np[:, 1]
            im_id = f"{im_id}#flip"

        # 【拡張3】 RGB欠損 (確率的に実施: 暗闇環境の学習)
        if self.augment_mode == 'train' and np.random.rand() < self.prob_rgb_drop:
            rgbd_image_float[:, :, :3] = 0.0

        # ===========================================================

        return {
            "img": rgbd_image_float,
            "labels": labels_np,
            "im_file": im_id,
        }

# ===================================================================
# 2. カスタムトレーナー (変更なし)
# ===================================================================
class RGBD_Trainer(DetectionTrainer):
    def _create_custom_collate_fn(self):
        def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            imgs = [item['img'] for item in batch]
            labels_np_list = [item['labels'] for item in batch]
            im_files = [item['im_file'] for item in batch]

            img_tensors = [torch.from_numpy(img).permute(2, 0, 1).contiguous() for img in imgs]
            stacked_images = torch.stack(img_tensors, 0)

            labels_batch_list = []
            for i, label_np in enumerate(labels_np_list):
                if label_np.shape[0] > 0:
                    label_tensor = torch.from_numpy(label_np)
                    batch_index = torch.full((label_tensor.shape[0], 1), float(i), dtype=torch.float32)
                    labels_batch_list.append(torch.cat((batch_index, label_tensor), dim=1))

            stacked_labels = torch.cat(labels_batch_list, 0) if labels_batch_list else torch.empty((0, 6), dtype=torch.float32)

            return {
                'img': stacked_images,
                'batch_idx': torch.arange(len(batch), dtype=torch.int32),
                'cls': stacked_labels[:, 1:2] if stacked_labels.numel() > 0 else torch.empty((0, 1), dtype=torch.float32),
                'bboxes': stacked_labels[:, 2:] if stacked_labels.numel() > 0 else torch.empty((0, 4), dtype=torch.float32),
                'im_file': im_files,
                'ori_shape': [img.shape[:2] for img in imgs],
                'ratio_pad': [(1.0, 1.0), None] * len(imgs),
            }
        return custom_collate_fn

    def build_dataset(self, img_path, mode="train", batch=None):
        label_path = str(Path(img_path)).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
        return RGBD_Dataset(img_dir=img_path, label_dir=label_path, augment_mode=mode)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        dataset = self.build_dataset(dataset_path, mode=mode, batch=batch_size)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == "train"),
            num_workers=NUM_WORKERS,
            pin_memory=True,
            collate_fn=self._create_custom_collate_fn(), 
        )
        def reset(): pass
        loader.reset = reset
        return loader

# ===================================================================
# 3. メイン処理
# ===================================================================
if __name__ == '__main__':
    # データセットパスの設定
    dataset_yaml_path = r'..//datasets16.01-11/rgbd_aligned/dataset.yaml'
    pretrained_weights_path = "yolov8n.pt"

    if not Path(dataset_yaml_path).exists():
        print(f"エラー: データYAMLが見つかりません: {dataset_yaml_path}"); exit()

    # モデル構築
    model = YOLO(pretrained_weights_path)
    
    print("\n🚀 Starting 4-channel training (RGB+Depth, Mosaic & Dark Augmentation)...")

    model.train(
        data=dataset_yaml_path,
        trainer=RGBD_Trainer,
        epochs=100,
        batch=8,
        imgsz=IMG_SIZE,
        project='runs/detect',
        name='RGBD_MOSAIC_DARK03', # 名前を変更して区別
        workers=NUM_WORKERS,
        cache=False,
        patience=20
    )

    print("\n🎉 4-channel training complete!")
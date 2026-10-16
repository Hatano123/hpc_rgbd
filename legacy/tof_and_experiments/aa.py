import os
import random
import shutil
from pathlib import Path

# --- 設定 ---
base_path = Path("/home/0/y263357/YOLORGBD/datasets/sunrgbd/sunrgbd_tiff")
image_dir = base_path / "images"
label_dir = base_path / "labels"

# 分割後の保存先
split_names = ['train', 'val']
for sn in split_names:
    (base_path / "images" / sn).mkdir(parents=True, exist_ok=True)
    (base_path / "labels" / sn).mkdir(parents=True, exist_ok=True)

# 1. ファイルリスト取得
images = sorted(list(image_dir.glob("*.tiff")))
random.seed(42)  # 再現性のためにシード固定
random.shuffle(images)

# 2. 分割計算 (7:3)
split_idx = int(len(images) * 0.7)
train_images = images[:split_idx]
val_images = images[split_idx:]

def move_files(file_list, target_subfolder):
    print(f"Moving {len(file_list)} files to {target_subfolder}...")
    for img_path in file_list:
        # 画像の移動
        shutil.move(str(img_path), str(base_path / "images" / target_subfolder / img_path.name))
        
        # 対応するラベルの移動
        label_path = label_dir / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.move(str(label_path), str(base_path / "labels" / target_subfolder / label_path.name))

# 3. 実行
move_files(train_images, 'train')
move_files(val_images, 'val')

print("✨ Done! Data split into 7:3.")
import os
import yaml
import argparse
from pathlib import Path
from collections import Counter

def inspect_dataset(data_yaml_path):
    print(f"🔍 Analyzing dataset: {data_yaml_path}")
    
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    path = data.get('path')
    train_dir = data.get('train')
    val_dir = data.get('val')
    nc = data.get('nc')
    names = data.get('names')
    
    print(f"  - Dataset Path: {path}")
    print(f"  - Classes: {nc} ({names})")
    
    # 実際にあるファイルをスキャン
    dataset_root = Path(path)
    if not dataset_root.exists():
        print(f"❌ Error: Dataset path does not exist: {path}")
        return

    # images と labels の対応チェック（簡易版）
    # train_dir が 'train_split.txt' のようなファイル名である可能性を考慮
    images_to_check = []
    if (dataset_root / train_dir).is_dir():
        images_to_check = list((dataset_root / train_dir).rglob("*"))
    elif (dataset_root / train_dir).is_file():
        # .txt ファイルの場合はその中のパスを読み込む
        with open(dataset_root / train_dir, 'r') as f:
             images_to_check = [dataset_root / line.strip() for line in f if line.strip()]
    
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [p for p in images_to_check if p.suffix.lower() in img_exts]
    
    print(f"  - Total images found (train entry): {len(image_files)}")
    
    label_counter = Counter()
    missing_labels = 0
    
    for img_path in image_files:
        # YOLOの標準的なラベルパス変換: images/ -> labels/, ext -> .txt
        label_path = str(img_path).replace('images', 'labels').replace('image', 'labels')
        label_path = Path(label_path).with_suffix('.txt')
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        label_counter[class_id] += 1
        else:
            missing_labels += 1
            
    print(f"  - Label statistics (Object counts):")
    for i, name in enumerate(names):
        print(f"    - {name} (class {i}): {label_counter[i]}")
    
    if missing_labels > 0:
        print(f"⚠️ Warning: {missing_labels} images are missing labels!")
    else:
        print(f"✅ All images have corresponding labels.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    args = parser.parse_args()
    
    try:
        inspect_dataset(args.data)
    except Exception as e:
        print(f"❌ Error during inspection: {e}")

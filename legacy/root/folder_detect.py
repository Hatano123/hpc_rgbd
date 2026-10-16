import os
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.nn import tasks
from pathlib import Path

# --- 1. 4ch構造強制パッチ（学習時と同じものが必要） ---
# これを入れないと、model.predictの中で入力チャンネルが不一致になります
original_init = tasks.DetectionModel.__init__
def patched_init(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):
    # 学習したモデルが4chなので、推論時もch=4を強制する
    original_init(self, cfg=cfg, ch=4, nc=nc, verbose=verbose)
tasks.DetectionModel.__init__ = patched_init

# --- 設定 ---
MODEL_PATH = "/home/0/y263357/YOLORGBD/runs/sunrgbd_4ch_final/weights/best.pt"
VAL_IMAGE_DIR = Path("/home/0/y263357/YOLORGBD/datasets/sunrgbd/sunrgbd_tiff/images/val")
RESULT_DIR = Path("/home/0/y263357/YOLORGBD/inference_results")

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    
    # モデルロード
    print(f"Loading Model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 検証用画像から先頭10枚を取得
    img_files = sorted(list(VAL_IMAGE_DIR.glob("*.tiff")))[:10]
    
    if not img_files:
        print(f"Error: No .tiff files found in {VAL_IMAGE_DIR}")
        return

    print(f"Found {len(img_files)} images. Starting Inference with 4ch Patch...")

    for i, path in enumerate(img_files):
        # 16bitで生読み込み
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None: continue
        
        # プリプロセス
        input_tensor = raw.astype(np.float32)
        # 学習時と同一の正規化 (RGB: 65535, Depth: 8000)
        input_tensor[:, :, :3] = (input_tensor[:, :, :3] / 65535.0) * 255.0
        input_tensor[:, :, 3] = np.clip(input_tensor[:, :, 3], 0, 8000) / 8000.0 * 255.0

        # 推論実行（confを少し下げて0.1に設定して、まずは枠が出るか確認しましょう）
        results = model.predict(input_tensor, imgsz=640, conf=0.1, verbose=False)

        # 可視化用：元の16bit RGBを 8bit に変換
        vis_base = (raw[:, :, :3] / 65535.0 * 255.0).astype(np.uint8)

        # 検出結果を上書き
        res_plotted = results[0].plot(
            conf=True,
            labels=True,
            line_width=2,
            img=vis_base
        )

        # 保存
        save_name = RESULT_DIR / f"val_result_{path.stem}.jpg"
        cv2.imwrite(str(save_name), res_plotted)
        print(f"[{i+1}/10] Saved: {save_name.name} (Found {len(results[0].boxes)} boxes)")

    print(f"\n✅ 推論完了！ '{RESULT_DIR}' を確認してください。")

if __name__ == "__main__":
    main()
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Try to import Colab-specific visualization tool
try:
    from google.colab.patches import cv2_imshow
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

class RGBDInferenceExperiment:
    def __init__(self, model_path: str, dataset_path: str):
        self.model = YOLO(model_path)
        self.dataset_path = Path(dataset_path)
        self.max_depth_mm = 5000.0
        
        # UI State
        self.dark_mode = False
        self.show_gt = True
        self.show_depth = True
        
        # Load Dataset Samples
        print(f"Scanning dataset at: {self.dataset_path}")
        self.samples = sorted(list(self.dataset_path.rglob("image/*.jpg")))
        print(f"Found {len(self.samples)} samples.")
        self.current_idx = 0

    def get_gt_boxes(self, img_path):
        """Parses SUN-RGBD index.json to retrieve bounding boxes."""
        gt_boxes = []
        # JSON is located in: .../image/0001.jpg -> .../annotation2Dfinal/index.json
        json_path = img_path.parent.parent / "annotation2Dfinal" / "index.json"
        
        if not json_path.exists():
            return []

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                objects_meta = data.get("objects", [])
                frames = data.get("frames", [])
                
                for frame in frames:
                    poly = frame.get("polygon", [{}])[0]
                    xs = poly.get("x", [])
                    ys = poly.get("y", [])
                    if xs and ys:
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                        obj_idx = frame.get("object", -1)
                        name = objects_meta[obj_idx]["name"] if 0 <= obj_idx < len(objects_meta) else "unknown"
                        gt_boxes.append({"name": name, "box": [int(x1), int(y1), int(x2), int(y2)]})
        except Exception as e:
            print(f"Error parsing GT: {e}")
        
        return gt_boxes

    def preprocess(self, rgb, depth_mm):
        # Simulated Dark Mode
        if self.dark_mode:
            rgb = np.zeros_like(rgb)
            
        rgb_norm = rgb.astype(np.float32) / 255.0
        depth_norm = np.clip(depth_mm.astype(np.float32) / self.max_depth_mm, 0.0, 1.0)
        return np.concatenate((rgb_norm, depth_norm[..., None]), axis=-1).astype(np.float32)

    def run(self):
        while self.current_idx < len(self.samples):
            img_path = self.samples[self.current_idx]
            depth_path = img_path.parent.parent / "depth" / f"{img_path.stem}.png"
            
            # 1. Read Data
            rgb = cv2.imread(str(img_path))
            if depth_path.exists():
                depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            else:
                depth_mm = np.full(rgb.shape[:2], 1000, dtype=np.uint16)

            # 2. Inference
            input_tensor = self.preprocess(rgb, depth_mm)
            results = self.model.predict(input_tensor, imgsz=640, conf=0.25, verbose=False)
            
            # 3. Handle Visualization (Prediction)
            annotated = rgb.copy()
            if self.dark_mode:
                annotated = np.zeros_like(rgb)

            # Draw Predictions (Green)
            for res in results:
                for box in res.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    name = self.model.names[cls_id]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"Pred: {name} {conf:.2f}", (x1, y1-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 4. Draw Ground Truth (Blue)
            if self.show_gt:
                gt_boxes = self.get_gt_boxes(img_path)
                for gt in gt_boxes:
                    x1, y1, x2, y2 = gt["box"]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(annotated, f"GT: {gt['name']}", (x1, y2+15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # 5. UI Layout
            depth_vis = cv2.applyColorMap(
                cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
                cv2.COLORMAP_JET
            )
            
            # Combined View
            h, w = annotated.shape[:2]
            depth_vis_resized = cv2.resize(depth_vis, (w, h))
            combined = np.hstack((annotated, depth_vis_resized))
            
            # Status Bar
            mode_str = "DARK" if self.dark_mode else "NORMAL"
            info_text = f"Sample: {self.current_idx + 1}/{len(self.samples)} | Mode: {mode_str} | [d]Toggle Dark [g]Toggle GT [Space]Next [q]Quit"
            cv2.putText(combined, info_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 6. Show
            if IS_COLAB:
                cv2_imshow(combined)
                # In Colab, we can't do waitKey interactively easily, so we just show one and increment loop
                print(info_text)
                self.current_idx += 1
                if self.current_idx >= len(self.samples): break
            else:
                cv2.imshow("RGB-D Inference Experiment", combined)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    self.dark_mode = not self.dark_mode
                elif key == ord('g'):
                    self.show_gt = not self.show_gt
                elif key == ord(' '):
                    self.current_idx += 1
                elif key == ord('n'):
                    self.current_idx += 1

        if not IS_COLAB:
            cv2.destroyAllWindows()

def main():
    # Environment-aware Paths
    colab_path = Path("/content/drive/MyDrive/Datasets/SUNRGBD/SUNRGBD/kv2")
    win_gdrive_path = Path(r"G:\マイドライブ\Datasets\SUNRGBD\SUNRGBD\kv2")
    
    default_path = win_gdrive_path if win_gdrive_path.exists() else colab_path

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--dataset", default=str(default_path))
    args = parser.parse_args()

    exp = RGBDInferenceExperiment(args.model, args.dataset)
    exp.run()

if __name__ == "__main__":
    main()

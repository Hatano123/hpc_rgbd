import os
import cv2
import numpy as np
import random
import glob

DATASET_DIR = r"c:\Users\chibi\programming\git\YOLORGBD\datasets\sunrgbd_tiff"
OUTPUT_DIR = r"c:\Users\chibi\programming\git\YOLORGBD\debug_depth_test"

def run_debug():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: {DATASET_DIR} not found.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_files = glob.glob(os.path.join(DATASET_DIR, "images", "*.tiff"))
    if not img_files:
        print("No .tiff images found.")
        return
        
    samples = random.sample(img_files, min(10, len(img_files)))
    print(f"Sampling {len(samples)} images...")
    
    for p in samples:
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None:
            print(f"Failed to read {p}")
            continue
            
        # Depth is expected at channel 4 (index 3)
        if im.shape[2] < 4:
            print(f"Skipping {p}: Expected 4 channels, got {im.shape[2]}")
            continue
            
        depth = im[:, :, 3].astype(np.float32)
        # Normalize 0-5000mm to 0-255
        depth_norm = np.clip(depth, 0, 5000) / 5000 * 255
        vis = cv2.cvtColor(depth_norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # Load label
        label_path = p.replace("images", "labels").replace(".tiff", ".txt")
        if os.path.exists(label_path):
            h, w = vis.shape[:2]
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id, x, y, nw, nh = map(float, parts)
                        x1 = int((x - nw/2) * w)
                        y1 = int((y - nh/2) * h)
                        x2 = int((x + nw/2) * w)
                        y2 = int((y + nh/2) * h)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(vis, f"ID:{int(cls_id)}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out_path = os.path.join(OUTPUT_DIR, os.path.basename(p).replace(".tiff", ".png"))
        cv2.imwrite(out_path, vis)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    run_debug()

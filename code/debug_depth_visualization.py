import os
import cv2
import numpy as np
import random
import glob

# --- Configuration ---
DRIVE_BASE = "/content/drive/MyDrive/2025_kenkyu/YOLORGBD"
DATASET_DIR = f"{DRIVE_BASE}/datasets/sunrgbd_tiff"
OUTPUT_DIR = f"{DRIVE_BASE}/debug_depth"

NUM_SAMPLES = 10

def visualize_depth_dynamic():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_files = glob.glob(os.path.join(DATASET_DIR, "images", "*.tiff"))
    if not img_files:
        print(f"ERROR: No .tiff files found in {DATASET_DIR}/images/")
        return

    samples = random.sample(img_files, min(NUM_SAMPLES, len(img_files)))

    for img_path in samples:
        base_name = os.path.basename(img_path)
        im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if im is None or im.shape[2] < 4: continue

        depth = im[:, :, 3].astype(np.float32)
        
        # --- DYNAMIC RANGE SCALING (Per-image Min/Max) ---
        mask = depth > 0
        if np.any(mask):
            # Using Percentile for robust min/max (0.5 to 99.5)
            d_min = np.percentile(depth[mask], 0.5)
            d_max = np.percentile(depth[mask], 99.5)
            # Clip and Normalize
            depth_norm = np.clip((depth - d_min) / (d_max - d_min) * 255, 0, 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth, dtype=np.uint8)
            d_min, d_max = 0, 0
        
        # JET Color Map
        vis_im = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        # 4. Load Labels
        label_path = img_path.replace("images", "labels").replace(".tiff", ".txt")
        if os.path.exists(label_path):
            h, w = depth_norm.shape
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if not parts: continue
                    cls_id, x, y, nw, nh = map(float, parts)
                    x1 = int((x - nw/2) * w)
                    y1 = int((y - nh/2) * h)
                    x2 = int((x + nw/2) * w)
                    y2 = int((y + nh/2) * h)
                    
                    cv2.rectangle(vis_im, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.putText(vis_im, f"ID:{int(cls_id)}", (x1, y1-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Info Overlay
        cv2.putText(vis_im, f"Range: {d_min/1000:.1f}m - {d_max/1000:.1f}m", (10, h-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out_name = base_name.replace(".tiff", "_dynamic_scale.png")
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), vis_im)

    print(f"✅ DONE: Check your Google Drive '{OUTPUT_DIR}' for perfectly scaled images.")

if __name__ == "__main__":
    visualize_depth_dynamic()

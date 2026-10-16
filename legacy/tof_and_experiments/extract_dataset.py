import json
import os
import random
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

class SUNRGBDExtractor:
    def __init__(self, root_path, output_path):
        self.root_path = Path(root_path)
        self.output_path = Path(output_path)
        self.img_dir = self.output_path / "images"
        self.label_dir = self.output_path / "labels"
        
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)

        # 0: chair, 1: cup, 2: desk
        self.class_names = ["chair", "cup", "desk"]

    def find_all_keys(self, data, key, results=None):
        """Recursively collect all values for a specific key."""
        if results is None: results = []
        if isinstance(data, dict):
            if key in data:
                results.append(data[key])
            for v in data.values():
                self.find_all_keys(v, key, results)
        elif isinstance(data, list):
            for item in data:
                self.find_all_keys(item, key, results)
        return results

    def extract_from_frame(self, frame, objs_meta, w, h):
        # 1. SUN-RGBD Kinect2 Compensation
        # The align_kv2 images (centered crop) need to subtract an offset from JSON (raw space)
        # Raw space was 960x540 (half-res), Aligned is w, h.
        dx = (960 - w) / 2
        dy = (540 - h) / 2

        # 2. Group polygons by object ID
        polygons = frame.get("polygon", [])
        if not polygons:
            return None

        obj_map = {} # obj_idx -> { 'xs': [], 'ys': [], 'cls_id': ... }

        for poly in polygons:
            obj_idx = poly.get("object")
            if obj_idx is None or obj_idx >= len(objs_meta):
                continue
            
            # Identify class
            name = objs_meta[obj_idx].get("name", "").lower() if objs_meta[obj_idx] else "unknown"
            target_cls = None
            if any(x in name for x in ["chair", "stool", "swivel_chair"]): target_cls = 0
            elif any(x in name for x in ["cup", "mug", "glass"]): target_cls = 1
            elif any(x in name for x in ["desk", "table"]): target_cls = 2
            
            if target_cls is None:
                continue

            # Group all coordinates for this specific object AND APPLY OFFSET
            if obj_idx not in obj_map:
                obj_map[obj_idx] = {"cls_id": target_cls, "xs": [], "ys": []}
            
            # --- FINAL ALIGNMENT (Zero Offset Revert) ---
            # Confirmation show that align_kv2 JSON is already aligned to images.
            obj_map[obj_idx]["xs"].extend([x for x in poly.get("x", [])])
            obj_map[obj_idx]["ys"].extend([y for y in poly.get("y", [])])
            
        # 3. Convert to final label list
        results = []
        for obj_idx, data in obj_map.items():
            if data["xs"] and data["ys"]:
                results.append(data)
                
        return results if results else None

    def safe_imread(self, path, flags=cv2.IMREAD_UNCHANGED):
        try:
            return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
        except Exception:
            return None

    def run(self, target_count=200):
        print(f"Scanning {self.root_path}...")
        subdirs = [d for d in self.root_path.iterdir() if d.is_dir()]
        
        samples_pool = []
        for subdir in tqdm(subdirs, desc="Searching Samples"):
            json_file = subdir / "annotation2Dfinal" / "index.json"
            if not json_file.exists(): continue
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    objs_meta = data.get("objects", [])
                    frames = data.get("frames", [])
                    
                    # Load RGB to get dimensions for offset calculation
                    imgs = list((subdir / "image").glob("*.jpg"))
                    depths = list((subdir / "depth").glob("*.png"))
                    if not imgs or not depths: continue
                    
                    rgb_tmp = self.safe_imread(imgs[0], cv2.IMREAD_COLOR)
                    if rgb_tmp is None: continue
                    img_h, img_w = rgb_tmp.shape[:2]

                    found_labels = []
                    for fr in frames:
                        res_list = self.extract_from_frame(fr, objs_meta, img_w, img_h)
                        if res_list:
                            found_labels.extend(res_list)
                    
                    if found_labels:
                        if imgs and depths:
                            samples_pool.append({
                                "img": imgs[0], "depth": depths[0],
                                "labels": found_labels,
                                "classes": {l["cls_id"] for l in found_labels}
                            })
            except Exception: continue

        print(f"Found {len(samples_pool)} valid candidate samples.")
        if not samples_pool: return

        # Balance & Select
        samples_pool.sort(key=lambda x: 1 in x["classes"], reverse=True)
        selected = samples_pool[:target_count]
        
        print(f"Saving {len(selected)} 4ch TIFFs...")
        for i, s in enumerate(tqdm(selected, desc="Exporting")):
            rgb = self.safe_imread(s["img"], cv2.IMREAD_COLOR)
            depth = self.safe_imread(s["depth"], cv2.IMREAD_UNCHANGED)
            if rgb is None or depth is None: continue
            
            h, w = rgb.shape[:2]
            base_name = f"sunrgbd_{i:04d}"
            
            # YOLO Label
            with open(self.label_dir / f"{base_name}.txt", "w") as f:
                for lbl in s["labels"]:
                    x1, y1, x2, y2 = min(lbl["xs"]), min(lbl["ys"]), max(lbl["xs"]), max(lbl["ys"])
                    bw, bh = (x2 - x1), (y2 - y1)
                    xc, yc = x1 + bw / 2, y1 + bh / 2
                    f.write(f"{lbl['cls_id']} {xc/w:.6f} {yc/h:.6f} {bw/w:.6f} {bh/h:.6f}\n")
            
            # 4ch TIFF
            depth_resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
            out_img = np.zeros((h, w, 4), dtype=np.uint16)
            out_img[:, :, :3] = rgb.astype(np.uint16) * 256
            out_img[:, :, 3] = depth_resized
            cv2.imwrite(str(self.img_dir / f"{base_name}.tiff"), out_img)

        # YAML
        yaml = f"path: {self.output_path.absolute()}\ntrain: images\nval: images\nnc: 3\nnames: {self.class_names}\n"
        with open(self.output_path / "data.yaml", "w") as f: f.write(yaml)
        print("Dataset ready.")

if __name__ == "__main__":
    extractor = SUNRGBDExtractor(r"G:\マイドライブ\Datasets\SUNRGBD\SUNRGBD\kv2\align_kv2", "./datasets/sunrgbd_tiff")
    extractor.run(200)

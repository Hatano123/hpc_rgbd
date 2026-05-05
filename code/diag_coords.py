import json
import os
import cv2
import numpy as np
from pathlib import Path

# Sample from align_kv2
SAMPLE_DIR = Path(r"G:\マイドライブ\Datasets\SUNRGBD\SUNRGBD\kv2\align_kv2\2014-12-18_12-03-40_260595134347")

def detailed_check():
    rgb_path = list((SAMPLE_DIR / "image").glob("*.jpg"))[0]
    depth_path = list((SAMPLE_DIR / "depth").glob("*.png"))[0]
    json_path = SAMPLE_DIR / "annotation2Dfinal" / "index.json"
    
    # Check Files
    print(f"RGB: {rgb_path.name}")
    print(f"Depth: {depth_path.name}")
    print(f"JSON: {json_path.name}")
    
    # Read Image sizes
    rgb = cv2.imdecode(np.fromfile(str(rgb_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    depth = cv2.imdecode(np.fromfile(str(depth_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    
    print(f"RGB Resolution: {rgb.shape if rgb is not None else 'FAILED'}")
    print(f"Depth Resolution: {depth.shape if depth is not None else 'FAILED'}")
    
    # Read JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
        frames = data.get("frames", [])
        for i, fr in enumerate(frames):
            polys = fr.get("polygon", [])
            print(f"Frame {i} has {len(polys)} polygons.")
            for p_idx, p in enumerate(polys):
                xs = p.get("x", [])
                ys = p.get("y", [])
                if xs and ys:
                    obj_idx = p.get("object")
                    obj_name = data['objects'][obj_idx]['name'] if obj_idx is not None else "Unknown"
                    print(f"  Poly {p_idx} ({obj_name}): X_range=[{min(xs)}, {max(xs)}], Y_range=[{min(ys)}, {max(ys)}]")

if __name__ == "__main__":
    detailed_check()

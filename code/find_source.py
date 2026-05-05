import json
import os
from pathlib import Path
from tqdm import tqdm

ROOT_PATH = Path(r"G:\マイドライブ\Datasets\SUNRGBD\SUNRGBD\kv2\align_kv2")
TARGET_INDEX = 50 # sunrgbd_0050

def find_source_of_sample(target_idx):
    print(f"Searching for sample index {target_idx} in {ROOT_PATH}...")
    subdirs = [d for d in ROOT_PATH.iterdir() if d.is_dir()]
    
    current_idx = 0
    samples_pool = []
    
    # We need to replicate the sorting and selection logic from extract_dataset.py
    for subdir in tqdm(subdirs):
        json_file = subdir / "annotation2Dfinal" / "index.json"
        if not json_file.exists(): continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                objs_meta = data.get("objects", [])
                frames = data.get("frames", [])
                
                has_target = False
                for fr in frames:
                    polygons = fr.get("polygon", [])
                    for poly in polygons:
                        obj_idx = poly.get("object")
                        if obj_idx is not None and obj_idx < len(objs_meta):
                            name = objs_meta[obj_idx].get("name", "").lower()
                            if any(x in name for x in ["chair", "cup", "desk", "table", "stool"]):
                                has_target = True
                                break
                    if has_target: break
                
                if has_target:
                    imgs = list((subdir / "image").glob("*.jpg"))
                    depths = list((subdir / "depth").glob("*.png"))
                    if imgs and depths:
                        samples_pool.append({
                            "subdir": subdir,
                            "img": imgs[0],
                            "depth": depths[0],
                            "json": json_file,
                            "has_cup": any("cup" in str(o).lower() or "mug" in str(o).lower() for o in objs_meta)
                        })
        except Exception: continue

    # Replicate sorting (cups first)
    samples_pool.sort(key=lambda x: x["has_cup"], reverse=True)
    
    if target_idx < len(samples_pool):
        sample = samples_pool[target_idx]
        print(f"\nSUCCESS! Target sunrgbd_{target_idx:04d} found:")
        print(f"Source Dir: {sample['subdir']}")
        print(f"Image: {sample['img']}")
        print(f"JSON: {sample['json']}")
        return sample
    else:
        print(f"Index {target_idx} out of range (Total valid: {len(samples_pool)})")
        return None

if __name__ == "__main__":
    find_source_of_sample(TARGET_INDEX)

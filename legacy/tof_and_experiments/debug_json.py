import json
from pathlib import Path

def debug_json():
    path = Path(r"G:\マイドライブ\Datasets\SUNRGBD\SUNRGBD\kv2\align_kv2\2014-12-18_11-31-54_260595134347\annotation2Dfinal\index.json")
    if not path.exists():
        print("Path not found")
        return

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Keys in JSON: {list(data.keys())}")
    
    objects = data.get("objects", [])
    print(f"Objects list (partial): {objects[:5]}")
    
    frames = data.get("frames", [])
    print(f"Number of frames: {len(frames)}")
    
    if frames:
        print(f"Type of frames[0]: {type(frames[0])}")
        if isinstance(frames[0], list):
            print(f"Length of frames[0]: {len(frames[0])}")
            sample_frame = frames[0][0]
        else:
            sample_frame = frames[0]
            
        print(f"Sample frame keys: {list(sample_frame.keys())}")
        obj_idx = sample_frame.get("object")
        print(f"Object index in frame: {obj_idx}")
        if obj_idx is not None and obj_idx < len(objects):
            print(f"Object name: {objects[obj_idx]}")
        
    # Check if any chair is present
    for i, obj in enumerate(objects):
        name = obj.get("name", "").lower()
        if "chair" in name:
            print(f"Found 'chair' at index {i}: {obj}")

if __name__ == "__main__":
    debug_json()

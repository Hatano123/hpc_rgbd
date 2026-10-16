import json
from pathlib import Path

def debug_structure():
    path = Path(r"G:\マイドライブ\Datasets\SUNRGBD\SUNRGBD\kv2\align_kv2\2014-12-18_11-31-54_260595134347\annotation2Dfinal\index.json")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    start = content.find('"frames":')
    if start != -1:
        print(f"Content around frames: {content[start:start+1000]}")
    
    data = json.loads(content)
    frames = data.get("frames", [])
    print(f"Type of frames: {type(frames)}")
    if frames:
        print(f"Length of frames: {len(frames)}")
        print(f"First frame: {frames[0]}")
        print(f"First frame keys: {list(frames[0].keys())}")

if __name__ == "__main__":
    debug_structure()

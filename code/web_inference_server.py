import argparse
import json
import time
import threading
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil

# Enable OpenCV optimizations
cv2.setUseOptimized(True)

app = FastAPI()

# Global state to store latest results and control the inference
class GlobalState:
    def __init__(self):
        self.frame_rgb = None
        self.frame_depth_vis = None
        self.detections = []
        self.fps = 0
        self.source_mode = "camera"  # "camera" or "dataset"
        self.stopped = False
        self.lock = threading.Lock()

state = GlobalState()

# ==========================================
# 1. Inference & Data Loading Logic
# ==========================================

class RGBDInferenceEngine:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.max_depth_mm = 5000.0

    def preprocess(self, rgb, depth_mm):
        # Normalize to 0-1 for 4-channel model
        rgb_norm = rgb.astype(np.float32) / 255.0
        depth_norm = np.clip(depth_mm.astype(np.float32) / self.max_depth_mm, 0.0, 1.0)
        # Stack to 4 channels
        return np.concatenate((rgb_norm, depth_norm[..., None]), axis=-1).astype(np.float32)

    def run_inference(self, input_tensor):
        results = self.model.predict(input_tensor, imgsz=640, conf=0.25, verbose=False)
        return results

    def draw_results(self, img, results):
        annotated = img.copy()
        current_detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                name = self.class_names[cls_id] if cls_id in self.class_names else str(cls_id)
                
                # Add to detections list
                current_detections.append({"name": name, "conf": round(conf, 2), "box": [x1, y1, x2, y2]})
                
                # Draw
                color = (0, 255, 127) # Spring Green
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
        return annotated, current_detections

def inference_loop(engine, state, args_dataset=None):
    # For now, use a simulated camera if no hardware is found
    # In a real scenario, integrate VideoStream and ToFStream classes
    cap = cv2.VideoCapture(0)
    
    # Environment detection and Path mapping
    # 1. Google Colab (MyDrive)
    # 2. Local Windows (G:\マイドライブ)
    # 3. Local Default (samples/sun_rgbd/kv2)
    
    colab_path = Path("/content/drive/MyDrive/Datasets/SUNRGBD/SUNRGBD/kv2")
    win_gdrive_path = Path(r"G:\マイドライブ\Datasets\SUNRGBD\SUNRGBD\kv2")
    local_default_path = Path("samples/sun_rgbd/kv2")

    if args_dataset:
        dataset_path = Path(args_dataset)
    elif colab_path.exists():
        dataset_path = colab_path
    elif win_gdrive_path.exists():
        dataset_path = win_gdrive_path
    else:
        dataset_path = local_default_path

    print(f"Dataset path set to: {dataset_path}")

    dataset_frames = []
    if dataset_path.exists():
        # SUN-RGBD structure: dataset_path / align_kv2 / {timestamp} / image / {name}.jpg
        # Search recursively for image folders
        image_files = sorted(list(dataset_path.rglob("image/*.jpg")))
        dataset_frames = image_files
        print(f"Found {len(dataset_frames)} samples in dataset.")

    frame_idx = 0
    
    while not state.stopped:
        t0 = time.time()
        
        rgb = None
        depth_mm = None
        
        if state.source_mode == "camera" and cap.isOpened():
            ret, rgb = cap.read()
            if ret:
                rgb = cv2.resize(rgb, (640, 480))
                # Simulated depth for demo
                h, w = rgb.shape[:2]
                depth_mm = np.full((h, w), 1000, dtype=np.uint16)
        elif state.source_mode == "dataset" and dataset_frames:
            img_path = dataset_frames[frame_idx % len(dataset_frames)]
            
            # Map image to depth path: .../image/0001.jpg -> .../depth/0001.png
            depth_path = img_path.parent.parent / "depth" / f"{img_path.stem}.png"
            
            rgb = cv2.imread(str(img_path))
            if depth_path.exists():
                depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            else:
                # Fallback if depth file is missing in the specific folder
                depth_mm = np.full((480, 640), 1000, dtype=np.uint16)
            
            rgb = cv2.resize(rgb, (640, 480))
            depth_mm = cv2.resize(depth_mm, (640, 480), interpolation=cv2.INTER_NEAREST)
            frame_idx += 1
            time.sleep(0.5) # Dataset slow playback

        if rgb is not None and depth_mm is not None:
            # Inference
            input_tensor = engine.preprocess(rgb, depth_mm)
            results = engine.run_inference(input_tensor)
            
            # Rendering
            annotated, detections = engine.draw_results(rgb, results)
            
            # Depth visualization
            depth_vis = cv2.applyColorMap(
                cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
                cv2.COLORMAP_JET
            )
            
            # Update state
            with state.lock:
                state.frame_rgb = annotated
                state.frame_depth_vis = depth_vis
                state.detections = detections
                state.fps = 1.0 / (time.time() - t0)
        
        time.sleep(0.01)

# ==========================================
# 2. Web Endpoints
# ==========================================

def gen_frames(state, mode="rgb"):
    while not state.stopped:
        with state.lock:
            frame = state.frame_rgb if mode == "rgb" else state.frame_depth_vis
            if frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04) # ~25 FPS stream

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(state, "rgb"), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/depth_feed")
async def depth_feed():
    return StreamingResponse(gen_frames(state, "depth"), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/stats")
async def get_stats():
    with state.lock:
        return {
            "fps": round(state.fps, 1),
            "detections": state.detections,
            "source": state.source_mode
        }

@app.post("/api/settings")
async def update_settings(request: Request):
    data = await request.json()
    if "source" in data:
        with state.lock:
            state.source_mode = data["source"]
    return {"status": "ok"}

app.mount("/", StaticFiles(directory="static", html=True), name="static")

# ==========================================
# 3. Entry Point
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dataset", default=None, help="Path to SUNRGBD 'kv2' root folder")
    args = parser.parse_args()

    engine = RGBDInferenceEngine(args.model)
    
    # Start inference thread
    threading.Thread(target=inference_loop, args=(engine, state, args.dataset), daemon=True).start()
    
    print(f"Starting server at http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Real-time RGBD detection optimized for FPS & Custom Drawing."""

import argparse
import json
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Enable OpenCV optimizations
cv2.setUseOptimized(True)

try:
    from ifm3dpy.device import Device, O3D
    from ifm3dpy.framegrabber import FrameGrabber, buffer_id
except ImportError as exc:
    raise SystemExit(
        "ifm3dpy is required. Install via 'pip install ifm3dpy'."
    ) from exc

FG_WAIT_TIMEOUT_MS = 1500
DEFAULT_R = np.array(
    [[0.99989512, 0.01158434, -0.00869217],
     [-0.01191274, 0.99917862, -0.03873198],
     [0.00823635, 0.03883147, 0.99921183]], dtype=np.float32
)
DEFAULT_T = np.array([[-0.0643181], [0.05334032], [0.03224605]], dtype=np.float32)

# ==========================================
# 1. Threaded Camera Classes (変更なし)
# ==========================================
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                self.stop()
                break
            with self.lock:
                self.ret, self.frame = ret, frame
            time.sleep(0.001)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()

class ToFStream:
    def __init__(self, fg, mode="xyz", unit_vectors=None, distance_scale=0.001):
        self.fg = fg
        self.mode = mode
        self.unit_vectors = unit_vectors
        self.distance_scale = distance_scale
        self.stopped = False
        self.xyz = None
        self.lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            try:
                frame_grab = self.fg.wait_for_frame().wait_for(1000)
                [ok, frame] = frame_grab
                if not ok: continue
                
                xyz_raw = None
                if self.mode == "raw":
                    raw_dist = frame.get_buffer(buffer_id.RAW_DISTANCE_IMAGE)
                    if raw_dist is not None and self.unit_vectors is not None:
                        dist_m = raw_dist.astype(np.float32) * self.distance_scale
                        xyz_raw = self.unit_vectors * dist_m[..., None]
                elif self.mode == "distance":
                    dist_img = frame.get_buffer(buffer_id.DISTANCE_IMAGE)
                    if dist_img is not None and self.unit_vectors is not None:
                        dist_m = dist_img.astype(np.float32) * self.distance_scale
                        xyz_raw = self.unit_vectors * dist_m[..., None]
                else:
                    xyz_raw = frame.get_buffer(buffer_id.XYZ)

                if xyz_raw is not None:
                    with self.lock:
                        self.xyz = xyz_raw
            except Exception:
                pass

    def read(self):
        with self.lock:
            return (self.xyz.copy() if self.xyz is not None else None), None

    def stop(self):
        self.stopped = True

# ==========================================
# 2. Setup Helper Logic (変更なし)
# ==========================================
def _parse_o3d_config(o3d_obj):
    try:
        cfg_raw = o3d_obj.to_json()
        return json.loads(str(cfg_raw)) if not isinstance(cfg_raw, dict) else cfg_raw
    except: return None

def _apply_app2_exposure(o3d_obj, cfg, exposure_us):
    if exposure_us is None or exposure_us <= 0 or not cfg: return
    try:
        apps = cfg.get("Apps") or cfg.get("apps")
        if not apps or len(apps) <= 2: return
        app2 = apps[2]
        imager = app2.get("Imager") or app2.get("imager")
        if imager:
            imager["ExposureTime"] = int(exposure_us)
            cfg["ActiveApplication"] = 2
            o3d_obj.from_json(cfg)
            print(f"Applied ExposureTime = {exposure_us} µs")
    except Exception as exc:
        print(f"Warning: Could not set exposure: {exc}")

def _extract_unit_vectors(frame, reference_buffer_id):
    try:
        reference = frame.get_buffer(reference_buffer_id)
        unit_raw = frame.get_buffer(buffer_id.UNIT_VECTOR_ALL)
        if reference is None or unit_raw is None: return None
        return unit_raw.reshape(reference.shape + (3,)).astype(np.float32)
    except: return None

def _raw_distance_scale(frame):
    try:
        info = frame.distance_image_info()
        resolution = getattr(info, "distance_resolution", None)
        return float(resolution) * 1e-3 if resolution is not None else 0.001
    except: return 0.001

def setup_tof(tof_ip, app2_exposure_us, use_raw=False):
    try:
        device = Device(tof_ip)
        o3d = O3D(tof_ip)
        cfg = _parse_o3d_config(o3d)
        _apply_app2_exposure(o3d, cfg, app2_exposure_us)
        fg = FrameGrabber(device)
        
        has_raw = hasattr(buffer_id, "RAW_DISTANCE_IMAGE") and hasattr(buffer_id, "UNIT_VECTOR_ALL")
        mode = "xyz"
        unit_vectors = None
        distance_scale = 0.001

        if use_raw and has_raw:
            print("Attempting RAW schema...")
            fg.start([buffer_id.RAW_DISTANCE_IMAGE, buffer_id.UNIT_VECTOR_ALL])
            ok, frame = fg.wait_for_frame().wait_for(FG_WAIT_TIMEOUT_MS)
            if ok:
                unit_vectors = _extract_unit_vectors(frame, buffer_id.RAW_DISTANCE_IMAGE)
                if unit_vectors is not None:
                    distance_scale = _raw_distance_scale(frame)
                    mode = "raw"
                    fg.stop()
                    fg.start([buffer_id.RAW_DISTANCE_IMAGE])

        if mode == "xyz":
            print("Fallback to XYZ schema.")
            fg.start([buffer_id.XYZ, buffer_id.NORM_AMPLITUDE_IMAGE])

        return device, fg, unit_vectors, distance_scale, mode
    except Exception as exc:
        print(f"Failed to setup ToF: {exc}")
        return None, None, None, 0.001, "xyz"

def xyz_to_xyzm(xyz):
    x = xyz[:, :, 0].astype(np.float32)
    y = xyz[:, :, 1].astype(np.float32)
    z = xyz[:, :, 2].astype(np.float32)
    z_finite = z[np.isfinite(z)]
    if z_finite.size > 0 and np.mean(z_finite) > 10.0:
        return x * 0.001, y * 0.001, z * 0.001
    return x, y, z

def load_calibration_simple(intr_path, extr_path=None):
    import scipy.io as sio 
    try:
        data = sio.loadmat(intr_path)
        cam_matrix = data["cameraMatrix"].astype(np.float32)
        dist_coeffs = data["distCoeffs"].flatten().astype(np.float32)
        R, T = DEFAULT_R, DEFAULT_T
        if extr_path and Path(extr_path).exists():
            ext = np.load(extr_path)
            if "R" in ext: R = ext["R"].astype(np.float32)
            if "T" in ext: T = ext["T"].astype(np.float32)
            elif "t" in ext: T = ext["t"].astype(np.float32)
        rvec, _ = cv2.Rodrigues(R)
        return cam_matrix, dist_coeffs, rvec, T
    except Exception as e:
        print(f"Calib load error: {e}")
        return np.eye(3, dtype=np.float32), np.zeros(5, dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

# ==========================================
# 3. Processing & Drawing Logic (ここを変更)
# ==========================================

def project_and_fill(rgb_shape, x_img, y_img, z_img, cam_matrix, dist_coeffs, rvec, tvec):
    h, w = rgb_shape
    points_3d = np.dstack((x_img, y_img, z_img)).reshape(-1, 3)
    valid_mask = (points_3d[:, 2] > 0) & np.isfinite(points_3d[:, 2])
    points_3d = points_3d[valid_mask]
    
    if points_3d.shape[0] == 0:
        return np.zeros((h, w), dtype=np.float32)

    projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, cam_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    
    u = np.rint(projected_points[:, 0]).astype(int)
    v = np.rint(projected_points[:, 1]).astype(int)
    
    valid_indices = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[valid_indices]
    v = v[valid_indices]
    z = points_3d[valid_indices, 2]
    
    sparse_depth = np.zeros((h, w), dtype=np.float32)
    sparse_depth[v, u] = z
    
    # Simple Dilation
    kernel = np.ones((3, 3), np.uint8) 
    filled_depth = cv2.dilate(sparse_depth, kernel, iterations=5)
    
    # 穴埋めロジック（0の部分だけdilate結果で埋める）
    mask = (sparse_depth > 0)
    final_depth = filled_depth.copy()
    final_depth[mask] = sparse_depth[mask] # 元データ優先

    return cv2.medianBlur(final_depth, 3)

def prepare_input(rgb_bgr, depth_mm):
    rgb_norm = rgb_bgr.astype(np.float32) / 255.0
    depth_norm = np.clip(depth_mm.astype(np.float32) / 5000.0, 0.0, 1.0)
    return np.concatenate((rgb_norm, depth_norm[..., None]), axis=-1).astype(np.float32)

def draw_detections(img, results, class_names):
    """
    YOLOの検出結果をOpenCVで描画する関数
    results: model.predict() の戻り値
    """
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 座標取得 (int変換)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # 信頼度とクラスID
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            name = class_names[cls_id] if cls_id in class_names else str(cls_id)
            
            label = f"{name} {conf:.2f}"
            
            # --- 描画設定 ---
            # 緑色の枠 (BGR)
            color = (0, 255, 0)
            text_color = (0, 0, 0) # 黒文字
            
            # バウンディングボックス
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # ラベルの背景帯を描画（文字が見やすくなるように）
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), color, -1)
            
            # テキスト描画
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
            
    return img

def draw_fps(image, fps):
    cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return image

# ==========================================
# 4. Main
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tof-ip", default="192.168.1.69")
    parser.add_argument("--webcam-index", type=int, default=1)
    parser.add_argument("--raw-tof", action="store_true")
    parser.add_argument("--show-depth", action="store_true")
    args = parser.parse_args()

    try:
        cam_mat, dist_coef, rvec, tvec = load_calibration_simple("camera_calibration.mat", "extrinsics.npz")
    except:
        cam_mat = np.eye(3, dtype=np.float32)
        dist_coef = np.zeros(5, dtype=np.float32)
        rvec = np.zeros(3, dtype=np.float32)
        tvec = DEFAULT_T

    print("Connecting to cameras...")
    device, fg, u_vec, d_scale, mode = setup_tof(args.tof_ip, 1000, use_raw=args.raw_tof)
    
    tof = ToFStream(fg, mode, u_vec, d_scale).start()
    rgb = VideoStream(args.webcam_index).start()
    
    time.sleep(1.0) 
    
    model = YOLO(args.model)
    print(f"Model loaded. Using ToF mode: {mode}")

    if args.show_depth:
        cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

    try:
        while True:
            t0 = time.time()
            
            ret, frame_rgb = rgb.read()
            frame_xyz, _ = tof.read()
            
            if not ret or frame_xyz is None:
                time.sleep(0.001)
                continue

            frame_rgb_small = cv2.resize(frame_rgb, (640, 480))
            x, y, z = xyz_to_xyzm(frame_xyz)
            
            # Depth作成
            depth_m = project_and_fill(
                (480, 640), x, y, z, cam_mat, dist_coef, rvec, tvec
            )
            depth_mm = (depth_m * 1000).astype(np.uint16)
            
            # 推論
            input_tensor = prepare_input(frame_rgb_small, depth_mm)
            results = model.predict(input_tensor, imgsz=640, conf=0.25, verbose=False)
            
            # --- 描画処理の変更点 ---
            # plot() は使わず、フレームに直接書き込む
            annotated_frame = frame_rgb_small.copy()
            draw_detections(annotated_frame, results, model.names)
            
            fps = 1.0 / (time.time() - t0)
            draw_fps(annotated_frame, fps)
            
            cv2.imshow("Detection (Custom Draw)", annotated_frame)
            
            if args.show_depth:
                depth_vis = cv2.applyColorMap(
                    cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
                    cv2.COLORMAP_JET
                )
                cv2.imshow("Depth", depth_vis)

            if cv2.waitKey(1) == ord('q'):
                break
                
    finally:
        rgb.stop()
        tof.stop()
        try: fg.stop() 
        except: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""Real-time RGBD confidence comparison with Threaded I/O (Matched to Normalized Training)."""

import argparse
import json
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
from scipy.interpolate import griddata
from ultralytics import YOLO

try:
    from ifm3dpy.device import Device, O3D
    from ifm3dpy.framegrabber import FrameGrabber, buffer_id
except ImportError as exc:
    raise SystemExit(
        "ifm3dpy is required. Install via 'pip install ifm3dpy'."
    ) from exc

FG_WAIT_TIMEOUT_MS = 1500
TOF_RETRIES = 2
APP2_EXPOSURE_US_DEFAULT = 1000
DEFAULT_R = np.array(
    [
        [0.99979081, 0.01975504, -0.0052994],
        [-0.01982122, 0.99972234, -0.0127423],
        [0.00504621, 0.01284468, 0.99990477],
    ],
    dtype=np.float32,
)
DEFAULT_T = np.array([[-0.06676424], [0.04272524], [0.03762958]], dtype=np.float32)

# ==========================================
# 1. Threaded Camera Classes
# ==========================================

class VideoStream:
    """Webカメラを別スレッドで読み込むクラス"""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
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
            time.sleep(0.005)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()

class ToFStream:
    """ToFカメラを別スレッドで読み込むクラス"""
    def __init__(self, fg):
        self.fg = fg
        self.stopped = False
        self.xyz = None
        self.ir = None
        self.lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            try:
                frame_grab = self.fg.wait_for_frame().wait_for(FG_WAIT_TIMEOUT_MS)
                [ok, frame] = frame_grab
                if not ok:
                    continue
                
                xyz_raw = frame.get_buffer(buffer_id.XYZ)
                ir_raw = frame.get_buffer(buffer_id.NORM_AMPLITUDE_IMAGE)
                
                with self.lock:
                    self.xyz = xyz_raw
                    self.ir = ir_raw
            except Exception:
                pass

    def read(self):
        with self.lock:
            if self.xyz is None:
                return None, None
            return self.xyz.copy(), self.ir.copy()

    def stop(self):
        self.stopped = True

# ==========================================
# 2. Helper Functions
# ==========================================

def _parse_o3d_config(o3d_obj):
    try:
        cfg_raw = o3d_obj.to_json()
        if isinstance(cfg_raw, dict):
            return cfg_raw
        if isinstance(cfg_raw, str):
            return json.loads(cfg_raw)
        return json.loads(str(cfg_raw))
    except Exception:
        return None


def _apply_app2_exposure(o3d_obj, cfg, exposure_us):
    if exposure_us is None or exposure_us <= 0 or not cfg:
        return
    apps = cfg.get("Apps") or cfg.get("apps")
    if not apps or len(apps) <= 2:
        print("Warning: Apps[2] not found; skipping exposure update.")
        return
    app2 = apps[2]
    imager = app2.get("Imager") or app2.get("imager")
    if not imager:
        print("Warning: Apps[2] has no Imager section; skipping exposure update.")
        return
    imager["ExposureTime"] = int(exposure_us)
    if "ActiveApplication" in cfg:
        cfg["ActiveApplication"] = 2
    elif "activeApplication" in cfg:
        cfg["activeApplication"] = 2
    else:
        cfg["ActiveApplication"] = 2
    try:
        o3d_obj.from_json(cfg)
        print(f"Applied Apps[2].Imager.ExposureTime = {exposure_us} µs")
    except Exception as exc:
        print(f"Failed to write exposure setting: {exc}")


def setup_tof(tof_ip, app2_exposure_us):
    try:
        device = Device(tof_ip)
        device_type = device.who_am_i()
        if device_type not in (device.device_family.O3D, device.device_family.O3X):
            print(f"Warning: device type {device_type} may behave differently.")
        o3d = O3D(tof_ip)
        cfg = _parse_o3d_config(o3d)
        _apply_app2_exposure(o3d, cfg, app2_exposure_us)
        fg = FrameGrabber(device)
        fg.start([buffer_id.XYZ, buffer_id.NORM_AMPLITUDE_IMAGE])
        print(f"Connected to ToF {tof_ip} and started FrameGrabber (Threaded Mode).")
        return device, fg
    except Exception as exc:
        print(f"Failed to setup ToF: {exc}")
        return None, None


def xyz_to_xyzm(xyz):
    x = xyz[:, :, 0].astype(np.float32)
    y = xyz[:, :, 1].astype(np.float32)
    z = xyz[:, :, 2].astype(np.float32)
    finite = np.isfinite(z)
    z_mean = np.nanmean(z[finite]) if np.any(finite) else 0.0
    divisor = 1000.0 if z_mean > 10.0 else 1.0
    if divisor != 1.0:
        x /= divisor
        y /= divisor
        z /= divisor
    return x, y, z


def load_calibration(intr_path, extr_path=None):
    data = sio.loadmat(intr_path)
    cam_matrix = data["cameraMatrix"]
    dist_coeffs = data["distCoeffs"].flatten()
    R_matrix = DEFAULT_R
    tvec = DEFAULT_T
    if extr_path:
        extr_path = Path(extr_path)
        if extr_path.exists():
            try:
                ext = np.load(extr_path)
                if "R" in ext:
                    R_matrix = ext["R"].astype(np.float32)
                if "T" in ext:
                    tvec = ext["T"].astype(np.float32)
                elif "t" in ext:
                    tvec = ext["t"].astype(np.float32)
            except Exception as exc:
                print(f"Failed to load extrinsics from {extr_path}: {exc}")
    rvec, _ = cv2.Rodrigues(R_matrix)
    return cam_matrix, dist_coeffs, rvec, tvec


def build_depth_map(rgb_shape, x_img, y_img, z_img, cam_matrix, dist_coeffs, rvec, tvec):
    h, w = rgb_shape
    aligned = np.zeros((h, w), dtype=np.float32)
    validity = np.zeros((h, w), dtype=np.uint8)
    depth_values = z_img.flatten()
    valid = np.isfinite(depth_values) & (depth_values > 0)
    if not np.any(valid):
        return aligned, validity
    points_3d = np.stack([x_img.flatten(), y_img.flatten(), z_img.flatten()], axis=-1)[valid]
    depth_values = depth_values[valid]
    projected, _ = cv2.projectPoints(points_3d, rvec, tvec, cam_matrix, dist_coeffs)
    uv = projected.reshape(-1, 2)
    for (u, v), depth in zip(uv, depth_values):
        col = int(round(u))
        row = int(round(v))
        if 0 <= col < w and 0 <= row < h:
            current = aligned[row, col]
            if current == 0 or depth < current:
                aligned[row, col] = depth
            validity[row, col] = 255
    if np.count_nonzero(aligned) == 0:
        return aligned, validity
    v_coords, u_coords = np.where(aligned > 0)
    if v_coords.size > 10:
        known_points = np.stack([v_coords, u_coords], axis=-1)
        known_values = aligned[v_coords, u_coords]
        grid_v, grid_u = np.mgrid[0:h, 0:w]
        dense = griddata(known_points, known_values, (grid_v, grid_u), method="nearest")
        dense[np.isnan(dense)] = 0.0
        aligned = dense.astype(np.float32)
    aligned = np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)
    aligned[aligned < 0] = 0.0
    return aligned, validity


# ==========================================
# 修正箇所: 正規化ロジックの適用
# ==========================================

def depth_to_uint16(depth_m, max_depth_m):
    # 安全のため 5.0m 以上はクリップ
    depth_clipped = np.clip(depth_m, 0.0, max_depth_m)
    return np.round(depth_clipped * 1000.0).astype(np.uint16)


def prepare_input(rgb_bgr, depth_mm):
    """
    学習コードと一致させた正規化関数
    RGB: 0-255 -> 0.0-1.0
    Depth: 0-5000 -> 0.0-1.0
    """
    # 1. RGB (uint8) -> Float (0.0-1.0)
    rgb_norm = rgb_bgr.astype(np.float32) / 255.0

    # 2. Depth (mm) -> Float (0.0-1.0)
    depth_float = depth_mm.astype(np.float32)
    depth_norm = depth_float / 5000.0
    depth_norm = np.clip(depth_norm, 0.0, 1.0)

    # 3. 結合 (H, W, 4)
    input_tensor = np.concatenate((rgb_norm, depth_norm[..., None]), axis=-1)
    
    return input_tensor.astype(np.float32)


def visualize_depth(depth_mm):
    if np.max(depth_mm) == 0:
        depth_norm = np.zeros_like(depth_mm, dtype=np.uint8)
    else:
        depth_norm = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)


def draw_detections(image, tof_boxes, coco_boxes, fps, cup_conf_4ch, cup_conf_rgb):
    for box in tof_boxes:
        coords = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf)
        cls = int(box.cls) if box.cls is not None else -1
        x1, y1, x2, y2 = coords
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, f"{cls}:{conf:.2f}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for box in coco_boxes:
        coords = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf)
        cls = int(box.cls) if box.cls is not None else -1
        x1, y1, x2, y2 = coords
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"COCO:{cls}:{conf:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(image, f"4ch cup conf: {cup_conf_4ch:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(image, f"COCO cup conf: {cup_conf_rgb:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    return image


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time RGBD capture + detection")
    parser.add_argument("--model", required=True, help="Path to YOLO model (.pt)")
    parser.add_argument("--tof-ip", default="192.168.1.69", help="ToF camera IP")
    parser.add_argument("--webcam-index", type=int, default=1, help="OpenCV webcam index")
    parser.add_argument("--intrinsics", default="camera_calibration.mat", help="Webcam intrinsics MAT file")
    parser.add_argument("--extrinsics", help="Optional extrinsics NPZ file containing R/T")
    parser.add_argument("--app2-exposure-us", type=int, default=APP2_EXPOSURE_US_DEFAULT, help="Exposure for App 2")
    # ★変更: デフォルトを 5.0m に修正
    parser.add_argument("--max-depth", type=float, default=5.0, help="Depth clipping distance in meters")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--device", default=None, help="YOLO device (auto by default)")
    parser.add_argument("--show-depth", action="store_true", help="Display a depth colormap window")
    parser.add_argument("--zero-rgb", action="store_true", help="Zero-out RGB channels before inference (depth-only)")
    parser.add_argument("--coco-model", help="Optional RGB-only YOLO model (e.g., COCO pretrain)")
    parser.add_argument("--coco-cup-class", type=int, default=41, help="Class index to treat as cup for COCO model")
    parser.add_argument("--coco-conf", type=float, default=0.25, help="Confidence threshold for COCO model")
    return parser.parse_args()


# ==========================================
# 3. Main Loop
# ==========================================

def main():
    args = parse_args()
    try:
        cam_matrix, dist_coeffs, rvec, tvec = load_calibration(args.intrinsics, args.extrinsics)
    except Exception as exc:
        raise SystemExit(f"Failed to load calibration: {exc}")

    # --- Setup ToF ---
    device, fg = setup_tof(args.tof_ip, args.app2_exposure_us)
    if fg is None:
        raise SystemExit("Could not initialize ToF camera.")
    
    tof_stream = ToFStream(fg).start()

    # --- Setup Webcam ---
    temp_cap = cv2.VideoCapture(args.webcam_index)
    if not temp_cap.isOpened():
        raise SystemExit(f"Cannot open webcam index {args.webcam_index}")
    temp_cap.release()

    rgb_stream = VideoStream(args.webcam_index).start()
    
    print("Waiting for cameras to warm up...")
    time.sleep(1.0) 

    # --- Load Models ---
    model = YOLO(args.model)
    coco_model = YOLO(args.coco_model) if args.coco_model else None
    
    if coco_model:
        print("Loaded 4ch + COCO models (Threaded I/O). Press 'q' to quit.")
    else:
        print("4ch model loaded (Threaded I/O). Press 'q' to quit.")

    if args.show_depth:
        cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)

    try:
        while True:
            iter_start = time.time()
            
            # --- 1. Non-blocking Read ---
            ret, rgb_frame = rgb_stream.read()
            xyz_raw, _ = tof_stream.read()
            
            if not ret or xyz_raw is None:
                time.sleep(0.001)
                continue

            # --- 2. Processing ---
            x_img, y_img, z_img = xyz_to_xyzm(xyz_raw)
            rgb_resized = cv2.resize(rgb_frame, (640, 480))
            
            depth_m, validity = build_depth_map(rgb_resized.shape[:2], x_img, y_img, z_img, cam_matrix, dist_coeffs, rvec, tvec)
            
            if np.count_nonzero(depth_m) == 0:
                depth_m = cv2.resize(z_img.astype(np.float32), (rgb_resized.shape[1], rgb_resized.shape[0]), interpolation=cv2.INTER_NEAREST)
                validity = (depth_m > 0).astype(np.uint8) * 255
                
            depth_mm = depth_to_uint16(depth_m, args.max_depth)

            # ★修正: 戻り値は1つだけ
            input_tensor = prepare_input(rgb_resized, depth_mm)
            
            if args.zero_rgb:
                input_tensor[:, :, :3] = 0.0
            
            # --- 3. Inference (4ch) ---
            results = model.predict(
                source=input_tensor,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )
            boxes = results[0].boxes if results else []
            cup_conf_4ch = max((float(box.conf) for box in boxes), default=0.0)

            # --- 4. Inference (COCO) - Optional ---
            coco_boxes = []
            cup_conf_rgb = 0.0
            if coco_model:
                coco_res = coco_model.predict(
                    source=rgb_resized,
                    imgsz=args.imgsz,
                    conf=args.coco_conf,
                    device=args.device,
                    verbose=False,
                )
                coco_boxes = coco_res[0].boxes if coco_res else []
                cup_conf_rgb = max(
                    (float(box.conf) for box in coco_boxes if int(box.cls) == args.coco_cup_class),
                    default=0.0,
                )
                
            # --- 5. Draw & Display ---
            fps = 1.0 / max(1e-6, time.time() - iter_start)
            vis_frame = draw_detections(rgb_resized.copy(), boxes, coco_boxes, fps, cup_conf_4ch, cup_conf_rgb)
            mask_ratio = validity.mean() / 255.0 if validity.size else 0.0
            cv2.putText(vis_frame, f"Mask cov: {mask_ratio:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Detections", vis_frame)
            if args.show_depth:
                depth_vis = visualize_depth(depth_mm)
                cv2.imshow("Depth", depth_vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        rgb_stream.stop()
        tof_stream.stop()
        if fg:
            try:
                fg.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
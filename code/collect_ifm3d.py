# -*- coding: utf-8 -*-
"""
collect_dataset_manual.py

Manual dataset capture script using ifm3dpy for ToF and OpenCV for a webcam.
Saves synchronized RGB (.png) and ToF (.npz) pairs whenever you press SPACE.

Usage:
    - Configure SAVE_DIR, WEBCAM_INDEX, and TOF_IP below.
    - Run: python collect_dataset_manual.py
    - Press [space] to capture, [q] to quit.
"""

import os
import time
import json
import cv2
import numpy as np
import traceback
import argparse

# Try to import ifm3dpy


try:
    from ifm3dpy.device import Device, O3D
    from ifm3dpy.framegrabber import FrameGrabber, buffer_id
except Exception as e:
    print("ifm3dpy is required for this script. Install it or run the PCIC socket version.")
    raise

# ----------------- User configuration -----------------
SAVE_DIR = "raw_train_dataset_1-15" # 保存先フォルダ名
WEBCAM_INDEX = 1
TOF_IP = "192.168.1.69"

# ★★★ 手動撮影の設定 ★★★
MAX_MANUAL_IMAGES = 1000   # 最大撮影枚数（この枚数に達すると終了）

# Amplitude (IR) threshold after normalization (0..255); set None to disable
AMPLITUDE_MIN = None

# Debug options
DEBUG_RAW = True

# Framegrabber wait timeout (ms)
FG_WAIT_TIMEOUT_MS = 1500

# Retries when ToF frame retrieval fails
TOF_RETRIES = 2

# Default App '2' exposure (µs).
APP2_EXPOSURE_US_DEFAULT = 1000

# Capture button UI (top-left)
BTN_POS = (10, 10)
BTN_SIZE = (180, 40)

# ------------------------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)
rgb_dir = os.path.join(SAVE_DIR, "rgb")
tof_dir = os.path.join(SAVE_DIR, "tof")
os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(tof_dir, exist_ok=True)


def _parse_o3d_config(o3d_obj):
    try:
        cfg_raw = o3d_obj.to_json()
        if isinstance(cfg_raw, dict):
            return cfg_raw
        if isinstance(cfg_raw, str):
            return json.loads(cfg_raw)
        return json.loads(str(cfg_raw))
    except Exception as exc:
        print(f"Failed to read device config via O3D: {exc}")
        return None


def _log_device_apps(cfg):
    if not cfg:
        print("No device config available for debugging app list.")
        return
    apps = cfg.get("Apps") or cfg.get("apps") or []
    active_app = cfg.get("ActiveApplication")
    if active_app is None:
        active_app = cfg.get("activeApplication")
    print(f"IFM configuration: {len(apps)} apps, active application = {active_app}")
    for idx, app in enumerate(apps):
        name = app.get("Name") or app.get("name") or "<unnamed>"
        app_type = app.get("Type") or app.get("type")
        imager = app.get("Imager") or app.get("imager") or {}
        exposure = imager.get("ExposureTime") or imager.get("exposure_time")
        print(f"  App[{idx}]: name={name}, type={app_type}, exposure={exposure}")


def _apply_app2_exposure(o3d_obj, cfg, exposure_us):
    if exposure_us is None:
        return False
    if not cfg:
        print("Cannot set exposure; device config was unavailable.")
        return False
    apps = cfg.get("Apps") or cfg.get("apps")
    if not apps or len(apps) <= 2:
        print("Warning: Apps[2] not found in device config; cannot set exposure.")
        return False
    app2 = apps[2]
    imager = app2.get("Imager") or app2.get("imager")
    if imager is None:
        print("Warning: Apps[2] does not contain 'Imager'; cannot adjust exposure.")
        return False
    imager["ExposureTime"] = int(exposure_us)
    if "ActiveApplication" in cfg:
        cfg["ActiveApplication"] = 2
    elif "activeApplication" in cfg:
        cfg["activeApplication"] = 2
    else:
        cfg["ActiveApplication"] = 2
    try:
        o3d_obj.from_json(cfg)
        print(f"Set Apps[2].Imager.ExposureTime = {exposure_us} µs and activated App 2.")
        return True
    except Exception as exc:
        print(f"Failed to write modified config to device: {exc}")
        traceback.print_exc()
        return False

# Helper: setup ToF FrameGrabber and device
def setup_tof( tof_ip=TOF_IP, app2_exposure_us=APP2_EXPOSURE_US_DEFAULT ):
    try:
        device = Device(tof_ip)
        device_type = device.who_am_i()
        if device_type not in (device.device_family.O3D, device.device_family.O3X):
            print(f"Warning: device type seems to be {device_type}. Proceeding but behavior may differ.")

        o3d = O3D(tof_ip)
        config = _parse_o3d_config(o3d)
        _log_device_apps(config)
        if app2_exposure_us is not None:
            if app2_exposure_us > 0:
                _apply_app2_exposure(o3d, config, app2_exposure_us)
            else:
                print("Skipping App 2 exposure because non-positive value was provided.")

        fg = FrameGrabber(device)
        fg.start([buffer_id.XYZ, buffer_id.NORM_AMPLITUDE_IMAGE])
        print(f"Connected to ToF {tof_ip} and started FrameGrabber.")
        return device, fg
    except Exception:
        traceback.print_exc()
        return None, None

# Helper: get tof data with retries
def get_tof_frame(fg, timeout_ms=FG_WAIT_TIMEOUT_MS, retries=TOF_RETRIES):
    for attempt in range(retries+1):
        try:
            [ok, frame] = fg.wait_for_frame().wait_for(timeout_ms)
            if not ok:
                print(f"ToF wait timeout (attempt {attempt+1}/{retries+1})")
                continue
            xyz = frame.get_buffer(buffer_id.XYZ)  # H x W x 3
            ir = frame.get_buffer(buffer_id.NORM_AMPLITUDE_IMAGE)  # H x W
            if xyz is None or ir is None:
                print("ToF returned empty buffers")
                continue
            return xyz.copy(), ir.copy()
        except Exception as e:
            print(f"Exception while getting ToF frame (attempt {attempt+1}): {e}")
            traceback.print_exc()
            time.sleep(0.05)
    return None, None

# Convert XYZ to separate x,y,z arrays (in meters)
def xyz_to_xyzm(xyz):
    x = xyz[:, :, 0].astype(np.float32)
    y = xyz[:, :, 1].astype(np.float32)
    z = xyz[:, :, 2].astype(np.float32)
    
    z_mean = np.nanmean(z[np.isfinite(z)]) if np.any(np.isfinite(z)) else 0.0
    if z_mean > 10.0:
        divisor = 1000.0 # likely mm
        x = x / divisor
        y = y / divisor
        z = z / divisor
    else:
        divisor = 1.0
    return x, y, z, divisor

# Normalize amplitude image to 0..255 uint8
def norm_amp(ir):
    ir_f = ir.astype(np.float32)
    if np.nanmax(ir_f) == np.nanmin(ir_f):
        return np.zeros_like(ir_f, dtype=np.uint8)
    irn = cv2.normalize(ir_f, None, 0, 255, cv2.NORM_MINMAX)
    return irn.astype(np.uint8)


def _mouse_capture(event, x, y, _flags, param):
    capture_request, btn_rect = param
    if event == cv2.EVENT_LBUTTONDOWN:
        x0, y0, w, h = btn_rect
        if x0 <= x <= x0 + w and y0 <= y <= y0 + h:
            capture_request[0] = True

# Main
def main():
    parser = argparse.ArgumentParser(description='collect_dataset_manual.py')
    parser.add_argument('--upsample', choices=['none', 'guided'], default='none', help='Upsample ToF depth/amp to RGB resolution')
    parser.add_argument('--amp-threshold', type=float, default=None, help='Normalized amplitude threshold (0..1)')
    parser.add_argument('--app2-exposure-us', type=int, default=APP2_EXPOSURE_US_DEFAULT, help='Target exposure for Apps[2]')
    args = parser.parse_args()

    do_upsample = (args.upsample == 'guided')
    amp_threshold = args.amp_threshold
    app2_exposure_us = args.app2_exposure_us

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Cannot open webcam index {WEBCAM_INDEX}")
        return

    device, fg = setup_tof(TOF_IP, app2_exposure_us)
    if fg is None:
        print("Failed to setup ToF. Exiting.")
        cap.release()
        return

    img_count = len(os.listdir(rgb_dir))
    print(f"Starting MANUAL capture. Existing count: {img_count}")
    print(f"Max Limit: {MAX_MANUAL_IMAGES}")
    print("Press [space] or click the button, [q] to quit.")

    window_name = 'Auto Capture (q to quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    capture_request = [False]
    btn_rect = (BTN_POS[0], BTN_POS[1], BTN_SIZE[0], BTN_SIZE[1])
    cv2.setMouseCallback(window_name, _mouse_capture, (capture_request, btn_rect))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read webcam frame")
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        should_capture = False
        if key == ord(' ') or capture_request[0]:
            capture_request[0] = False
            if img_count < MAX_MANUAL_IMAGES:
                should_capture = True
            else:
                print("Max image limit reached.")

        preview = frame.copy()
        status_color = (0, 255, 255) if should_capture else (255, 255, 255)
        cv2.putText(preview, "SPACE or BUTTON to CAPTURE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(preview, f"Count: {img_count}/{MAX_MANUAL_IMAGES}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Draw capture button
        x0, y0, w, h = btn_rect
        cv2.rectangle(preview, (x0, y0), (x0 + w, y0 + h), (0, 200, 255), 2)
        cv2.putText(preview, "CLICK to CAPTURE", (x0 + 8, y0 + int(h * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.imshow(window_name, preview)

        if should_capture:
            ret_web, frame_to_save = cap.read()
            if not ret_web:
                print("Failed to re-read webcam frame at capture time")
                continue

            xyz_raw, ir_raw = get_tof_frame(fg)
            if xyz_raw is None or ir_raw is None:
                print("ToF capture failed. Skipping this shot.")
                continue

            try:
                x_img, y_img, z_img, divisor = xyz_to_xyzm(xyz_raw)
                ir_u8 = norm_amp(ir_raw)
            except Exception as e:
                print(f"Error processing ToF buffers: {e}")
                continue

            depth_up = None
            ir_up = None
            mask_up = None
            if do_upsample:
                try:
                    h_rgb, w_rgb = frame_to_save.shape[:2]
                    z_for = z_img.astype(np.float32)
                    finite_mask_src = np.isfinite(z_for) & (z_for > 0)
                    z_tmp = np.where(finite_mask_src, z_for, 0.0).astype(np.float32)
                    z_resized = cv2.resize(z_tmp, (w_rgb, h_rgb), interpolation=cv2.INTER_LINEAR)
                    mask_resized = cv2.resize(finite_mask_src.astype(np.uint8), (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)
                    ir_resized = cv2.resize(ir_u8, (w_rgb, h_rgb), interpolation=cv2.INTER_LINEAR)

                    guide = cv2.cvtColor(frame_to_save, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                    try:
                        gf = cv2.ximgproc.guidedFilter
                        depth_guided = gf(guide.astype(np.float32), z_resized.astype(np.float32), 8, 1e-6)
                        amp_guided = gf(guide.astype(np.float32), ir_resized.astype(np.float32), 8, 1e-2)
                    except:
                        depth_guided = cv2.medianBlur((z_resized).astype(np.float32), 3)
                        amp_guided = cv2.medianBlur((ir_resized).astype(np.float32), 3)

                    depth_guided[mask_resized == 0] = np.nan
                    amp_guided_u8 = np.clip(np.round(cv2.normalize(amp_guided, None, 0, 255, cv2.NORM_MINMAX)), 0, 255).astype(np.uint8)

                    depth_up = depth_guided
                    ir_up = amp_guided_u8
                    mask_up = (mask_resized.astype(np.uint8) > 0)
                except Exception as e:
                    print(f"Upsample failed: {e}")

            if mask_up is None:
                mask = np.isfinite(z_img) & (z_img > 0)
            else:
                mask = mask_up
            if amp_threshold is not None:
                thr_u8 = int(np.clip(amp_threshold, 0.0, 1.0) * 255)
                if do_upsample and ir_up is not None:
                    mask = mask & (ir_up > thr_u8)
                else:
                    mask = mask & (ir_u8 > thr_u8)

            finite_mask = np.isfinite(z_img) & (z_img > 0)
            depth_mean = float(np.nanmean(z_img[finite_mask])) if np.any(finite_mask) else float('nan')
            depth_median = float(np.nanmedian(z_img[finite_mask])) if np.any(finite_mask) else float('nan')
            amp_mean = float(np.nanmean(ir_u8))

            meta = {
                'ts_web': time.time(),
                'ts_saved': time.time(),
                'depth_mean': depth_mean,
                'depth_median': depth_median,
                'amp_mean': amp_mean,
                'tof_ip': TOF_IP
            }

            img_count += 1
            base = f"{img_count:06d}"
            rgb_path = os.path.join(rgb_dir, base + ".png")
            tof_path = os.path.join(tof_dir, base + ".npz")

            try:
                if do_upsample and depth_up is not None and ir_up is not None:
                    np.savez_compressed(tof_path, x=x_img, y=y_img, z=z_img, ir=ir_u8,
                                        depth_up=depth_up, amp_up=ir_up, mask=mask.astype(np.uint8), meta=json.dumps(meta))
                else:
                    np.savez_compressed(tof_path, x=x_img, y=y_img, z=z_img, ir=ir_u8, mask=mask.astype(np.uint8), meta=json.dumps(meta))
                
                cv2.imwrite(rgb_path, frame_to_save)

                try:
                    depth_mm = np.zeros_like(z_img, dtype=np.uint16)
                    finite = np.isfinite(z_img) & (z_img > 0)
                    if np.any(finite):
                        depth_vals_mm = np.round(z_img[finite].astype(np.float64) * 1000.0).astype(np.int64)
                        depth_vals_mm = np.clip(depth_vals_mm, 0, 65535)
                        depth_mm[finite] = depth_vals_mm.astype(np.uint16)
                    depth_path = os.path.join(tof_dir, base + "_depth.png")
                    cv2.imwrite(depth_path, depth_mm)
                except Exception as e:
                    print(f"Warning depth png: {e}")

                print(f"Manual save [{img_count}]: {base}")
                
            except Exception as e:
                print(f"Failed to save data: {e}")
            if img_count >= MAX_MANUAL_IMAGES:
                print("Max image limit reached.")

    # Cleanup
    cap.release()
    if fg:
        try:
            fg.stop()
        except:
            pass
    cv2.destroyAllWindows()
    print("Finished.")

if __name__ == '__main__':
    main()
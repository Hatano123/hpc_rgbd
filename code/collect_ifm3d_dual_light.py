# -*- coding: utf-8 -*-
"""
collect_ifm3d_dual_light.py

Capture bright/dark RGB + ToF pairs without moving the rig.
Workflow per pair:
 1. Keep lights ON (bright) and press [Space] to grab the first frame.
 2. Without touching the rig, turn lights OFF (dark) and press [Space] again.

The script saves two synchronized captures per pair with the same numeric ID and
phase suffix (_bright / _dark). Later, you can annotate the bright image and
copy its label file to the matching dark file name.
"""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

try:
    from ifm3dpy.device import Device, O3D
    from ifm3dpy.framegrabber import FrameGrabber, buffer_id
except Exception as exc:  # pragma: no cover - hardware dependency
    print("ifm3dpy is required for this script. Install it before running.")
    raise

# ---------------------------------------------------------------------------
# Defaults (override via CLI if needed)
# ---------------------------------------------------------------------------
DEFAULT_SAVE_DIR = "LED_light_dataset1-8"
DEFAULT_WEBCAM_INDEX = 1
DEFAULT_TOF_IP = "192.168.1.69"
APP2_EXPOSURE_US_DEFAULT = 1000
FG_WAIT_TIMEOUT_MS = 1500
TOF_RETRIES = 2
DEBUG_RAW = True
PAIR_SEQUENCE = [
    ("bright", "Lights ON (明るい) → press [Space]"),
    ("dark", "Lights OFF (暗い) → press [Space]"), 
]


# ---------------------------------------------------------------------------
# Device helpers (derived from collect_ifm3d.py)
# ---------------------------------------------------------------------------
def _parse_o3d_config(o3d_obj):
    try:
        cfg_raw = o3d_obj.to_json()
        if isinstance(cfg_raw, dict):
            return cfg_raw
        if isinstance(cfg_raw, str):
            return json.loads(cfg_raw)
        return json.loads(str(cfg_raw))
    except Exception as exc:  # pragma: no cover
        print(f"Failed to read device config via O3D: {exc}")
        return None


def _log_device_apps(cfg):
    if not cfg:
        print("No device config available for app listing.")
        return
    apps = cfg.get("Apps") or cfg.get("apps") or []
    active_app = cfg.get("ActiveApplication") or cfg.get("activeApplication")
    print(f"IFM configuration: {len(apps)} apps, active application = {active_app}")
    for idx, app in enumerate(apps):
        name = app.get("Name") or app.get("name") or "<unnamed>"
        app_type = app.get("Type") or app.get("type")
        imager = app.get("Imager") or app.get("imager") or {}
        exposure = imager.get("ExposureTime") or imager.get("exposure_time")
        print(f"  App[{idx}]: name={name}, type={app_type}, exposure={exposure}")


def _apply_app2_exposure(o3d_obj, cfg, exposure_us):
    if exposure_us is None or exposure_us <= 0:
        return False
    if not cfg:
        print("Cannot set exposure; device config unavailable.")
        return False
    apps = cfg.get("Apps") or cfg.get("apps")
    if not apps or len(apps) <= 2:
        print("Warning: Apps[2] missing; cannot set exposure.")
        return False
    app2 = apps[2]

    imager = app2.get("Imager") or app2.get("imager")
    if not imager:
        print("Warning: Apps[2] lacks 'Imager'; cannot adjust exposure.")
        return False
    imager["ExposureTime"] = int(exposure_us)
    cfg["ActiveApplication"] = 2
    try:
        o3d_obj.from_json(cfg)
        print(f"Set Apps[2].Imager.ExposureTime = {exposure_us} µs and activated App 2.")
        return True
    except Exception as exc:  # pragma: no cover
        print(f"Failed to push exposure config: {exc}")
        traceback.print_exc()
        return False


def setup_tof(tof_ip, app2_exposure_us):
    try:
        device = Device(tof_ip)
        o3d = O3D(tof_ip)
        config = _parse_o3d_config(o3d)
        _log_device_apps(config)
        _apply_app2_exposure(o3d, config, app2_exposure_us)

        fg = FrameGrabber(device)
        fg.start([buffer_id.XYZ, buffer_id.NORM_AMPLITUDE_IMAGE])
        print(f"Connected to ToF {tof_ip} and started FrameGrabber.")
        return device, fg
    except Exception:  # pragma: no cover
        traceback.print_exc()
        return None, None


def get_tof_frame(fg, timeout_ms=FG_WAIT_TIMEOUT_MS, retries=TOF_RETRIES):
    for attempt in range(retries + 1):
        try:
            ok, frame = fg.wait_for_frame().wait_for(timeout_ms)
            if not ok:
                print(f"ToF wait timeout ({attempt + 1}/{retries + 1})")
                continue
            xyz = frame.get_buffer(buffer_id.XYZ)
            ir = frame.get_buffer(buffer_id.NORM_AMPLITUDE_IMAGE)
            if xyz is None or ir is None:
                print("ToF returned empty buffers")
                continue
            return xyz.copy(), ir.copy()
        except Exception as exc:
            print(f"Error retrieving ToF frame ({attempt + 1}): {exc}")
            traceback.print_exc()
            time.sleep(0.05)
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
    return x, y, z, divisor


def norm_amp(ir):
    ir_f = ir.astype(np.float32)
    if np.nanmax(ir_f) == np.nanmin(ir_f):
        return np.zeros_like(ir_f, dtype=np.uint8)
    irn = cv2.normalize(ir_f, None, 0, 255, cv2.NORM_MINMAX)
    return irn.astype(np.uint8)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def ensure_dirs(save_dir: Path):
    (save_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (save_dir / "tof").mkdir(parents=True, exist_ok=True)


def next_pair_index(rgb_dir: Path) -> int:
    existing = []
    for file in rgb_dir.glob("*_bright.png"):
        try:
            existing.append(int(file.stem.split("_")[0]))
        except Exception:
            continue
    return max(existing, default=0) + 1


def save_capture(base_name: str, phase: str, rgb_img, x_img, y_img, z_img,
                 ir_u8, mask, save_root: Path, meta: dict):
    rgb_path = save_root / "rgb" / f"{base_name}_{phase}.png"
    tof_path = save_root / "tof" / f"{base_name}_{phase}.npz"
    depth_path = save_root / "tof" / f"{base_name}_{phase}_depth.png"

    np.savez_compressed(
        tof_path,
        x=x_img,
        y=y_img,
        z=z_img,
        ir=ir_u8,
        mask=mask.astype(np.uint8),
        meta=json.dumps(meta)
    )
    cv2.imwrite(str(rgb_path), rgb_img)

    depth_mm = np.zeros_like(z_img, dtype=np.uint16)
    finite = np.isfinite(z_img) & (z_img > 0)
    if np.any(finite):
        depth_vals_mm = np.clip(np.round(z_img[finite] * 1000.0), 0, 65535)
        depth_mm[finite] = depth_vals_mm.astype(np.uint16)
    cv2.imwrite(str(depth_path), depth_mm)

    return {
        "rgb": str(rgb_path),
        "tof": str(tof_path),
        "depth_png": str(depth_path),
        "meta": meta,
    }


def append_pair_log(log_path: Path, entry: dict):
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Capture workflow
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Capture bright/dark ToF pairs")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--webcam-index", type=int, default=DEFAULT_WEBCAM_INDEX)
    parser.add_argument("--tof-ip", default=DEFAULT_TOF_IP)
    parser.add_argument("--app2-exposure-us", type=int, default=APP2_EXPOSURE_US_DEFAULT)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    ensure_dirs(save_dir)
    rgb_dir = save_dir / "rgb"
    tof_dir = save_dir / "tof"
    log_path = save_dir / "pair_log.jsonl"

    cap = cv2.VideoCapture(args.webcam_index)
    if not cap.isOpened():
        print(f"Cannot open webcam index {args.webcam_index}")
        return

    device, fg = setup_tof(args.tof_ip, args.app2_exposure_us)
    if fg is None:
        print("Failed to setup ToF. Exiting.")
        cap.release()
        return

    pair_idx = next_pair_index(rgb_dir)
    phase_idx = 0
    current_pair = {}
    window = "Dual Light Capture"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    print("\nInstructions:")
    print("  - Keep rig absolutely fixed.")
    print("  - Follow on-screen prompt for each phase.")
    print("  - [Space] captures, [q] quits.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read webcam frame.")
            break

        phase, prompt = PAIR_SEQUENCE[phase_idx]
        preview = frame.copy()
        cv2.putText(preview, f"Pair {pair_idx:04d} / {phase}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(preview, prompt, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(window, preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key != ord(' '):
            continue

        timestamp_web = time.time()
        ret_capture, rgb_frame = cap.read()
        if not ret_capture:
            print("Failed to grab RGB frame at capture time.")
            continue
        xyz_raw, ir_raw = get_tof_frame(fg)
        if xyz_raw is None:
            print("ToF frame unavailable; retry phase.")
            continue

        if DEBUG_RAW:
            try:
                z_raw = xyz_raw[:, :, 2].astype(np.float32)
                finite_raw = np.isfinite(z_raw)
                raw_mean = float(np.nanmean(z_raw[finite_raw])) if np.any(finite_raw) else float('nan')
                raw_min = float(np.nanmin(z_raw[finite_raw])) if np.any(finite_raw) else float('nan')
                raw_max = float(np.nanmax(z_raw[finite_raw])) if np.any(finite_raw) else float('nan')
                print(f"Raw XYZ: mean={raw_mean:.3f}, min={raw_min:.3f}, max={raw_max:.3f}")
            except Exception:
                pass

        try:
            x_img, y_img, z_img, divisor = xyz_to_xyzm(xyz_raw)
            ir_u8 = norm_amp(ir_raw)
        except Exception as exc:
            print(f"Processing error: {exc}")
            traceback.print_exc()
            continue

        mask = np.isfinite(z_img) & (z_img > 0)
        depth_mean = float(np.nanmean(z_img[mask])) if np.any(mask) else float('nan')
        depth_median = float(np.nanmedian(z_img[mask])) if np.any(mask) else float('nan')
        amp_mean = float(np.nanmean(ir_u8))

        base_name = f"{pair_idx:06d}"
        meta = {
            "pair_id": pair_idx,
            "phase": phase,
            "ts_rgb": timestamp_web,
            "depth_mean": depth_mean,
            "depth_median": depth_median,
            "amp_mean": amp_mean,
            "tof_ip": args.tof_ip,
            "unit_divisor": divisor,
        }

        result = save_capture(
            base_name,
            phase,
            rgb_frame,
            x_img,
            y_img,
            z_img,
            ir_u8,
            mask,
            save_dir,
            meta,
        )
        current_pair[phase] = result
        print(f"Saved {phase} capture for pair {pair_idx:04d}: {result['rgb']}")

        phase_idx = (phase_idx + 1) % len(PAIR_SEQUENCE)
        if phase_idx == 0:
            append_pair_log(log_path, {
                "pair_id": pair_idx,
                "bright": current_pair.get("bright"),
                "dark": current_pair.get("dark"),
                "label_hint": f"Copy {base_name}_bright.txt -> {base_name}_dark.txt after annotation",
            })
            print(f"Pair {pair_idx:04d} complete! Annotate the bright image then copy its label to the dark twin.")
            current_pair = {}
            pair_idx += 1

    cap.release()
    if fg:
        try:
            fg.stop()
        except Exception:
            pass
    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()

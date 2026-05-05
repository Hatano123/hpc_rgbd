"""Run YOLO detections on RGB-only images (3ch) in a folder.

Supports TIFF (with potential 4ch input; depth channel is dropped) and common image formats.
Provides headless (--no-gui) and interactive viewing modes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
from ultralytics import YOLO


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
WINDOW_NAME = "Folder Detection Viewer (3ch)"
MAX_RGB_VAL = 65535.0  # TIFF 16-bit対応


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO detections on RGB-only images in a folder.")
    parser.add_argument("--image-dir", type=Path, default=Path("input_images"), help="Directory that holds images.")
    parser.add_argument("--model", type=Path, default=Path("yolov8n.pt"), help="Path to a YOLO model weight file.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution passed to YOLO.")
    parser.add_argument("--conf", type=float, default=0.25, help="Score threshold for predictions.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. '0' or 'cpu'.")
    parser.add_argument("--save-dir", type=Path, help="If set, annotated images are written here.")
    parser.add_argument("--no-gui", action="store_true", help="Run headless: no viewer, just predict and optionally save.")
    return parser.parse_args()


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Image directory not found: {folder}")
    files: List[Path] = []
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            files.append(path)
    files.sort()
    return files


def draw_detections(frame, boxes) -> cv2.Mat:
    vis = frame.copy()
    for box in boxes:
        coords = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords.tolist()
        conf = float(box.conf.item()) if box.conf is not None else 0.0
        cls_id = int(box.cls.item()) if box.cls is not None else -1
        label = f"cls{cls_id} {conf:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return vis


def annotate_image(vis, path: Path, index: int, total: int) -> None:
    text = f"{path.name}  [{index + 1}/{total}]"
    cv2.putText(vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


def save_result(vis, out_dir: Path, path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / path.name
    cv2.imwrite(str(out_path), vis)


def load_rgb_tensor(path: Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if raw.ndim == 2:
        raw = np.expand_dims(raw, axis=2)
    # If 4ch (e.g., RGBD TIFF), drop depth
    if raw.shape[2] > 3:
        raw = raw[:, :, :3]
    # Normalize to 0-1 with dtype-aware scaling
    if raw.dtype == np.uint8:
        raw = raw.astype(np.float32) / 255.0
    elif raw.dtype == np.uint16:
        raw = raw.astype(np.float32) / MAX_RGB_VAL
    else:
        raw = raw.astype(np.float32)
    return raw


def prepare_display_image(rgb_tensor: np.ndarray) -> cv2.Mat:
    rgb = (rgb_tensor * 255.0).clip(0, 255).astype(np.uint8)
    return rgb


def interactive_loop(image_files: Iterable[Path], model: YOLO, args: argparse.Namespace) -> None:
    files = list(image_files)
    if not files:
        print(f"No images with extensions {sorted(SUPPORTED_EXTS)} found in {args.image_dir}")
        return

    if args.no_gui:
        total = len(files)
        for idx, path in enumerate(files):
            try:
                rgb_tensor = load_rgb_tensor(path)
            except Exception as exc:
                print(f"Warning: {exc}. Skipping {path.name}.")
                continue

            results = model.predict(rgb_tensor, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
            boxes = results[0].boxes if results else []
            vis = draw_detections(prepare_display_image(rgb_tensor), boxes)
            annotate_image(vis, path, idx, total)
            if args.save_dir:
                save_result(vis, args.save_dir, path)
            print(f"[{idx+1}/{total}] {path.name} done")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    idx = 0
    total = len(files)

    while 0 <= idx < total:
        path = files[idx]
        try:
            rgb_tensor = load_rgb_tensor(path)
        except Exception as exc:
            print(f"Warning: {exc}. Skipping.")
            idx += 1
            continue

        display_image = prepare_display_image(rgb_tensor)

        results = model.predict(rgb_tensor, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
        boxes = results[0].boxes if results else []
        vis = draw_detections(display_image, boxes)
        annotate_image(vis, path, idx, total)

        if args.save_dir:
            save_result(vis, args.save_dir, path)

        cv2.imshow(WINDOW_NAME, vis)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key in (ord("p"), ord("b")):
            idx = max(0, idx - 1)
        else:
            idx += 1

    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    image_files = list_images(args.image_dir)
    print(f"Loaded {len(image_files)} images from {args.image_dir}.")

    model = YOLO(str(args.model))
    print("Model ready. Controls: N/Space for next, P for previous, Q to quit.")

    interactive_loop(image_files, model, args)


if __name__ == "__main__":
    main()
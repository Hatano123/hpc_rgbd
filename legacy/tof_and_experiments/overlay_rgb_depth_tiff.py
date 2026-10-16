"""Interactive viewer: overlay RGB and depth from 4-channel TIFFs.

Controls: n/space/right -> next, p/left -> previous, q/esc -> quit
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import tifffile
from ultralytics import YOLO


MAX_RGB_VAL = 65535.0
MAX_DEPTH_VAL = 5000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-file",
        type=Path,
        default=None,
        help="Single 4-channel TIFF (RGB + depth) to view; overrides --image-dir",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("datasets/datasets_dark_4ch_map/images/train"),
        help="Directory containing 4-channel TIFF files (RGB + depth)",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Directory with YOLO-format txt labels; auto-resolved from image dir if omitted",
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=["cup"],
        help="Optional class names list (space-separated). Example: --class-names cup bottle",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/best.pt"),
        help="YOLO model weights (.pt). If set, run inference instead of reading labels.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution for YOLO model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for YOLO model")
    parser.add_argument("--device", default=None, help="Torch device for YOLO, e.g., '0' or 'cpu'")
    parser.add_argument("--alpha", type=float, default=0.45, help="Blend ratio for depth overlay")
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Optional fixed max depth in meters; auto 1st-99th percentile if omitted",
    )
    parser.add_argument("--start-index", type=int, default=0, help="0-based index to start from")
    return parser.parse_args()


def list_tiffs(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    return sorted(image_dir.glob("*.tiff"))


def load_rgb_depth(tiff_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = tifffile.imread(str(tiff_path))
    if data.ndim != 3 or data.shape[2] < 4:
        raise ValueError(f"Expected HxWx4 TIFF, got shape {data.shape} at {tiff_path}")

    rgb16 = data[..., :3].astype(np.float32)
    depth_mm = data[..., 3].astype(np.float32)

    # Convert 16-bit RGB to 8-bit for display.
    rgb = cv2.convertScaleAbs(rgb16, alpha=1.0 / 256.0)
    depth_m = depth_mm / 1000.0

    # Prepare 4ch tensor for YOLO inference (RGB normalized to 0-1, depth normalized to 0-1 by 5 m cap)
    rgbd_tensor = np.zeros_like(data, dtype=np.float32)
    rgbd_tensor[..., :3] = rgb16 / MAX_RGB_VAL
    depth_norm = np.nan_to_num(depth_mm, nan=MAX_DEPTH_VAL)
    depth_norm = np.clip(depth_norm, 0.0, MAX_DEPTH_VAL)
    rgbd_tensor[..., 3] = depth_norm / MAX_DEPTH_VAL

    return rgb, depth_m, rgbd_tensor


def depth_to_colormap(depth_m: np.ndarray, max_depth: float | None) -> np.ndarray:
    valid = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
    if valid.size == 0:
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)

    vmin = float(np.percentile(valid, 1))
    vmax = float(np.percentile(valid, 99)) if max_depth is None else max_depth
    vmax = max(vmax, vmin + 1e-3)

    depth_norm = np.clip((depth_m - vmin) / (vmax - vmin), 0.0, 1.0)
    depth_u8 = (depth_norm * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    depth_vis[depth_m <= 0] = 0
    return depth_vis


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> List[Tuple[int, Optional[float], int, int, int, int]]:
    labels: List[Tuple[int, Optional[float], int, int, int, int]] = []
    if not label_path.exists():
        return labels

    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls = int(float(parts[0]))
        cx, cy, w_rel, h_rel = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) >= 6 else None

        x1 = (cx - w_rel / 2.0) * img_w
        y1 = (cy - h_rel / 2.0) * img_h
        x2 = (cx + w_rel / 2.0) * img_w
        y2 = (cy + h_rel / 2.0) * img_h

        x1_i = int(max(0, min(img_w - 1, x1)))
        y1_i = int(max(0, min(img_h - 1, y1)))
        x2_i = int(max(0, min(img_w - 1, x2)))
        y2_i = int(max(0, min(img_h - 1, y2)))
        if x2_i <= x1_i or y2_i <= y1_i:
            continue

        labels.append((cls, conf, x1_i, y1_i, x2_i, y2_i))

    return labels


def boxes_from_ultralytics(result_boxes, img_w: int, img_h: int) -> List[Tuple[int, Optional[float], int, int, int, int]]:
    boxes: List[Tuple[int, Optional[float], int, int, int, int]] = []
    if result_boxes is None or len(result_boxes) == 0:
        return boxes

    for box in result_boxes:
        coords = box.xyxy[0].cpu().numpy().astype(float)
        x1, y1, x2, y2 = coords.tolist()
        conf = float(box.conf.item()) if box.conf is not None else None
        cls = int(box.cls.item()) if box.cls is not None else 0

        x1_i = int(max(0, min(img_w - 1, x1)))
        y1_i = int(max(0, min(img_h - 1, y1)))
        x2_i = int(max(0, min(img_w - 1, x2)))
        y2_i = int(max(0, min(img_h - 1, y2)))
        if x2_i <= x1_i or y2_i <= y1_i:
            continue
        boxes.append((cls, conf, x1_i, y1_i, x2_i, y2_i))
    return boxes


def draw_boxes(img: np.ndarray, boxes: List[Tuple[int, Optional[float], int, int, int, int]], class_names: Optional[Sequence[str]]) -> np.ndarray:
    if not boxes:
        return img

    out = img.copy()
    for cls, conf, x1, y1, x2, y2 in boxes:
        color = (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = class_names[cls] if class_names and cls < len(class_names) else str(cls)
        if conf is not None:
            label = f"{label}:{conf:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def show_samples(
    files: List[Path],
    start_index: int,
    alpha: float,
    max_depth: float | None,
    labels_dir: Path | None,
    class_names: Optional[Sequence[str]],
    model: Optional[YOLO],
    imgsz: int,
    conf: float,
    device: Optional[str],
) -> None:
    if not files:
        print("No TIFF files found.")
        return

    idx = min(max(0, start_index), len(files) - 1)
    total = len(files)
    cv2.namedWindow("RGB-Depth Overlay", cv2.WINDOW_NORMAL)

    while True:
        tiff_path = files[idx]
        try:
            rgb, depth_m, rgbd_tensor = load_rgb_depth(tiff_path)
        except Exception as exc:
            print(exc)
            idx = (idx + 1) % total
            continue

        depth_vis = depth_to_colormap(depth_m, max_depth)
        if depth_vis.shape[:2] != rgb.shape[:2]:
            depth_vis = cv2.resize(depth_vis, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        overlay = cv2.addWeighted(rgb, 1 - alpha, depth_vis, alpha, 0.0)

        if model is not None:
            results = model.predict(rgbd_tensor, imgsz=imgsz, conf=conf, device=device, verbose=False)
            result_boxes = results[0].boxes if results else []
            boxes = boxes_from_ultralytics(result_boxes, rgb.shape[1], rgb.shape[0])
        else:
            label_folder = labels_dir
            if label_folder is None:
                label_folder = tiff_path.parent.parent / "labels" / tiff_path.parent.name
            label_path = label_folder / f"{tiff_path.stem}.txt"
            boxes = load_yolo_labels(label_path, rgb.shape[1], rgb.shape[0])

        rgb_draw = draw_boxes(rgb, boxes, class_names)
        overlay_draw = draw_boxes(overlay, boxes, class_names)

        combined = np.hstack([rgb_draw, depth_vis, overlay_draw])
        title = f"[{idx + 1}/{total}] {tiff_path.name}"
        cv2.imshow("RGB-Depth Overlay", combined)
        cv2.setWindowTitle("RGB-Depth Overlay", title)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("n"), ord(" "), 83):
            idx = (idx + 1) % total
        elif key in (ord("p"), 81):
            idx = (idx - 1) % total

    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if args.image_file is not None:
        files = [args.image_file]
        print(f"Viewing single file: {args.image_file}")
    else:
        files = list_tiffs(args.image_dir)
        print(f"Found {len(files)} TIFFs under {args.image_dir}.")
    print("Controls: n/space/right -> next, p/left -> previous, q/esc -> quit")
    model = YOLO(str(args.model)) if args.model is not None else None
    if model is not None:
        print(f"Loaded model: {args.model}")
    else:
        print("Model not provided; using label files if present.")

    show_samples(
        files,
        args.start_index,
        args.alpha,
        args.max_depth,
        args.labels_dir,
        args.class_names,
        model,
        args.imgsz,
        args.conf,
        args.device,
    )


if __name__ == "__main__":
    main()

"""Run YOLO detections on every image inside a folder.

Usage example:
    python folder_detect.py --image-dir input_images --model yolov8n.pt --save-dir detection_results

Controls while the viewer window is focused:
    N / Space / Right arrow : show next image
    P / Left arrow          : show previous image
    Q / Esc                 : quit viewer

The script optionally writes annotated outputs into --save-dir.
"""

import argparse
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
from ultralytics import YOLO

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
WINDOW_NAME = "Folder Detection Viewer"
MAX_RGB_VAL = 65535.0
MAX_DEPTH_VAL = 5000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO detections on images stored in a folder.")
    parser.add_argument("--image-dir", type=Path, default=Path("input_images"), help="Directory that holds images.")
    parser.add_argument("--model", type=Path, default=Path("yolov8n.pt"), help="Path to a YOLO model weight file.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution passed to YOLO.")
    parser.add_argument("--conf", type=float, default=0.25, help="Score threshold for predictions.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. '0' or 'cpu'.")
    parser.add_argument("--save-dir", type=Path, help="If set, annotated images are written here.")
    parser.add_argument("--dark", action="store_true", help="Zero RGB channels before inference.")
    parser.add_argument("--view-mode", choices=["rgb", "overlay", "depth-blend"], default="overlay", help="Viewer base: rgb, overlay(depth on rgb), or depth-blend (depth as semi-transparent)")
    parser.add_argument("--depth-overlay", type=float, default=0.3, help="Alpha to blend a depth colormap onto RGB when view-mode=overlay (0 disables overlay).")
    parser.add_argument("--depth-blend-alpha", type=float, default=0.6, help="Alpha of depth when view-mode=depth-blend (0-1).")
    parser.add_argument("--depth-min", type=float, default=0.0, help="Minimum depth [m]; values below are set to 0 (invalid).")
    parser.add_argument("--depth-max", type=float, default=5.0, help="Maximum depth [m]; values above are set to 0 (invalid).")
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


def load_rgbd_tensor(path: Path, args: argparse.Namespace) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if raw.ndim == 2:
        raw = np.expand_dims(raw, axis=2)
    if raw.dtype != np.float32:
        raw = raw.astype(np.float32)
    if raw.shape[2] < 4:
        h, w, c = raw.shape
        pad = np.zeros((h, w, 4 - c), dtype=np.float32)
        raw = np.concatenate([raw, pad], axis=2)

    # Normalize RGB and Depth channels to match training pipeline
    raw[:, :, :3] /= MAX_RGB_VAL
    depth = raw[:, :, 3]
    depth = np.nan_to_num(depth, nan=MAX_DEPTH_VAL)
    depth = np.clip(depth, 0.0, MAX_DEPTH_VAL)

    # Keep inference depth untouched (only normalize). Filtering is applied later for display only.
    raw[:, :, 3] = depth / MAX_DEPTH_VAL
    return raw


def filter_depth_for_display(depth_norm: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Apply depth range filter only for visualization; inference uses unfiltered depth."""

    depth_mm = depth_norm * MAX_DEPTH_VAL
    d_min = max(0.0, float(args.depth_min)) * 1000.0
    d_max = max(d_min + 1.0, float(args.depth_max) * 1000.0)
    mask = (depth_mm < d_min) | (depth_mm > d_max)
    depth_mm_filtered = depth_mm.copy()
    depth_mm_filtered[mask] = 0.0
    return depth_mm_filtered / MAX_DEPTH_VAL


def prepare_display_image(rgbd_tensor: np.ndarray, args: argparse.Namespace) -> cv2.Mat:
    rgb = (rgbd_tensor[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8)
    depth = filter_depth_for_display(np.nan_to_num(rgbd_tensor[:, :, 3], nan=0.0), args)
    depth_vis = (depth * 255.0).clip(0, 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)

    if args.view_mode == "rgb":
        return rgb
    if args.view_mode == "overlay":
        alpha = max(0.0, min(1.0, args.depth_overlay))
        if alpha <= 0.0:
            return rgb
        return cv2.addWeighted(rgb, 1.0 - alpha, depth_vis, alpha, 0.0)
    if args.view_mode == "depth-blend":
        alpha = max(0.0, min(1.0, args.depth_blend_alpha))
        # Depth as the main layer, but keep some RGB context for edges
        return cv2.addWeighted(depth_vis, alpha, rgb, 1.0 - alpha, 0.0)
    return rgb


def setup_trackbars(args: argparse.Namespace) -> None:
    """Create trackbars for depth filtering and blending when GUI is enabled."""

    if args.no_gui:
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Depth range in millimeters (0-10000)
    cv2.createTrackbar("DepthMin(mm)", WINDOW_NAME, int(args.depth_min * 1000), 10000, lambda v: None)
    cv2.createTrackbar("DepthMax(mm)", WINDOW_NAME, int(args.depth_max * 1000), 10000, lambda v: None)

    # Alphas in percent (0-100)
    cv2.createTrackbar("OverlayAlpha(%)", WINDOW_NAME, int(args.depth_overlay * 100), 100, lambda v: None)
    cv2.createTrackbar("BlendAlpha(%)", WINDOW_NAME, int(args.depth_blend_alpha * 100), 100, lambda v: None)


def read_trackbar_settings(args: argparse.Namespace) -> None:
    """Update args in-place from trackbars (no effect in no-gui mode)."""

    if args.no_gui:
        return

    dmin_mm = cv2.getTrackbarPos("DepthMin(mm)", WINDOW_NAME)
    dmax_mm = cv2.getTrackbarPos("DepthMax(mm)", WINDOW_NAME)
    if dmax_mm <= dmin_mm:
        dmax_mm = dmin_mm + 1

    args.depth_min = dmin_mm / 1000.0
    args.depth_max = dmax_mm / 1000.0
    args.depth_overlay = cv2.getTrackbarPos("OverlayAlpha(%)", WINDOW_NAME) / 100.0
    args.depth_blend_alpha = cv2.getTrackbarPos("BlendAlpha(%)", WINDOW_NAME) / 100.0


def interactive_loop(image_files: Iterable[Path], model: YOLO, args: argparse.Namespace) -> None:
    files = list(image_files)
    if not files:
        print(f"No images with extensions {sorted(SUPPORTED_EXTS)} found in {args.image_dir}")
        return

    if args.no_gui:
        total = len(files)
        for idx, path in enumerate(files):
            try:
                rgbd_tensor = load_rgbd_tensor(path, args)
            except Exception as exc:
                print(f"Warning: {exc}. Skipping {path.name}.")
                continue
            if args.dark:
                rgbd_tensor[:, :, :3] = 0.0

            results = model.predict(rgbd_tensor, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
            boxes = results[0].boxes if results else []
            vis = draw_detections(prepare_display_image(rgbd_tensor, args), boxes)
            annotate_image(vis, path, idx, total)
            if args.save_dir:
                save_result(vis, args.save_dir, path)
            print(f"[{idx+1}/{total}] {path.name} done")
        return

    setup_trackbars(args)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    idx = 0
    total = len(files)

    while 0 <= idx < total:
        path = files[idx]
        read_trackbar_settings(args)
        try:
            rgbd_tensor = load_rgbd_tensor(path, args)
        except Exception as exc:
            print(f"Warning: {exc}. Skipping.")
            idx += 1
            continue
        if args.dark:
            rgbd_tensor[:, :, :3] = 0.0

        display_image = prepare_display_image(rgbd_tensor, args)

        results = model.predict(rgbd_tensor, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
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

"""Provisional RealSense adapter. Hardware operation has not been verified."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import cv2
import numpy as np
import yaml
from .preprocessing import PreprocessConfig, prepare_aligned


def validate_frame(rgb, depth_mm):
    depth = np.asarray(depth_mm, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or depth.shape != rgb.shape[:2]:
        raise ValueError('Expected aligned uint8 BGR and single-channel depth in mm')
    if not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError('Depth must contain finite, non-negative millimetres')
    return depth


def save_capture(directory, index, rgb, depth_mm, metadata):
    depth = validate_frame(rgb, depth_mm)
    directory = Path(directory)
    name = f'frame_{index:06d}'
    encoded = np.empty((*depth.shape, 4), dtype=np.uint16)
    encoded[..., :3] = rgb.astype(np.uint16) * 256
    encoded[..., 3] = np.rint(np.clip(depth, 0, 8000) * 8).astype(np.uint16)
    images = {'rgb': rgb, 'depth_mm': np.rint(np.clip(depth, 0, 65535)).astype(np.uint16), 'rgbd': encoded}
    for kind in images:
        suffix = '.tiff' if kind == 'rgbd' else '.png'
        path = directory / kind / (name + suffix)
        if path.exists():
            raise FileExistsError(path)
    for kind, image in images.items():
        suffix = '.tiff' if kind == 'rgbd' else '.png'
        path = directory / kind / (name + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise OSError(f'Cannot save {path}')
    (directory / 'metadata').mkdir(exist_ok=True)
    (directory / 'metadata' / (name + '.json')).write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (directory / 'preprocess.yaml').write_text(yaml.safe_dump(PreprocessConfig().to_dict()), encoding='utf-8')
    return directory / 'rgbd' / (name + '.tiff')


def realsense_frames(width, height, fps, serial):
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError('RealSense capture requires pyrealsense2 and a compatible SDK/Python environment. Use --replay for a camera-free test.') from exc
    pipeline = rs.pipeline()
    settings = rs.config()
    if serial:
        settings.enable_device(serial)
    settings.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    settings.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    started = False
    try:
        profile = pipeline.start(settings)
        started = True
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        alignment = rs.align(rs.stream.color)
        serial_number = profile.get_device().get_info(rs.camera_info.serial_number)
        while True:
            frames = alignment.process(pipeline.wait_for_frames(5000))
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                continue
            intr = color.profile.as_video_stream_profile().get_intrinsics()
            metadata = {'adapter': 'RealSense (provisional)', 'serial': serial_number,
                'aligned_to': 'color', 'depth_scale_m': float(scale),
                'color_timestamp_ms': float(color.get_timestamp()), 'depth_timestamp_ms': float(depth.get_timestamp()),
                'color_intrinsics': {'width': intr.width, 'height': intr.height,
                    'fx': intr.fx, 'fy': intr.fy, 'ppx': intr.ppx, 'ppy': intr.ppy,
                    'distortion_model': str(intr.model), 'coefficients': list(intr.coeffs)}}
            yield (np.asanyarray(color.get_data()).copy(),
                   np.asanyarray(depth.get_data()).astype(np.float32) * float(scale) * 1000, metadata)
    finally:
        if started:
            pipeline.stop()


def draw_live(rgb, detections):
    canvas = rgb.copy()
    for item in detections:
        x1, y1, x2, y2 = map(int, item['xyxy'])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas, f"{item['name']} {item['confidence']:.2f}", (x1, max(15, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description='Provisional RealSense capture: Space saves, Q quits. No hole filling or artificial missing data.')
    parser.add_argument('--output', help='Empty capture directory; default: work/captures/timestamp')
    parser.add_argument('--model', help='Optional trained 4ch checkpoint for live detection')
    parser.add_argument('--config', help='Preprocessing config for an older checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=.25)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--serial')
    parser.add_argument('--frames', type=int, default=0, help='Automatically save N frames and exit; default: manual capture')
    parser.add_argument('--no-preview', action='store_true', help='No camera window; requires --frames')
    parser.add_argument('--replay', help='Camera-free check: read an encoded 4ch TIFF, save one frame, then exit')
    args = parser.parse_args(argv)
    if args.frames < 0 or min(args.width, args.height, args.fps) < 1:
        parser.error('frames must be non-negative; width, height and fps must be positive')
    if args.no_preview and not args.frames and not args.replay:
        parser.error('--no-preview requires --frames N')
    timestamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S_%f')
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve() if args.output else root / 'work/captures' / timestamp
    if output.exists() and any(output.iterdir()):
        parser.error(f'Choose an empty output directory: {output}')
    config, model = PreprocessConfig(), None
    if args.model:
        from ultralytics import YOLO
        from .cli import checkpoint_config
        model = YOLO(args.model)
        if model.model.model[0].conv.in_channels != 4:
            parser.error('--model must be a trained four-channel model')
        config = checkpoint_config(model, args.config)
    if args.replay:
        raw = cv2.imread(args.replay, cv2.IMREAD_UNCHANGED)
        if raw is None or raw.ndim != 3 or raw.shape[-1] != 4 or raw.dtype != np.uint16:
            parser.error('--replay expects an encoded uint16 four-channel TIFF')
        # This adapter's recording format: BGR intensity x256, depth millimetres x8.
        frames = iter([(np.clip(raw[..., :3] / 256, 0, 255).astype(np.uint8),
                        raw[..., 3].astype(np.float32) / 8, {'adapter': 'file replay', 'source': str(Path(args.replay).resolve())})])
    else:
        print('Provisional RealSense adapter; hardware behavior has not been verified. Space: save, Q: exit.', flush=True)
        frames = realsense_frames(args.width, args.height, args.fps, args.serial)
    saved = 0
    try:
        for rgb, depth, metadata in frames:
            depth = validate_frame(rgb, depth)
            detections = []
            if model:
                from .inference import predict_aligned
                detections = predict_aligned(model, prepare_aligned(rgb, depth, config), args.imgsz, args.conf, args.device)
            key = -1
            if not args.no_preview and not args.replay:
                scaled = np.clip(depth / 8000 * 255, 0, 255).astype(np.uint8)
                canvas = np.hstack([draw_live(rgb, detections), cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)])
                cv2.imshow('RGB-D capture (provisional) | Space save | Q quit', canvas)
                key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if args.replay or args.frames or key == 32:
                path = save_capture(output, saved, rgb, depth, metadata)
                if model:
                    from .inference import save_prediction
                    save_prediction(prepare_aligned(rgb, depth, config), detections, output / 'detections' / path.stem)
                saved += 1
                print(f'Saved: {path}', flush=True)
            if args.replay or (args.frames and saved >= args.frames):
                break
    except KeyboardInterrupt:
        print('Capture stopped.', flush=True)
    finally:
        if hasattr(frames, 'close'):
            frames.close()
        if not args.no_preview and not args.replay:
            cv2.destroyAllWindows()
    print(f'{saved} frame(s) saved. These captures have no training labels.', flush=True)

"""Stable entry points for aligned RGB-D workflows."""
import argparse
from functools import partial
import json
from pathlib import Path
import platform
import sys
import cv2
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from .preprocessing import PreprocessConfig, prepare_aligned, read_tiff

ROOT = Path(__file__).resolve().parents[1]


def parser():
    result = argparse.ArgumentParser(description='YOLO on pixel-aligned BGR + depth; no camera registration')
    commands = result.add_subparsers(dest='command', required=True)
    for command in ['prepare', 'demo']:
        cmd = commands.add_parser(command, help='Prepare a checked subset' if command == 'prepare' else 'Run input -> preparation -> training -> detection -> evaluation')
        cmd.add_argument('--source', help='Local dataset root; demo can discover ../YOLORGBD/datasets/sunrgbd_top5')
        cmd.add_argument('--output', default='work/sunrgbd_demo' if command == 'demo' else 'work/sunrgbd_subset')
        cmd.add_argument('--config', default=str(ROOT / 'configs/sunrgbd_tiff.yaml'))
        cmd.add_argument('--classes', default=str(ROOT / 'sunrgbd_top5/data.yaml'), help='Fallback class YAML when source/data.yaml is absent')
        cmd.add_argument('--train-count', type=int, default=8)
        cmd.add_argument('--val-count', type=int, default=4)
        cmd.add_argument('--seed', type=int, default=0)
        if command == 'demo':
            cmd.add_argument('--epochs', type=int, default=2)
            cmd.add_argument('--imgsz', type=int, default=160)
            cmd.add_argument('--device', default='cpu')
    train = commands.add_parser('train')
    train.add_argument('--mode', choices=['rgb', 'rgbd'], default='rgbd')
    train.add_argument('--data', required=True)
    train.add_argument('--cfg', default='yolo11n.yaml')
    train.add_argument('--weights', default=str(ROOT / 'yolo11n.pt') if (ROOT / 'yolo11n.pt').is_file() else 'yolo11n.pt')
    train.add_argument('--config', help='Preprocessing YAML (default: encoded SUN RGB-D TIFF profile)')
    train.add_argument('--epochs', type=int, default=150)
    train.add_argument('--imgsz', type=int, default=640)
    train.add_argument('--batch', type=int, default=16)
    train.add_argument('--workers', type=int, default=0)
    train.add_argument('--device', default='cpu')
    train.add_argument('--project', default=str(ROOT / 'runs'))
    train.add_argument('--name', default='aligned_rgbd')
    train.add_argument('--lr0', type=float, default=.002)
    train.add_argument('--mosaic', type=float, default=.5)
    train.add_argument('--mixup', type=float, default=0.0)
    train.add_argument('--close-mosaic', type=int, default=10)
    train.add_argument('--log-file')
    train.add_argument('--no-plots', action='store_true')
    for name in ['val', 'predict']:
        cmd = commands.add_parser(name)
        cmd.add_argument('--model', required=True)
        cmd.add_argument('--config', help='Required for old checkpoints without preprocessing metadata')
        cmd.add_argument('--imgsz', type=int, default=640)
        cmd.add_argument('--device', default='cpu')
        if name == 'val':
            cmd.add_argument('--data', required=True)
            cmd.add_argument('--batch', type=int, default=4)
            cmd.add_argument('--split', choices=['val', 'test'], default='val')
            cmd.add_argument('--mode', choices=['rgb', 'rgbd'], default='rgbd')
            cmd.add_argument('--output', help='Write metrics.json, per-class summary and evaluation plots here')
        else:
            inputs = cmd.add_mutually_exclusive_group(required=True)
            inputs.add_argument('--image', help='Encoded aligned 4ch TIFF')
            inputs.add_argument('--images', help='Directory containing aligned 4ch TIFFs; save one result directory per image')
            inputs.add_argument('--rgb', help='BGR-readable RGB image, paired with --depth in millimetres')
            cmd.add_argument('--depth', help='Single-channel depth image in millimetres')
            cmd.add_argument('--conf', type=float, default=.25)
            cmd.add_argument('--output', required=True)
    inspect_cmd = commands.add_parser('inspect')
    inspect_cmd.add_argument('--image', required=True)
    inspect_cmd.add_argument('--config')
    return result


def checkpoint_config(model, path):
    stored = getattr(model.model, 'rgbd_config', None)
    if path:
        return PreprocessConfig.load(path)
    if stored:
        if set(stored) - {'rgb_divisor', 'depth_divisor', 'max_depth_mm'}:
            raise ValueError('This checkpoint uses an older experimental preprocessing profile. Retrain with the simple 4-channel workflow.')
        return PreprocessConfig(**stored)
    raise ValueError('Checkpoint has no preprocessing metadata; specify --config explicitly')


def train(args):
    from .training import RGBDTrainer
    data = yaml.safe_load(Path(args.data).read_text(encoding='utf-8'))
    expected_channels = 4 if args.mode == 'rgbd' else 3
    if data.get('channels', 3) != expected_channels:
        raise ValueError(f'{args.mode} training requires channels: {expected_channels} in dataset YAML')
    config = PreprocessConfig.load(args.config) if args.config else PreprocessConfig()
    model = YOLO(args.cfg)
    if args.weights:
        model.load(args.weights)
    kwargs = {}
    if args.mode == 'rgbd':
        kwargs['trainer'] = partial(RGBDTrainer, preprocess=config)
    model.train(**kwargs, data=str(Path(args.data).resolve()), epochs=args.epochs,
        imgsz=args.imgsz, batch=args.batch, workers=args.workers, device=args.device,
        project=str(Path(args.project).resolve()), name=args.name, exist_ok=True,
        lr0=args.lr0, lrf=.01, warmup_epochs=5, weight_decay=.002, optimizer='AdamW',
        box=10, cls=.5, mosaic=args.mosaic, mixup=args.mixup, close_mosaic=args.close_mosaic,
        fliplr=.5, hsv_h=0, hsv_s=0, hsv_v=0, amp=False, cache=False, save=True, val=True,
        plots=not args.no_plots)
    manifest = {'python': platform.python_version(), 'torch': torch.__version__,
        'ultralytics': ultralytics.__version__, 'opencv': cv2.__version__,
        'arguments': vars(args), 'preprocess': config.to_dict() if args.mode == 'rgbd' else None,
        'alignment': 'pre-aligned input; no projection performed', 'channel_order': 'BGRD' if args.mode == 'rgbd' else 'RGB'}
    (Path(model.trainer.save_dir) / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return model


def run(args):
    if args.command in {'prepare', 'demo'}:
        from .pipeline import prepare_command, demo
        return demo(args) if args.command == 'demo' else prepare_command(args)
    if args.command == 'train':
        return train(args)
    if args.command == 'inspect':
        image = read_tiff(args.image, PreprocessConfig.load(args.config) if args.config else PreprocessConfig())
        print(json.dumps({'shape': list(image.shape), 'dtype': str(image.dtype),
            'order': 'BGRD', 'ranges': [[float(image[..., c].min()), float(image[..., c].max())] for c in range(4)]}, indent=2))
        return
    model = YOLO(args.model)
    if args.command == 'val' and args.mode == 'rgb':
        options = {'project': str(Path(args.output).resolve()), 'name': 'plots', 'exist_ok': True} if args.output else {}
        metrics = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device, split=args.split, workers=0, **options)
        if args.output:
            from .evaluation import save_evaluation
            save_evaluation(metrics, args.output, args.model, args.data, args.split)
        return metrics
    config = checkpoint_config(model, args.config)
    if model.model.model[0].conv.in_channels != 4:
        raise ValueError('RGB-D command requires a 4-channel checkpoint')
    if args.command == 'val':
        from .training import RGBDValidator
        options = {'project': str(Path(args.output).resolve()), 'name': 'plots', 'exist_ok': True} if args.output else {}
        metrics = model.val(validator=partial(RGBDValidator, preprocess=config), data=args.data,
            imgsz=args.imgsz, batch=args.batch, device=args.device, split=args.split, workers=0, **options)
        if args.output:
            from .evaluation import save_evaluation
            save_evaluation(metrics, args.output, args.model, args.data, args.split)
        return metrics
    from .inference import predict_aligned, save_prediction
    if args.images:
        if args.depth:
            raise ValueError('--depth is only used with --rgb')
        paths = sorted(path for path in Path(args.images).glob('*') if path.suffix.lower() in {'.tif', '.tiff'})
        if not paths:
            raise FileNotFoundError(f'No TIFF files found: {args.images}')
        if len({path.stem for path in paths}) != len(paths):
            raise ValueError('TIFF stems must be unique to create separate result directories')
        index = []
        for path in paths:
            image = read_tiff(path, config)
            detections = predict_aligned(model, image, args.imgsz, args.conf, args.device)
            save_prediction(image, detections, Path(args.output) / path.stem)
            index.append({'image': str(path.resolve()), 'detections': len(detections), 'output': path.stem})
        (Path(args.output) / 'index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
        print(f'Saved predictions for {len(paths)} images to {args.output}')
        return
    if args.image:
        if args.depth:
            raise ValueError('--depth is only used with --rgb')
        image = read_tiff(args.image, config)
    else:
        if not args.depth:
            raise ValueError('--rgb requires --depth; depth must be aligned and in millimetres')
        rgb = cv2.imread(args.rgb)
        depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise FileNotFoundError('Could not read RGB/depth pair')
        image = prepare_aligned(rgb, depth, config)
    detections = predict_aligned(model, image, args.imgsz, args.conf, args.device)
    save_prediction(image, detections, args.output)
    print(f'Saved {len(detections)} detections to {args.output}')


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file
    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()
    def flush(self):
        self.stream.flush()
        self.file.flush()
    def isatty(self):
        return False


def main(argv=None):
    args = parser().parse_args(argv)
    if getattr(args, 'log_file', None):
        path = Path(args.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        original = sys.stdout, sys.stderr
        with path.open('a', encoding='utf-8') as file:
            try:
                sys.stdout, sys.stderr = Tee(original[0], file), Tee(original[1], file)
                return run(args)
            finally:
                sys.stdout, sys.stderr = original
    return run(args)

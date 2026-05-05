import argparse
import os

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO with RGB-only (3ch) images on HPC")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--weights", default="yolo11n.pt", help="Initial model weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--name", default="hpc_rgb_3ch_baseline", help="Run name")
    parser.add_argument("--project", default=None, help="Output project directory (default: ./runs)")
    parser.add_argument("--device", default=None, help="Device id or string (default: auto)")
    return parser.parse_args()


def resolve_device(device_arg: str | None):
    if device_arg is not None:
        return device_arg
    return 0 if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = parse_args()

    data_yaml = os.path.abspath(args.data)
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = args.project or os.path.join(script_dir, "runs")
    device = resolve_device(args.device)

    print(f"[3ch] data: {data_yaml}")
    print(f"[3ch] weights: {args.weights}")
    print(f"[3ch] epochs: {args.epochs}")
    print(f"[3ch] project/name: {runs_dir}/{args.name}")
    print(f"[3ch] device: {device}")

    model = YOLO(args.weights)
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=runs_dir,
        name=args.name,
        exist_ok=True,
        device=device,
        cache=False,
        augment=False,
        mosaic=0.0,
        mixup=0.0,
        amp=False,
        save=True,
        val=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
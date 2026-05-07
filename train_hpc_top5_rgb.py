#!/usr/bin/env python3
"""
Train YOLO with RGB-only (3ch) images on HPC - Optimized for SUN RGBD Top5 Dataset
"""
import argparse
import os
import torch
from ultralytics import YOLO

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO with RGB-only (3ch) images on HPC - SUN RGBD Top5"
    )
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--weights", default="yolo11n.pt", help="Initial model weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=2, help="Data loader workers")
    parser.add_argument("--name", default="hpc_sunrgbd_top5_rgb_e100", help="Run name")
    parser.add_argument("--project", default=None, help="Output project directory (default: ./runs)")
    parser.add_argument("--device", default=None, help="Device id or string (default: auto)")
    parser.add_argument("--augment", action="store_true", help="Enable augmentation")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation ratio")
    parser.add_argument("--mixup", type=float, default=0.5, help="Mixup augmentation ratio")
    parser.add_argument("--cache", choices=["false", "disk", "ram"], default="false", help="Dataset cache strategy: false/disk/ram")
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

    print("\n" + "=" * 60)
    print("[RGB-Only Training] SUN RGBD Top5 Dataset")
    print("=" * 60)
    print(f"Data:     {data_yaml}")
    print(f"Weights:  {args.weights}")
    print(f"Epochs:   {args.epochs}")
    print(f"Batch:    {args.batch}")
    print(f"Image sz: {args.imgsz}")
    print(f"Workers:  {args.workers}")
    print(f"Project:  {runs_dir}/{args.name}")
    print(f"Device:   {device}")
    print(f"Augment:  {args.augment}")
    print(f"Mosaic:   {args.mosaic}")
    print(f"Mixup:    {args.mixup}")
    print(f"Cache:    {args.cache}")
    print("=" * 60 + "\n")

    model = YOLO(args.weights)
    
    # Resolve cache parameter for ultralytics: False | True (disk) | 'ram'
    cache_param = False
    if args.cache == "disk":
        cache_param = True
    elif args.cache == "ram":
        cache_param = "ram"

    # Basic safety check for ram cache
    if cache_param == "ram":
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        avail_kb = int(line.split()[1])
                        avail_gb = avail_kb / 1024.0 / 1024.0
                        if avail_gb < 8.0:
                            print(f"WARNING: Available memory ~{avail_gb:.1f}GB — ram cache may cause OOM. Consider using disk cache or reducing workers/batch.")
                        break
        except Exception:
            print("WARNING: Could not read /proc/meminfo to assess ram availability. Use ram cache only if you know the machine has sufficient RAM.")

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=runs_dir,
        name=args.name,
        exist_ok=True,
        device=device,
        cache=cache_param,
        augment=args.augment,
        mosaic=args.mosaic,
        mixup=args.mixup,
        amp=False,
        save=True,
        val=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("[Training Complete] Results saved to:")
    print(f"{runs_dir}/{args.name}")
    print("=" * 60 + "\n")

    return results

if __name__ == "__main__":
    main()
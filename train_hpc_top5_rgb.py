#!/usr/bin/env python3
"""
Train YOLO with RGB-only (3ch) images on HPC - Optimized for SUN RGBD Top5 Dataset
(Aligned with 4ch Full-Boost settings for fair ablation study)
"""
import argparse
import os
import torch
import random
from ultralytics import YOLO

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO with RGB-only (3ch) images on HPC - SUN RGBD Top5"
    )
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    # 【修正1】4chと揃えるため yolov8n.pt をデフォルトに変更
    parser.add_argument("--weights", default="yolov8n.pt", help="Initial model weights")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers (0 for HPC safety)")
    parser.add_argument("--name", default="hpc_sunrgbd_top5_rgb_baseline", help="Run name")
    parser.add_argument("--project", default=None, help="Output project directory (default: ./runs_rgb)")
    parser.add_argument("--device", default=None, help="Device id or string (default: auto)")
    # 【修正2】Augmentationのデフォルト値を4chと完全に一致させる
    parser.add_argument("--augment", action="store_true", default=True, help="Enable augmentation")
    parser.add_argument("--mosaic", type=float, default=0.5, help="Mosaic augmentation ratio")
    parser.add_argument("--mixup", type=float, default=0.0, help="Mixup augmentation ratio")
    parser.add_argument("--cache", choices=["false", "disk", "ram"], default="false", help="Dataset cache strategy")
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
    runs_dir = args.project or os.path.join(script_dir, "runs_rgb")
    device = resolve_device(args.device)

    print("\n" + "=" * 60)
    print("🔥 [RGB-Only Baseline] SUN RGBD Top5 Dataset")
    print("🔥 Aligned with 4ch hyperparameters for Fair Comparison")
    print("=" * 60)

    model = YOLO(args.weights)
    
    # ---------------------------------------------------------
    # HPC Mosaicバグ回避のパッチ (4chコードと同じ安全装置)
    # ---------------------------------------------------------
    try:
        import ultralytics.data.augment as ua
        import inspect

        def make_safe_get_indexes(orig):
            def safe_get_indexes(self, *args, **kwargs):
                try:
                    return orig(self, *args, **kwargs)
                except IndexError:
                    try:
                        n = getattr(self, 'n', None) or kwargs.get('n', None)
                        k = (n - 1) if n else kwargs.get('k', 1)
                        ds = getattr(self, 'dataset', None)
                        if ds is None:
                            return []
                        ds_len = len(getattr(ds, 'im_files', getattr(ds, 'images', [])))
                        if ds_len <= 0:
                            return []
                        return [random.randrange(ds_len) for _ in range(max(1, int(k)))]
                    except Exception:
                        return []
            return safe_get_indexes

        for name, obj in inspect.getmembers(ua, inspect.isclass):
            if hasattr(obj, 'get_indexes'):
                orig = getattr(obj, 'get_indexes')
                try:
                    setattr(obj, 'get_indexes', make_safe_get_indexes(orig))
                except Exception:
                    pass
        print("✅ DEBUG: Applied augment get_indexes guard for HPC.")
    except Exception as e:
        print("⚠️ DEBUG: Could not apply augment guard:", e)

    cache_param = False if args.cache == "false" else (True if args.cache == "disk" else "ram")

    # ---------------------------------------------------------
    # Training Execution
    # ---------------------------------------------------------
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

        # --- 【修正3】4chフルブーストと揃えたハイパーパラメータ ---
        cos_lr=True,          
        weight_decay=0.001,   
        patience=0,           
        amp=False,            
        # --------------------------------------------------

        # --- 【修正4】4chと揃えたAugmentation ---
        augment=args.augment,
        mosaic=args.mosaic,
        mixup=args.mixup,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        fliplr=0.5,
        # --------------------------------------------------

        save_period=-1,
        save=True,
        val=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("[Training Complete] RGB Baseline Results saved to:")
    print(f"{runs_dir}/{args.name}")
    print("=" * 60 + "\n")

    return results

if __name__ == "__main__":
    main()
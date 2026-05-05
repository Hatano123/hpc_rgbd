import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


MAX_RGB_VAL = 65535.0
MAX_DEPTH_VAL = 5000.0
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resolve_dataset_root(yaml_path: Path, data_cfg: dict) -> Path:
    root = data_cfg.get("path", "")
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = (yaml_path.parent / root_path).resolve()
    return root_path


def collect_image_files(dataset_root: Path, train_entry: str) -> list[Path]:
    entry_path = (dataset_root / train_entry).resolve()
    if entry_path.is_dir():
        return sorted([p for p in entry_path.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if entry_path.is_file():
        if entry_path.suffix.lower() == ".txt":
            files = []
            for line in entry_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                p = Path(line)
                if not p.is_absolute():
                    p = (dataset_root / p).resolve()
                if p.suffix.lower() in IMG_EXTS:
                    files.append(p)
            return files
        if entry_path.suffix.lower() in IMG_EXTS:
            return [entry_path]
    return []


def ensure_4ch(im: np.ndarray) -> np.ndarray:
    if im.ndim == 2:
        return np.stack([im, im, im, np.zeros_like(im)], axis=-1)
    if im.ndim == 3 and im.shape[2] < 4:
        pad = np.zeros((im.shape[0], im.shape[1], 4 - im.shape[2]), dtype=im.dtype)
        return np.concatenate([im, pad], axis=2)
    if im.ndim == 3 and im.shape[2] > 4:
        return im[:, :, :4]
    return im


def normalize_like_train_hpc(im: np.ndarray) -> np.ndarray:
    im = ensure_4ch(im)
    im = im.astype(np.float32)

    im[:, :, :3] = (im[:, :, :3] / MAX_RGB_VAL) * 255.0

    depth = im[:, :, 3]
    depth = np.nan_to_num(depth, nan=0.0)
    depth = (np.clip(depth, 0, MAX_DEPTH_VAL) / MAX_DEPTH_VAL) * 255.0
    im[:, :, 3] = depth

    im = np.nan_to_num(im, nan=0.0, posinf=255.0, neginf=0.0)
    im = np.clip(im, 0.0, 255.0)
    return im


def channel_stats(arr: np.ndarray, ch_idx: int) -> dict:
    x = arr[:, :, ch_idx].astype(np.float32).reshape(-1)
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "p1": float(np.percentile(x, 1)),
        "p99": float(np.percentile(x, 99)),
    }


def print_stats(label: str, stats: dict) -> None:
    print(
        f"{label}: min={stats['min']:.3f}, max={stats['max']:.3f}, "
        f"mean={stats['mean']:.3f}, p1={stats['p1']:.3f}, p99={stats['p99']:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify RGBD normalization scale in TMP workspace.")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--samples", type=int, default=32, help="Number of sampled images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tmp-out", required=True, help="TMP output directory")
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        print(f"[ERROR] data yaml not found: {data_yaml}")
        return 1

    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_root = resolve_dataset_root(data_yaml, cfg)
    train_entry = cfg.get("train")
    if not train_entry:
        print("[ERROR] 'train' entry not found in data yaml")
        return 1

    all_images = collect_image_files(dataset_root, train_entry)
    if not all_images:
        print(f"[ERROR] no images found from train entry: {train_entry}")
        print(f"        resolved dataset root: {dataset_root}")
        return 1

    random.seed(args.seed)
    n = min(args.samples, len(all_images))
    sampled = random.sample(all_images, n)

    out_dir = Path(args.tmp_out).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Normalization scale check (same formula as train_hpc.py)")
    print(f"data yaml      : {data_yaml}")
    print(f"dataset root   : {dataset_root}")
    print(f"train entry    : {train_entry}")
    print(f"total images   : {len(all_images)}")
    print(f"sampled images : {n}")
    print(f"tmp output     : {out_dir}")
    print("=" * 70)

    raw_rgb_stats = []
    raw_depth_stats = []
    norm_rgb_stats = []
    norm_depth_stats = []

    depth_clip_count = 0
    rgb_low_dynamic_count = 0

    for idx, p in enumerate(sampled):
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is None:
            print(f"[WARN] skip unreadable image: {p}")
            continue

        raw_4ch = ensure_4ch(raw)
        norm = normalize_like_train_hpc(raw)

        r0 = channel_stats(raw_4ch, 0)
        g0 = channel_stats(raw_4ch, 1)
        b0 = channel_stats(raw_4ch, 2)
        d0 = channel_stats(raw_4ch, 3)

        rn = channel_stats(norm, 0)
        gn = channel_stats(norm, 1)
        bn = channel_stats(norm, 2)
        dn = channel_stats(norm, 3)

        raw_rgb_stats.extend([r0, g0, b0])
        raw_depth_stats.append(d0)
        norm_rgb_stats.extend([rn, gn, bn])
        norm_depth_stats.append(dn)

        if max(r0["p99"], g0["p99"], b0["p99"]) <= 255.0:
            rgb_low_dynamic_count += 1
        if d0["max"] > MAX_DEPTH_VAL:
            depth_clip_count += 1

        if idx < 3:
            vis = norm.astype(np.uint8)
            cv2.imwrite(str(out_dir / f"sample_{idx:02d}_norm_rgb.png"), vis[:, :, :3])
            cv2.imwrite(str(out_dir / f"sample_{idx:02d}_norm_depth.png"), vis[:, :, 3])

    def merge_stats(stats_list: list[dict]) -> dict:
        if not stats_list:
            return {"min": np.nan, "max": np.nan, "mean": np.nan, "p1": np.nan, "p99": np.nan}
        keys = ["min", "max", "mean", "p1", "p99"]
        return {k: float(np.mean([d[k] for d in stats_list])) for k in keys}

    print("\n[RAW RGB avg stats over sampled channels]")
    print_stats("RAW_RGB", merge_stats(raw_rgb_stats))
    print("[RAW DEPTH avg stats]")
    print_stats("RAW_DEPTH", merge_stats(raw_depth_stats))

    print("\n[NORMALIZED RGB avg stats over sampled channels]")
    print_stats("NORM_RGB", merge_stats(norm_rgb_stats))
    print("[NORMALIZED DEPTH avg stats]")
    print_stats("NORM_DEPTH", merge_stats(norm_depth_stats))

    print("\n[HEURISTIC CHECK]")
    if rgb_low_dynamic_count > 0:
        print(
            f"[WARN] {rgb_low_dynamic_count}/{n} images look <=8-bit on RGB p99; "
            "using MAX_RGB_VAL=65535 may over-compress RGB."
        )
    else:
        print("[OK] RGB dynamic range appears consistent with >8-bit data.")

    if depth_clip_count > 0:
        print(
            f"[WARN] {depth_clip_count}/{n} images have depth max > {MAX_DEPTH_VAL}; "
            "depth clipping is occurring."
        )
    else:
        print("[OK] Depth max is within MAX_DEPTH_VAL for sampled images.")

    print("\nArtifacts written to TMP:")
    print(f"  - {out_dir}")
    print("  - sample_*_norm_rgb.png / sample_*_norm_depth.png")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

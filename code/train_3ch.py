import os
from pathlib import Path
import yaml
from ultralytics import YOLO

"""
Train YOLO on RGB-only PNGs converted from datasets16.01-11 4ch TIFFs.
1) Run make_rgb_dataset_16_01_11.py first to create rgb_only_3ch.
2) Then run this script.
"""

DEFAULT_DATASET_YAML = Path("datasets/datasets16.01-11/rgb_only_3ch/dataset_rgb3.yaml").resolve()


def resolve_dataset_yaml() -> Path:
    dataset_yaml_env = os.getenv("TRAIN3CH_DATASET_YAML")
    dataset_yaml_path = Path(dataset_yaml_env) if dataset_yaml_env else DEFAULT_DATASET_YAML
    if not dataset_yaml_path.exists():
        raise FileNotFoundError(f"Data YAML not found: {dataset_yaml_path}")
    return dataset_yaml_path


if __name__ == "__main__":
    dataset_yaml = resolve_dataset_yaml()
    pretrained_weights = os.getenv("TRAIN3CH_WEIGHTS", "yolov8n.pt")
    epochs = int(os.getenv("TRAIN3CH_EPOCHS", "100"))
    batch = int(os.getenv("TRAIN3CH_BATCH", "16"))
    imgsz = int(os.getenv("TRAIN3CH_IMGSZ", "640"))
    workers = int(os.getenv("TRAIN3CH_WORKERS", "0"))
    project = os.getenv("TRAIN3CH_PROJECT", "runs/detect")
    run_name = os.getenv("TRAIN3CH_RUN_NAME", "train_3ch_test")

    with open(dataset_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    if data_cfg.get("channels") not in (None, 3):
        print("⚠ Please remove 'channels' from the YAML or set it to 3.")

    print(f"✅ Training on: {dataset_yaml}")
    print(f"✅ Images: {data_cfg.get('train', 'n/a')}")

    model = YOLO(pretrained_weights)

    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=project,
        name=run_name,
        workers=workers,
        cache=False,
    )

    print("\n🎉 Training complete on RGB-only data!")

import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn import tasks

# --- Detect Environment ---
IS_COLAB = "COLAB_GPU" in os.environ
if IS_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT_DATA = "/content/drive/MyDrive/Datasets/SUNRGBD/SUNRGBD/kv2/align_kv2"
    OUTPUT_DATA = "/content/yolor_dataset"
else:
    ROOT_DATA = r"G:\マイドライブ\Datasets\SUNRGBD\SUNRGBD\kv2\align_kv2"
    OUTPUT_DATA = "./datasets/sunrgbd_tiff"

# --- Monkey Patching Dataset to support 4-channel TIFF (BGRD) ---

def custom_load_image(self, i):
    im_path = self.im_files[i]
    im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"Image Not Found: {im_path}")
    
    im = im.astype(np.float32)
    depth = im[:, :, 3]
    # Normalize Depth (Scale 0-8000mm to 0-255 range)
    depth = np.clip(depth, 0, 8000) / 8000.0 * 255.0
    im[:, :, 3] = depth
    
    h0, w0 = im.shape[:2]
    return im, (h0, w0), im.shape[:2]

YOLODataset.load_image = custom_load_image

# --- Monkey Patching DetectionModel to force 4-channel setup ---

original_init = tasks.DetectionModel.__init__
def patched_init(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):
    print(f"DEBUG: DetectionModel.__init__ forcing ch=4")
    original_init(self, cfg=cfg, ch=4, nc=nc, verbose=verbose)
tasks.DetectionModel.__init__ = patched_init

original_parse_model = tasks.parse_model
def patched_parse_model(d, ch, verbose=True):
    return original_parse_model(d, ch=4, verbose=verbose)
tasks.parse_model = patched_parse_model

# --- Training Script ---

def start_training():
    print(f"Starting training in {'COLAB' if IS_COLAB else 'LOCAL'}...")
    # Use the prepared 4ch YAML
    model = YOLO("yolov8n-4ch.yaml")
    
    # Manual weight transfer from standard yolov8n
    pretrained_model = YOLO("yolov8n.pt").model
    new_model = model.model
    new_dict = new_model.state_dict()
    old_dict = pretrained_model.state_dict()
    
    update_dict = {}
    for k, v in old_dict.items():
        if k in new_dict:
            if v.shape == new_dict[k].shape:
                update_dict[k] = v
            elif k == "model.0.conv.weight":
                new_v = torch.zeros(new_dict[k].shape)
                new_v[:, :3, :, :] = v
                new_v[:, 3, :, :] = torch.mean(v, dim=1)
                update_dict[k] = new_v
    
    new_dict.update(update_dict)
    new_model.load_state_dict(new_dict)
    
    model.train(
        data=os.path.join(OUTPUT_DATA, "data.yaml"),
        epochs=50,
        imgsz=640,
        batch=16 if IS_COLAB else 4,
        name="sunrgbd_4ch_colab" if IS_COLAB else "sunrgbd_4ch_local",
        device=0 if torch.cuda.is_available() else "cpu",
        mosaic=0.0,
        mixup=0.0,
        augment=False,
        cache=True if IS_COLAB else False,
        workers=4 if IS_COLAB else 0
    )

if __name__ == "__main__":
    start_training()

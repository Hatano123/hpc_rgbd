import torch
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn import tasks

# --- Monkey Patching Dataset to support 4-channel TIFF (BGRD) ---

def custom_load_image(self, i):
    im_path = self.im_files[i]
    im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"Image Not Found: {im_path}")
    
    im = im.astype(np.float32)
    depth = im[:, :, 3]
    depth = np.clip(depth, 0, 8000) / 8000.0 * 255.0
    im[:, :, 3] = depth
    
    h0, w0 = im.shape[:2]
    return im, (h0, w0), im.shape[:2]

YOLODataset.load_image = custom_load_image

# --- Monkey Patching DetectionModel to force 4-channel setup ---

original_init = tasks.DetectionModel.__init__

def patched_init(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):
    """Force ch=4 during model initialization."""
    print(f"DEBUG: DetectionModel.__init__ called with ch={ch}, forcing ch=4")
    # Call original init but pass ch=4
    original_init(self, cfg=cfg, ch=4, nc=nc, verbose=verbose)

tasks.DetectionModel.__init__ = patched_init

# Also patch parse_model just in case
original_parse_model = tasks.parse_model
def patched_parse_model(d, ch, verbose=True):
    return original_parse_model(d, ch=4, verbose=verbose)
tasks.parse_model = patched_parse_model

# --- Training Script ---

if __name__ == "__main__":
    # Initialize from 4-ch YAML
    model = YOLO("yolov8n-4ch.yaml")
    
    # Manual weight transfer
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
                print(f"Manual weight transfer for {k}")
                new_v = torch.zeros(new_dict[k].shape)
                new_v[:, :3, :, :] = v
                new_v[:, 3, :, :] = torch.mean(v, dim=1)
                update_dict[k] = new_v
    
    new_dict.update(update_dict)
    new_model.load_state_dict(new_dict)
    
    # Train
    model.train(
        data="datasets/sunrgbd_tiff/data.yaml",
        epochs=30,
        imgsz=640,
        batch=4,
        name="sunrgbd_4ch_final_v2",
        mosaic=0.0,
        mixup=0.0,
        augment=False,
        cache=False,
        workers=0
    )
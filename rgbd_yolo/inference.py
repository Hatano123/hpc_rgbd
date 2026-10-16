"""Inference on pre-aligned data with explicit tensor normalization."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch


def predict_aligned(model, image, imgsz=640, conf=.25, device='cpu'):
    """Accept the shared HWC BGRD [0,255] representation and return original-pixel boxes."""
    if imgsz % 32:
        raise ValueError('imgsz must be a multiple of 32')
    h, w = image.shape[:2]
    ratio = min(imgsz / h, imgsz / w)
    nw, nh = round(w * ratio), round(h * ratio)
    rgb = cv2.resize(image[..., :3], (nw, nh), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(image[..., 3], (nw, nh), interpolation=cv2.INTER_NEAREST)
    resized = np.dstack([rgb, depth]).astype(np.float32)
    left, top = (imgsz - nw) // 2, (imgsz - nh) // 2
    padded = cv2.copyMakeBorder(resized, top, imgsz - nh - top, left, imgsz - nw - left,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114, 114))
    tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).unsqueeze(0) / 255
    results = model.predict(tensor, imgsz=imgsz, conf=conf, device=device, verbose=False)
    detections = []
    for box in results[0].boxes:
        coords = box.xyxy[0].cpu().numpy().copy()
        coords[[0, 2]] = np.clip((coords[[0, 2]] - left) / (nw / w), 0, w)
        coords[[1, 3]] = np.clip((coords[[1, 3]] - top) / (nh / h), 0, h)
        cls = int(box.cls.item())
        detections.append({'class_id': cls, 'name': model.names[cls],
                           'confidence': float(box.conf.item()), 'xyxy': coords.tolist()})
    return detections


def save_prediction(image, detections, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    canvas = np.clip(image[..., :3], 0, 255).astype(np.uint8)
    for item in detections:
        x1, y1, x2, y2 = map(int, item['xyxy'])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas, f"{item['name']} {item['confidence']:.2f}", (x1, max(15, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
    if not cv2.imwrite(str(output / 'prediction.jpg'), canvas):
        raise OSError('Could not save prediction image')
    (output / 'detections.json').write_text(json.dumps(detections, indent=2, ensure_ascii=False), encoding='utf-8')

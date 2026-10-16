"""Ultralytics dataset extension; no global monkey patches or epoch files."""
import math
import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset
from .preprocessing import read_tiff


class AlignedRGBDDataset(YOLODataset):
    def __init__(self, *args, preprocess, **kwargs):
        self.preprocess_config = preprocess
        super().__init__(*args, **kwargs)

    def load_image(self, i, rect_mode=True):
        if self.ims[i] is not None:
            return self.ims[i].copy(), self.im_hw0[i], self.im_hw[i]
        image = read_tiff(self.im_files[i], self.preprocess_config)
        h0, w0 = image.shape[:2]
        ratio = self.imgsz / max(h0, w0)
        size = (min(math.ceil(w0 * ratio), self.imgsz), min(math.ceil(h0 * ratio), self.imgsz)) if rect_mode else (self.imgsz, self.imgsz)
        rgb = cv2.resize(image[..., :3], size, interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(image[..., 3], size, interpolation=cv2.INTER_NEAREST)
        image = np.empty((*depth.shape, 4), dtype=np.float32)
        image[..., :3], image[..., 3] = rgb, depth
        if self.augment:
            self.ims[i], self.im_hw0[i], self.im_hw[i] = image.copy(), (h0, w0), image.shape[:2]
            self.buffer.append(i)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                j = self.buffer.pop(0)
                self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
        return image, (h0, w0), image.shape[:2]

def build_dataset(args, img_path, batch, data, mode, stride, preprocess):
    return AlignedRGBDDataset(img_path=img_path, imgsz=args.imgsz, batch_size=batch,
        augment=mode == 'train', hyp=args, rect=args.rect or mode != 'train', cache=False,
        single_cls=args.single_cls, stride=stride, pad=0.0 if mode == 'train' else 0.5,
        prefix=f'{mode}: ', task='detect', classes=args.classes, data=data,
        fraction=args.fraction if mode == 'train' else 1.0,
        preprocess=preprocess)

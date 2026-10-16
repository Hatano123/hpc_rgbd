"""One explicit input contract shared by training, validation and prediction."""
from dataclasses import asdict, dataclass
from pathlib import Path
import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class PreprocessConfig:
    rgb_divisor: float = 256.0
    depth_divisor: float = 8.0
    max_depth_mm: float = 8000.0

    def __post_init__(self):
        values = [self.rgb_divisor, self.depth_divisor, self.max_depth_mm]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Scale factors must be finite and positive")

    @classmethod
    def load(cls, path):
        return cls(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def to_dict(self):
        return asdict(self)


def scale_depth(depth_mm, config):
    depth = np.asarray(depth_mm, dtype=np.float32)
    if not np.isfinite(depth).all():
        raise ValueError('Depth contains NaN/Inf; provide valid numeric input')
    return (np.clip(depth, 0, config.max_depth_mm) / config.max_depth_mm).astype(np.float32)


def prepare_aligned(bgr_8bit, depth_mm, config):
    """Return HWC B,G,R,D float32 in [0,255]; never project or re-align pixels.

    RGB is already in 8-bit intensity units and depth is already millimetres.
    Encoded TIFF scale factors are deliberately not applied here.
    """
    rgb = np.asarray(bgr_8bit, dtype=np.float32)
    depth = np.asarray(depth_mm, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]
    if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
        raise ValueError("Expected aligned HxWx3 BGR and HxW depth with identical spatial dimensions")
    if not np.isfinite(rgb).all():
        raise ValueError('RGB contains NaN/Inf; provide valid numeric input')
    rgb = np.clip(rgb, 0, 255)
    return np.concatenate([rgb, (scale_depth(depth, config) * 255)[..., None]], axis=-1).astype(np.float32)


def decode_tiff(raw, config):
    if raw is None or raw.ndim != 3 or raw.shape[2] != 4:
        raise ValueError("Expected a four-channel TIFF; do not fabricate missing RGB or depth channels")
    return prepare_aligned(raw[..., :3].astype(np.float32) / config.rgb_divisor,
                           raw[..., 3].astype(np.float32) / config.depth_divisor, config)


def read_tiff(path, config):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    return decode_tiff(raw, config)

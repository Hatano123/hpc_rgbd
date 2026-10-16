"""Model-specific training and validation extensions."""
from copy import copy
import torch
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from .dataset import build_dataset
from .preprocessing import PreprocessConfig


class RGBDValidator(DetectionValidator):
    def __init__(self, *args, preprocess=None, **kwargs):
        self.preprocess_config = preprocess or PreprocessConfig()
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode='val', batch=None):
        return build_dataset(self.args, img_path, batch, self.data, mode, self.stride,
                             self.preprocess_config)


class RGBDTrainer(DetectionTrainer):
    def __init__(self, *args, preprocess=None, **kwargs):
        self.preprocess_config = preprocess or PreprocessConfig()
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        if self.data.get('channels') != 4:
            raise ValueError('RGB-D dataset YAML must declare channels: 4')
        model = DetectionModel(cfg, nc=self.data['nc'], ch=4, verbose=verbose)
        if weights:
            model.load(weights)
            source = weights.state_dict().get('model.0.conv.weight')
            target = model.model[0].conv.weight
            if source is not None and source.shape[1] == 3 and source.shape[0] == target.shape[0] and source.shape[2:] == target.shape[2:]:
                with torch.no_grad():
                    target[:, :3].copy_(source)
                    target[:, 3:4].copy_(source.mean(dim=1, keepdim=True))
        model.rgbd_config = self.preprocess_config.to_dict()
        model.rgbd_channel_order = 'BGRD'
        return model

    def build_dataset(self, img_path, mode='train', batch=None):
        stride = max(int(self.model.stride.max()), 32)
        return build_dataset(self.args, img_path, batch, self.data, mode, stride,
                             self.preprocess_config)

    def get_validator(self):
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return RGBDValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args),
                             _callbacks=self.callbacks, preprocess=self.preprocess_config)

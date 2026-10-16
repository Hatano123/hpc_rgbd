"""Run a small real-data training, depth-gradient and checkpoint roundtrip test."""
import argparse
from functools import partial
import json
from pathlib import Path
import shutil
import sys
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rgbd_yolo.preprocessing import PreprocessConfig, read_tiff
from rgbd_yolo.training import RGBDTrainer
from rgbd_yolo.inference import predict_aligned, save_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Dataset directory with images/train,val and labels/train,val')
    parser.add_argument('--output', default=str(ROOT / 'work/smoke_refactored'))
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = PreprocessConfig.load(ROOT / 'configs/sunrgbd_tiff.yaml')
    subset = output / 'data'
    names = yaml.safe_load((source / 'data.yaml').read_text(encoding='utf-8'))['names'] if (source / 'data.yaml').is_file() else ['chair', 'cabinet', 'picture', 'books', 'table']
    for split, count in [('train', 8), ('val', 4)]:
        images = sorted((source / 'images' / split).glob('*.tiff'))[:count]
        if len(images) != count:
            raise ValueError(f'Need {count} {split} TIFFs')
        for kind in ['images', 'labels']:
            (subset / kind / split).mkdir(parents=True, exist_ok=True)
        for image in images:
            label = source / 'labels' / split / f'{image.stem}.txt'
            if not label.is_file():
                raise FileNotFoundError(label)
            shutil.copy2(image, subset / 'images' / split / image.name)
            shutil.copy2(label, subset / 'labels' / split / label.name)
    data = subset / 'data.yaml'
    data.write_text(yaml.safe_dump({'path': str(subset), 'train': 'images/train', 'val': 'images/val',
                                  'names': names, 'nc': len(names), 'channels': 4}), encoding='utf-8')
    evidence = {'inputs': [], 'depth_gradients': [], 'losses': []}
    initial = []

    def instrument(trainer):
        conv = trainer.model.model[0].conv
        initial.append(conv.weight.detach().clone())
        def forward(module, inputs):
            if trainer.model.training:
                x = inputs[0].detach()
                assert x.shape[1] == 4 and torch.isfinite(x).all()
                evidence['inputs'].append(list(x.shape))
        def gradient(grad):
            assert torch.isfinite(grad).all()
            evidence['depth_gradients'].append(float(grad[:, 3].norm()))
        conv.register_forward_pre_hook(forward)
        conv.weight.register_hook(gradient)

    def batch(trainer):
        loss = trainer.loss_items.detach()
        assert torch.isfinite(loss).all()
        evidence['losses'].append(loss.tolist())

    def epoch(trainer):
        delta = (trainer.model.model[0].conv.weight.detach()[:, 3] - initial[0][:, 3]).abs().max()
        evidence['depth_weight_max_change'] = float(delta)

    model = YOLO('yolo11n.yaml').load(str(ROOT / 'yolo11n.pt'))
    model.add_callback('on_pretrain_routine_end', instrument)
    model.add_callback('on_train_batch_end', batch)
    model.add_callback('on_train_epoch_end', epoch)
    model.train(trainer=partial(RGBDTrainer, preprocess=config),
        data=str(data), epochs=2, imgsz=160, batch=2, workers=args.workers, device='cpu',
        project=str(output), name='train', exist_ok=True, amp=False, plots=False,
        optimizer='AdamW', lr0=.002, close_mosaic=0, mosaic=.5, mixup=0,
        hsv_h=0, hsv_s=0, hsv_v=0)
    for name in ['best.pt', 'last.pt']:
        loaded = YOLO(str(output / 'train/weights' / name))
        assert loaded.model.model[0].conv.in_channels == 4
        assert loaded.model.rgbd_config == config.to_dict()
    image = read_tiff(next((subset / 'images/val').glob('*.tiff')), config)
    detections = predict_aligned(loaded, image, imgsz=160)
    save_prediction(image, detections, output / 'prediction')
    assert len(evidence['losses']) == 8
    assert max(evidence['depth_gradients']) > 0
    assert evidence['depth_weight_max_change'] > 0
    evidence['checkpoint_preprocess_roundtrip'] = True
    evidence['passed'] = True
    (output / 'evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print('PASS:', output / 'evidence.json')


if __name__ == '__main__':
    main()

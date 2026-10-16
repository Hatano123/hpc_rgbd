"""A visible, reproducible route from local SUN RGB-D samples to evaluation."""
import hashlib
import json
from pathlib import Path
import random
import shutil
import cv2
import numpy as np
import yaml
from .preprocessing import PreprocessConfig, read_tiff


def find_source(source, repository):
    if source:
        path = Path(source).resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    candidate = repository.parent / 'YOLORGBD/datasets/sunrgbd_top5'
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError('No local SUN RGB-D data found. Supply --source /path/to/sunrgbd_top5; images are not bundled in Git.')


def read_labels(path, class_count):
    if not path.is_file():
        raise FileNotFoundError(f'Missing label: {path}')
    rows = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        fields = [float(item) for item in line.split()]
        if len(fields) != 5 or not np.isfinite(fields).all():
            raise ValueError(f'{path}:{line_number}: expected class x_center y_center width height')
        cls, x, y, w, h = fields
        if cls != int(cls) or not 0 <= cls < class_count or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f'{path}:{line_number}: invalid class or normalized box')
        rows.append(fields)
    return rows


def preview(image, labels, names, output):
    rgb = np.clip(image[..., :3], 0, 255).astype(np.uint8)
    h, w = rgb.shape[:2]
    for cls, x, y, bw, bh in labels:
        p1 = (round((x - bw / 2) * w), round((y - bh / 2) * h))
        p2 = (round((x + bw / 2) * w), round((y + bh / 2) * h))
        cv2.rectangle(rgb, p1, p2, (0, 255, 0), 2)
        cv2.putText(rgb, names[int(cls)], (max(0, p1[0]), max(35, p1[1])), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
    depth = cv2.applyColorMap(np.clip(image[..., 3], 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    for canvas, title in [(rgb, 'RGB + ground truth'), (depth, 'Scaled depth (not metric)')]:
        cv2.rectangle(canvas, (0, 0), (w, 25), (0, 0, 0), -1)
        cv2.putText(canvas, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    combined = np.hstack([rgb, depth])
    if not cv2.imwrite(str(output), combined):
        raise OSError(f'Cannot write {output}')


def prepare_subset(source, output, config, names, train_count=8, val_count=4, seed=0):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Source and output must be separate directories, not nested')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output is not empty: {output}. Choose a new --output directory.')
    if min(train_count, val_count) < 1:
        raise ValueError('train-count and val-count must be positive')
    rng = random.Random(seed)
    selected = {}
    records = []
    hashes = {}
    for split, count in [('train', train_count), ('val', val_count)]:
        images = sorted(path for path in (source / 'images' / split).glob('*') if path.suffix.lower() in {'.tiff', '.tif'})
        if len(images) < count:
            raise ValueError(f'{split}: need {count} TIFFs, found {len(images)}')
        selected[split] = sorted(rng.sample(images, count))
        hashes[split] = set()
        for path in selected[split]:
            label_path = source / 'labels' / split / (path.stem + '.txt')
            labels = read_labels(label_path, len(names))
            image = read_tiff(path, config)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[split].add(sha)
            records.append({'split': split, 'image': path.name, 'sha256': sha,
                            'shape': list(image.shape), 'objects': len(labels), 'classes': [int(row[0]) for row in labels]})
    if hashes['train'] & hashes['val']:
        raise ValueError('Selected train and val sets contain identical TIFFs')
    output.mkdir(parents=True, exist_ok=True)
    for split, images in selected.items():
        for kind in ['images', 'labels']:
            (output / kind / split).mkdir(parents=True, exist_ok=True)
        for path in images:
            shutil.copy2(path, output / 'images' / split / path.name)
            label = source / 'labels' / split / (path.stem + '.txt')
            shutil.copy2(label, output / 'labels' / split / label.name)
    # No absolute `path`: Ultralytics resolves these splits relative to this YAML.
    data = {'train': 'images/train', 'val': 'images/val', 'nc': len(names), 'names': names, 'channels': 4}
    (output / 'data.yaml').write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')
    (output / 'preprocess.yaml').write_text(yaml.safe_dump(config.to_dict()), encoding='utf-8')
    image_path = selected['val'][0]
    image = read_tiff(image_path, config)
    labels = read_labels(source / 'labels/val' / (image_path.stem + '.txt'), len(names))
    preview(image, labels, names, output / 'preview.jpg')
    np.savez_compressed(output / 'sample_preprocessed.npz', image=image, channel_order='BGRD', value_range=np.array([0, 255]))
    counts = {split: {name: sum(row['classes'].count(i) for row in records if row['split'] == split) for i, name in enumerate(names)} for split in ['train', 'val']}
    report = {'source': str(source), 'seed': seed, 'counts': {'train': train_count, 'val': val_count},
              'objects_per_class': counts, 'preprocess': config.to_dict(), 'samples': records,
              'alignment': 'assumed pre-aligned; shape validation does not verify sensor calibration'}
    (output / 'input_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def prepare_command(args):
    repository = Path(__file__).resolve().parents[1]
    source = find_source(args.source, repository)
    class_yaml = source / 'data.yaml' if (source / 'data.yaml').is_file() else Path(args.classes)
    names = yaml.safe_load(class_yaml.read_text(encoding='utf-8'))['names']
    if isinstance(names, dict):
        if sorted(names) != list(range(len(names))):
            raise ValueError('Class dictionary keys must be contiguous integers starting at zero')
        names = [names[i] for i in range(len(names))]
    result = prepare_subset(source, args.output, PreprocessConfig.load(args.config), names,
                            args.train_count, args.val_count, args.seed)
    print(f'Prepared {args.train_count} train + {args.val_count} val images: {Path(args.output).resolve()}', flush=True)
    return result


def demo(args):
    from .cli import main
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {output}')
    from types import SimpleNamespace
    print('[1/5] Input: find and validate local SUN RGB-D samples', flush=True)
    prepared = prepare_command(SimpleNamespace(**{**vars(args), 'output': str(output / 'dataset')}))
    print('[2/5] Preparation: preview.jpg, sample_preprocessed.npz, input_report.json saved', flush=True)
    print('[3/5] Training: small 4-channel YOLO11 run', flush=True)
    data = str(output / 'dataset/data.yaml')
    main(['train', '--data', data, '--config', str(output / 'dataset/preprocess.yaml'),
          '--epochs', str(args.epochs), '--imgsz', str(args.imgsz), '--batch', '2',
          '--device', args.device, '--project', str(output), '--name', 'training', '--no-plots'])
    checkpoint = str(output / 'training/weights/best.pt')
    print('[4/5] Detection: draw predictions on one held-out image', flush=True)
    sample = sorted((output / 'dataset/images/val').iterdir())[0]
    main(['predict', '--model', checkpoint, '--image', str(sample), '--imgsz', str(args.imgsz),
          '--device', args.device, '--output', str(output / 'detection')])
    print('[5/5] Evaluation: measure the held-out validation split', flush=True)
    destination = output / 'evaluation'
    main(['val', '--model', checkpoint, '--data', data, '--imgsz', str(args.imgsz),
          '--batch', '2', '--device', args.device, '--output', str(destination)])
    evaluation = json.loads((destination / 'metrics.json').read_text(encoding='utf-8'))
    result = {'input': prepared, 'checkpoint': checkpoint, 'evaluation': evaluation,
              'purpose': 'Pipeline smoke test; not an accuracy benchmark'}
    (output / 'pipeline_report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    summary = ['# SUN RGB-D quick test', '', 'Input → preparation → training → detection → evaluation completed.', '',
               f"- Samples: {prepared['counts']['train']} train / {prepared['counts']['val']} validation",
               '- Input/ground truth/depth preview: [preview](dataset/preview.jpg)',
               '- Detection: [prediction](detection/prediction.jpg)',
               '- Dataset contract: [input_report.json](dataset/input_report.json)',
               '- Full report: [pipeline_report.json](pipeline_report.json)', '',
               '| mAP50 | mAP50-95 |', '|---:|---:|']
    m = evaluation['metrics']
    summary.append(f"| {m['metrics/mAP50(B)']:.4f} | {m['metrics/mAP50-95(B)']:.4f} |")
    summary += ['', 'These numbers use a tiny validation set and do not establish accuracy or superiority over RGB.']
    (output / 'REPORT.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    print(f'Completed. Open {output / "REPORT.md"}', flush=True)
    return result

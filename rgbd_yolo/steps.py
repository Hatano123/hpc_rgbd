"""Numbered Python entry points sharing one readable settings file."""
import argparse
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import yaml
from .cli import main as run_command
from .pipeline import find_source, prepare_command
from .preprocessing import PreprocessConfig

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    source_dataset: str | None = None
    data_directory: str = 'work/quickstart/data'
    class_names: str = 'configs/class_names.yaml'
    preprocessing: str = 'configs/sunrgbd_tiff.yaml'
    train_count: int = 8
    val_count: int = 4
    seed: int = 0
    training_directory: str = 'runs/quickstart'
    epochs: int = 2
    image_size: int = 160
    batch_size: int = 2
    workers: int = 0
    device: str = 'cpu'
    model_file: str | None = None
    detection_directory: str = 'work/quickstart/detections'
    evaluation_directory: str = 'work/quickstart/evaluation'

    @classmethod
    def load(cls, path):
        values = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown settings: {sorted(unknown)}')
        result = cls(**values)
        if min(result.train_count, result.val_count, result.epochs, result.batch_size) < 1:
            raise ValueError('Image counts, epochs and batch_size must be positive')
        if result.image_size < 32 or result.image_size % 32:
            raise ValueError('image_size must be a positive multiple of 32')
        return result


def resolve_path(base, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def require(path, message):
    if not path.is_file():
        raise FileNotFoundError(f'{path}\n{message}')


def reuse_prepared_data(directory, source, settings, config, classes):
    """Reuse only the same checked subset; reject changed settings or changed copies."""
    if not directory.is_dir() or not any(directory.iterdir()):
        return False
    required = ['input_report.json', 'data.yaml', 'preprocess.yaml']
    if not all((directory / name).is_file() for name in required):
        raise ValueError('Prepared directory is incomplete. Choose another data_directory in settings.yaml.')
    report = json.loads((directory / 'input_report.json').read_text(encoding='utf-8'))
    data = yaml.safe_load((directory / 'data.yaml').read_text(encoding='utf-8'))
    class_file = source / 'data.yaml' if (source / 'data.yaml').is_file() else classes
    names = yaml.safe_load(class_file.read_text(encoding='utf-8'))['names']
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    matches = (report['source'] == str(source) and report['seed'] == settings.seed
               and report['counts'] == {'train': settings.train_count, 'val': settings.val_count}
               and report['preprocess'] == config.to_dict() and data['names'] == names
               and PreprocessConfig.load(directory / 'preprocess.yaml') == config)
    if not matches:
        raise ValueError('Preparation settings changed. Choose a new data_directory to keep the earlier data.')
    for split, count in report['counts'].items():
        actual = [p for p in (directory / 'images' / split).iterdir() if p.suffix.lower() in {'.tif', '.tiff'}]
        if len(actual) != count:
            raise ValueError('Prepared image count changed; use a new data_directory.')
    for sample in report['samples']:
        image = Path('images') / sample['split'] / sample['image']
        label = Path('labels') / sample['split'] / (Path(sample['image']).stem + '.txt')
        if (hashlib.sha256((directory / image).read_bytes()).hexdigest() != sample['sha256']
                or hashlib.sha256((source / image).read_bytes()).hexdigest() != sample['sha256']
                or (directory / label).read_bytes() != (source / label).read_bytes()):
            raise ValueError('Input or prepared sample changed; use a new data_directory.')
    print(f'01: Checked and reused the existing prepared data: {directory}', flush=True)
    return True


def run_step(step, settings_file):
    settings_file = Path(settings_file).resolve()
    settings = Settings.load(settings_file)
    base = settings_file.parent
    data_dir = resolve_path(base, settings.data_directory)
    train_dir = resolve_path(base, settings.training_directory)
    data_yaml = data_dir / 'data.yaml'
    config_yaml = data_dir / 'preprocess.yaml'
    model = resolve_path(base, settings.model_file) if settings.model_file else train_dir / 'weights/best.pt'
    if step == 'prepare':
        supplied = str(resolve_path(base, settings.source_dataset)) if settings.source_dataset else None
        source = find_source(supplied, ROOT)
        config_path = resolve_path(base, settings.preprocessing)
        classes = resolve_path(base, settings.class_names)
        if reuse_prepared_data(data_dir, source, settings, PreprocessConfig.load(config_path), classes):
            return
        return prepare_command(SimpleNamespace(source=str(source), output=str(data_dir), config=str(config_path),
            classes=str(classes), train_count=settings.train_count, val_count=settings.val_count, seed=settings.seed))
    require(data_yaml, 'Run python 01_prepare_data.py first.')
    require(config_yaml, 'Run python 01_prepare_data.py first.')
    if step == 'train':
        print(f'02: Train a 4-channel model. Results: {train_dir}', flush=True)
        return run_command(['train', '--data', str(data_yaml), '--config', str(config_yaml),
            '--project', str(train_dir.parent), '--name', train_dir.name,
            '--epochs', str(settings.epochs), '--imgsz', str(settings.image_size),
            '--batch', str(settings.batch_size), '--workers', str(settings.workers),
            '--device', str(settings.device), '--no-plots'])
    require(model, 'Run python 02_train_model.py first, or set model_file in settings.yaml.')
    if step == 'detect':
        destination = resolve_path(base, settings.detection_directory)
        print(f'03: Detect objects. Results: {destination}', flush=True)
        return run_command(['predict', '--model', str(model), '--images', str(data_dir / 'images/val'),
            '--imgsz', str(settings.image_size), '--device', str(settings.device), '--output', str(destination)])
    if step == 'evaluate':
        destination = resolve_path(base, settings.evaluation_directory)
        print(f'04: Evaluate against ground truth. Results: {destination}', flush=True)
        return run_command(['val', '--model', str(model), '--data', str(data_yaml), '--batch', str(settings.batch_size),
            '--imgsz', str(settings.image_size), '--device', str(settings.device), '--output', str(destination)])
    raise ValueError(f'Unknown step: {step}')


def main(step, argv=None):
    parser = argparse.ArgumentParser(description=f'{step}: execute one stage using settings.yaml')
    parser.add_argument('--settings', default=str(ROOT / 'settings.yaml'), help='Shared settings file')
    args = parser.parse_args(argv)
    return run_step(step, args.settings)

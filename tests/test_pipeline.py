from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from dataclasses import replace
from rgbd_yolo.pipeline import prepare_subset, read_labels
from rgbd_yolo.preprocessing import PreprocessConfig
from rgbd_yolo.steps import Settings, reuse_prepared_data


class PipelineTests(unittest.TestCase):
    def test_preparation_validates_copies_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / 'source', root / 'subset'
            for split, offset in [('train', 1), ('val', 10)]:
                for kind in ['images', 'labels']:
                    (source / kind / split).mkdir(parents=True)
                for i in range(3):
                    raw = np.full((32, 40, 4), (offset + i) * 256, np.uint16)
                    self.assertTrue(cv2.imwrite(str(source / 'images' / split / f'{i}.tiff'), raw))
                    (source / 'labels' / split / f'{i}.txt').write_text('0 0.5 0.5 0.2 0.3\n')
            before = {path: path.read_bytes() for path in source.rglob('*') if path.is_file()}
            report = prepare_subset(source, output, PreprocessConfig(), ['chair'], 2, 1, 3)
            self.assertEqual(report['counts'], {'train': 2, 'val': 1})
            self.assertTrue((output / 'preview.jpg').is_file())
            self.assertEqual(np.load(output / 'sample_preprocessed.npz')['image'].shape, (32, 40, 4))
            self.assertEqual(before, {path: path.read_bytes() for path in source.rglob('*') if path.is_file()})
            classes = root / 'classes.yaml'
            classes.write_text('names: [chair]\n')
            settings = Settings(train_count=2, val_count=1, seed=3)
            self.assertTrue(reuse_prepared_data(output, source, settings, PreprocessConfig(), classes))
            with self.assertRaises(ValueError):
                reuse_prepared_data(output, source, replace(settings, seed=4), PreprocessConfig(), classes)
            copied = next((output / 'images/train').glob('*.tiff'))
            copied.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                reuse_prepared_data(output, source, settings, PreprocessConfig(), classes)
            with self.assertRaises(FileExistsError):
                prepare_subset(source, output, PreprocessConfig(), ['chair'], 2, 1)
            with self.assertRaises(ValueError):
                prepare_subset(source, source / 'nested', PreprocessConfig(), ['chair'])

    def test_invalid_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            label = Path(temp) / 'label.txt'
            for row in ['5 .5 .5 .2 .2', '0 .5 .5 nan .2', '0 .5 .5 -.2 .2', '0 .5 .5']:
                label.write_text(row)
                with self.assertRaises(ValueError):
                    read_labels(label, 1)


if __name__ == '__main__':
    unittest.main()

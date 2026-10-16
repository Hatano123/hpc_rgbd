from pathlib import Path
import tempfile
import unittest
import numpy as np
from rgbd_yolo.camera import save_capture
from rgbd_yolo.preprocessing import PreprocessConfig, prepare_aligned, read_tiff


class CameraRecordingTests(unittest.TestCase):
    def test_capture_roundtrip_preserves_color_and_depth_without_filling(self):
        bgr = np.array([[[12, 34, 56], [120, 180, 220], [5, 6, 7]]], dtype=np.uint8)
        depth = np.array([[0, 1234.5, 9000]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp:
            path = save_capture(Path(temp), 0, bgr, depth, {'adapter': 'test'})
            restored = read_tiff(path, PreprocessConfig())
            expected = prepare_aligned(bgr, depth, PreprocessConfig())
            np.testing.assert_allclose(restored, expected)
            self.assertEqual(restored[0, 0, 3], 0)
            self.assertTrue((Path(temp) / 'rgb/frame_000000.png').is_file())
            self.assertTrue((Path(temp) / 'depth_mm/frame_000000.png').is_file())
            with self.assertRaises(FileExistsError):
                save_capture(Path(temp), 0, bgr, depth, {})

    def test_capture_rejects_unaligned_or_invalid_depth(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                save_capture(temp, 0, np.zeros((3, 4, 3), np.uint8), np.zeros((2, 4)), {})
            with self.assertRaises(ValueError):
                save_capture(temp, 0, np.zeros((1, 1, 3), np.uint8), np.array([[np.nan]]), {})


if __name__ == '__main__':
    unittest.main()

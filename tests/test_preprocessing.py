import unittest
import numpy as np
from rgbd_yolo.preprocessing import PreprocessConfig, decode_tiff, prepare_aligned, scale_depth


class PreprocessingTests(unittest.TestCase):
    def test_encoded_tiff_matches_millimetre_pair(self):
        bgr = np.array([[[10, 20, 30], [100, 150, 200]]], dtype=np.uint16)
        mm = np.array([[1000, 8000]], dtype=np.uint16)
        raw = np.dstack([bgr * 256, mm * 8])
        config = PreprocessConfig()
        np.testing.assert_allclose(decode_tiff(raw, config), prepare_aligned(bgr, mm, config))
        np.testing.assert_array_equal(decode_tiff(raw, config)[..., :3], bgr)
        np.testing.assert_array_equal(prepare_aligned(bgr, mm[..., None], config), prepare_aligned(bgr, mm, config))

    def test_depth_scale_limits(self):
        output = scale_depth(np.array([0, 1000, 8000, 10000]), PreprocessConfig())
        np.testing.assert_allclose(output, [0, .125, 1, 1])
        with self.assertRaises(ValueError):
            scale_depth(np.array([np.nan]), PreprocessConfig())

    def test_reject_misaligned_shapes_and_missing_channels(self):
        with self.assertRaises(ValueError):
            prepare_aligned(np.zeros((4, 5, 3)), np.zeros((3, 5)), PreprocessConfig())
        with self.assertRaises(ValueError):
            decode_tiff(np.zeros((4, 5, 3)), PreprocessConfig())

    def test_depth_is_only_scaled_and_never_filled(self):
        config = PreprocessConfig()
        result = prepare_aligned(np.zeros((1, 3, 3)), np.array([[1000, 0, 2000]]), config)
        np.testing.assert_allclose(result[..., 3], [[31.875, 0, 63.75]])


if __name__ == '__main__':
    unittest.main()

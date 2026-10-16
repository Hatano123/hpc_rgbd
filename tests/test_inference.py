from types import SimpleNamespace
import unittest
import numpy as np
import torch
from rgbd_yolo.inference import predict_aligned


class InferenceContractTests(unittest.TestCase):
    def test_bgrd_tensor_and_original_coordinates(self):
        captured = []
        class Model:
            names = {0: 'chair'}
            def predict(self, image, **kwargs):
                captured.append(image)
                box = SimpleNamespace(xyxy=torch.tensor([[16., 48., 144., 112.]]),
                                      cls=torch.tensor([0.]), conf=torch.tensor([.8]))
                return [SimpleNamespace(boxes=[box])]
        image = np.empty((100, 200, 4), dtype=np.float32)
        image[...] = [10, 20, 30, 100]
        boxes = predict_aligned(Model(), image, imgsz=160)
        np.testing.assert_allclose(captured[0][0, :, 80, 80].numpy(), np.array([10, 20, 30, 100]) / 255)
        np.testing.assert_allclose(boxes[0]['xyxy'], [20, 10, 180, 90])
        self.assertEqual(list(captured[0].shape), [1, 4, 160, 160])


if __name__ == '__main__':
    unittest.main()

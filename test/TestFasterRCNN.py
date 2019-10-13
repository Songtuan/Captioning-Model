import unittest
import torch

from modules.FasterRCNN import FasterRCNN


class MyTestCase(unittest.TestCase):
    def test_faster_rcnn(self):
        faster_rcnn = FasterRCNN()
        batch_size = 2
        test_input = [torch.rand(3, 255, 255), torch.rand(3, 255, 255)]
        features = faster_rcnn(test_input)
        features = features.view(batch_size, 100, -1)
        self.assertEqual(features.shape, torch.Size([2, 100, 2048]))


if __name__ == '__main__':
    unittest.main()

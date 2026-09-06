import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from detection import MODEL_NAME as DETECTION_MODEL, reading_order
from main import models
from recognition import available_models


class ReadingOrderTest(unittest.TestCase):
    def test_top_to_bottom_then_right_to_left(self):
        boxes = [
            np.array([[10, 50], [30, 50], [30, 70], [10, 70]], dtype=np.float32),
            np.array([[70, 10], [90, 10], [90, 30], [70, 30]], dtype=np.float32),
            np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float32),
        ]
        self.assertEqual(reading_order(boxes), [1, 2, 0])


class ModelDiscoveryTest(unittest.TestCase):
    def test_only_paddle_detector_is_available(self):
        self.assertEqual(models()["detection"], [DETECTION_MODEL])

    def test_default_and_named_models(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "recognition" / "model_config.yaml").parent.mkdir()
            (root / "recognition" / "model_config.yaml").touch()
            (root / "recognition" / "ptdr").mkdir()
            (root / "recognition" / "ptdr" / "model_config.yaml").touch()
            self.assertEqual(list(available_models(root / "recognition")), ["default", "ptdr"])


if __name__ == "__main__":
    unittest.main()

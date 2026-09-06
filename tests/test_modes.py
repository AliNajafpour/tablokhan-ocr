import unittest
from unittest.mock import patch

import numpy as np

from main import run_ocr


class ModeTest(unittest.TestCase):
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    quad = np.float32([[1, 1], [30, 1], [30, 10], [1, 10]])

    @patch("main.detect", return_value=([quad], [0.9]))
    @patch("main.recognize")
    def test_detection_only_skips_recognition(self, recognize, _detect):
        result = run_ocr(self.image, "detection")
        recognize.assert_not_called()
        self.assertEqual((result["mode"], result["n_boxes"]), ("detection", 1))

    @patch("main.recognition_models", return_value={"test": None})
    @patch("main.recognize", return_value=["متن"])
    @patch("main.detect")
    def test_recognition_only_skips_detection(self, detect, _recognize, _models):
        result = run_ocr(self.image, "recognition", recognition_model="test")
        detect.assert_not_called()
        self.assertEqual(result["text"], "متن")

    @patch("main.recognize", return_value=["متن"])
    @patch("main.detect", return_value=([quad], [0.9]))
    def test_full_ocr_runs_both(self, detect, recognize):
        result = run_ocr(self.image, "ocr")
        detect.assert_called_once()
        recognize.assert_called_once()
        self.assertEqual(result["text"], "متن")


if __name__ == "__main__":
    unittest.main()

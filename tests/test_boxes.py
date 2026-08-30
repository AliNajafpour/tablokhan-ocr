import unittest

import numpy as np

from app.boxes import reading_order


class ReadingOrderTest(unittest.TestCase):
    def test_top_to_bottom_then_right_to_left(self):
        boxes = [
            np.array([[10, 50], [30, 50], [30, 70], [10, 70]], dtype=np.float32),
            np.array([[70, 10], [90, 10], [90, 30], [70, 30]], dtype=np.float32),
            np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float32),
        ]
        self.assertEqual(reading_order(boxes), [1, 2, 0])


if __name__ == "__main__":
    unittest.main()

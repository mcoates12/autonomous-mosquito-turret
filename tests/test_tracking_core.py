import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from tracking_core import EveryNFrames, TargetObservation


class EveryNFramesTests(unittest.TestCase):
    def test_first_frame_is_due_then_repeats_at_requested_cadence(self):
        cadence = EveryNFrames(3)
        self.assertEqual(
            [cadence.step() for _ in range(7)],
            [True, False, False, True, False, False, True],
        )

    def test_rejects_invalid_cadence(self):
        with self.assertRaises(ValueError):
            EveryNFrames(0)


class TargetObservationTests(unittest.TestCase):
    def test_centroid_rounds_model_output_to_image_pixels(self):
        target = TargetObservation(
            x=100.6,
            y=50.4,
            confidence=0.9,
            timestamp=123.0,
            label="mosquito",
        )
        self.assertEqual(target.centroid, (101, 50))


if __name__ == "__main__":
    unittest.main()

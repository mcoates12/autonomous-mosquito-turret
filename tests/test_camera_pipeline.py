import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from camera_pipeline import capture_pipeline_candidates, rolling_rate_hz


class CapturePipelineTests(unittest.TestCase):
    def test_prefers_nvidia_vic_and_retains_software_fallback(self):
        candidates = capture_pipeline_candidates(
            "/dev/video1", 1920, 1200, 60
        )

        self.assertEqual(
            [candidate.name for candidate in candidates],
            ["nvidia-vic", "software-videoconvert"],
        )
        accelerated = candidates[0].pipeline
        self.assertIn("nvv4l2camerasrc device=/dev/video1", accelerated)
        self.assertIn(
            "video/x-raw(memory:NVMM), format=(string)UYVY",
            accelerated,
        )
        self.assertIn("nvvidconv", accelerated)
        self.assertIn(
            "width=(int)1920, height=(int)1200, "
            "framerate=(fraction)60/1",
            accelerated,
        )
        self.assertTrue(
            accelerated.endswith("sync=false processing-deadline=0")
        )

        fallback = candidates[1].pipeline
        self.assertIn("v4l2src device=/dev/video1 io-mode=2", fallback)
        self.assertIn("format=(string)UYVY", fallback)
        self.assertIn("videoconvert", fallback)

    def test_rolling_rate_uses_elapsed_capture_time(self):
        timestamps = [10.0, 10.02, 10.04, 10.06]
        self.assertAlmostEqual(rolling_rate_hz(timestamps), 50.0)
        self.assertEqual(rolling_rate_hz([10.0]), 0.0)


if __name__ == "__main__":
    unittest.main()

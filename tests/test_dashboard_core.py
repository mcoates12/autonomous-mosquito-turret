import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dashboard_core import apply_parameter_updates, coerce_parameter


class FakeStore:
    def __init__(self):
        self.values = {}

    def set_attr(self, name, value):
        self.values[name] = value


class ParameterValidationTests(unittest.TestCase):
    def test_normalizes_float_and_accepts_valid_choice(self):
        self.assertEqual(coerce_parameter("smoothing_tau_ms", 60), 60.0)
        self.assertEqual(coerce_parameter("track_source", "Right"), "Right")
        self.assertEqual(coerce_parameter("local_red_contrast_gate", 6), 6)
        self.assertEqual(coerce_parameter("laser_edge_margin_px", 48), 48)

    def test_rejects_unknown_wrong_type_and_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "unknown parameter"):
            coerce_parameter("not_a_real_control", 1)
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            coerce_parameter("manual_exposure", 1)
        with self.assertRaisesRegex(ValueError, "at most 160"):
            coerce_parameter("exposure_time_absolute", 312)
        with self.assertRaisesRegex(ValueError, "at most 255"):
            coerce_parameter("local_contrast_gate", 256)
        with self.assertRaisesRegex(ValueError, "at most 255"):
            coerce_parameter("local_red_contrast_gate", 256)
        with self.assertRaisesRegex(ValueError, "at most 480"):
            coerce_parameter("laser_edge_margin_px", 481)

    def test_validates_entire_request_before_mutating_store(self):
        store = FakeStore()
        with self.assertRaises(ValueError):
            apply_parameter_updates(
                store,
                {"v_thresh": 90, "profile_velocity": 5000},
            )
        self.assertEqual(store.values, {})

    def test_applies_valid_batch(self):
        store = FakeStore()
        result = apply_parameter_updates(
            store,
            {"v_thresh": 90, "servos_enabled": False},
        )
        self.assertEqual(result, {"v_thresh": 90, "servos_enabled": False})
        self.assertEqual(store.values, result)


if __name__ == "__main__":
    unittest.main()

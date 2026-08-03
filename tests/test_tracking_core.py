import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from tracking_core import (
    EveryNFrames,
    TargetObservation,
    TimeBasedTargetFilter,
    degrees_to_position_ticks,
    position_ticks_to_degrees,
    sanitize_motion_profile,
    tracking_delta_degrees,
    within_reacquisition_radius,
)


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


class TimeBasedTargetFilterTests(unittest.TestCase):
    def test_smoothing_depends_on_elapsed_time_not_frame_count(self):
        def run(rate_hz):
            point_filter = TimeBasedTargetFilter(lock_time_sec=0.0)
            point_filter.update(0.0, 0.0, 0.0)
            result = None
            for frame in range(1, int(rate_hz * 0.1) + 1):
                result = point_filter.update(100.0, 0.0, frame / rate_hz)
            return result.x

        self.assertAlmostEqual(run(30.0), run(60.0), places=6)

    def test_lock_uses_elapsed_time(self):
        point_filter = TimeBasedTargetFilter(lock_time_sec=0.05)
        self.assertFalse(point_filter.update(10.0, 10.0, 1.0).locked)
        self.assertFalse(point_filter.update(10.0, 10.0, 1.04).locked)
        self.assertTrue(point_filter.update(10.0, 10.0, 1.05).locked)

    def test_outlier_is_rejected_then_reacquired_after_timeout(self):
        point_filter = TimeBasedTargetFilter(
            reset_after_sec=0.15,
            max_speed_px_sec=100.0,
            jump_allowance_px=5.0,
        )
        point_filter.update(10.0, 10.0, 1.0)
        rejected = point_filter.update(1000.0, 1000.0, 1.01)
        self.assertFalse(rejected.accepted)
        self.assertTrue(rejected.outlier)
        reacquired = point_filter.update(1000.0, 1000.0, 1.20)
        self.assertTrue(reacquired.accepted)
        self.assertAlmostEqual(reacquired.x, 1000.0)


class PositionConversionTests(unittest.TestCase):
    def test_inclusive_degree_endpoints_do_not_wrap(self):
        self.assertEqual(degrees_to_position_ticks(0.0), 0)
        self.assertEqual(degrees_to_position_ticks(360.0), 4095)

    def test_conversion_clamps_and_round_trips(self):
        self.assertEqual(degrees_to_position_ticks(-1.0), 0)
        self.assertEqual(degrees_to_position_ticks(361.0), 4095)
        self.assertAlmostEqual(position_ticks_to_degrees(4095), 360.0)

    def test_invalid_position_cannot_turn_into_an_endpoint_command(self):
        with self.assertRaises(ValueError):
            degrees_to_position_ticks(float("nan"))


class MotionProfileTests(unittest.TestCase):
    def test_profile_never_uses_zero_or_excessive_acceleration(self):
        self.assertEqual(sanitize_motion_profile(200, 30), (200, 30))
        self.assertEqual(sanitize_motion_profile(200, 150), (200, 100))
        self.assertEqual(sanitize_motion_profile(0, 0), (2, 1))


class TrackingDeltaTests(unittest.TestCase):
    def test_motion_is_independent_of_control_frequency(self):
        at_60_hz = tracking_delta_degrees(100, 0.006, 1, 1 / 60, 2.0)
        at_120_hz = tracking_delta_degrees(100, 0.006, 1, 1 / 120, 2.0)
        at_5_hz = tracking_delta_degrees(100, 0.006, 1, 1 / 5, 2.0)
        self.assertAlmostEqual(at_60_hz, 2 * at_120_hz)
        self.assertAlmostEqual(at_5_hz, 12 * at_60_hz)

    def test_motion_respects_legacy_step_limit(self):
        delta = tracking_delta_degrees(1000, 0.05, 1, 1 / 60, 2.0)
        self.assertAlmostEqual(delta, 2.0)

    def test_invalid_detector_error_commands_no_motion(self):
        delta = tracking_delta_degrees(float("nan"), 0.006, 1, 1 / 60, 2.0)
        self.assertEqual(delta, 0.0)


class ReacquisitionRadiusTests(unittest.TestCase):
    def test_unrestricted_acquisition_without_existing_identity(self):
        self.assertTrue(within_reacquisition_radius(900, 700, None, 300))
        self.assertTrue(
            within_reacquisition_radius(900, 700, (100, 100), None)
        )

    def test_accepts_nearby_candidate_and_rejects_distant_distractor(self):
        self.assertTrue(
            within_reacquisition_radius(400, 500, (100, 500), 300)
        )
        self.assertFalse(
            within_reacquisition_radius(401, 500, (100, 500), 300)
        )

    def test_active_gate_fails_closed_for_invalid_coordinates(self):
        self.assertFalse(
            within_reacquisition_radius(float("nan"), 500, (100, 500), 300)
        )


if __name__ == "__main__":
    unittest.main()

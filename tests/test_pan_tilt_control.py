import sys
import threading
import time
import types
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    import dynamixel_sdk  # noqa: F401
except ModuleNotFoundError:
    sdk_stub = types.ModuleType("dynamixel_sdk")
    sdk_stub.GroupSyncRead = object
    sdk_stub.GroupSyncWrite = object
    sdk_stub.PacketHandler = object
    sdk_stub.PortHandler = object
    sys.modules["dynamixel_sdk"] = sdk_stub

from pan_tilt_control import FixedRatePanTiltController
from tracking_core import ControlTarget


class FakeParams:
    rate_hz = 120.0
    profile_velocity = 200
    profile_acceleration = 30
    tracking_enabled = True
    servos_enabled = True
    deadband_px = 0
    deg_per_px_pan = 0.006
    deg_per_px_tilt = 0.006
    pan_dir = 1
    tilt_dir = 1
    max_step_deg = 2.0


class FakeStore:
    def __init__(self):
        self.params = FakeParams()

    def get(self):
        return self.params


class FakeTurret:
    def __init__(self):
        self.pan_cmd = 180.0
        self.tilt_cmd = 120.0
        self.pan_actual = 180.0
        self.tilt_actual = 120.0
        self.pan_velocity = 0.0
        self.tilt_velocity = 0.0
        self.torque = False
        self.move_event = threading.Event()
        self.torque_off_event = threading.Event()
        self.last_profile = None

    def read_feedback(self):
        return (
            self.pan_actual,
            self.tilt_actual,
            self.pan_velocity,
            self.tilt_velocity,
        )

    def set_motion_profile(self, velocity, acceleration):
        self.last_profile = (velocity, acceleration)

    def send(self, pan, tilt):
        moved = pan != self.pan_cmd or tilt != self.tilt_cmd
        self.pan_cmd = pan
        self.tilt_cmd = tilt
        if moved:
            self.move_event.set()

    def torque_on(self):
        self.torque = True

    def torque_off(self, best_effort=True):
        self.torque = False
        self.torque_off_event.set()


class FixedRateControllerTests(unittest.TestCase):
    def test_controller_moves_latest_target_then_really_torques_off(self):
        turret = FakeTurret()
        store = FakeStore()
        controller = FixedRatePanTiltController(turret, store)
        controller.publish_target(
            ControlTarget(100.0, 50.0, time.monotonic(), locked=True)
        )
        controller.start()
        try:
            self.assertTrue(turret.move_event.wait(0.5))
            self.assertEqual(turret.last_profile, (200, 30))
            self.assertGreater(turret.pan_cmd, 180.0)

            turret.torque_off_event.clear()
            store.params.tracking_enabled = False
            self.assertTrue(turret.torque_off_event.wait(0.5))
            self.assertFalse(turret.torque)
        finally:
            controller.stop()

    def test_stale_target_is_held_without_motion(self):
        turret = FakeTurret()
        store = FakeStore()
        controller = FixedRatePanTiltController(turret, store)
        controller.publish_target(
            ControlTarget(100.0, 50.0, time.monotonic() - 1.0, locked=True)
        )
        controller.start()
        try:
            self.assertFalse(turret.move_event.wait(0.1))
            self.assertEqual(turret.pan_cmd, 180.0)
            self.assertEqual(turret.tilt_cmd, 120.0)
        finally:
            controller.stop()


if __name__ == "__main__":
    unittest.main()

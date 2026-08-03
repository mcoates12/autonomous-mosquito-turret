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

from pan_tilt_control import (
    ADDR_HARDWARE_ERROR_STATUS,
    ADDR_MAX_POSITION_LIMIT,
    ADDR_MIN_POSITION_LIMIT,
    APPLICATION_TILT_MIN_TICKS,
    DynamixelPanTilt,
    DynamixelCommunicationError,
    FixedRatePanTiltController,
    HARDWARE_ERROR_VALID_MASK,
    ServoHealth,
)
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
    pan_damping_gain = 0.0
    tilt_damping_gain = 0.0
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
        self.pan_health = ServoHealth()
        self.tilt_health = ServoHealth()
        self.move_event = threading.Event()
        self.torque_off_event = threading.Event()
        self.last_profile = None
        self.feedback_error = None
        self.feedback_calls = 0
        self.health_calls = 0
        self.send_calls = 0

    def read_feedback(self):
        self.feedback_calls += 1
        if self.feedback_error is not None:
            raise self.feedback_error
        return (
            self.pan_actual,
            self.tilt_actual,
            self.pan_velocity,
            self.tilt_velocity,
        )

    def set_motion_profile(self, velocity, acceleration):
        self.last_profile = (velocity, acceleration)

    def read_health(self):
        self.health_calls += 1
        return self.pan_health, self.tilt_health

    def send(self, pan, tilt):
        self.send_calls += 1
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
    def test_command_rate_is_decoupled_from_feedback_and_health_reads(self):
        turret = FakeTurret()
        store = FakeStore()
        controller = FixedRatePanTiltController(turret, store)
        controller.publish_target(
            ControlTarget(100.0, 50.0, time.monotonic(), locked=True)
        )
        controller.start()
        try:
            self.assertTrue(turret.move_event.wait(0.5))
            time.sleep(0.12)
            self.assertGreater(turret.send_calls, turret.feedback_calls)
            self.assertLessEqual(turret.feedback_calls, 8)
            self.assertLessEqual(turret.health_calls, 2)
        finally:
            controller.stop()

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

    def test_hardware_error_torques_off_and_latches_controller_error(self):
        turret = FakeTurret()
        turret.pan_health = ServoHealth(hardware_error=0x20)
        store = FakeStore()
        controller = FixedRatePanTiltController(turret, store)
        controller.start()
        try:
            self.assertTrue(turret.torque_off_event.wait(0.5))
            deadline = time.monotonic() + 0.5
            while controller.snapshot().error is None and time.monotonic() < deadline:
                time.sleep(0.005)
            snapshot = controller.snapshot()
            self.assertIsNotNone(snapshot.error)
            self.assertIn("hardware error", snapshot.error)
            self.assertFalse(snapshot.error_recoverable)
            self.assertFalse(snapshot.torque_enabled)
        finally:
            controller.stop()

    def test_inactive_communication_loss_is_recoverable(self):
        turret = FakeTurret()
        turret.feedback_error = DynamixelCommunicationError(
            "sync_read COMM FAIL: no status packet"
        )
        store = FakeStore()
        store.params.tracking_enabled = False
        controller = FixedRatePanTiltController(turret, store)
        controller.start()
        try:
            deadline = time.monotonic() + 0.5
            while (
                controller.snapshot().error is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            snapshot = controller.snapshot()
            self.assertIsNotNone(snapshot.error)
            self.assertTrue(snapshot.error_recoverable)
            self.assertFalse(snapshot.torque_enabled)
        finally:
            controller.stop()

    def test_active_communication_loss_stays_latched(self):
        turret = FakeTurret()
        turret.feedback_error = DynamixelCommunicationError(
            "sync_read COMM FAIL: no status packet"
        )
        store = FakeStore()
        controller = FixedRatePanTiltController(turret, store)
        controller.start()
        try:
            deadline = time.monotonic() + 0.5
            while (
                controller.snapshot().error is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            snapshot = controller.snapshot()
            self.assertIsNotNone(snapshot.error)
            self.assertFalse(snapshot.error_recoverable)
            self.assertFalse(snapshot.torque_enabled)
        finally:
            controller.stop()


class ConfiguredLimitTests(unittest.TestCase):
    def test_tilt_uses_conservative_application_minimum(self):
        turret = object.__new__(DynamixelPanTilt)
        turret.pan_id = 1
        turret.tilt_id = 2
        turret.position_limits_ticks = {1: (0, 4095), 2: (0, 4095)}
        turret.configured_position_limits_ticks = dict(turret.position_limits_ticks)

        values = {
            (1, ADDR_MIN_POSITION_LIMIT): 0,
            (1, ADDR_MAX_POSITION_LIMIT): 4095,
            (2, ADDR_MIN_POSITION_LIMIT): 0,
            (2, ADDR_MAX_POSITION_LIMIT): 2190,
        }
        turret._read4 = lambda servo_id, address, what: values[(servo_id, address)]

        turret._read_configured_limits()

        self.assertEqual(turret.position_limits_ticks[1], (0, 4095))
        self.assertEqual(
            turret.position_limits_ticks[2],
            (APPLICATION_TILT_MIN_TICKS, 2190),
        )


class HealthReadTests(unittest.TestCase):
    def test_health_sync_read_also_refreshes_position_and_velocity(self):
        class FakeSyncRead:
            @staticmethod
            def txRxPacket():
                return 0

            @staticmethod
            def isAvailable(servo_id, address, length):
                return True

            @staticmethod
            def getData(servo_id, address, length):
                values = {
                    (1, 126): 100,
                    (1, 128): 10,
                    (1, 132): 2048,
                    (1, 144): 120,
                    (1, 146): 40,
                    (2, 126): 0xFFFF,
                    (2, 128): 0xFFFFFFF6,
                    (2, 132): 1024,
                    (2, 144): 119,
                    (2, 146): 41,
                }
                return values[(servo_id, address)]

        turret = object.__new__(DynamixelPanTilt)
        turret.pan_id = 1
        turret.tilt_id = 2
        turret.sync_read_health = FakeSyncRead()
        turret.packet_handler = types.SimpleNamespace(
            getTxRxResult=lambda comm: "communication error"
        )
        turret._read1 = lambda *args, **kwargs: 0

        pan_health, tilt_health = turret.read_health()

        self.assertAlmostEqual(pan_health.load_percent, 10.0)
        self.assertAlmostEqual(tilt_health.load_percent, -0.1)
        self.assertAlmostEqual(pan_health.input_voltage, 12.0)
        self.assertAlmostEqual(tilt_health.input_voltage, 11.9)
        self.assertAlmostEqual(turret.pan_actual, 180.043956, places=5)
        self.assertAlmostEqual(turret.tilt_actual, 90.021978, places=5)
        self.assertGreater(turret.pan_velocity, 0.0)
        self.assertLess(turret.tilt_velocity, 0.0)


class SyncReadRetryTests(unittest.TestCase):
    def test_feedback_read_retries_one_missed_status_packet(self):
        class FakeSyncRead:
            def __init__(self):
                self.results = [-3001, 0]
                self.calls = 0

            def txRxPacket(self):
                result = self.results[min(self.calls, len(self.results) - 1)]
                self.calls += 1
                return result

            def isAvailable(self, servo_id, address, length):
                return True

            def getData(self, servo_id, address, length):
                return 0 if address == 128 else 2048

        class FakePacketHandler:
            @staticmethod
            def getTxRxResult(comm):
                return "There is no status packet!"

        turret = object.__new__(DynamixelPanTilt)
        turret.pan_id = 1
        turret.tilt_id = 2
        turret.sync_read_state = FakeSyncRead()
        turret.packet_handler = FakePacketHandler()

        pan, tilt, pan_velocity, tilt_velocity = turret.read_feedback()

        self.assertEqual(turret.sync_read_state.calls, 2)
        self.assertAlmostEqual(pan, tilt)
        self.assertEqual((pan_velocity, tilt_velocity), (0.0, 0.0))

    def test_hardware_status_read_recovers_from_transient_bad_packets(self):
        class FakePacketHandler:
            def __init__(self):
                self.results = [
                    (0, -3002, 0),
                    (0, -3002, 0),
                    (0x20, 0, 0),
                ]
                self.calls = 0

            def read1ByteTxRx(self, port_handler, servo_id, address):
                result = self.results[self.calls]
                self.calls += 1
                return result

            @staticmethod
            def getTxRxResult(comm):
                return "Incorrect status packet!"

        turret = object.__new__(DynamixelPanTilt)
        turret.port_handler = object()
        turret.packet_handler = FakePacketHandler()

        value = turret._read1(
            1,
            ADDR_HARDWARE_ERROR_STATUS,
            "read hardware error status",
            valid_mask=HARDWARE_ERROR_VALID_MASK,
        )

        self.assertEqual(value, 0x20)
        self.assertEqual(turret.packet_handler.calls, 3)

    def test_hardware_status_read_raises_after_persistent_bad_packets(self):
        class FakePacketHandler:
            def __init__(self):
                self.calls = 0

            def read1ByteTxRx(self, port_handler, servo_id, address):
                self.calls += 1
                return 0, -3002, 0

            @staticmethod
            def getTxRxResult(comm):
                return "Incorrect status packet!"

        turret = object.__new__(DynamixelPanTilt)
        turret.port_handler = object()
        turret.packet_handler = FakePacketHandler()

        with self.assertRaisesRegex(
            DynamixelCommunicationError,
            "Incorrect status packet",
        ):
            turret._read1(
                1,
                ADDR_HARDWARE_ERROR_STATUS,
                "read hardware error status",
                valid_mask=HARDWARE_ERROR_VALID_MASK,
            )

        self.assertEqual(turret.packet_handler.calls, 3)

    def test_hardware_status_read_does_not_retry_device_error(self):
        class FakePacketHandler:
            def __init__(self):
                self.calls = 0

            def read1ByteTxRx(self, port_handler, servo_id, address):
                self.calls += 1
                return 0, 0, 1

            @staticmethod
            def getRxPacketError(err):
                return "device-reported error"

        turret = object.__new__(DynamixelPanTilt)
        turret.port_handler = object()
        turret.packet_handler = FakePacketHandler()

        with self.assertRaisesRegex(RuntimeError, "device-reported error"):
            turret._read1(
                1,
                ADDR_HARDWARE_ERROR_STATUS,
                "read hardware error status",
                valid_mask=HARDWARE_ERROR_VALID_MASK,
            )

        self.assertEqual(turret.packet_handler.calls, 1)

    def test_hardware_status_read_retries_reserved_bits(self):
        class FakePacketHandler:
            def __init__(self):
                self.results = [
                    (0xF3, 0, 0),
                    (0x13, 0, 0),
                    (0x00, 0, 0),
                ]
                self.calls = 0

            def read1ByteTxRx(self, port_handler, servo_id, address):
                result = self.results[self.calls]
                self.calls += 1
                return result

        turret = object.__new__(DynamixelPanTilt)
        turret.port_handler = object()
        turret.packet_handler = FakePacketHandler()

        value = turret._read1(
            1,
            ADDR_HARDWARE_ERROR_STATUS,
            "read hardware error status",
            valid_mask=HARDWARE_ERROR_VALID_MASK,
        )

        self.assertEqual(value, 0)
        self.assertEqual(turret.packet_handler.calls, 3)

    def test_hardware_status_read_rejects_persistent_reserved_bits(self):
        class FakePacketHandler:
            def __init__(self):
                self.calls = 0

            def read1ByteTxRx(self, port_handler, servo_id, address):
                self.calls += 1
                return 0xF3, 0, 0

        turret = object.__new__(DynamixelPanTilt)
        turret.port_handler = object()
        turret.packet_handler = FakePacketHandler()

        with self.assertRaisesRegex(
            DynamixelCommunicationError,
            "reserved bits 0xc2 were set",
        ):
            turret._read1(
                1,
                ADDR_HARDWARE_ERROR_STATUS,
                "read hardware error status",
                valid_mask=HARDWARE_ERROR_VALID_MASK,
            )

        self.assertEqual(turret.packet_handler.calls, 3)

    def test_hardware_status_read_accepts_real_error_bits(self):
        class FakePacketHandler:
            def __init__(self):
                self.calls = 0

            def read1ByteTxRx(self, port_handler, servo_id, address):
                self.calls += 1
                return 0x20, 0, 0

        turret = object.__new__(DynamixelPanTilt)
        turret.port_handler = object()
        turret.packet_handler = FakePacketHandler()

        value = turret._read1(
            1,
            ADDR_HARDWARE_ERROR_STATUS,
            "read hardware error status",
            valid_mask=HARDWARE_ERROR_VALID_MASK,
        )

        self.assertEqual(value, 0x20)
        self.assertEqual(turret.packet_handler.calls, 1)

if __name__ == "__main__":
    unittest.main()

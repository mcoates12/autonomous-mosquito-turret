"""Qt-free Dynamixel pan/tilt hardware and fixed-rate tracking control."""

import threading
import time
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

from dynamixel_sdk import (  # type: ignore
    GroupSyncRead,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
)

from tracking_core import (
    ControlTarget,
    clamp,
    degrees_to_position_ticks,
    position_ticks_to_degrees,
    sanitize_motion_profile,
    tracking_delta_degrees,
)


# XL430 Control Table (Protocol 2.0)
ADDR_OPERATING_MODE = 11
ADDR_MAX_POSITION_LIMIT = 48
ADDR_MIN_POSITION_LIMIT = 52
ADDR_TORQUE_ENABLE = 64
ADDR_BUS_WATCHDOG = 98
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132

LEN_GOAL_POSITION = 4
LEN_PRESENT_STATE = 8

OPERATING_MODE_POSITION = 3
TORQUE_ON, TORQUE_OFF = 1, 0

DXL_VELOCITY_UNIT_DEG_S = 0.229 * 6.0
BUS_WATCHDOG_TICKS = 25  # 25 * 20 ms = 500 ms


class ControlParams(Protocol):
    rate_hz: float
    profile_velocity: int
    profile_acceleration: int
    tracking_enabled: bool
    servos_enabled: bool
    deadband_px: int
    deg_per_px_pan: float
    deg_per_px_tilt: float
    pan_dir: int
    tilt_dir: int
    max_step_deg: float


class ParamProvider(Protocol):
    def get(self) -> ControlParams:
        ...


def int_to_le_bytes(value: int, length: int) -> bytes:
    return int(value).to_bytes(length, byteorder="little", signed=False)


class DynamixelPanTilt:
    """Low-level XL430 interface; callers must serialize all bus access."""

    def __init__(self, port: str, baud: int, pan_id: int, tilt_id: int):
        self.port = port
        self.baud = baud
        self.pan_id = pan_id
        self.tilt_id = tilt_id

        self.protocol = 2.0
        self.port_handler = PortHandler(self.port)
        self.packet_handler = PacketHandler(self.protocol)
        self.sync_write_goal = GroupSyncWrite(
            self.port_handler,
            self.packet_handler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )
        self.sync_read_state = GroupSyncRead(
            self.port_handler,
            self.packet_handler,
            ADDR_PRESENT_VELOCITY,
            LEN_PRESENT_STATE,
        )
        if not self.sync_read_state.addParam(self.pan_id):
            raise RuntimeError(f"Failed to add pan ID {self.pan_id} to sync read")
        if not self.sync_read_state.addParam(self.tilt_id):
            raise RuntimeError(f"Failed to add tilt ID {self.tilt_id} to sync read")

        # Populated from the control table after open(). Dynamixel Wizard is
        # authoritative; this application does not duplicate its limits.
        self.position_limits_ticks = {
            self.pan_id: (0, 4095),
            self.tilt_id: (0, 4095),
        }
        self.pan_min, self.pan_max = 0.0, 360.0
        self.tilt_min, self.tilt_max = 0.0, 360.0
        self.pan_cmd = 0.0
        self.tilt_cmd = 0.0
        self.pan_actual = 0.0
        self.tilt_actual = 0.0
        self.pan_velocity = 0.0
        self.tilt_velocity = 0.0

        self.opened = False
        self.torque = False
        self.motion_profile = None

    def _check(self, servo_id: int, comm: int, err: int, what: str) -> None:
        if comm != 0:
            raise RuntimeError(
                f"[ID:{servo_id}] {what} COMM FAIL: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )
        if err != 0:
            raise RuntimeError(
                f"[ID:{servo_id}] {what} DXL ERROR: "
                f"{self.packet_handler.getRxPacketError(err)}"
            )

    def open(self) -> None:
        if self.opened:
            return
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open port {self.port}")
        if not self.port_handler.setBaudRate(self.baud):
            self.port_handler.closePort()
            raise RuntimeError(f"Failed to set baudrate {self.baud}")
        self.opened = True
        try:
            # Setting the mode resets the motion profile; the controller
            # reapplies it immediately before torque is enabled.
            self.torque_off(best_effort=False)
            for servo_id in (self.pan_id, self.tilt_id):
                comm, err = self.packet_handler.write1ByteTxRx(
                    self.port_handler,
                    servo_id,
                    ADDR_OPERATING_MODE,
                    OPERATING_MODE_POSITION,
                )
                self._check(servo_id, comm, err, "set operating mode")

            self._read_configured_limits()
            self.read_feedback()
            self.pan_cmd = self.pan_actual
            self.tilt_cmd = self.tilt_actual
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        try:
            self.torque_off()
        except Exception:
            pass
        try:
            if self.opened:
                self.port_handler.closePort()
        finally:
            self.opened = False

    def _read4(self, servo_id: int, address: int, what: str) -> int:
        value, comm, err = self.packet_handler.read4ByteTxRx(
            self.port_handler, servo_id, address
        )
        self._check(servo_id, comm, err, what)
        return int(value)

    def _read_configured_limits(self) -> None:
        for servo_id in (self.pan_id, self.tilt_id):
            maximum = self._read4(
                servo_id, ADDR_MAX_POSITION_LIMIT, "read max position limit"
            )
            minimum = self._read4(
                servo_id, ADDR_MIN_POSITION_LIMIT, "read min position limit"
            )
            if not (0 <= minimum <= maximum <= 4095):
                raise RuntimeError(
                    f"[ID:{servo_id}] invalid configured position limits: "
                    f"{minimum}..{maximum}"
                )
            self.position_limits_ticks[servo_id] = (minimum, maximum)

        pan_limits = self.position_limits_ticks[self.pan_id]
        tilt_limits = self.position_limits_ticks[self.tilt_id]
        self.pan_min, self.pan_max = (
            position_ticks_to_degrees(pan_limits[0]),
            position_ticks_to_degrees(pan_limits[1]),
        )
        self.tilt_min, self.tilt_max = (
            position_ticks_to_degrees(tilt_limits[0]),
            position_ticks_to_degrees(tilt_limits[1]),
        )

    def set_motion_profile(self, velocity: int, acceleration: int) -> None:
        safe_profile = sanitize_motion_profile(velocity, acceleration)
        if safe_profile == self.motion_profile:
            return
        safe_velocity, safe_acceleration = safe_profile
        current_velocity = None if self.motion_profile is None else self.motion_profile[0]
        for servo_id in (self.pan_id, self.tilt_id):
            # Preserve acceleration <= velocity/2 even between the two writes
            # while changing a live profile.
            if (
                current_velocity is not None
                and safe_acceleration > current_velocity // 2
            ):
                writes = (
                    (ADDR_PROFILE_VELOCITY, safe_velocity, "velocity"),
                    (ADDR_PROFILE_ACCELERATION, safe_acceleration, "acceleration"),
                )
            else:
                writes = (
                    (ADDR_PROFILE_ACCELERATION, safe_acceleration, "acceleration"),
                    (ADDR_PROFILE_VELOCITY, safe_velocity, "velocity"),
                )
            for address, value, label in writes:
                comm, err = self.packet_handler.write4ByteTxRx(
                    self.port_handler, servo_id, address, value
                )
                self._check(servo_id, comm, err, f"set profile {label}")
        self.motion_profile = safe_profile

    def _set_bus_watchdog(self, watchdog_ticks: int) -> None:
        for servo_id in (self.pan_id, self.tilt_id):
            comm, err = self.packet_handler.write1ByteTxRx(
                self.port_handler,
                servo_id,
                ADDR_BUS_WATCHDOG,
                int(watchdog_ticks),
            )
            self._check(servo_id, comm, err, "set bus watchdog")

    def torque_on(self) -> None:
        if self.torque:
            return
        try:
            self._set_bus_watchdog(0)
            for servo_id in (self.pan_id, self.tilt_id):
                comm, err = self.packet_handler.write1ByteTxRx(
                    self.port_handler, servo_id, ADDR_TORQUE_ENABLE, TORQUE_ON
                )
                self._check(servo_id, comm, err, "torque on")
            self._set_bus_watchdog(BUS_WATCHDOG_TICKS)
            self.torque = True
        except Exception:
            self.torque_off(best_effort=True)
            raise

    def torque_off(self, best_effort: bool = True) -> None:
        first_error = None
        for servo_id in (self.pan_id, self.tilt_id):
            try:
                comm, err = self.packet_handler.write1ByteTxRx(
                    self.port_handler, servo_id, ADDR_TORQUE_ENABLE, TORQUE_OFF
                )
                self._check(servo_id, comm, err, "torque off")
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        self.torque = False
        if first_error is not None and not best_effort:
            raise first_error

    @staticmethod
    def _signed32(value: int) -> int:
        return value - (1 << 32) if value & (1 << 31) else value

    def read_feedback(self) -> Tuple[float, float, float, float]:
        comm = self.sync_read_state.txRxPacket()
        if comm != 0:
            raise RuntimeError(
                f"sync_read COMM FAIL: {self.packet_handler.getTxRxResult(comm)}"
            )

        values = []
        for servo_id in (self.pan_id, self.tilt_id):
            if not self.sync_read_state.isAvailable(
                servo_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_STATE
            ):
                raise RuntimeError(f"[ID:{servo_id}] present state unavailable")
            raw_velocity = self.sync_read_state.getData(
                servo_id, ADDR_PRESENT_VELOCITY, 4
            )
            raw_position = self.sync_read_state.getData(
                servo_id, ADDR_PRESENT_POSITION, 4
            )
            velocity_deg_s = (
                self._signed32(int(raw_velocity)) * DXL_VELOCITY_UNIT_DEG_S
            )
            position_deg = position_ticks_to_degrees(int(raw_position))
            values.append((position_deg, velocity_deg_s))

        self.pan_actual, self.pan_velocity = values[0]
        self.tilt_actual, self.tilt_velocity = values[1]
        return (
            self.pan_actual,
            self.tilt_actual,
            self.pan_velocity,
            self.tilt_velocity,
        )

    def _bounded_ticks(self, servo_id: int, degrees: float) -> int:
        ticks = degrees_to_position_ticks(degrees)
        minimum, maximum = self.position_limits_ticks[servo_id]
        return int(clamp(ticks, minimum, maximum))

    def send(self, pan_deg: float, tilt_deg: float) -> None:
        pan_ticks = self._bounded_ticks(self.pan_id, pan_deg)
        tilt_ticks = self._bounded_ticks(self.tilt_id, tilt_deg)

        self.sync_write_goal.clearParam()
        ok_pan = self.sync_write_goal.addParam(
            self.pan_id, int_to_le_bytes(pan_ticks, LEN_GOAL_POSITION)
        )
        ok_tilt = self.sync_write_goal.addParam(
            self.tilt_id, int_to_le_bytes(tilt_ticks, LEN_GOAL_POSITION)
        )
        if not (ok_pan and ok_tilt):
            raise RuntimeError("Failed to add params to sync write")

        comm = self.sync_write_goal.txPacket()
        if comm != 0:
            raise RuntimeError(
                f"sync_write COMM FAIL: {self.packet_handler.getTxRxResult(comm)}"
            )

        self.pan_cmd = position_ticks_to_degrees(pan_ticks)
        self.tilt_cmd = position_ticks_to_degrees(tilt_ticks)


@dataclass(frozen=True)
class ControllerSnapshot:
    pan_command_deg: float
    tilt_command_deg: float
    pan_actual_deg: float
    tilt_actual_deg: float
    pan_velocity_deg_s: float
    tilt_velocity_deg_s: float
    torque_enabled: bool
    move_updates: int
    target_age_s: Optional[float] = None
    error: Optional[str] = None


class FixedRatePanTiltController:
    """Own the servo bus and consume only the most recent target error."""

    TARGET_STALE_SEC = 0.25
    INACTIVE_FEEDBACK_PERIOD_SEC = 0.10
    MIN_MAX_CONTROL_DT_SEC = 0.05

    def __init__(self, turret: DynamixelPanTilt, store: ParamProvider):
        self.turret = turret
        self.store = store
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._target = None
        self._move_updates = 0
        self._error = None
        self._snapshot = self._make_snapshot()
        self._thread = threading.Thread(
            target=self._run,
            name="pan-tilt-control",
            daemon=True,
        )

    def _make_snapshot(
        self, target_age_s: Optional[float] = None
    ) -> ControllerSnapshot:
        return ControllerSnapshot(
            pan_command_deg=self.turret.pan_cmd,
            tilt_command_deg=self.turret.tilt_cmd,
            pan_actual_deg=self.turret.pan_actual,
            tilt_actual_deg=self.turret.tilt_actual,
            pan_velocity_deg_s=self.turret.pan_velocity,
            tilt_velocity_deg_s=self.turret.tilt_velocity,
            torque_enabled=self.turret.torque,
            move_updates=self._move_updates,
            target_age_s=target_age_s,
            error=self._error,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def publish_target(self, target: ControlTarget) -> None:
        with self._lock:
            self._target = target

    def clear_target(self) -> None:
        with self._lock:
            self._target = None

    def snapshot(self) -> ControllerSnapshot:
        with self._lock:
            return self._snapshot

    def _latest_target(self) -> Optional[ControlTarget]:
        with self._lock:
            return self._target

    def _publish_snapshot(self, target_age_s: Optional[float] = None) -> None:
        snapshot = self._make_snapshot(target_age_s)
        with self._lock:
            self._snapshot = snapshot

    def _arm_without_jump(self, params: ControlParams) -> None:
        self.turret.read_feedback()
        self.turret.pan_cmd = self.turret.pan_actual
        self.turret.tilt_cmd = self.turret.tilt_actual
        self.turret.set_motion_profile(
            params.profile_velocity, params.profile_acceleration
        )
        self.turret.send(self.turret.pan_cmd, self.turret.tilt_cmd)
        self.turret.torque_on()

    def _run(self) -> None:
        last_tick = time.monotonic()
        next_tick = last_tick
        next_inactive_feedback = last_tick
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if now < next_tick:
                    self._stop.wait(next_tick - now)
                    continue

                params = self.store.get()
                period = 1.0 / clamp(params.rate_hz, 5.0, 120.0)
                max_dt = max(self.MIN_MAX_CONTROL_DT_SEC, 2.0 * period)
                dt = clamp(now - last_tick, 0.0, max_dt)
                last_tick = now
                next_tick += period
                if next_tick <= now:
                    next_tick = now + period

                active = params.tracking_enabled and params.servos_enabled
                target_age = None

                if active:
                    if not self.turret.torque:
                        self._arm_without_jump(params)
                    else:
                        self.turret.set_motion_profile(
                            params.profile_velocity,
                            params.profile_acceleration,
                        )

                    self.turret.read_feedback()
                    target = self._latest_target()
                    if target is not None:
                        target_age = max(0.0, now - target.timestamp)

                    if (
                        target is not None
                        and target.locked
                        and target_age is not None
                        and target_age <= self.TARGET_STALE_SEC
                    ):
                        error_x = target.error_x_px
                        error_y = target.error_y_px
                        if abs(error_x) < params.deadband_px:
                            error_x = 0.0
                        if abs(error_y) < params.deadband_px:
                            error_y = 0.0

                        delta_pan = tracking_delta_degrees(
                            error_x,
                            params.deg_per_px_pan,
                            params.pan_dir,
                            dt,
                            params.max_step_deg,
                        )
                        delta_tilt = tracking_delta_degrees(
                            error_y,
                            params.deg_per_px_tilt,
                            params.tilt_dir,
                            dt,
                            params.max_step_deg,
                        )
                        if delta_pan != 0.0 or delta_tilt != 0.0:
                            self.turret.send(
                                self.turret.pan_cmd + delta_pan,
                                self.turret.tilt_cmd + delta_tilt,
                            )
                            self._move_updates += 1
                else:
                    if self.turret.torque:
                        self.turret.torque_off(best_effort=False)
                    if now >= next_inactive_feedback:
                        self.turret.read_feedback()
                        self.turret.pan_cmd = self.turret.pan_actual
                        self.turret.tilt_cmd = self.turret.tilt_actual
                        next_inactive_feedback = (
                            now + self.INACTIVE_FEEDBACK_PERIOD_SEC
                        )

                self._publish_snapshot(target_age)
        except Exception as exc:
            self._error = str(exc)
            try:
                self.turret.torque_off(best_effort=True)
            finally:
                self._publish_snapshot()
        finally:
            try:
                self.turret.torque_off(best_effort=True)
            except Exception:
                pass

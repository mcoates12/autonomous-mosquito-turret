#!/usr/bin/env python3
"""
Dynamixel XL430 (Protocol 2.0) pan/tilt control via U2D2 (TTL).

Features:
- Soft limits (tilt bounds prefilled)
- Degrees -> ticks conversion (0-360 deg => 0-4095)
- Torque enable/disable
- Operating mode set to Position Control
- Profile velocity control
- Commands: center, goto, step, sweep, status

Wiring:
- U2D2 USB to Jetson
- U2D2 TTL to Dynamixel chain + external power to servos (NOT from USB)

Author: ChatGPT
"""

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass

from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, GroupSyncRead  # type: ignore

# -----------------------------
# XL430 Control Table (Protocol 2.0)
# -----------------------------
ADDR_OPERATING_MODE   = 11   # 1 byte
ADDR_TORQUE_ENABLE    = 64   # 1 byte
ADDR_PROFILE_VELOCITY = 112  # 4 bytes (profile velocity, used in position mode)
ADDR_GOAL_POSITION    = 116  # 4 bytes
ADDR_PRESENT_POSITION = 132  # 4 bytes

LEN_OPERATING_MODE   = 1
LEN_TORQUE_ENABLE    = 1
LEN_PROFILE_VELOCITY = 4
LEN_GOAL_POSITION    = 4
LEN_PRESENT_POSITION = 4

OPERATING_MODE_POSITION = 3  # Position Control Mode

TORQUE_ON  = 1
TORQUE_OFF = 0

# XL430 resolution: 0..4095 ticks for 0..360 degrees
TICKS_PER_REV = 4096.0
DEG_PER_TICK = 360.0 / TICKS_PER_REV


# -----------------------------
# Configuration (edit these)
# -----------------------------
@dataclass
class ServoConfig:
    servo_id: int
    name: str

    # Soft limits in DEGREES (logical, after applying invert/offset)
    min_deg: float
    max_deg: float

    # Home position in degrees (logical)
    home_deg: float

    # Optional: invert direction if your servo rotates opposite your desired sign
    invert: bool = False

    # Optional: offset in degrees (logical 0deg maps to physical position)
    # Example: if "0 deg" in your logic corresponds to servo sitting at physical 180 deg, set offset_deg=180
    offset_deg: float = 0.0


@dataclass
class TurretConfig:
    port: str = "/dev/ttyUSB0"
    baud: int = 1000000  # common default; change if you set different in Dynamixel Wizard
    protocol: float = 2.0

    # Limit rate of change to prevent whipping around
    max_step_deg_per_cmd: float = 15.0

    # Default profile velocity (0 = max, but don't do that). This value is in Dynamixel units.
    # Typical safe starting points: 50–200. Tune later.
    profile_velocity: int = 120

    # Servos
    pan: ServoConfig = ServoConfig(
        servo_id=1,
        name="pan",
        min_deg=0.0,
        max_deg=360.0,
        home_deg=180.0,
        invert=False,
        offset_deg=0.0,
    )

    tilt: ServoConfig = ServoConfig(
        servo_id=2,
        name="tilt",
        # Using YOUR safe tilt bounds:
        min_deg=62.23,
        max_deg=191.78,
        home_deg=120.0,
        invert=False,
        offset_deg=0.0,
    )


# -----------------------------
# Helpers
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_deg_0_360(deg: float) -> float:
    # keep in [0, 360)
    d = deg % 360.0
    if d < 0:
        d += 360.0
    return d


def int_to_little_endian_bytes(value: int, length: int) -> bytes:
    return int(value).to_bytes(length, byteorder="little", signed=False)


# -----------------------------
# Dynamixel Controller
# -----------------------------
class DynamixelTurret:
    def __init__(self, cfg: TurretConfig):
        self.cfg = cfg
        self.port_handler = PortHandler(cfg.port)
        self.packet_handler = PacketHandler(cfg.protocol)

        self.sync_write_goal = GroupSyncWrite(
            self.port_handler, self.packet_handler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION
        )
        self.sync_read_present = GroupSyncRead(
            self.port_handler, self.packet_handler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
        )

        self._torque_enabled = False
        self._last_cmd_deg = {
            cfg.pan.name: None,
            cfg.tilt.name: None,
        }

    def open(self) -> None:
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open port {self.cfg.port}")
        if not self.port_handler.setBaudRate(self.cfg.baud):
            raise RuntimeError(f"Failed to set baudrate {self.cfg.baud} on {self.cfg.port}")

        # Register servos for sync read
        for s in (self.cfg.pan, self.cfg.tilt):
            ok = self.sync_read_present.addParam(s.servo_id)
            if not ok:
                raise RuntimeError(f"Failed to add servo {s.servo_id} to sync read")

    def close(self) -> None:
        try:
            self.torque_off_all()
        except Exception:
            pass
        try:
            self.port_handler.closePort()
        except Exception:
            pass

    # ---------- Low-level read/write ----------
    def write1(self, servo_id: int, addr: int, value: int) -> None:
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, servo_id, addr, value
        )
        self._check_comm(servo_id, dxl_comm_result, dxl_error, f"write1 addr={addr} val={value}")

    def write4(self, servo_id: int, addr: int, value: int) -> None:
        dxl_comm_result, dxl_error = self.packet_handler.write4ByteTxRx(
            self.port_handler, servo_id, addr, value
        )
        self._check_comm(servo_id, dxl_comm_result, dxl_error, f"write4 addr={addr} val={value}")

    def read4(self, servo_id: int, addr: int) -> int:
        value, dxl_comm_result, dxl_error = self.packet_handler.read4ByteTxRx(
            self.port_handler, servo_id, addr
        )
        self._check_comm(servo_id, dxl_comm_result, dxl_error, f"read4 addr={addr}")
        return int(value)

    def _check_comm(self, servo_id: int, comm_result: int, dxl_error: int, what: str) -> None:
        if comm_result != 0:
            msg = self.packet_handler.getTxRxResult(comm_result)
            raise RuntimeError(f"[ID:{servo_id}] {what} COMM FAIL: {msg}")
        if dxl_error != 0:
            msg = self.packet_handler.getRxPacketError(dxl_error)
            raise RuntimeError(f"[ID:{servo_id}] {what} DXL ERROR: {msg}")

    # ---------- Setup ----------
    def ping(self) -> None:
        for s in (self.cfg.pan, self.cfg.tilt):
            model, comm, err = self.packet_handler.ping(self.port_handler, s.servo_id)
            self._check_comm(s.servo_id, comm, err, "ping")
            print(f"[OK] {s.name} ID={s.servo_id} model_number={model}")

    def set_position_mode_all(self) -> None:
        # Must disable torque to change operating mode
        self.torque_off_all()
        for s in (self.cfg.pan, self.cfg.tilt):
            self.write1(s.servo_id, ADDR_OPERATING_MODE, OPERATING_MODE_POSITION)
        print("[OK] Operating mode set to POSITION for both servos")

    def set_profile_velocity_all(self, profile_vel: int) -> None:
        # profile velocity can be written with torque on/off; safer with torque off initially
        for s in (self.cfg.pan, self.cfg.tilt):
            self.write4(s.servo_id, ADDR_PROFILE_VELOCITY, int(profile_vel))
        print(f"[OK] Profile velocity set to {profile_vel} (both)")

    def torque_on_all(self) -> None:
        for s in (self.cfg.pan, self.cfg.tilt):
            self.write1(s.servo_id, ADDR_TORQUE_ENABLE, TORQUE_ON)
        self._torque_enabled = True
        print("[OK] Torque ON (both)")

    def torque_off_all(self) -> None:
        for s in (self.cfg.pan, self.cfg.tilt):
            try:
                self.write1(s.servo_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)
            except Exception:
                # if one servo isn't reachable, don't block shutdown
                pass
        self._torque_enabled = False
        print("[OK] Torque OFF (both)")

    # ---------- Conversions ----------
    def deg_to_ticks(self, servo: ServoConfig, deg_logical: float) -> int:
        # apply clamp in logical space first
        deg_logical = clamp(deg_logical, servo.min_deg, servo.max_deg)

        # apply rate limit (per-servo)
        last = self._last_cmd_deg.get(servo.name)
        if last is not None:
            max_step = self.cfg.max_step_deg_per_cmd
            deg_logical = clamp(deg_logical, last - max_step, last + max_step)

        # apply invert + offset mapping to physical deg (0..360 domain)
        d = deg_logical
        if servo.invert:
            # invert around 0 in logical space; you can change this behavior later if needed
            d = -d
        d = d + servo.offset_deg
        d = wrap_deg_0_360(d)

        ticks = int(round(d / DEG_PER_TICK)) % int(TICKS_PER_REV)
        self._last_cmd_deg[servo.name] = deg_logical
        return ticks

    def ticks_to_deg_physical(self, ticks: int) -> float:
        ticks = ticks % int(TICKS_PER_REV)
        return ticks * DEG_PER_TICK

    # ---------- Motion Commands ----------
    def set_pan_tilt_deg(self, pan_deg: float, tilt_deg: float) -> None:
        pan_ticks = self.deg_to_ticks(self.cfg.pan, pan_deg)
        tilt_ticks = self.deg_to_ticks(self.cfg.tilt, tilt_deg)

        # Build sync write packet
        self.sync_write_goal.clearParam()
        ok1 = self.sync_write_goal.addParam(self.cfg.pan.servo_id, int_to_little_endian_bytes(pan_ticks, 4))
        ok2 = self.sync_write_goal.addParam(self.cfg.tilt.servo_id, int_to_little_endian_bytes(tilt_ticks, 4))
        if not (ok1 and ok2):
            raise RuntimeError("Failed to add params to sync write")

        dxl_comm_result = self.sync_write_goal.txPacket()
        if dxl_comm_result != 0:
            msg = self.packet_handler.getTxRxResult(dxl_comm_result)
            raise RuntimeError(f"sync_write_goal COMM FAIL: {msg}")

    def center(self) -> None:
        self.set_pan_tilt_deg(self.cfg.pan.home_deg, self.cfg.tilt.home_deg)

    def read_present_positions(self) -> dict:
        dxl_comm_result = self.sync_read_present.txRxPacket()
        if dxl_comm_result != 0:
            msg = self.packet_handler.getTxRxResult(dxl_comm_result)
            raise RuntimeError(f"sync_read_present COMM FAIL: {msg}")

        out = {}
        for s in (self.cfg.pan, self.cfg.tilt):
            if not self.sync_read_present.isAvailable(s.servo_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                raise RuntimeError(f"Present position not available for servo {s.servo_id}")
            pos = self.sync_read_present.getData(s.servo_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            out[s.name] = int(pos)
        return out


# -----------------------------
# CLI routines
# -----------------------------
def cmd_status(ctrl: DynamixelTurret) -> None:
    pos = ctrl.read_present_positions()
    pan_deg = ctrl.ticks_to_deg_physical(pos["pan"])
    tilt_deg = ctrl.ticks_to_deg_physical(pos["tilt"])
    print(f"PAN:  ticks={pos['pan']:4d}  physical_deg={pan_deg:7.2f}")
    print(f"TILT: ticks={pos['tilt']:4d}  physical_deg={tilt_deg:7.2f}")


def cmd_goto(ctrl: DynamixelTurret, pan: float, tilt: float, settle: float) -> None:
    ctrl.set_pan_tilt_deg(pan, tilt)
    time.sleep(settle)
    cmd_status(ctrl)


def cmd_step(ctrl: DynamixelTurret, axis: str, step_deg: float, n: int, settle: float) -> None:
    # Step around home
    pan = ctrl.cfg.pan.home_deg
    tilt = ctrl.cfg.tilt.home_deg

    for i in range(n):
        sign = 1 if (i % 2 == 0) else -1
        if axis == "pan":
            cmd_goto(ctrl, pan + sign * step_deg, tilt, settle)
        else:
            cmd_goto(ctrl, pan, tilt + sign * step_deg, settle)


def cmd_sweep(ctrl: DynamixelTurret, axis: str, step_deg: float, settle: float) -> None:
    if axis == "pan":
        s = ctrl.cfg.pan
        fixed = ctrl.cfg.tilt.home_deg
        d = s.min_deg
        while d <= s.max_deg + 1e-6:
            ctrl.set_pan_tilt_deg(d, fixed)
            time.sleep(settle)
            d += step_deg
    else:
        s = ctrl.cfg.tilt
        fixed = ctrl.cfg.pan.home_deg
        d = s.min_deg
        while d <= s.max_deg + 1e-6:
            ctrl.set_pan_tilt_deg(fixed, d)
            time.sleep(settle)
            d += step_deg
    cmd_status(ctrl)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dynamixel XL430 pan/tilt controller (U2D2 TTL).")
    p.add_argument("--port", default=TurretConfig.port, help="Serial port (e.g., /dev/ttyUSB0)")
    p.add_argument("--baud", type=int, default=TurretConfig.baud, help="Baudrate (e.g., 57600, 1000000)")
    p.add_argument("--pan_id", type=int, default=TurretConfig.pan.servo_id, help="Pan servo ID")
    p.add_argument("--tilt_id", type=int, default=TurretConfig.tilt.servo_id, help="Tilt servo ID")
    p.add_argument("--profile_velocity", type=int, default=TurretConfig.profile_velocity, help="Profile velocity (both)")
    p.add_argument("--max_step", type=float, default=TurretConfig.max_step_deg_per_cmd, help="Max deg change per command")

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ping", help="Ping both servos and print model numbers")
    sub.add_parser("center", help="Move to home positions")
    sub.add_parser("status", help="Read present positions and print")

    g = sub.add_parser("goto", help="Move to pan/tilt degrees (logical)")
    g.add_argument("--pan", type=float, required=True)
    g.add_argument("--tilt", type=float, required=True)
    g.add_argument("--settle", type=float, default=0.5)

    st = sub.add_parser("step", help="Step test around home on one axis")
    st.add_argument("--axis", choices=["pan", "tilt"], required=True)
    st.add_argument("--step_deg", type=float, default=5.0)
    st.add_argument("--n", type=int, default=6)
    st.add_argument("--settle", type=float, default=0.5)

    sw = sub.add_parser("sweep", help="Sweep one axis through its soft limits")
    sw.add_argument("--axis", choices=["pan", "tilt"], required=True)
    sw.add_argument("--step_deg", type=float, default=5.0)
    sw.add_argument("--settle", type=float, default=0.4)

    return p


def main() -> int:
    args = build_argparser().parse_args()

    cfg = TurretConfig()
    cfg.port = args.port
    cfg.baud = args.baud
    cfg.profile_velocity = args.profile_velocity
    cfg.max_step_deg_per_cmd = args.max_step

    cfg.pan.servo_id = args.pan_id
    cfg.tilt.servo_id = args.tilt_id

    ctrl = DynamixelTurret(cfg)

    def _sigint_handler(sig, frame):
        print("\n[CTRL+C] Disabling torque and exiting...")
        try:
            ctrl.torque_off_all()
        finally:
            try:
                ctrl.port_handler.closePort()
            except Exception:
                pass
        sys.exit(130)

    signal.signal(signal.SIGINT, _sigint_handler)

    ctrl.open()

    # Basic setup
    ctrl.ping()
    ctrl.set_position_mode_all()
    ctrl.set_profile_velocity_all(cfg.profile_velocity)
    ctrl.torque_on_all()

    if args.cmd == "ping":
        pass
    elif args.cmd == "center":
        ctrl.center()
        time.sleep(0.7)
        cmd_status(ctrl)
    elif args.cmd == "status":
        cmd_status(ctrl)
    elif args.cmd == "goto":
        cmd_goto(ctrl, args.pan, args.tilt, args.settle)
    elif args.cmd == "step":
        cmd_step(ctrl, args.axis, args.step_deg, args.n, args.settle)
    elif args.cmd == "sweep":
        cmd_sweep(ctrl, args.axis, args.step_deg, args.settle)
    else:
        raise RuntimeError("Unknown command")

    ctrl.torque_off_all()
    ctrl.port_handler.closePort()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

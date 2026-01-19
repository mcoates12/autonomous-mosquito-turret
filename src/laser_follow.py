#!/usr/bin/env python3
"""
Laser follow (red dot) -> Dynamixel pan/tilt

Requires:
  pip install opencv-python dynamixel-sdk numpy

Run example:
  python3 laser_follow.py --port /dev/ttyUSB0 --baud 1000000 --cam 0 --width 1920 --height 1200 --v_thresh 60 --s_thresh 40 --deadband_px 20 --lost_frames_hold 1 --max_step_deg 1.0 --pan_dir -1

Controls:
  q / ESC  : quit (torque off)
"""

import argparse
import signal
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, GroupSyncRead  # type: ignore


# -----------------------------
# Dynamixel XL430 Control Table (Protocol 2.0)
# -----------------------------
ADDR_OPERATING_MODE   = 11
ADDR_TORQUE_ENABLE    = 64
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION    = 116
ADDR_PRESENT_POSITION = 132

LEN_GOAL_POSITION     = 4
LEN_PRESENT_POSITION  = 4

OPERATING_MODE_POSITION = 3
TORQUE_ON, TORQUE_OFF = 1, 0

TICKS_PER_REV = 4096.0
DEG_PER_TICK = 360.0 / TICKS_PER_REV


# -----------------------------
# Config
# -----------------------------
@dataclass
class ServoConfig:
    servo_id: int
    name: str
    min_deg: float
    max_deg: float
    home_deg: float
    invert: bool = False
    offset_deg: float = 0.0


@dataclass
class TurretConfig:
    port: str = "/dev/ttyUSB0"
    baud: int = 1000000
    protocol: float = 2.0
    profile_velocity: int = 120

    # Safety: how far a single update is allowed to change angle
    max_step_deg_per_update: float = 2.0

    pan: ServoConfig = ServoConfig(
        servo_id=1, name="pan",
        min_deg=0.0, max_deg=360.0,
        home_deg=180.0,
        invert=False, offset_deg=0.0
    )
    tilt: ServoConfig = ServoConfig(
        servo_id=2, name="tilt",
        min_deg=62.23, max_deg=191.78,
        home_deg=120.0,
        invert=False, offset_deg=0.0
    )


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_deg_0_360(deg: float) -> float:
    d = deg % 360.0
    if d < 0:
        d += 360.0
    return d


def int_to_le_bytes(value: int, length: int) -> bytes:
    return int(value).to_bytes(length, byteorder="little", signed=False)


# -----------------------------
# Red laser detection (HSV)
# -----------------------------
def find_laser_centroid_red(frame_bgr,
                            v_thresh=80,
                            s_thresh=40,
                            min_area=1,
                            max_area=2000):
    """
    Red-laser dot detection:
    - HSV threshold for red (two hue ranges)
    - Allows low saturation (important at distance on walls)
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lower1 = np.array([0,   s_thresh, v_thresh], dtype=np.uint8)
    upper1 = np.array([10,  255,      255],      dtype=np.uint8)
    lower2 = np.array([170, s_thresh, v_thresh], dtype=np.uint8)
    upper2 = np.array([179, 255,      255],      dtype=np.uint8)

    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                          cv2.inRange(hsv, lower2, upper2))

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    best = None
    best_score = -1.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, w, h2 = cv2.boundingRect(c)
        roi_v = v[y:y+h2, x:x+w]
        mean_v = float(np.mean(roi_v)) if roi_v.size else 0.0
        score = mean_v - 0.03 * area
        if score > best_score:
            best_score = score
            best = c

    if best is None:
        return None, mask
        
    # Confidence gate: require a very bright core inside the detected blob
    x, y, w, h2 = cv2.boundingRect(best)
    roi_v = v[y:y+h2, x:x+w]
    peak_v = int(np.max(roi_v)) if roi_v.size else 0
    
    area = cv2.contourArea(best)
    
    if peak_v < 215:   # tune: +30 to +80
        return None, mask
    if area < 2 or area > 300:
    	return None, mask

    M = cv2.moments(best)
    if M["m00"] == 0:
        return None, mask
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


# -----------------------------
# Dynamixel turret control
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

        self.pan_cmd = cfg.pan.home_deg
        self.tilt_cmd = cfg.tilt.home_deg

    def _check(self, servo_id: int, comm: int, err: int, what: str):
        if comm != 0:
            raise RuntimeError(f"[ID:{servo_id}] {what} COMM FAIL: {self.packet_handler.getTxRxResult(comm)}")
        if err != 0:
            raise RuntimeError(f"[ID:{servo_id}] {what} DXL ERROR: {self.packet_handler.getRxPacketError(err)}")

    def open(self):
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open port {self.cfg.port}")
        if not self.port_handler.setBaudRate(self.cfg.baud):
            raise RuntimeError(f"Failed to set baudrate {self.cfg.baud} on {self.cfg.port}")

        for s in (self.cfg.pan, self.cfg.tilt):
            if not self.sync_read_present.addParam(s.servo_id):
                raise RuntimeError(f"Failed to add servo {s.servo_id} to sync read")

    def close(self):
        try:
            self.torque_off()
        except Exception:
            pass
        try:
            self.port_handler.closePort()
        except Exception:
            pass

    def ping(self):
        for s in (self.cfg.pan, self.cfg.tilt):
            model, comm, err = self.packet_handler.ping(self.port_handler, s.servo_id)
            self._check(s.servo_id, comm, err, "ping")
            print(f"[OK] {s.name} ID={s.servo_id} model_number={model}")

    def write1(self, servo_id: int, addr: int, value: int):
        comm, err = self.packet_handler.write1ByteTxRx(self.port_handler, servo_id, addr, value)
        self._check(servo_id, comm, err, f"write1 addr={addr}")

    def write4(self, servo_id: int, addr: int, value: int):
        comm, err = self.packet_handler.write4ByteTxRx(self.port_handler, servo_id, addr, value)
        self._check(servo_id, comm, err, f"write4 addr={addr}")

    def set_position_mode(self):
        self.torque_off()
        for s in (self.cfg.pan, self.cfg.tilt):
            self.write1(s.servo_id, ADDR_OPERATING_MODE, OPERATING_MODE_POSITION)
        print("[OK] Operating mode set to POSITION for both")

    def set_profile_velocity(self, vel: int):
        for s in (self.cfg.pan, self.cfg.tilt):
            self.write4(s.servo_id, ADDR_PROFILE_VELOCITY, int(vel))
        print(f"[OK] Profile velocity set to {vel} both")

    def torque_on(self):
        for s in (self.cfg.pan, self.cfg.tilt):
            self.write1(s.servo_id, ADDR_TORQUE_ENABLE, TORQUE_ON)
        print("[OK] Torque ON both")

    def torque_off(self):
        for s in (self.cfg.pan, self.cfg.tilt):
            try:
                self.write1(s.servo_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)
            except Exception:
                pass
        print("[OK] Torque OFF both")

    def deg_to_ticks(self, servo: ServoConfig, deg_logical: float) -> int:
        deg_logical = clamp(deg_logical, servo.min_deg, servo.max_deg)

        d = deg_logical
        if servo.invert:
            d = -d
        d = d + servo.offset_deg
        d = wrap_deg_0_360(d)

        ticks = int(round(d / DEG_PER_TICK)) % int(TICKS_PER_REV)
        return ticks

    def send(self, pan_deg: float, tilt_deg: float):
        # safety: limit per-update step
        pan_deg = clamp(pan_deg, self.cfg.pan.min_deg, self.cfg.pan.max_deg)
        tilt_deg = clamp(tilt_deg, self.cfg.tilt.min_deg, self.cfg.tilt.max_deg)

        pan_deg = clamp(pan_deg, self.pan_cmd - self.cfg.max_step_deg_per_update,
                        self.pan_cmd + self.cfg.max_step_deg_per_update)
        tilt_deg = clamp(tilt_deg, self.tilt_cmd - self.cfg.max_step_deg_per_update,
                         self.tilt_cmd + self.cfg.max_step_deg_per_update)

        pan_ticks = self.deg_to_ticks(self.cfg.pan, pan_deg)
        tilt_ticks = self.deg_to_ticks(self.cfg.tilt, tilt_deg)

        self.sync_write_goal.clearParam()
        ok1 = self.sync_write_goal.addParam(self.cfg.pan.servo_id, int_to_le_bytes(pan_ticks, 4))
        ok2 = self.sync_write_goal.addParam(self.cfg.tilt.servo_id, int_to_le_bytes(tilt_ticks, 4))
        if not (ok1 and ok2):
            raise RuntimeError("Failed to add params to sync write")

        comm = self.sync_write_goal.txPacket()
        if comm != 0:
            raise RuntimeError(f"sync_write_goal COMM FAIL: {self.packet_handler.getTxRxResult(comm)}")

        self.pan_cmd = pan_deg
        self.tilt_cmd = tilt_deg


# -----------------------------
# Main loop
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Follow a red laser dot with Dynamixel pan/tilt.")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=1000000)
    ap.add_argument("--pan_id", type=int, default=1)
    ap.add_argument("--tilt_id", type=int, default=2)

    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1200)

    ap.add_argument("--v_thresh", type=int, default=80)
    ap.add_argument("--s_thresh", type=int, default=40)
    ap.add_argument("--min_area", type=float, default=1)
    ap.add_argument("--max_area", type=float, default=2000)
    ap.add_argument("--show_mask", action="store_true")

    # Control tuning:
    # deg_per_px is the main “gain”: how many degrees to move per pixel of error.
    ap.add_argument("--deg_per_px_pan", type=float, default=0.0040)
    ap.add_argument("--deg_per_px_tilt", type=float, default=0.0040)

    # Flip directions if motion is opposite your expectation:
    ap.add_argument("--pan_dir", type=int, default=+1, help="Use -1 if pan moves wrong way")
    ap.add_argument("--tilt_dir", type=int, default=+1, help="Use -1 if tilt moves wrong way")

    ap.add_argument("--deadband_px", type=int, default=8, help="Ignore small jitter within this many pixels")
    ap.add_argument("--lost_frames_hold", type=int, default=10, help="Hold position if target lost this many frames")
    ap.add_argument("--rate_hz", type=float, default=30.0)
    ap.add_argument("--profile_velocity", type=int, default=120)
    ap.add_argument("--max_step_deg", type=float, default=2.0)

    args = ap.parse_args()

    # Camera
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) #0.25 = manual (backend dependant)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7) # try -5. -7. -9
    cap.set(cv2.CAP_PROP_GAIN, 0) 
    
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try --cam 0/1 or check /dev/video*")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Turret
    cfg = TurretConfig(port=args.port, baud=args.baud, profile_velocity=args.profile_velocity)
    cfg.pan.servo_id = args.pan_id
    cfg.tilt.servo_id = args.tilt_id
    cfg.max_step_deg_per_update = args.max_step_deg

    turret = DynamixelTurret(cfg)

    def shutdown(*_):
        print("\n[EXIT] torque off + cleanup")
        try:
            turret.close()
        finally:
            try:
                cap.release()
                cv2.destroyAllWindows()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    turret.open()
    turret.ping()
    turret.set_position_mode()
    turret.set_profile_velocity(args.profile_velocity)
    turret.torque_on()

    # Start centered
    turret.send(cfg.pan.home_deg, cfg.tilt.home_deg)
    time.sleep(0.5)

    target_lost = 0
    dt = 1.0 / max(1e-6, args.rate_hz)
    last = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        cx0, cy0 = w // 2, h // 2

        centroid, mask = find_laser_centroid_red(
            frame,
            v_thresh=args.v_thresh,
            s_thresh=args.s_thresh,
            min_area=args.min_area,
            max_area=args.max_area,
        )

        cv2.drawMarker(frame, (cx0, cy0), (255, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

        if centroid is not None:
            target_lost = 0
            cx, cy = centroid
            err_x = cx - cx0
            err_y = cy - cy0

            # deadband
            if abs(err_x) < args.deadband_px:
                err_x = 0
            if abs(err_y) < args.deadband_px:
                err_y = 0

            # Convert pixel error to angle delta (degrees)
            dpan = args.pan_dir * (args.deg_per_px_pan * err_x)
            dtilt = args.tilt_dir * (args.deg_per_px_tilt * err_y)

            # Typical camera coords: y increases downward.
            # Many rigs want "dot above center" -> tilt up, so you might need to flip tilt_dir.
            new_pan = turret.pan_cmd + dpan
            new_tilt = turret.tilt_cmd + dtilt

            turret.send(new_pan, new_tilt)

            cv2.circle(frame, (cx, cy), 6, (0, 255, 255), 2)
            cv2.putText(frame, f"err=({cx-cx0},{cy-cy0}) pan={turret.pan_cmd:.1f} tilt={turret.tilt_cmd:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            target_lost += 1
            cv2.putText(frame, f"laser NOT FOUND (lost={target_lost})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # If lost for a while, just hold position (safe default)
            # (We can add slow recenter later if you want.)
            if target_lost > args.lost_frames_hold:
                pass

        cv2.imshow("laser_follow", frame)
        if args.show_mask:
            cv2.imshow("mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

        # Rate limit loop
        now = time.time()
        elapsed = now - last
        if elapsed < dt:
            time.sleep(dt - elapsed)
        last = time.time()

    shutdown()


if __name__ == "__main__":
    main()


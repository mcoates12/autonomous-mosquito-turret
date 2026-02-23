#!/usr/bin/env python3

import os
from turtle import right
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/aarch64-linux-gnu/qt5/plugins"

import sys
import time
import threading
from dataclasses import dataclass

import cv2
import numpy as np

import logging
from collections import deque


from PyQt5 import QtCore, QtGui, QtWidgets
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite  # type: ignore

# -----------------------------
# Statistics tracking
# -----------------------------
class Stats:
    def __init__(self):
        self.t0 = time.time()
        self.last_log = time.time()

        # counts
        self.frames = 0
        self.found = 0

        self.depth_attempt = 0
        self.depth_roi_calls = 0
        self.depth_roi_ok = 0
        self.depth_roi_fail = 0
        self.depth_full_calls = 0
        self.depth_full_ok = 0
        self.depth_full_fail = 0

        self.locked_frames = 0
        self.move_frames = 0

        # timing (keep last ~120 samples)
        self.ms_cap = deque(maxlen=120)
        self.ms_rect = deque(maxlen=120)
        self.ms_det = deque(maxlen=120)
        self.ms_depth_roi = deque(maxlen=120)
        self.ms_depth_full = deque(maxlen=120)
        self.ms_loop = deque(maxlen=120)

    @staticmethod
    def _avg(dq):
        return (sum(dq) / len(dq)) if dq else 0.0

    def hud(self):
        # quick one-liner for overlay
        return (f"FPS~{self.frames/max(1e-6,(time.time()-self.t0)):.1f} "
                f"found={self.found}/{self.frames} "
                f"ROI ok={self.depth_roi_ok}/{self.depth_roi_calls} "
                f"FULL={self.depth_full_calls} "
                f"lock={self.locked_frames} move={self.move_frames}")

    def log_once_per_sec(self, logger):
        now = time.time()
        if now - self.last_log < 1.0:
            return
        self.last_log = now

        logger.info(
            "fps=%.1f found=%d/%d depthAttempt=%d ROI=%d ok=%d fail=%d FULL=%d ok=%d fail=%d "
            "ms(cap=%.1f rect=%.1f det=%.1f roi=%.1f full=%.1f loop=%.1f) "
            "locked=%d move=%d",
            self.frames / max(1e-6, (now - self.t0)),
            self.found, self.frames,
            self.depth_attempt,
            self.depth_roi_calls, self.depth_roi_ok, self.depth_roi_fail,
            self.depth_full_calls, self.depth_full_ok, self.depth_full_fail,
            self._avg(self.ms_cap), self._avg(self.ms_rect), self._avg(self.ms_det),
            self._avg(self.ms_depth_roi), self._avg(self.ms_depth_full), self._avg(self.ms_loop),
            self.locked_frames, self.move_frames
        )

# -----------------------------
# Latest frame grabber thread
# -----------------------------
class LatestFrame:
    def __init__(self, cap: cv2.VideoCapture, name: str):
        self.cap = cap
        self.name = name
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.stop_evt = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.th.start()

    def stop(self):
        self.stop_evt.set()
        self.th.join(timeout=1.0)

    def _run(self):
        while not self.stop_evt.is_set():
            ok, f = self.cap.read()
            with self.lock:
                self.ok = ok
                if ok:
                    self.frame = f

    def get(self):
        with self.lock:
            return self.ok, (None if self.frame is None else self.frame.copy())

# -----------------------------
# GStreamer pipeline helper
# -----------------------------
def gst_v4l2_bgr_pipeline(device: str, width: int, height: int, fps: int = 60) -> str:
    """
    V4L2 -> (optionally hw convert) -> BGR -> OpenCV appsink.
    Notes:
      - drop/max-buffers=1 keeps latency low (your main goal).
      - sync=false prevents GStreamer from buffering to "sync" timestamps.
      - If your camera doesn't support the requested caps, GStreamer will fail to preroll.
    """
    return (
        f"v4l2src device={device} io-mode=2 ! "
        f"video/x-raw, width={width}, height={height}, framerate={fps}/1 ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )

# -----------------------------
# Dynamixel XL430 Control Table (Protocol 2.0)
# -----------------------------
ADDR_OPERATING_MODE   = 11
ADDR_TORQUE_ENABLE    = 64
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION    = 116

LEN_GOAL_POSITION     = 4

OPERATING_MODE_POSITION = 3
TORQUE_ON, TORQUE_OFF = 1, 0

TICKS_PER_REV = 4096.0
DEG_PER_TICK = 360.0 / TICKS_PER_REV


# -----------------------------
# Utility
# -----------------------------
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
# Shared live parameters (GUI -> worker)
# -----------------------------
@dataclass
class LiveParams:
    # camera / detection
    display_mode: str = "Left"      # Left | Right | Side-by-side | Combined
    track_source: str = "Left"      # Left | Right

    v_thresh: int = 80
    s_thresh: int = 60
    min_area: float = 1.0
    max_area: float = 2000

    # confidence gates
    peak_v_gate: int = 140         # require bright core (210-245 range)
    area_hi_gate: float = 40.0    # reject huge blobs

    # control
    deg_per_px_pan: float = 0.0060
    deg_per_px_tilt: float = 0.0060
    max_step_deg: float = 2.0
    deadband_px: int = 25
    rate_hz: float = 60.0

    profile_velocity: int = 200
    pan_dir: int = -1
    tilt_dir: int = 1

    # servo enable + tracking enable
    tracking_enabled: bool = False
    servos_enabled: bool = True


class ParamStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._p = LiveParams()

    def get(self) -> LiveParams:
        with self._lock:
            return LiveParams(**self._p.__dict__)

    def set_attr(self, name: str, value):
        with self._lock:
            setattr(self._p, name, value)


# -----------------------------
# Red laser detection (HSV)
# -----------------------------
def find_laser_centroid_red(frame_bgr, p: LiveParams):
    """
    Red-laser dot detection:
    - HSV threshold for red (two hue ranges)
    - Allows low saturation (important at distance on walls)
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lower1 = np.array([0,   p.s_thresh, p.v_thresh], dtype=np.uint8)
    upper1 = np.array([10,  255,        255],        dtype=np.uint8)
    lower2 = np.array([170, p.s_thresh, p.v_thresh], dtype=np.uint8)
    upper2 = np.array([179, 255,        255],        dtype=np.uint8)

    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                          cv2.inRange(hsv, lower2, upper2))

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    best, best_score = None, -1.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < p.min_area or area > p.max_area:
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

    # USE YOUR LIVE KNOBS HERE
    area = float(cv2.contourArea(best))
    if area > float(p.area_hi_gate):
        return None, mask

    x, y, w, h2 = cv2.boundingRect(best)
    roi_v = v[y:y+h2, x:x+w]
    peak_v = int(np.max(roi_v)) if roi_v.size else 0
    if peak_v < int(p.peak_v_gate):
        return None, mask

    M = cv2.moments(best)
    if M["m00"] == 0:
        return None, mask
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


# -----------------------------
# Dynamixel pan/tilt minimal controller
# -----------------------------
class DynamixelPanTilt:
    def __init__(self, port: str, baud: int, pan_id: int, tilt_id: int):
        self.port = port
        self.baud = baud
        self.pan_id = pan_id
        self.tilt_id = tilt_id

        self.protocol = 2.0
        self.port_handler = PortHandler(self.port)
        self.packet_handler = PacketHandler(self.protocol)

        self.sync_write_goal = GroupSyncWrite(
            self.port_handler, self.packet_handler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION
        )

        # your safe limits
        self.pan_min, self.pan_max = 0.0, 360.0
        self.tilt_min, self.tilt_max = 62.23, 191.78
        self.pan_cmd = 180.0
        self.tilt_cmd = 120.0

        self.opened = False
        self.torque = False
        self.profile_velocity = 200

    def _check(self, servo_id: int, comm: int, err: int, what: str):
        if comm != 0:
            raise RuntimeError(f"[ID:{servo_id}] {what} COMM FAIL: {self.packet_handler.getTxRxResult(comm)}")
        if err != 0:
            raise RuntimeError(f"[ID:{servo_id}] {what} DXL ERROR: {self.packet_handler.getRxPacketError(err)}")

    def open(self):
        if self.opened:
            return
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open port {self.port}")
        if not self.port_handler.setBaudRate(self.baud):
            raise RuntimeError(f"Failed to set baudrate {self.baud}")
        self.opened = True

        # position mode
        self.torque_off()
        for sid in (self.pan_id, self.tilt_id):
            comm, err = self.packet_handler.write1ByteTxRx(self.port_handler, sid, ADDR_OPERATING_MODE, OPERATING_MODE_POSITION)
            self._check(sid, comm, err, "set operating mode")

    def close(self):
        try:
            self.torque_off()
        except Exception:
            pass
        try:
            if self.opened:
                self.port_handler.closePort()
        finally:
            self.opened = False

    def set_profile_velocity(self, vel: int):
        self.profile_velocity = int(vel)
        for sid in (self.pan_id, self.tilt_id):
            comm, err = self.packet_handler.write4ByteTxRx(self.port_handler, sid, ADDR_PROFILE_VELOCITY, int(vel))
            self._check(sid, comm, err, "set profile velocity")

    def torque_on(self):
        for sid in (self.pan_id, self.tilt_id):
            comm, err = self.packet_handler.write1ByteTxRx(self.port_handler, sid, ADDR_TORQUE_ENABLE, TORQUE_ON)
            self._check(sid, comm, err, "torque on")
        self.torque = True

    def torque_off(self):
        for sid in (self.pan_id, self.tilt_id):
            try:
                comm, err = self.packet_handler.write1ByteTxRx(self.port_handler, sid, ADDR_TORQUE_ENABLE, TORQUE_OFF)
                self._check(sid, comm, err, "torque off")
            except Exception:
                pass
        self.torque = False

    def _deg_to_ticks(self, deg: float) -> int:
        d = wrap_deg_0_360(deg)
        return int(round(d / DEG_PER_TICK)) % int(TICKS_PER_REV)

    def send(self, pan_deg: float, tilt_deg: float, max_step_deg: float):
        pan_deg = clamp(pan_deg, self.pan_min, self.pan_max)
        tilt_deg = clamp(tilt_deg, self.tilt_min, self.tilt_max)

        # per-update clamp to reduce jerk
        pan_deg = clamp(pan_deg, self.pan_cmd - max_step_deg, self.pan_cmd + max_step_deg)
        tilt_deg = clamp(tilt_deg, self.tilt_cmd - max_step_deg, self.tilt_cmd + max_step_deg)

        pan_ticks = self._deg_to_ticks(pan_deg)
        tilt_ticks = self._deg_to_ticks(tilt_deg)

        self.sync_write_goal.clearParam()
        ok1 = self.sync_write_goal.addParam(self.pan_id, int_to_le_bytes(pan_ticks, 4))
        ok2 = self.sync_write_goal.addParam(self.tilt_id, int_to_le_bytes(tilt_ticks, 4))
        if not (ok1 and ok2):
            raise RuntimeError("Failed to add params to sync write")

        comm = self.sync_write_goal.txPacket()
        if comm != 0:
            raise RuntimeError(f"sync_write COMM FAIL: {self.packet_handler.getTxRxResult(comm)}")

        self.pan_cmd = pan_deg
        self.tilt_cmd = tilt_deg

# -----------------------------
# stereo capture thread: depth from stereovision
# -----------------------------
class StereoBundle:
    def __init__(self, npz_path: str):
        d = np.load(npz_path)

        self.mtx_l = d["mtx_l"]
        self.dist_l = d["dist_l"]
        self.mtx_r = d["mtx_r"]
        self.dist_r = d["dist_r"]

        # Rectification + reprojection
        self.R1 = d["R1"]; self.R2 = d["R2"]
        self.P1 = d["P1"]; self.P2 = d["P2"]
        self.Q  = d["Q"]

        # Precomputed rectification maps
        self.map1_l = d["map1_l"]; self.map2_l = d["map2_l"]
        self.map1_r = d["map1_r"]; self.map2_r = d["map2_r"]

        self.image_size = tuple(d["image_size"])  # (w, h)
        self.square_size = float(d["square_size"])  # calibration unit

        # SGBM parameters
        min_disp = 0
        num_disp = 128  # divisible by 16
        block = 7
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block,
            P1=8 * 1 * block * block,
            P2=32 * 1 * block * block,
            disp12MaxDiff=1,
            uniquenessRatio=8,
            speckleWindowSize=80,
            speckleRange=2,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def rectify(self, left_bgr, right_bgr):
        left_r  = cv2.remap(left_bgr,  self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        right_r = cv2.remap(right_bgr, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
        return left_r, right_r

    def depth_at(self, left_rect_bgr, right_rect_bgr, x: int, y: int):
        """
        Returns (depth_m, disparity_map) or (None, disparity_map).
        Uses Q. Units depend on calibration; we convert to meters if it looks like mm.
        """
        gL = cv2.cvtColor(left_rect_bgr, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(right_rect_bgr, cv2.COLOR_BGR2GRAY)

        disp = self.sgbm.compute(gL, gR).astype(np.float32) / 16.0
        d = float(disp[y, x]) if 0 <= x < disp.shape[1] and 0 <= y < disp.shape[0] else -1.0
        if not np.isfinite(d) or d <= 0.5:
            return None, disp

        vec = np.array([x, y, d, 1.0], dtype=np.float32)
        X, Y, Z, W = (self.Q @ vec)
        if W == 0:
            return None, disp
        Z = float(Z / W)

        # Heuristic unit fix:
        # If Z is huge, assume it's mm -> convert to meters.
        depth_m = Z
        if depth_m > 50.0:
            depth_m = depth_m / 1000.0

        if depth_m <= 0 or depth_m > 50:
            return None, disp
        return depth_m, disp
    
    #crops 1920x1200 picture into smaller roi for faster depth calculation reducing latency
    def depth_at_roi(self, left_rect_bgr, right_rect_bgr, x: int, y: int,
                 roi: int = 256, patch: int = 5):
        """
        ROI stereo depth around (x,y).
        Returns (depth_m, disp_roi) or (None, disp_roi).
        """
        h, w = left_rect_bgr.shape[:2]
        half = roi // 2

        x1 = max(0, x - half); x2 = min(w, x + half)
        y1 = max(0, y - half); y2 = min(h, y + half)

        # minimum window for SGBM to behave
        min_w = self.sgbm.getNumDisparities() + 32  # e.g. 160
        if (x2 - x1) < min_w or (y2 - y1) < 80:
            return None, None

        L = left_rect_bgr[y1:y2, x1:x2]
        R = right_rect_bgr[y1:y2, x1:x2]

        gL = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
        disp = self.sgbm.compute(gL, gR).astype(np.float32) / 16.0

        cx = x - x1
        cy = y - y1
        if not (0 <= cx < disp.shape[1] and 0 <= cy < disp.shape[0]):
            return None, disp

        k = int(patch)
        if k < 1: k = 1
        if k % 2 == 0: k += 1
        r = k // 2

        px1 = max(0, cx - r); px2 = min(disp.shape[1], cx + r + 1)
        py1 = max(0, cy - r); py2 = min(disp.shape[0], cy + r + 1)

        vals = disp[py1:py2, px1:px2].reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None, disp

        d = float(np.median(vals))
        if not np.isfinite(d) or d <= 0.5:
            return None, disp

        vec = np.array([x, y, d, 1.0], dtype=np.float32)
        X, Y, Z, W = (self.Q @ vec)
        if W == 0:
            return None, disp

        depth_m = float(Z / W)

        # same heuristic already used
        if depth_m > 50.0:
            depth_m /= 1000.0
        if depth_m <= 0 or depth_m > 50:
            return None, disp
        return depth_m, disp

# -----------------------------
# Worker thread: capture frames, optionally track & drive servos
# -----------------------------
class LaserWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray, np.ndarray, str)  # frame_bgr, mask, status text
    error_signal = QtCore.pyqtSignal(str)

    def __init__(self, store: ParamStore, cam_left: int, cam_right: int, width: int, height: int,
             port: str, baud: int, pan_id: int, tilt_id: int, calib_path: str):
        super().__init__()
        self.store = store
        self.cam_left = cam_left
        self.cam_right = cam_right
        self.calib_path = calib_path
        self.capL = None
        self.capR = None
        self.stereo = None
        self.width = width
        self.height = height
        self.port = port
        self.baud = baud
        self.pan_id = pan_id
        self.tilt_id = tilt_id
        self._stop = threading.Event()
        self.turret = None


        # centroid smoothing (EMA)
        self.cx_f = None
        self.cy_f = None
        self.alpha = 0.25

        # lock-on gating
        self.lock_count = 0
        self.LOCK_N = 1
        #depth
        self.depth_i = 0
        self.roi_fail = 0
        self.full_cooldown = 0

        # tuning knobs 
        self.DEPTH_EVERY_N = 30          # compute depth every frame
        self.ROI_SIZE = 320              # ROI window size for fast depth
        self.ROI_PATCH = 5

        self.ROI_FAIL_TO_FULL = 10       # after N ROI failures in a row -> full depth once
        self.FULL_COOLDOWN_FRAMES = 90   # prevent full depth spamming (~1.5s at 30Hz)
        
        
    def stop(self):
        self._stop.set()

    def run(self):
        try:
            pipeL = gst_v4l2_bgr_pipeline("/dev/video1", self.width, self.height, fps=60)
            pipeR = gst_v4l2_bgr_pipeline("/dev/video0", self.width, self.height, fps=60)

            self.capL = cv2.VideoCapture(pipeL, cv2.CAP_GSTREAMER)
            self.capR = cv2.VideoCapture(pipeR, cv2.CAP_GSTREAMER)
            
            self.readerL = LatestFrame(self.capL, "L")
            self.readerR = LatestFrame(self.capR, "R")
            self.readerL.start()
            self.readerR.start()

            if not self.capL.isOpened() or not self.capR.isOpened():
                raise RuntimeError("Could not open both cameras (left/right). Check /dev/video* indexes.")
 
            for cap in (self.capL, self.capR):
               # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
               # cap.set(cv2.CAP_PROP_EXPOSURE, -7)
               # cap.set(cv2.CAP_PROP_GAIN, 0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
               # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
            # Load stereo calibration + build rectify maps + SGBM
            self.stereo = StereoBundle(self.calib_path)

            self.turret = DynamixelPanTilt(self.port, self.baud, self.pan_id, self.tilt_id)
            self.turret.open()
            
            # don’t torque on until user hits Start Tracking
            last = time.time()
            
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(message)s",
                filename="laser_follow_debug.log",
                filemode="a",
            )
            logger = logging.getLogger("turret")
            stats = Stats()

            while not self._stop.is_set():
                t_loop0 = time.time()
                p = self.store.get()

                t0 = time.time()
                okL, left = self.readerL.get()
                okR, right = self.readerR.get()
                if left is None or right is None:
                    continue

                stats.ms_cap.append((time.time() - t0) * 1000.0)
                if not okL or not okR:
                    raise RuntimeError("Camera read failed (left or right).")

                t0 = time.time()
                left_rect, right_rect = left, right #previously self.stereo.rectify(left, right)
                stats.ms_rect.append((time.time() - t0) * 1000.0)

                # Pick which image to track on
                track_img = left_rect if p.track_source == "Left" else right_rect

                # Build what to DISPLAY
                if p.display_mode == "Left":
                    frame = left_rect
                    x_offset = 0
                elif p.display_mode == "Right":
                    frame = right_rect
                    x_offset = 0
                elif p.display_mode == "Side-by-side":
                    frame = np.hstack([left_rect, right_rect])
                    x_offset = 0 if p.track_source == "Left" else self.width
                    # optional divider line
                    cv2.line(frame, (self.width, 0), (self.width, self.height - 1), (255, 255, 255), 2)
                elif p.display_mode == "Combined":
                    # Simple overlay blend. Good for “both combined” preview.
                    frame = cv2.addWeighted(left_rect, 0.5, right_rect, 0.5, 0.0)
                    x_offset = 0
                else:
                    frame = left_rect
                    x_offset = 0

                # Laser detection on LEFT rectified
                t0 = time.time()
                centroid, mask = find_laser_centroid_red(track_img, p)
                stats.ms_det.append((time.time() - t0) * 1000.0)

                cx0, cy0 = self.width // 2, self.height // 2
                
                # DEBUG overlay to visually confirm it updates
                cv2.putText(frame, f"MODE={p.display_mode} TRACK={p.track_source}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)


                # draw crosshair
                cv2.drawMarker(frame, (cx0 + x_offset, cy0), (255, 255, 255),
                               markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)


                status = ""
                stats.frames += 1
                if centroid is not None:
                    stats.found += 1
                    cx, cy = centroid

                    # EMA smoothing
                    if self.cx_f is None:
                        self.cx_f, self.cy_f = float(cx), float(cy)
                    else:
                        self.cx_f = (1 - self.alpha) * self.cx_f + self.alpha * float(cx)
                        self.cy_f = (1 - self.alpha) * self.cy_f + self.alpha * float(cy)

                    cx_use = int(round(self.cx_f))
                    cy_use = int(round(self.cy_f))

                    err_x = cx_use - cx0
                    err_y = cy_use - cy0
                    
                    # lock-on gating
                    self.lock_count += 1
                    locked = (self.lock_count >= self.LOCK_N)
                    
                    depth_m = None

                    # cooldown tick
                    if self.full_cooldown > 0:
                        self.full_cooldown -= 1

                    # only compute depth every N frames
                    self.depth_i = (self.depth_i + 1) % self.DEPTH_EVERY_N
                    do_depth = (self.depth_i == 0)
                    do_depth = do_depth and locked and (p.track_source == "Left")  # only if locked

                    if do_depth:
                        stats.depth_attempt += 1
                        stats.depth_roi_calls += 1
                        t0 = time.time()
                        # FAST PATH: ROI depth
                        depth_m, _ = self.stereo.depth_at_roi(
                            left_rect, right_rect, cx_use, cy_use,
                            roi=self.ROI_SIZE, patch=self.ROI_PATCH,
                        )
                        stats.ms_depth_roi.append((time.time() - t0) * 1000.0)
                    if depth_m is None:
                        self.roi_fail += 1
                        stats.depth_roi_fail += 1
                    else:
                        self.roi_fail = 0
                        stats.depth_roi_ok += 1

                    # SLOW PATH: full depth ONCE if ROI keeps failing, with cooldown
                    if (depth_m is None and
                        self.roi_fail >= self.ROI_FAIL_TO_FULL and
                        self.full_cooldown == 0):
                        
                        stats.depth_full_calls += 1
                        t0 = time.time()
                        depth_m, _ = self.stereo.depth_at(left_rect, right_rect, cx_use, cy_use)  # slow
                        self.full_cooldown = self.FULL_COOLDOWN_FRAMES
                        self.roi_fail = 0  # reset after escalation
                        stats.ms_depth_full.append((time.time() - t0) * 1000.0)
                    
                    if depth_m is not None:
                        depth_str = f"depth={depth_m:.2f} m"
                    else:
                        depth_str = "depth=NA"

                    cv2.putText(frame, depth_str, (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    status = status + "  " + depth_str

                    
                    # deadband
                    if abs(err_x) < p.deadband_px:
                        err_x = 0
                    if abs(err_y) < p.deadband_px:
                        err_y = 0

                    if locked:
                        stats.locked_frames += 1

                    # show dot
                    cv2.circle(frame, (cx_use + x_offset, cy_use), 6, (0, 255, 255), 2)
                    status = f"FOUND  lock={self.lock_count}/{self.LOCK_N}  err=({err_x},{err_y})  pan={self.turret.pan_cmd:.1f} tilt={self.turret.tilt_cmd:.1f}"

                    # drive servos only if enabled + tracking enabled + locked + nonzero error
                    if p.tracking_enabled and p.servos_enabled and locked and (err_x != 0 or err_y != 0):
                        # ensure torque on + profile velocity applied
                        if not self.turret.torque:
                            self.turret.set_profile_velocity(p.profile_velocity)
                            self.turret.torque_on()

                        dpan = p.pan_dir * (p.deg_per_px_pan * err_x)
                        dtilt = p.tilt_dir * (p.deg_per_px_tilt * err_y)

                        new_pan = self.turret.pan_cmd + dpan
                        new_tilt = self.turret.tilt_cmd + dtilt
                        stats.move_frames += 1
                        self.turret.send(new_pan, new_tilt, p.max_step_deg)
                    else:
                        # if either tracking OR servos disabled, keep torque off
                        if (not p.tracking_enabled or not p.servos_enabled) and self.turret.torque:
                            pass #stops error and keeps torque on
                else:
                    self.lock_count = 0
                    status = "NOT FOUND"
                    pass #keeps torque on021

                # emit frames (throttled UI)
                cv2.putText(frame, stats.hud(), (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                self.ui_i = getattr(self, "ui_i", 0) + 1
                UI_EVERY_N = 2  # show ~30 fps UI while backend can run faster
                if (self.ui_i % UI_EVERY_N) == 0:
                    self.frame_ready.emit(frame, mask, status)
                stats.ms_loop.append((time.time() - t_loop0) * 1000.0)
                stats.log_once_per_sec(logger)

                # rate limit
                #dt = 1.0 / max(1e-6, p.rate_hz)
                #now = time.time()
                #elapsed = now - last
                #if elapsed < dt:
                    #time.sleep(dt - elapsed)
                last = time.time()
    
        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            # stop camera threads first
            try:
                if hasattr(self, "readerL"):
                    self.readerL.stop()
                if hasattr(self, "readerR"):
                    self.readerR.stop()
            except Exception:
                pass
            # close turret
            try:
                if self.turret is not None:
                    self.turret.close()
            except Exception:
                pass
            # release cameras
            try:
                if self.capL is not None:
                    self.capL.release()
                if self.capR is not None:
                    self.capR.release()
            except Exception:
                pass
        
# -----------------------------
# GUI
# -----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Laser Follow Tuner (live knobs)")

        # Defaults
        self.store = ParamStore()
        self.cam_left = 1
        self.cam_right = 0
        self.calib_path = "/home/myles/Documents/autonomous-mosquito-turret/src/stereo_calib/stereo_calibration_full.npz"
        self.width = 1920
        self.height = 1200
        self.port = "/dev/ttyUSB0"
        self.baud = 1000000
        self.pan_id = 1
        self.tilt_id = 2

        self.worker = None

        self._build_ui()
        self._apply_tooltips()

        # start camera preview worker immediately (servos stay off until Start)
        self.start_worker()

    def _build_ui(self):
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)

        layout = QtWidgets.QHBoxLayout(root)

        # Left: controls
        ctrl = QtWidgets.QVBoxLayout()
        layout.addLayout(ctrl, 0)

        # Right: video
        video = QtWidgets.QVBoxLayout()
        layout.addLayout(video, 1)
        
        # Adds ability to change between left/right/side-by-side/combined camera views
        self.cb_display = QtWidgets.QComboBox()
        self.cb_display.addItems(["Left", "Right", "Side-by-side", "Combined"])
        ctrl.addWidget(QtWidgets.QLabel("Display"))
        ctrl.addWidget(self.cb_display)

        def on_display_changed(t):
            print("Display changed to:", t)
            self.store.set_attr("display_mode", t)
        
        self.cb_track = QtWidgets.QComboBox()
        self.cb_track.addItems(["Left", "Right"])
        ctrl.addWidget(QtWidgets.QLabel("Track source"))
        ctrl.addWidget(self.cb_track)
        
        self.cb_display.currentTextChanged.connect(lambda t: self.store.set_attr("display_mode", t))
        self.cb_track.currentTextChanged.connect(lambda t: self.store.set_attr("track_source", t))
        
        self.cb_display.setToolTip("Choose what the GUI shows: Left, Right, both side-by-side, or a combined overlay.")
        self.cb_track.setToolTip("Which camera to use for laser detection + servo control.")

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        ctrl.addLayout(btn_row)

        self.btn_start = QtWidgets.QPushButton("Start Tracking")
        self.btn_stop = QtWidgets.QPushButton("Stop Tracking (Torque Off)")
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)

        self.chk_servos = QtWidgets.QCheckBox("Servo Enable")
        self.chk_servos.setChecked(True)
        ctrl.addWidget(self.chk_servos)

        self.lbl_status = QtWidgets.QLabel("Status: (waiting)")
        ctrl.addWidget(self.lbl_status)

        ctrl.addSpacing(10)

        # Knobs (spinboxes)
        form = QtWidgets.QFormLayout()
        ctrl.addLayout(form)

        def dbl(minv, maxv, step, val):
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(minv, maxv)
            sb.setSingleStep(step)
            sb.setDecimals(4)
            sb.setValue(val)
            return sb

        def integer(minv, maxv, val):
            sb = QtWidgets.QSpinBox()
            sb.setRange(minv, maxv)
            sb.setValue(val)
            return sb

        p = self.store.get()

        self.sb_deg_pan = dbl(0.0001, 0.0500, 0.0005, p.deg_per_px_pan)
        self.sb_deg_tilt = dbl(0.0001, 0.0500, 0.0005, p.deg_per_px_tilt)
        self.sb_max_step = dbl(0.1, 20.0, 0.1, p.max_step_deg)
        self.sb_deadband = integer(0, 200, p.deadband_px)
        self.sb_rate = dbl(5.0, 120.0, 1.0, p.rate_hz)

        self.sb_profile_vel = integer(0, 1500, p.profile_velocity)

        self.sb_v = integer(0, 255, p.v_thresh)
        self.sb_s = integer(0, 255, p.s_thresh)
        self.sb_min_area = dbl(0.0, 200.0, 1.0, p.min_area)
        self.sb_max_area = dbl(1.0, 5000.0, 10.0, p.max_area)

        self.sb_peak_gate = integer(0, 255, p.peak_v_gate)
        self.sb_area_hi_gate = dbl(1.0, 5000.0, 10.0, p.area_hi_gate)

        self.cb_pan_dir = QtWidgets.QComboBox()
        self.cb_pan_dir.addItems(["+1", "-1"])
        self.cb_pan_dir.setCurrentText(str(p.pan_dir))
        self.cb_tilt_dir = QtWidgets.QComboBox()
        self.cb_tilt_dir.addItems(["+1", "-1"])
        self.cb_tilt_dir.setCurrentText(str(p.tilt_dir))

        form.addRow("deg_per_px_pan", self.sb_deg_pan)
        form.addRow("deg_per_px_tilt", self.sb_deg_tilt)
        form.addRow("max_step_deg", self.sb_max_step)
        form.addRow("deadband_px", self.sb_deadband)
        form.addRow("rate_hz", self.sb_rate)
        form.addRow("profile_velocity", self.sb_profile_vel)
        form.addRow("pan_dir", self.cb_pan_dir)
        form.addRow("tilt_dir", self.cb_tilt_dir)

        form.addRow(QtWidgets.QLabel("— Detection —"), QtWidgets.QLabel(""))
        form.addRow("v_thresh", self.sb_v)
        form.addRow("s_thresh", self.sb_s)
        form.addRow("min_area", self.sb_min_area)
        form.addRow("max_area", self.sb_max_area)

        form.addRow(QtWidgets.QLabel("— Confidence Gates —"), QtWidgets.QLabel(""))
        form.addRow("peak_v_gate", self.sb_peak_gate)
        form.addRow("area_hi_gate", self.sb_area_hi_gate)

        ctrl.addStretch(1)

        # Video widgets
        self.video_label = QtWidgets.QLabel()
        self.video_label.setMinimumSize(900, 560)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        video.addWidget(self.video_label, 1)

        self.mask_label = QtWidgets.QLabel()
        self.mask_label.setMinimumSize(450, 280)
        self.mask_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.mask_label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        video.addWidget(self.mask_label, 0)

        # Wire events
        self.btn_start.clicked.connect(self.on_start_tracking)
        self.btn_stop.clicked.connect(self.on_stop_tracking)
        self.chk_servos.stateChanged.connect(self.on_servos_toggle)

        # Live knob updates
        self.sb_deg_pan.valueChanged.connect(lambda v: self.store.set_attr("deg_per_px_pan", float(v)))
        self.sb_deg_tilt.valueChanged.connect(lambda v: self.store.set_attr("deg_per_px_tilt", float(v)))
        self.sb_max_step.valueChanged.connect(lambda v: self.store.set_attr("max_step_deg", float(v)))
        self.sb_deadband.valueChanged.connect(lambda v: self.store.set_attr("deadband_px", int(v)))
        self.sb_rate.valueChanged.connect(lambda v: self.store.set_attr("rate_hz", float(v)))
        self.sb_profile_vel.valueChanged.connect(lambda v: self.store.set_attr("profile_velocity", int(v)))

        self.sb_v.valueChanged.connect(lambda v: self.store.set_attr("v_thresh", int(v)))
        self.sb_s.valueChanged.connect(lambda v: self.store.set_attr("s_thresh", int(v)))
        self.sb_min_area.valueChanged.connect(lambda v: self.store.set_attr("min_area", float(v)))
        self.sb_max_area.valueChanged.connect(lambda v: self.store.set_attr("max_area", float(v)))

        self.sb_peak_gate.valueChanged.connect(lambda v: self.store.set_attr("peak_v_gate", int(v)))
        self.sb_area_hi_gate.valueChanged.connect(lambda v: self.store.set_attr("area_hi_gate", float(v)))

        self.cb_pan_dir.currentTextChanged.connect(lambda t: self.store.set_attr("pan_dir", int(t)))
        self.cb_tilt_dir.currentTextChanged.connect(lambda t: self.store.set_attr("tilt_dir", int(t)))

    def _apply_tooltips(self):
        self.sb_deg_pan.setToolTip("Degrees moved per pixel of horizontal error. Higher = faster/more aggressive, but can overshoot/jitter.")
        self.sb_deg_tilt.setToolTip("Degrees moved per pixel of vertical error. Higher = faster/more aggressive, but can overshoot/jitter.")
        self.sb_max_step.setToolTip("Max degrees the command is allowed to change per update. Caps snap/jerk. Lower = smoother/safer.")
        self.sb_deadband.setToolTip("If error magnitude is below this many pixels, treat it as zero (no movement). Bigger = less jitter near center.")
        self.sb_rate.setToolTip("Update loop rate in Hz. Higher = more responsive but can amplify jitter/buzz. 20–60 typical.")
        self.sb_profile_vel.setToolTip("Dynamixel Profile Velocity. Higher = servo can physically move faster. Too high can sound/feel harsh.")
        self.cb_pan_dir.setToolTip("Flip pan direction if it moves the wrong way (+1 or -1).")
        self.cb_tilt_dir.setToolTip("Flip tilt direction if it moves the wrong way (+1 or -1).")

        self.sb_v.setToolTip("HSV V (brightness) threshold. Higher = ignores dim stuff; lower = sees farther but may pick up noise.")
        self.sb_s.setToolTip("HSV S (saturation) threshold. Lower helps on distant walls; higher rejects weak pinkish noise.")
        self.sb_min_area.setToolTip("Reject blobs smaller than this area. Increase to ignore speckle noise.")
        self.sb_max_area.setToolTip("Reject blobs larger than this area. Prevents big red objects from being treated as the dot.")

        self.sb_peak_gate.setToolTip("Hard gate: require the detected blob to contain pixels with V >= this value. Strongly reduces false positives.")
        self.sb_area_hi_gate.setToolTip("Hard gate: reject any detection larger than this area even if it is red. Helps with big red patches.")

        self.chk_servos.setToolTip("If unchecked, tracking can run and display, but servos will never move.")
        self.btn_start.setToolTip("Enable tracking & torque on (servos will follow dot).")
        self.btn_stop.setToolTip("Disable tracking and torque off (GUI + camera stay running).")

    def start_worker(self):
        if self.worker is not None:
            return
        self.worker = LaserWorker(
            store=self.store,
            cam_left=self.cam_left,
            cam_right=self.cam_right,
            width=self.width,
            height=self.height,
            port=self.port,
            baud=self.baud,
            pan_id=self.pan_id,
            tilt_id=self.tilt_id,
            calib_path=self.calib_path
        )
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_start_tracking(self):
        self.store.set_attr("tracking_enabled", True)

    def on_stop_tracking(self):
        self.store.set_attr("tracking_enabled", False)

    def on_servos_toggle(self, _):
        self.store.set_attr("servos_enabled", bool(self.chk_servos.isChecked()))

    def on_error(self, msg: str):
        self.lbl_status.setText(f"ERROR: {msg}")

    def on_frame(self, frame_bgr: np.ndarray, mask: np.ndarray, status: str):
        self.lbl_status.setText(f"Status: {status}")

        # show camera
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.FastTransformation
        )
        self.video_label.setPixmap(pix)

        # show mask
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mh, mw, _ = mask_vis.shape
        qimg2 = QtGui.QImage(mask_vis.data, mw, mh, 3 * mw, QtGui.QImage.Format_RGB888)
        pix2 = QtGui.QPixmap.fromImage(qimg2).scaled(
            self.mask_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.FastTransformation
        )
        self.mask_label.setPixmap(pix2)

    def closeEvent(self, event):
        # stop tracking first
        self.store.set_attr("tracking_enabled", False)
        try:
            if self.worker is not None:
                self.worker.stop()
                self.worker.wait(1500)
        except Exception:
            pass
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1400, 800)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

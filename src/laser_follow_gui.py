#!/usr/bin/env python3

import os
from pathlib import Path

JETSON_QT_PLUGIN_PATH = "/usr/lib/aarch64-linux-gnu/qt5/plugins"
MAX_60_FPS_EXPOSURE_ABSOLUTE = 160
if os.path.isdir(JETSON_QT_PLUGIN_PATH):
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", JETSON_QT_PLUGIN_PATH)

import sys
import time
import threading
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple

import cv2
import numpy as np

import logging
from logging.handlers import RotatingFileHandler
from collections import deque


from PyQt5 import QtCore, QtGui, QtWidgets
from camera_pipeline import (
    CapturePipeline,
    capture_pipeline_candidates,
    gst_v4l2_bgr_pipeline,
    rolling_rate_hz,
)
from pan_tilt_control import DynamixelPanTilt, FixedRatePanTiltController
from tracking_core import (
    ControlTarget,
    EveryNFrames,
    TargetObservation,
    TimeBasedTargetFilter,
)

# -----------------------------
# Statistics tracking
# -----------------------------
class Stats:
    def __init__(self):
        self.t0 = time.monotonic()
        self.last_log = time.monotonic()

        # counts
        self.frames = 0
        self.found = 0
        self.outliers = 0
        self.frame_times = deque(maxlen=240)

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

    def record_frame(self) -> None:
        self.frames += 1
        now = time.monotonic()
        self.frame_times.append(now)
        while self.frame_times and now - self.frame_times[0] > 2.0:
            self.frame_times.popleft()

    def rolling_fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        elapsed = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / max(1e-6, elapsed)

    def hud(self, source_fps: float = 0.0):
        # quick one-liner for overlay
        return (f"FPS~{self.rolling_fps():.1f} src~{source_fps:.1f} "
                f"found={self.found}/{self.frames} outlier={self.outliers} "
                f"ROI ok={self.depth_roi_ok}/{self.depth_roi_calls} "
                f"FULL={self.depth_full_calls} "
                f"lock={self.locked_frames} move={self.move_frames}")

    def log_once_per_sec(
        self, logger, controller_state=None, camera_health=None
    ):
        now = time.monotonic()
        if now - self.last_log < 1.0:
            return
        self.last_log = now

        logger.info(
            "fps=%.1f found=%d/%d outlier=%d depthAttempt=%d ROI=%d ok=%d fail=%d FULL=%d ok=%d fail=%d "
            "ms(cap=%.1f rect=%.1f det=%.1f roi=%.1f full=%.1f loop=%.1f) "
            "locked=%d move=%d",
            self.rolling_fps(),
            self.found, self.frames,
            self.outliers,
            self.depth_attempt,
            self.depth_roi_calls, self.depth_roi_ok, self.depth_roi_fail,
            self.depth_full_calls, self.depth_full_ok, self.depth_full_fail,
            self._avg(self.ms_cap), self._avg(self.ms_rect), self._avg(self.ms_det),
            self._avg(self.ms_depth_roi), self._avg(self.ms_depth_full), self._avg(self.ms_loop),
            self.locked_frames, self.move_frames
        )
        if controller_state is not None:
            logger.info(
                "control hz=%.1f misses=%d io_ms(read=%.2f write=%.2f) "
                "error_deg(pan=%.2f tilt=%.2f) "
                "health(load=%.1f/%.1f%% voltage=%.1f/%.1fV temp=%d/%dC hw=0x%02x/0x%02x) "
                "target_age=%s",
                controller_state.controller_rate_hz,
                controller_state.deadline_misses,
                controller_state.feedback_read_ms,
                controller_state.command_write_ms,
                controller_state.pan_command_deg - controller_state.pan_actual_deg,
                controller_state.tilt_command_deg - controller_state.tilt_actual_deg,
                controller_state.pan_load_percent,
                controller_state.tilt_load_percent,
                controller_state.pan_voltage,
                controller_state.tilt_voltage,
                controller_state.pan_temperature_c,
                controller_state.tilt_temperature_c,
                controller_state.pan_hardware_error,
                controller_state.tilt_hardware_error,
                (
                    "NA"
                    if controller_state.target_age_s is None
                    else f"{controller_state.target_age_s:.3f}s"
                ),
            )
        if camera_health is not None:
            left_health, right_health = camera_health
            logger.info(
                "camera left(ok=%s fps=%.1f age=%s failures=%d reconnects=%d) "
                "right(ok=%s fps=%.1f age=%s failures=%d reconnects=%d)",
                left_health.connected,
                left_health.capture_fps,
                "NA" if left_health.frame_age_sec is None else f"{left_health.frame_age_sec:.3f}s",
                left_health.consecutive_failures,
                left_health.reconnects,
                right_health.connected,
                right_health.capture_fps,
                "NA" if right_health.frame_age_sec is None else f"{right_health.frame_age_sec:.3f}s",
                right_health.consecutive_failures,
                right_health.reconnects,
            )

# -----------------------------
# Latest frame grabber thread
# -----------------------------
@dataclass(frozen=True)
class CameraHealth:
    connected: bool
    consecutive_failures: int
    reconnects: int
    frame_age_sec: Optional[float]
    capture_fps: float


class LatestFrame:
    REOPEN_AFTER_FAILURES = 5
    REOPEN_BACKOFF_SEC = 0.50

    def __init__(
        self,
        cap: Optional[cv2.VideoCapture],
        name: str,
        reopen_factory: Optional[Callable[[], cv2.VideoCapture]] = None,
    ):
        self.cap = cap
        self.name = name
        self.reopen_factory = reopen_factory
        self.condition = threading.Condition()
        self.frame = None
        self.ok = False
        self.timestamp = 0.0
        self.sequence = 0
        self.consecutive_failures = 0
        self.reconnects = 0
        self.frame_times = deque(maxlen=240)
        self._last_reopen_attempt = 0.0
        self.stop_evt = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.th.start()

    def stop(self):
        self.stop_evt.set()
        cap = self.cap
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        with self.condition:
            self.condition.notify_all()
        self.th.join(timeout=3.0)
        return not self.th.is_alive()

    def _reopen(self) -> bool:
        if self.reopen_factory is None or self.stop_evt.is_set():
            return False
        now = time.monotonic()
        if now - self._last_reopen_attempt < self.REOPEN_BACKOFF_SEC:
            return False
        self._last_reopen_attempt = now
        old_cap = self.cap
        if old_cap is not None:
            try:
                old_cap.release()
            except Exception:
                pass
        try:
            new_cap = self.reopen_factory()
        except Exception:
            self.cap = None
            return False
        if self.stop_evt.is_set():
            new_cap.release()
            self.cap = None
            return False
        if not new_cap.isOpened():
            new_cap.release()
            self.cap = None
            return False
        self.cap = new_cap
        self.consecutive_failures = 0
        self.reconnects += 1
        return True

    def _run(self):
        while not self.stop_evt.is_set():
            cap = self.cap
            if cap is None or not cap.isOpened():
                self.ok = False
                self._reopen()
                self.stop_evt.wait(0.05)
                continue

            try:
                ok, f = cap.read()
            except Exception:
                ok, f = False, None
            captured_at = time.monotonic()
            with self.condition:
                self.ok = ok
                if ok:
                    # VideoCapture returns a new ndarray. Replacing this
                    # reference makes snapshots safe to share without copying.
                    self.frame = f
                    self.timestamp = captured_at
                    self.sequence += 1
                    self.frame_times.append(captured_at)
                    while (
                        self.frame_times
                        and captured_at - self.frame_times[0] > 2.0
                    ):
                        self.frame_times.popleft()
                    self.consecutive_failures = 0
                else:
                    self.consecutive_failures += 1
                self.condition.notify_all()
            if not ok:
                if (
                    not self.stop_evt.is_set()
                    and self.consecutive_failures >= self.REOPEN_AFTER_FAILURES
                ):
                    self._reopen()
                self.stop_evt.wait(0.005)

    def get(self):
        """Return the immutable latest-frame reference without copying pixels."""
        with self.condition:
            return (
                self.ok,
                self.frame,
                self.timestamp,
                self.sequence,
            )

    def wait_for_new(self, last_sequence: int, timeout: float = 0.05):
        """Wait briefly for a sequence newer than ``last_sequence``."""
        with self.condition:
            if self.sequence == last_sequence and not self.stop_evt.is_set():
                self.condition.wait(timeout=timeout)
            return self.ok, self.frame, self.timestamp, self.sequence

    def health(self) -> CameraHealth:
        with self.condition:
            age = (
                None
                if self.timestamp <= 0.0
                else max(0.0, time.monotonic() - self.timestamp)
            )
            return CameraHealth(
                connected=self.ok,
                consecutive_failures=self.consecutive_failures,
                reconnects=self.reconnects,
                frame_age_sec=age,
                capture_fps=rolling_rate_hz(self.frame_times),
            )

# -----------------------------
# Shared live parameters (GUI -> worker)
# -----------------------------
@dataclass
class LiveParams:
    # camera / detection
    display_mode: str = "Left"      # Left | Right | Side-by-side | Combined
    track_source: str = "Left"      # Left | Right

    v_thresh: int = 80
    s_thresh: int = 35
    min_area: float = 1.0
    max_area: float = 2000

    # confidence gates
    peak_v_gate: int = 140         # require bright core (210-245 range)
    local_contrast_gate: int = 25  # peak V above nearby background
    local_red_contrast_gate: int = 6  # red excess above nearby wall
    laser_edge_margin_px: int = 48  # ignore incomplete targets at frame edge
    area_hi_gate: float = 40.0    # reject huge blobs
    smoothing_tau_ms: float = 60.0
    lock_time_ms: float = 50.0
    outlier_speed_px_s: float = 8000.0
    laser_roi_half_size: int = 160

    # Start in the camera's automatic modes so the troubleshooting preview is
    # usable across lighting conditions. Manual controls remain available for
    # repeatable detector tuning after a suitable exposure has been found.
    manual_exposure: bool = False
    # Standard V4L2 units are 100 us. 100 = 10 ms; keep the GUI at or below
    # 16 ms so exposure cannot silently force a nominal 60 FPS stream slower.
    exposure_time_absolute: int = 100
    camera_gain: int = 1
    auto_white_balance: bool = True
    white_balance_temperature: int = 4600
    low_latency_mode: bool = True

    # control
    deg_per_px_pan: float = 0.0060
    deg_per_px_tilt: float = 0.0060
    max_step_deg: float = 2.0
    deadband_px: int = 25
    rate_hz: float = 60.0

    profile_velocity: int = 200
    profile_acceleration: int = 30
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


@dataclass(frozen=True)
class RuntimeConfig:
    cam_left: int
    cam_right: int
    width: int
    height: int
    port: str
    baud: int
    pan_id: int
    tilt_id: int
    calib_path: str


def default_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        cam_left=1,
        cam_right=0,
        width=1920,
        height=1200,
        port="/dev/ttyUSB0",
        baud=1000000,
        pan_id=1,
        tilt_id=2,
        calib_path=str(
            Path(__file__).resolve().parent
            / "stereo_calib"
            / "stereo_calibration_full.npz"
        ),
    )


def v4l2_control_signature(p: LiveParams) -> Tuple[object, ...]:
    """Return the camera settings whose changes require a V4L2 update."""
    return (
        p.manual_exposure,
        p.exposure_time_absolute,
        p.camera_gain,
        p.auto_white_balance,
        p.white_balance_temperature,
        p.low_latency_mode,
    )


def apply_v4l2_camera_controls(device: str, p: LiveParams) -> Optional[str]:
    """Apply known AR0234 controls without making camera startup depend on them.

    Mode controls are sent first because exposure time and white-balance
    temperature are only meaningful after their automatic modes are disabled.
    A failure is returned to the caller for diagnostics; capture may still run.
    """
    mode_controls = (
        f"exposure_auto={1 if p.manual_exposure else 0}",
        f"white_balance_automatic={1 if p.auto_white_balance else 0}",
        f"low_latency_mode={1 if p.low_latency_mode else 0}",
    )
    value_controls = [f"gain={p.camera_gain}"]
    if p.manual_exposure:
        value_controls.append(
            f"exposure_time_absolute={p.exposure_time_absolute}"
        )
    if not p.auto_white_balance:
        value_controls.append(
            f"white_balance_temperature={p.white_balance_temperature}"
        )

    for controls in (mode_controls, tuple(value_controls)):
        command = [
            "v4l2-ctl",
            "-d",
            device,
            "--set-ctrl",
            ",".join(controls),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"{device}: v4l2-ctl failed: {exc}"
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            return (
                f"{device}: v4l2-ctl exited {result.returncode}"
                + (f": {detail}" if detail else "")
            )
    return None


class TargetDetector(Protocol):
    """Interface implemented by the laser detector and a future AI detector."""

    def detect(
        self,
        frame_bgr: np.ndarray,
        params: LiveParams,
        frame_timestamp: float,
    ) -> Tuple[Optional[TargetObservation], np.ndarray]:
        ...


# -----------------------------
# Red laser detection (HSV + local chroma)
# -----------------------------
LASER_MORPH_KERNEL = np.ones((3, 3), np.uint8)


def find_laser_target_red(
    frame_bgr: np.ndarray,
    p: LiveParams,
    frame_timestamp: Optional[float] = None,
    frame_origin: Tuple[int, int] = (0, 0),
    full_frame_shape: Optional[Tuple[int, int]] = None,
    expected_position: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[TargetObservation], np.ndarray]:
    """
    Red-laser dot detection:
    - HSV threshold for red (two hue ranges)
    - Local brightness and red-chroma contrast against the nearby wall
    - Temporal proximity preference while an existing track is active
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    lower1 = np.array([0,   p.s_thresh, p.v_thresh], dtype=np.uint8)
    upper1 = np.array([10,  255,        255],        dtype=np.uint8)
    lower2 = np.array([170, p.s_thresh, p.v_thresh], dtype=np.uint8)
    upper2 = np.array([179, 255,        255],        dtype=np.uint8)

    threshold_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2),
    )

    # Do not use MORPH_OPEN here. Its erosion stage completely removes a real
    # one- or two-pixel laser return before the contour/temporal gates can
    # evaluate it. A single dilation makes those tiny candidates measurable;
    # peak brightness, area, lock time, and outlier rejection still suppress
    # isolated sensor noise before motion is allowed.
    mask = cv2.dilate(
        threshold_mask, LASER_MORPH_KERNEL, iterations=1
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    best, best_score = None, float("-inf")
    origin_x, origin_y = frame_origin
    if full_frame_shape is None:
        frame_height, frame_width = v.shape
    else:
        frame_height, frame_width = full_frame_shape
    edge_margin = max(0, int(p.laser_edge_margin_px))
    for c in contours:
        area = cv2.contourArea(c)
        if (
            area < p.min_area
            or area > p.max_area
            or area > float(p.area_hi_gate)
        ):
            continue
        x, y, w, h2 = cv2.boundingRect(c)

        # Candidates clipped by a frame edge do not have a complete shape or
        # a reliable local background. They also commonly fall outside the
        # valid stereo-rectification area. Reject this strip before scoring so
        # a bright cardboard edge cannot steal lock from an interior laser dot.
        global_x = x + origin_x
        global_y = y + origin_y
        if (
            global_x < edge_margin
            or global_y < edge_margin
            or global_x + w > frame_width - edge_margin
            or global_y + h2 > frame_height - edge_margin
        ):
            continue

        roi_v = v[y:y+h2, x:x+w]
        roi_threshold = threshold_mask[y:y+h2, x:x+w]
        candidate_v = roi_v[roi_threshold != 0]
        if candidate_v.size == 0:
            continue
        peak_v = int(np.max(candidate_v))
        if peak_v < int(p.peak_v_gate):
            continue

        # A laser return is a sharp local brightness peak. Brown cardboard and
        # other reddish surfaces can have the right HSV hue, but their nearby
        # background is usually almost as bright as the candidate itself.
        radius = 8
        local_x1 = max(0, x - radius)
        local_y1 = max(0, y - radius)
        local_x2 = min(v.shape[1], x + w + radius)
        local_y2 = min(v.shape[0], y + h2 + radius)
        local_background_v = float(
            np.median(v[local_y1:local_y2, local_x1:local_x2])
        )
        local_contrast = float(peak_v) - local_background_v
        if local_contrast < float(p.local_contrast_gate):
            continue

        # Compute chroma only in these tiny candidate/local patches. A
        # full-frame signed conversion here would add several large memory
        # allocations to every 60 FPS detection pass.
        roi_bgr = frame_bgr[y:y+h2, x:x+w]
        candidate_bgr = roi_bgr[roi_threshold != 0].astype(np.int16)
        candidate_red_signal = (
            candidate_bgr[:, 2]
            - (candidate_bgr[:, 1] + candidate_bgr[:, 0]) // 2
        )
        peak_red_signal = float(np.max(candidate_red_signal))
        local_bgr = frame_bgr[
            local_y1:local_y2,
            local_x1:local_x2,
        ].astype(np.int16)
        local_red_signal = (
            local_bgr[:, :, 2]
            - (local_bgr[:, :, 1] + local_bgr[:, :, 0]) // 2
        )
        local_background_red = float(
            np.median(local_red_signal)
        )
        local_red_contrast = peak_red_signal - local_background_red
        if local_red_contrast < float(p.local_red_contrast_gate):
            continue

        score = (
            2.0 * local_contrast
            + 3.0 * local_red_contrast
            + 0.25 * peak_v
            - 0.03 * area
        )
        if expected_position is not None:
            candidate_x = global_x + 0.5 * w
            candidate_y = global_y + 0.5 * h2
            distance = float(
                np.hypot(
                    candidate_x - expected_position[0],
                    candidate_y - expected_position[1],
                )
            )
            # Within the tracking ROI, prefer continuity over a slightly
            # brighter distractor. Fast real motion remains possible; after
            # three misses the detector performs an unbiased full reacquire.
            score -= 0.5 * distance
        if score > best_score:
            best_score = score
            best = c

    if best is None:
        return None, mask

    x, y, w, h2 = cv2.boundingRect(best)
    roi_v = v[y:y+h2, x:x+w]
    roi_threshold = threshold_mask[y:y+h2, x:x+w]
    candidate_v = roi_v[roi_threshold != 0]
    peak_v = int(np.max(candidate_v))

    # Brightness-weighted subpixel centroid. The raw (pre-dilation) threshold
    # mask prevents the morphology halo from shifting the measurement.
    roi_binary = roi_threshold.astype(np.float32) / 255.0
    brightness = np.maximum(
        roi_v.astype(np.float32) - float(p.v_thresh) + 1.0,
        0.0,
    )
    weights = roi_binary * brightness
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return None, mask
    column_weights = np.sum(weights, axis=0)
    row_weights = np.sum(weights, axis=1)
    cx = float(x) + float(np.dot(column_weights, np.arange(w))) / weight_sum
    cy = float(y) + float(np.dot(row_weights, np.arange(h2))) / weight_sum
    target = TargetObservation(
        x=cx,
        y=cy,
        confidence=peak_v / 255.0,
        timestamp=time.monotonic() if frame_timestamp is None else frame_timestamp,
        label="red_laser",
        bbox_xyxy=(float(x), float(y), float(x + w), float(y + h2)),
    )
    return target, mask


class RedLaserDetector:
    """Full-frame acquisition followed by expanding, low-cost ROI tracking."""

    FULL_SEARCH_AFTER_MISSES = 3

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._last_x = None
        self._last_y = None
        self._misses = 0

    def detect(
        self,
        frame_bgr: np.ndarray,
        params: LiveParams,
        frame_timestamp: float,
    ) -> Tuple[Optional[TargetObservation], np.ndarray]:
        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = 0, 0, width, height
        using_tracking_roi = (
            self._last_x is not None
            and self._misses < self.FULL_SEARCH_AFTER_MISSES
        )
        if using_tracking_roi:
            base_half_size = max(32, int(params.laser_roi_half_size))
            half_size = base_half_size * (2 ** self._misses)
            center_x = int(round(self._last_x))
            center_y = int(round(self._last_y))
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(width, center_x + half_size)
            y2 = min(height, center_y + half_size)

        target, mask = find_laser_target_red(
            frame_bgr[y1:y2, x1:x2],
            params,
            frame_timestamp,
            frame_origin=(x1, y1),
            full_frame_shape=(height, width),
            expected_position=(
                (self._last_x, self._last_y)
                if using_tracking_roi
                else None
            ),
        )
        if target is None:
            self._misses += 1
            return None, mask

        bbox = target.bbox_xyxy
        adjusted_bbox = None
        if bbox is not None:
            adjusted_bbox = (
                bbox[0] + x1,
                bbox[1] + y1,
                bbox[2] + x1,
                bbox[3] + y1,
            )
        adjusted = TargetObservation(
            x=target.x + x1,
            y=target.y + y1,
            confidence=target.confidence,
            timestamp=target.timestamp,
            label=target.label,
            bbox_xyxy=adjusted_bbox,
        )
        self._last_x = adjusted.x
        self._last_y = adjusted.y
        self._misses = 0
        return adjusted, mask


def find_laser_centroid_red(frame_bgr, p: LiveParams):
    """Backward-compatible wrapper for callers that only need a centroid."""
    target, mask = find_laser_target_red(frame_bgr, p)
    return (None if target is None else target.centroid), mask


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

        # Convert the saved floating-point maps once at startup. OpenCV's
        # fixed-point map representation is substantially faster for remap().
        self.map1_l, self.map2_l = cv2.convertMaps(
            d["map1_l"], d["map2_l"], cv2.CV_16SC2
        )
        self.map1_r, self.map2_r = cv2.convertMaps(
            d["map1_r"], d["map2_r"], cv2.CV_16SC2
        )

        self.image_size = tuple(int(v) for v in d["image_size"])  # (w, h)
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
        left_size = (left_bgr.shape[1], left_bgr.shape[0])
        right_size = (right_bgr.shape[1], right_bgr.shape[0])
        if left_size != self.image_size or right_size != self.image_size:
            raise ValueError(
                "Stereo frame size does not match calibration: "
                f"left={left_size} right={right_size} calibration={self.image_size}"
            )
        left_r  = cv2.remap(left_bgr,  self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        right_r = cv2.remap(right_bgr, self.map1_r, self.map2_r, cv2.INTER_LINEAR)
        return left_r, right_r

    def rectify_left_point(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Map one raw left-camera point into the rectified image cheaply."""
        point = np.array([[[float(x), float(y)]]], dtype=np.float32)
        rectified = cv2.undistortPoints(
            point,
            self.mtx_l,
            self.dist_l,
            R=self.R1,
            P=self.P1,
        )
        rx, ry = (float(v) for v in rectified[0, 0])
        if not np.isfinite(rx) or not np.isfinite(ry):
            return None
        px, py = int(round(rx)), int(round(ry))
        width, height = self.image_size
        if not (0 <= px < width and 0 <= py < height):
            return None
        return px, py

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


@dataclass(frozen=True)
class DepthRequest:
    left_bgr: np.ndarray
    right_bgr: np.ndarray
    target: TargetObservation
    captured_at_l: float
    captured_at_r: float


@dataclass(frozen=True)
class DepthResult:
    source_timestamp: float
    depth_m: Optional[float] = None
    note: str = ""
    rect_ms: float = 0.0
    roi_ms: float = 0.0
    full_ms: float = 0.0
    roi_attempted: bool = False
    roi_ok: bool = False
    full_attempted: bool = False
    full_ok: bool = False


@dataclass(frozen=True)
class PreviewFrame:
    frame_bgr: np.ndarray
    mask: np.ndarray
    status: str


class StereoDepthWorker:
    """Run bounded, latest-only stereo work away from the control loop."""

    def __init__(
        self,
        stereo: StereoBundle,
        roi_size: int,
        roi_patch: int,
        roi_fail_to_full: int,
        full_cooldown_sec: float,
        max_stereo_skew_sec: float,
    ):
        self.stereo = stereo
        self.roi_size = roi_size
        self.roi_patch = roi_patch
        self.roi_fail_to_full = roi_fail_to_full
        self.full_cooldown_sec = full_cooldown_sec
        self.max_stereo_skew_sec = max_stereo_skew_sec

        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._busy = False
        self._pending = None
        self._latest_result = None
        self._roi_fail = 0
        self._next_full_allowed_at = 0.0
        self._thread = threading.Thread(
            target=self._run,
            name="stereo-depth",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> bool:
        with self._lock:
            self._stop = True
            self._pending = None
        self._wake.set()
        self._thread.join(timeout=2.0)
        return not self._thread.is_alive()

    def submit(self, request: DepthRequest) -> bool:
        """Accept work only when idle so depth can never build a backlog."""
        with self._lock:
            if self._stop or self._busy or self._pending is not None:
                return False
            self._pending = request
        self._wake.set()
        return True

    def take_result(self) -> Optional[DepthResult]:
        with self._lock:
            result = self._latest_result
            self._latest_result = None
        return result

    def _run(self) -> None:
        while True:
            self._wake.wait()
            with self._lock:
                if self._stop:
                    return
                request = self._pending
                self._pending = None
                self._busy = request is not None
                self._wake.clear()

            if request is None:
                continue

            result = self._process(request)
            with self._lock:
                self._latest_result = result
                self._busy = False

    def _process(self, request: DepthRequest) -> DepthResult:
        source_timestamp = request.target.timestamp
        stereo_skew = abs(request.captured_at_l - request.captured_at_r)
        if stereo_skew > self.max_stereo_skew_sec:
            return DepthResult(
                source_timestamp=source_timestamp,
                note=f"pair skew={stereo_skew * 1000.0:.0f}ms",
            )

        try:
            t0 = time.perf_counter()
            left_depth, right_depth = self.stereo.rectify(
                request.left_bgr, request.right_bgr
            )
            rect_ms = (time.perf_counter() - t0) * 1000.0

            depth_point = self.stereo.rectify_left_point(
                request.target.x, request.target.y
            )
            if depth_point is None:
                self._roi_fail += 1
                return DepthResult(
                    source_timestamp=source_timestamp,
                    note="target outside rectified image",
                    rect_ms=rect_ms,
                )

            depth_x, depth_y = depth_point
            t0 = time.perf_counter()
            depth_m, _ = self.stereo.depth_at_roi(
                left_depth,
                right_depth,
                depth_x,
                depth_y,
                roi=self.roi_size,
                patch=self.roi_patch,
            )
            roi_ms = (time.perf_counter() - t0) * 1000.0
            roi_ok = depth_m is not None

            if roi_ok:
                self._roi_fail = 0
            else:
                self._roi_fail += 1

            full_attempted = False
            full_ok = False
            full_ms = 0.0
            now = time.monotonic()
            if (depth_m is None and
                self._roi_fail >= self.roi_fail_to_full and
                now >= self._next_full_allowed_at):
                full_attempted = True
                t0 = time.perf_counter()
                depth_m, _ = self.stereo.depth_at(
                    left_depth, right_depth, depth_x, depth_y
                )
                full_ms = (time.perf_counter() - t0) * 1000.0
                full_ok = depth_m is not None
                self._roi_fail = 0
                self._next_full_allowed_at = now + self.full_cooldown_sec

            note = "" if depth_m is not None else "stereo depth unavailable"
            return DepthResult(
                source_timestamp=source_timestamp,
                depth_m=depth_m,
                note=note,
                rect_ms=rect_ms,
                roi_ms=roi_ms,
                full_ms=full_ms,
                roi_attempted=True,
                roi_ok=roi_ok,
                full_attempted=full_attempted,
                full_ok=full_ok,
            )
        except Exception as exc:
            return DepthResult(
                source_timestamp=source_timestamp,
                note=f"depth error: {exc}",
            )

# -----------------------------
# Worker thread: capture frames, optionally track & drive servos
# -----------------------------
class LaserWorker(QtCore.QThread):
    error_signal = QtCore.pyqtSignal(str)
    status_signal = QtCore.pyqtSignal(str)

    def __init__(self, store: ParamStore, cam_left: int, cam_right: int, width: int, height: int,
             port: str, baud: int, pan_id: int, tilt_id: int, calib_path: str,
             detector: Optional[TargetDetector] = None):
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
        self.controller = None
        self.servo_note = "servo not connected"
        self._next_servo_retry_at = 0.0
        self._servo_fault_latched = False
        self._last_camera_status_at = 0.0
        self._last_camera_control_signature = None
        self.camera_control_note = ""
        self.detector = detector if detector is not None else RedLaserDetector()
        self._last_track_sequence = {"Left": 0, "Right": 0}
        self._active_track_source = None

        # Preview delivery is a latest-only, demand-driven side channel. A
        # headless caller that never requests a preview pays no composition cost.
        self._preview_lock = threading.Lock()
        self._preview_requested = False
        self._latest_preview = None


        self.target_filter = TimeBasedTargetFilter(
            smoothing_tau_sec=0.06,
            lock_time_sec=0.05,
            reset_after_sec=0.15,
            max_speed_px_sec=8000.0,
            jump_allowance_px=40.0,
        )
        # Depth runs much more slowly than detection/control. It consumes a
        # model-neutral TargetObservation, so a future AI detector can use the
        # same rectification and depth path.
        self.DEPTH_EVERY_N = 30
        self.depth_cadence = EveryNFrames(self.DEPTH_EVERY_N)
        self.depth_worker = None
        self.last_depth_m = None
        self.last_depth_at = 0.0
        self.last_depth_note = ""

        # Depth tuning
        self.ROI_SIZE = 320              # ROI window size for fast depth
        self.ROI_PATCH = 5

        self.ROI_FAIL_TO_FULL = 10       # after N ROI failures in a row -> full depth once
        self.FULL_COOLDOWN_SEC = 1.5
        self.DEPTH_STALE_SEC = 2.0
        self.MAX_STEREO_SKEW_SEC = 0.020
        
        
    def stop(self):
        self._stop.set()

    def clear_servo_fault(self) -> None:
        self._servo_fault_latched = False
        self._next_servo_retry_at = 0.0

    def _disconnect_servo(self) -> None:
        controller = self.controller
        turret = self.turret
        self.controller = None
        self.turret = None
        safe_to_close = True
        if controller is not None:
            try:
                controller.stop()
            except Exception:
                # Do not close a serial port that a blocked controller thread
                # could still be using. Its stop flag and servo watchdog remain
                # armed, so it will torque off when the SDK call returns.
                safe_to_close = False
        if turret is not None and safe_to_close:
            try:
                turret.close()
            except Exception:
                pass

    def _try_connect_servo(self, logger) -> None:
        if self.controller is not None:
            return
        turret = None
        try:
            turret = DynamixelPanTilt(
                self.port, self.baud, self.pan_id, self.tilt_id
            )
            turret.open()
            controller = FixedRatePanTiltController(turret, self.store)
            controller.start()
            self.turret = turret
            self.controller = controller
            self.servo_note = ""
            logger.info(
                "configured servo limits pan=%.2f..%.2f deg "
                "tilt=%.2f..%.2f deg; initial pan=%.2f deg tilt=%.2f deg",
                turret.pan_min,
                turret.pan_max,
                turret.tilt_min,
                turret.tilt_max,
                turret.pan_actual,
                turret.tilt_actual,
            )
        except Exception as exc:
            if turret is not None:
                try:
                    turret.close()
                except Exception:
                    pass
            self.servo_note = str(exc)
            self._next_servo_retry_at = time.monotonic() + 2.0
            logger.warning("servo unavailable; vision remains active: %s", exc)

    def _apply_camera_controls(self, p: LiveParams, logger, force=False) -> None:
        signature = v4l2_control_signature(p)
        if not force and signature == self._last_camera_control_signature:
            return

        errors = []
        for camera_index in (self.cam_left, self.cam_right):
            device = f"/dev/video{camera_index}"
            error = apply_v4l2_camera_controls(device, p)
            if error is not None:
                errors.append(error)

        self._last_camera_control_signature = signature
        if errors:
            self.camera_control_note = "; ".join(errors)
            logger.warning(
                "camera controls could not be fully applied; capture will continue: %s",
                self.camera_control_note,
            )
        else:
            self.camera_control_note = ""
            logger.info(
                "camera controls manual_exposure=%s exposure=%d gain=%d "
                "auto_wb=%s wb_temp=%d low_latency=%s",
                p.manual_exposure,
                p.exposure_time_absolute,
                p.camera_gain,
                p.auto_white_balance,
                p.white_balance_temperature,
                p.low_latency_mode,
            )

    def request_preview(self) -> None:
        with self._preview_lock:
            self._preview_requested = True

    def take_preview(self) -> Optional[PreviewFrame]:
        with self._preview_lock:
            preview = self._latest_preview
            self._latest_preview = None
        return preview

    def _consume_preview_request(self) -> bool:
        with self._preview_lock:
            requested = self._preview_requested
            self._preview_requested = False
        return requested

    def _publish_preview(self, frame_bgr: np.ndarray, mask: np.ndarray, status: str) -> None:
        with self._preview_lock:
            self._latest_preview = PreviewFrame(frame_bgr, mask, status)

    def _compose_preview_frame(
        self,
        left: Optional[np.ndarray],
        right: Optional[np.ndarray],
        p: LiveParams,
    ):
        if left is None and right is None:
            raise RuntimeError("no camera frame is available for preview")
        if left is None:
            return right.copy(), 0
        if right is None:
            return left.copy(), 0
        if p.display_mode == "Left":
            return left.copy(), 0
        if p.display_mode == "Right":
            return right.copy(), 0
        if p.display_mode == "Side-by-side":
            frame = np.hstack([left, right])
            cv2.line(
                frame,
                (self.width, 0),
                (self.width, self.height - 1),
                (255, 255, 255),
                2,
            )
            x_offset = 0 if p.track_source == "Left" else self.width
            return frame, x_offset
        if p.display_mode == "Combined":
            return cv2.addWeighted(left, 0.5, right, 0.5, 0.0), 0
        return left.copy(), 0

    def run(self):
        try:
            logger = logging.getLogger("turret")
            logger.setLevel(logging.INFO)
            logger.propagate = False
            if not logger.handlers:
                log_handler = RotatingFileHandler(
                    "laser_follow_debug.log",
                    maxBytes=5 * 1024 * 1024,
                    backupCount=3,
                )
                log_handler.setFormatter(
                    logging.Formatter("%(asctime)s %(levelname)s %(message)s")
                )
                logger.addHandler(log_handler)

            self._apply_camera_controls(self.store.get(), logger, force=True)

            device_l = f"/dev/video{self.cam_left}"
            device_r = f"/dev/video{self.cam_right}"
            pipelines_by_device = {
                device_l: capture_pipeline_candidates(
                    device_l, self.width, self.height, fps=60
                ),
                device_r: capture_pipeline_candidates(
                    device_r, self.width, self.height, fps=60
                ),
            }
            pipeline_index = {device_l: 0, device_r: 0}

            def open_capture(
                device: str,
                pipelines: Tuple[CapturePipeline, ...],
                advance: bool = False,
            ) -> cv2.VideoCapture:
                # A reconnect can reset driver controls, so reapply them before
                # every new GStreamer capture instance.
                error = apply_v4l2_camera_controls(device, self.store.get())
                if error is not None:
                    logger.warning("camera reconnect controls: %s", error)
                start_index = pipeline_index[device]
                if advance:
                    start_index = (start_index + 1) % len(pipelines)
                last_capture = None
                for offset in range(len(pipelines)):
                    index = (start_index + offset) % len(pipelines)
                    candidate = pipelines[index]
                    if last_capture is not None:
                        last_capture.release()
                    capture = cv2.VideoCapture(
                        candidate.pipeline, cv2.CAP_GSTREAMER
                    )
                    if capture.isOpened():
                        pipeline_index[device] = index
                        logger.info(
                            "camera %s opened with %s pipeline",
                            device,
                            candidate.name,
                        )
                        return capture
                    logger.warning(
                        "camera %s could not open %s pipeline",
                        device,
                        candidate.name,
                    )
                    last_capture = capture
                return last_capture

            self.capL = open_capture(device_l, pipelines_by_device[device_l])
            self.capR = open_capture(device_r, pipelines_by_device[device_r])

            self.readerL = LatestFrame(
                self.capL,
                "L",
                reopen_factory=lambda: open_capture(
                    device_l,
                    pipelines_by_device[device_l],
                    advance=True,
                ),
            )
            self.readerR = LatestFrame(
                self.capR,
                "R",
                reopen_factory=lambda: open_capture(
                    device_r,
                    pipelines_by_device[device_r],
                    advance=True,
                ),
            )
            self.readerL.start()
            self.readerR.start()

            # Load stereo calibration + build rectify maps + SGBM
            try:
                self.stereo = StereoBundle(self.calib_path)
                self.depth_worker = StereoDepthWorker(
                    stereo=self.stereo,
                    roi_size=self.ROI_SIZE,
                    roi_patch=self.ROI_PATCH,
                    roi_fail_to_full=self.ROI_FAIL_TO_FULL,
                    full_cooldown_sec=self.FULL_COOLDOWN_SEC,
                    max_stereo_skew_sec=self.MAX_STEREO_SKEW_SEC,
                )
                self.depth_worker.start()
            except Exception as exc:
                self.stereo = None
                self.depth_worker = None
                self.last_depth_note = f"stereo disabled: {exc}"

            stats = Stats()
            if self.last_depth_note:
                logger.warning(self.last_depth_note)
            self._try_connect_servo(logger)

            while not self._stop.is_set():
                t_loop0 = time.perf_counter()
                p = self.store.get()
                self._apply_camera_controls(p, logger)

                if p.track_source != self._active_track_source:
                    self._active_track_source = p.track_source
                    self.target_filter.reset()
                    if hasattr(self.detector, "reset"):
                        self.detector.reset()
                    self.depth_cadence.reset(due_immediately=True)
                    if self.controller is not None:
                        self.controller.clear_target()

                controller_state = (
                    None if self.controller is None else self.controller.snapshot()
                )
                if (
                    controller_state is not None
                    and controller_state.error is not None
                ):
                    self.servo_note = controller_state.error
                    logger.error("servo controller stopped: %s", self.servo_note)
                    self._disconnect_servo()
                    self._servo_fault_latched = True
                    controller_state = None
                if (
                    self.controller is None
                    and p.servos_enabled
                    and not self._servo_fault_latched
                    and time.monotonic() >= self._next_servo_retry_at
                ):
                    self._try_connect_servo(logger)
                    controller_state = (
                        None
                        if self.controller is None
                        else self.controller.snapshot()
                    )
                if controller_state is not None:
                    stats.move_frames = controller_state.move_updates

                depth_result = (
                    None
                    if self.depth_worker is None
                    else self.depth_worker.take_result()
                )
                if depth_result is not None:
                    self.last_depth_note = depth_result.note
                    if depth_result.rect_ms > 0.0:
                        stats.ms_rect.append(depth_result.rect_ms)
                    if depth_result.roi_attempted:
                        stats.depth_roi_calls += 1
                        stats.ms_depth_roi.append(depth_result.roi_ms)
                        if depth_result.roi_ok:
                            stats.depth_roi_ok += 1
                        else:
                            stats.depth_roi_fail += 1
                    if depth_result.full_attempted:
                        stats.depth_full_calls += 1
                        stats.ms_depth_full.append(depth_result.full_ms)
                        if depth_result.full_ok:
                            stats.depth_full_ok += 1
                        else:
                            stats.depth_full_fail += 1
                    if depth_result.depth_m is not None:
                        self.last_depth_m = depth_result.depth_m
                        self.last_depth_at = depth_result.source_timestamp
                    elif depth_result.note.startswith("depth error:"):
                        logger.warning(depth_result.note)

                t0 = time.perf_counter()
                if p.track_source == "Left":
                    okL, left, captured_at_l, sequence_l = self.readerL.wait_for_new(
                        self._last_track_sequence["Left"]
                    )
                    okR, right, captured_at_r, sequence_r = self.readerR.get()
                    track_sequence = sequence_l
                else:
                    okR, right, captured_at_r, sequence_r = self.readerR.wait_for_new(
                        self._last_track_sequence["Right"]
                    )
                    okL, left, captured_at_l, sequence_l = self.readerL.get()
                    track_sequence = sequence_r
                track_frame = left if p.track_source == "Left" else right
                if track_frame is None:
                    now = time.monotonic()
                    if now - self._last_camera_status_at >= 0.5:
                        selected_reader = (
                            self.readerL if p.track_source == "Left" else self.readerR
                        )
                        health = selected_reader.health()
                        self.status_signal.emit(
                            f"Waiting for {p.track_source} camera; "
                            f"failures={health.consecutive_failures} "
                            f"reconnects={health.reconnects}"
                        )
                        self._last_camera_status_at = now
                    continue

                stats.ms_cap.append((time.perf_counter() - t0) * 1000.0)
                if track_sequence == self._last_track_sequence[p.track_source]:
                    now = time.monotonic()
                    selected_reader = (
                        self.readerL if p.track_source == "Left" else self.readerR
                    )
                    health = selected_reader.health()
                    if (
                        health.frame_age_sec is not None
                        and health.frame_age_sec >= 0.5
                        and now - self._last_camera_status_at >= 0.5
                    ):
                        self.status_signal.emit(
                            f"Reconnecting {p.track_source} camera; "
                            f"last frame={health.frame_age_sec:.1f}s ago "
                            f"reconnects={health.reconnects}"
                        )
                        self._last_camera_status_at = now
                    continue
                self._last_track_sequence[p.track_source] = track_sequence

                # Detection and control stay on raw frames. Full-frame stereo
                # rectification happens only on scheduled depth updates.
                track_img = track_frame
                track_timestamp = captured_at_l if p.track_source == "Left" else captured_at_r

                # Detector output is model-neutral. A future AI detector only
                # needs to implement TargetDetector.detect().
                t0 = time.perf_counter()
                target, mask = self.detector.detect(track_img, p, track_timestamp)
                stats.ms_det.append((time.perf_counter() - t0) * 1000.0)

                filtered_target = None
                if target is not None:
                    stats.found += 1
                    self.target_filter.configure(
                        smoothing_tau_sec=p.smoothing_tau_ms / 1000.0,
                        lock_time_sec=p.lock_time_ms / 1000.0,
                        max_speed_px_sec=p.outlier_speed_px_s,
                    )
                    filtered_target = self.target_filter.update(
                        target.x, target.y, target.timestamp
                    )
                    if not filtered_target.accepted:
                        if filtered_target.outlier:
                            stats.outliers += 1
                        target = None

                cx0, cy0 = self.width // 2, self.height // 2
                status = ""
                preview_target = None
                depth_str = "depth=NA"
                stats.record_frame()
                if target is not None:
                    cx_use = int(round(filtered_target.x))
                    cy_use = int(round(filtered_target.y))
                    preview_target = (cx_use, cy_use)

                    err_x = filtered_target.x - cx0
                    err_y = filtered_target.y - cy0
                    
                    locked = filtered_target.locked

                    if self.controller is not None:
                        self.controller.publish_target(
                            ControlTarget(
                                error_x_px=float(err_x),
                                error_y_px=float(err_y),
                                timestamp=target.timestamp,
                                locked=locked,
                            )
                        )
                    
                    depth_eligible = locked and (p.track_source == "Left")
                    if not depth_eligible:
                        self.depth_cadence.reset(due_immediately=True)
                    do_depth = (
                        depth_eligible
                        and self.depth_worker is not None
                        and left is not None
                        and right is not None
                        and self.depth_cadence.step()
                    )

                    if do_depth:
                        accepted = self.depth_worker.submit(
                            DepthRequest(
                                left_bgr=left,
                                right_bgr=right,
                                target=target,
                                captured_at_l=captured_at_l,
                                captured_at_r=captured_at_r,
                            )
                        )
                        if accepted:
                            stats.depth_attempt += 1

                    depth_age = time.monotonic() - self.last_depth_at
                    if self.last_depth_m is not None and depth_age <= self.DEPTH_STALE_SEC:
                        depth_str = f"depth={self.last_depth_m:.2f} m age={depth_age:.1f}s"
                    elif self.last_depth_note:
                        depth_str = f"depth=NA ({self.last_depth_note})"
                    else:
                        depth_str = "depth=NA"

                    # deadband
                    if abs(err_x) < p.deadband_px:
                        err_x = 0
                    if abs(err_y) < p.deadband_px:
                        err_y = 0

                    if locked:
                        stats.locked_frames += 1

                    if controller_state is None:
                        servo_status = f"servo=OFFLINE ({self.servo_note})"
                    else:
                        servo_status = (
                            f"torque={'ON' if controller_state.torque_enabled else 'OFF'} "
                            f"pan={controller_state.pan_actual_deg:.1f}/"
                            f"{controller_state.pan_command_deg:.1f} "
                            f"tilt={controller_state.tilt_actual_deg:.1f}/"
                            f"{controller_state.tilt_command_deg:.1f} "
                            f"ctrl={controller_state.controller_rate_hz:.1f}Hz"
                        )
                    status = (
                        f"FOUND lock={'YES' if locked else 'ACQUIRING'} "
                        f"err=({err_x:.1f},{err_y:.1f}) {servo_status} {depth_str}"
                    )
                else:
                    if self.controller is not None:
                        self.controller.clear_target()
                    self.target_filter.miss(track_timestamp)
                    self.depth_cadence.reset(due_immediately=True)
                    if controller_state is None:
                        status = f"NOT FOUND servo=OFFLINE ({self.servo_note})"
                    else:
                        status = (
                            "NOT FOUND "
                            f"torque={'ON' if controller_state.torque_enabled else 'OFF'} "
                            f"pan={controller_state.pan_actual_deg:.1f} "
                            f"tilt={controller_state.tilt_actual_deg:.1f}"
                        )

                if self._consume_preview_request():
                    frame, x_offset = self._compose_preview_frame(left, right, p)
                    cv2.putText(
                        frame,
                        f"MODE={p.display_mode} TRACK={p.track_source}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                    cv2.drawMarker(
                        frame,
                        (cx0 + x_offset, cy0),
                        (255, 255, 255),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=18,
                        thickness=2,
                    )
                    if preview_target is not None:
                        preview_x, preview_y = preview_target
                        cv2.circle(
                            frame,
                            (preview_x + x_offset, preview_y),
                            6,
                            (0, 255, 255),
                            2,
                        )
                        cv2.putText(
                            frame,
                            depth_str,
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (255, 255, 255),
                            2,
                        )
                    selected_health = (
                        self.readerL.health()
                        if p.track_source == "Left"
                        else self.readerR.health()
                    )
                    cv2.putText(
                        frame,
                        stats.hud(selected_health.capture_fps),
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    self._publish_preview(frame, mask, status)
                stats.ms_loop.append((time.perf_counter() - t_loop0) * 1000.0)
                stats.log_once_per_sec(
                    logger,
                    controller_state,
                    (self.readerL.health(), self.readerR.health()),
                )


        except Exception as e:
            logging.getLogger("turret").exception("laser worker stopped")
            self.error_signal.emit(str(e))
        finally:
            # The controller owns the serial bus. Stop it before closing the
            # port or tearing down the rest of the worker.
            self._disconnect_servo()
            # Stop background stereo work before tearing down shared state.
            try:
                if self.depth_worker is not None:
                    self.depth_worker.stop()
            except Exception:
                pass
            # stop camera threads first
            try:
                if hasattr(self, "readerL"):
                    self.readerL.stop()
                if hasattr(self, "readerR"):
                    self.readerR.stop()
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
    PREVIEW_FPS = 30

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Laser Follow Tuner (live knobs)")

        # Defaults
        self.store = ParamStore()
        config = default_runtime_config()
        self.cam_left = config.cam_left
        self.cam_right = config.cam_right
        self.calib_path = config.calib_path
        self.width = config.width
        self.height = config.height
        self.port = config.port
        self.baud = config.baud
        self.pan_id = config.pan_id
        self.tilt_id = config.tilt_id

        self.worker = None
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setInterval(max(1, round(1000 / self.PREVIEW_FPS)))
        self.preview_timer.timeout.connect(self.poll_preview)

        self._build_ui()
        self._apply_tooltips()

        # start camera preview worker immediately (servos stay off until Start)
        self.start_worker()
        self.preview_timer.start()

    def _build_ui(self):
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)

        layout = QtWidgets.QHBoxLayout(root)

        # Left: controls. Keep every tuning control reachable on smaller remote
        # desktops instead of allowing the form to extend below the window.
        controls_widget = QtWidgets.QWidget()
        ctrl = QtWidgets.QVBoxLayout(controls_widget)
        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        controls_scroll.setMinimumWidth(390)
        controls_scroll.setWidget(controls_widget)
        layout.addWidget(controls_scroll, 0)

        # Right: video
        video = QtWidgets.QVBoxLayout()
        layout.addLayout(video, 1)
        
        # Adds ability to change between left/right/side-by-side/combined camera views
        self.cb_display = QtWidgets.QComboBox()
        self.cb_display.addItems(["Left", "Right", "Side-by-side", "Combined"])
        ctrl.addWidget(QtWidgets.QLabel("Display"))
        ctrl.addWidget(self.cb_display)

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

        self.sb_profile_vel = integer(2, 1500, p.profile_velocity)
        self.sb_profile_accel = integer(1, 750, p.profile_acceleration)

        self.sb_v = integer(0, 255, p.v_thresh)
        self.sb_s = integer(0, 255, p.s_thresh)
        self.sb_min_area = dbl(0.0, 200.0, 1.0, p.min_area)
        self.sb_max_area = dbl(1.0, 5000.0, 10.0, p.max_area)

        self.sb_peak_gate = integer(0, 255, p.peak_v_gate)
        self.sb_local_contrast_gate = integer(
            0, 255, p.local_contrast_gate
        )
        self.sb_local_red_contrast_gate = integer(
            0, 255, p.local_red_contrast_gate
        )
        self.sb_laser_edge_margin = integer(
            0, 480, p.laser_edge_margin_px
        )
        self.sb_area_hi_gate = dbl(1.0, 5000.0, 10.0, p.area_hi_gate)
        self.sb_smoothing_tau = dbl(5.0, 500.0, 5.0, p.smoothing_tau_ms)
        self.sb_lock_time = dbl(0.0, 1000.0, 10.0, p.lock_time_ms)
        self.sb_outlier_speed = dbl(
            100.0, 20000.0, 100.0, p.outlier_speed_px_s
        )
        self.sb_laser_roi = integer(32, 960, p.laser_roi_half_size)

        self.chk_manual_exposure = QtWidgets.QCheckBox()
        self.chk_manual_exposure.setChecked(p.manual_exposure)
        self.sb_exposure = integer(
            1, MAX_60_FPS_EXPOSURE_ABSOLUTE, p.exposure_time_absolute
        )
        self.sb_camera_gain = integer(1, 40, p.camera_gain)
        self.chk_auto_wb = QtWidgets.QCheckBox()
        self.chk_auto_wb.setChecked(p.auto_white_balance)
        self.sb_wb_temp = integer(10, 10000, p.white_balance_temperature)
        self.sb_wb_temp.setSingleStep(10)
        self.chk_low_latency = QtWidgets.QCheckBox()
        self.chk_low_latency.setChecked(p.low_latency_mode)

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
        form.addRow("profile_acceleration", self.sb_profile_accel)
        form.addRow("pan_dir", self.cb_pan_dir)
        form.addRow("tilt_dir", self.cb_tilt_dir)

        form.addRow(QtWidgets.QLabel("— Detection —"), QtWidgets.QLabel(""))
        form.addRow("v_thresh", self.sb_v)
        form.addRow("s_thresh", self.sb_s)
        form.addRow("min_area", self.sb_min_area)
        form.addRow("max_area", self.sb_max_area)

        form.addRow(QtWidgets.QLabel("— Confidence Gates —"), QtWidgets.QLabel(""))
        form.addRow("peak_v_gate", self.sb_peak_gate)
        form.addRow("local_contrast_gate", self.sb_local_contrast_gate)
        form.addRow(
            "local_red_contrast_gate", self.sb_local_red_contrast_gate
        )
        form.addRow("laser_edge_margin_px", self.sb_laser_edge_margin)
        form.addRow("area_hi_gate", self.sb_area_hi_gate)
        form.addRow("smoothing_tau_ms", self.sb_smoothing_tau)
        form.addRow("lock_time_ms", self.sb_lock_time)
        form.addRow("outlier_speed_px_s", self.sb_outlier_speed)
        form.addRow("laser_roi_half_size", self.sb_laser_roi)

        form.addRow(QtWidgets.QLabel("— Camera —"), QtWidgets.QLabel(""))
        form.addRow("manual_exposure", self.chk_manual_exposure)
        form.addRow("exposure_time_absolute", self.sb_exposure)
        form.addRow("camera_gain", self.sb_camera_gain)
        form.addRow("auto_white_balance", self.chk_auto_wb)
        form.addRow("white_balance_temperature", self.sb_wb_temp)
        form.addRow("low_latency_mode", self.chk_low_latency)

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
        self.sb_profile_accel.valueChanged.connect(
            lambda v: self.store.set_attr("profile_acceleration", int(v))
        )

        self.sb_v.valueChanged.connect(lambda v: self.store.set_attr("v_thresh", int(v)))
        self.sb_s.valueChanged.connect(lambda v: self.store.set_attr("s_thresh", int(v)))
        self.sb_min_area.valueChanged.connect(lambda v: self.store.set_attr("min_area", float(v)))
        self.sb_max_area.valueChanged.connect(lambda v: self.store.set_attr("max_area", float(v)))

        self.sb_peak_gate.valueChanged.connect(lambda v: self.store.set_attr("peak_v_gate", int(v)))
        self.sb_local_contrast_gate.valueChanged.connect(
            lambda v: self.store.set_attr("local_contrast_gate", int(v))
        )
        self.sb_local_red_contrast_gate.valueChanged.connect(
            lambda v: self.store.set_attr(
                "local_red_contrast_gate", int(v)
            )
        )
        self.sb_laser_edge_margin.valueChanged.connect(
            lambda v: self.store.set_attr("laser_edge_margin_px", int(v))
        )
        self.sb_area_hi_gate.valueChanged.connect(lambda v: self.store.set_attr("area_hi_gate", float(v)))
        self.sb_smoothing_tau.valueChanged.connect(
            lambda v: self.store.set_attr("smoothing_tau_ms", float(v))
        )
        self.sb_lock_time.valueChanged.connect(
            lambda v: self.store.set_attr("lock_time_ms", float(v))
        )
        self.sb_outlier_speed.valueChanged.connect(
            lambda v: self.store.set_attr("outlier_speed_px_s", float(v))
        )
        self.sb_laser_roi.valueChanged.connect(
            lambda v: self.store.set_attr("laser_roi_half_size", int(v))
        )
        self.chk_manual_exposure.stateChanged.connect(
            lambda _: self.store.set_attr(
                "manual_exposure", bool(self.chk_manual_exposure.isChecked())
            )
        )
        self.sb_exposure.valueChanged.connect(
            lambda v: self.store.set_attr("exposure_time_absolute", int(v))
        )
        self.sb_camera_gain.valueChanged.connect(
            lambda v: self.store.set_attr("camera_gain", int(v))
        )
        self.chk_auto_wb.stateChanged.connect(
            lambda _: self.store.set_attr(
                "auto_white_balance", bool(self.chk_auto_wb.isChecked())
            )
        )
        self.sb_wb_temp.valueChanged.connect(
            lambda v: self.store.set_attr("white_balance_temperature", int(v))
        )
        self.chk_low_latency.stateChanged.connect(
            lambda _: self.store.set_attr(
                "low_latency_mode", bool(self.chk_low_latency.isChecked())
            )
        )

        self.cb_pan_dir.currentTextChanged.connect(lambda t: self.store.set_attr("pan_dir", int(t)))
        self.cb_tilt_dir.currentTextChanged.connect(lambda t: self.store.set_attr("tilt_dir", int(t)))

    def _apply_tooltips(self):
        self.sb_deg_pan.setToolTip("Degrees moved per pixel of horizontal error. Higher = faster/more aggressive, but can overshoot/jitter.")
        self.sb_deg_tilt.setToolTip("Degrees moved per pixel of vertical error. Higher = faster/more aggressive, but can overshoot/jitter.")
        self.sb_max_step.setToolTip("Maximum command change at the 60 Hz reference rate. It is time-scaled when rate_hz changes. Lower = smoother/safer.")
        self.sb_deadband.setToolTip("If error magnitude is below this many pixels, treat it as zero (no movement). Bigger = less jitter near center.")
        self.sb_rate.setToolTip("Fixed servo controller rate in Hz. It is independent of camera and detector FPS; 60 Hz is the recommended starting point.")
        self.sb_profile_vel.setToolTip("Dynamixel Profile Velocity. Higher = servo can physically move faster. Too high can sound/feel harsh.")
        self.sb_profile_accel.setToolTip(
            "Dynamixel Profile Acceleration. Lower values ramp speed more gently. "
            "It is automatically capped at half of Profile Velocity."
        )
        self.cb_pan_dir.setToolTip("Flip pan direction if it moves the wrong way (+1 or -1).")
        self.cb_tilt_dir.setToolTip("Flip tilt direction if it moves the wrong way (+1 or -1).")

        self.sb_v.setToolTip("HSV V (brightness) threshold. Higher = ignores dim stuff; lower = sees farther but may pick up noise.")
        self.sb_s.setToolTip("HSV S (saturation) threshold. Lower helps on distant walls; higher rejects weak pinkish noise.")
        self.sb_min_area.setToolTip("Reject blobs smaller than this area. Increase to ignore speckle noise.")
        self.sb_max_area.setToolTip("Reject blobs larger than this area. Prevents big red objects from being treated as the dot.")

        self.sb_peak_gate.setToolTip("Hard gate: require the detected blob to contain pixels with V >= this value. Strongly reduces false positives.")
        self.sb_local_contrast_gate.setToolTip(
            "Require the candidate peak to be this much brighter than its "
            "local background. Raise it to reject broad reddish objects."
        )
        self.sb_local_red_contrast_gate.setToolTip(
            "Require the candidate to be redder than the nearby wall by this "
            "amount. This adapts to bright and dark wall regions."
        )
        self.sb_laser_edge_margin.setToolTip(
            "Ignore detections this many source-image pixels from an edge. "
            "Edge-clipped targets are unreliable and may be outside the "
            "valid rectified stereo image."
        )
        self.sb_area_hi_gate.setToolTip("Hard gate: reject any detection larger than this area even if it is red. Helps with big red patches.")
        self.sb_smoothing_tau.setToolTip(
            "Time-based laser smoothing in milliseconds. Lower follows faster; higher removes more jitter but adds lag."
        )
        self.sb_lock_time.setToolTip(
            "Continuous valid-detection time required before servo commands are allowed."
        )
        self.sb_outlier_speed.setToolTip(
            "Maximum plausible dot speed in pixels/second before a measurement is rejected as a jump."
        )
        self.sb_laser_roi.setToolTip(
            "Half-width of the fast tracking search window. It expands automatically after misses."
        )
        self.chk_manual_exposure.setToolTip(
            "Use a fixed exposure so changing room light does not continuously move the detector thresholds."
        )
        self.sb_exposure.setToolTip(
            "AR0234 exposure in 100-microsecond units. The GUI caps it at 160 (16 ms) to preserve 60 FPS; "
            "lower values reduce motion blur and ambient light. Start at 100 and tune on the Jetson."
        )
        self.sb_camera_gain.setToolTip(
            "Sensor gain. Keep near 1 when possible; higher gain brightens the dot but also amplifies image noise."
        )
        self.chk_auto_wb.setToolTip(
            "Automatic white balance can change red-dot color while tracking. Leave off for repeatable HSV detection."
        )
        self.sb_wb_temp.setToolTip(
            "Fixed white-balance temperature used while automatic white balance is off."
        )
        self.chk_low_latency.setToolTip(
            "Enable the Jetson camera driver's low-latency capture mode."
        )

        self.chk_servos.setToolTip("If unchecked, tracking can run and display, but servos will never move.")
        self.btn_start.setToolTip("Enable tracking & torque on (servos will follow dot).")
        self.btn_stop.setToolTip("Disable tracking and torque off (GUI + camera stay running).")
        self.lbl_status.setToolTip(
            "Pan and tilt are shown as actual/commanded degrees; velocity is measured servo feedback."
        )

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
        self.worker.error_signal.connect(self.on_error)
        self.worker.status_signal.connect(self.on_status)
        self.worker.start()
        self.worker.request_preview()

    def on_start_tracking(self):
        if self.worker is not None:
            self.worker.clear_servo_fault()
        self.store.set_attr("tracking_enabled", True)

    def on_stop_tracking(self):
        self.store.set_attr("tracking_enabled", False)

    def on_servos_toggle(self, _):
        self.store.set_attr("servos_enabled", bool(self.chk_servos.isChecked()))

    def on_error(self, msg: str):
        self.preview_timer.stop()
        self.lbl_status.setText(f"ERROR: {msg}")

    def on_status(self, msg: str):
        self.lbl_status.setText(f"Status: {msg}")

    def poll_preview(self):
        if self.worker is None:
            return
        preview = self.worker.take_preview()
        self.worker.request_preview()
        if preview is not None:
            self.on_frame(preview.frame_bgr, preview.mask, preview.status)

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
        self.preview_timer.stop()
        self.store.set_attr("tracking_enabled", False)
        try:
            if self.worker is not None:
                self.worker.stop()
                if not self.worker.wait(10000):
                    self.lbl_status.setText(
                        "Status: waiting for camera/control threads to stop safely"
                    )
                    event.ignore()
                    return
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

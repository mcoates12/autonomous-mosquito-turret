"""Small, dependency-free types shared by target detectors and the control loop."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple


POSITION_TICK_MAX = 4095
DISPLAY_DEGREES_MAX = 360.0
CONTROL_REFERENCE_HZ = 60.0


@dataclass(frozen=True)
class TargetObservation:
    """A detector-neutral observation in source-image pixel coordinates."""

    x: float
    y: float
    confidence: float
    timestamp: float
    label: str
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None

    @property
    def centroid(self) -> Tuple[int, int]:
        return int(round(self.x)), int(round(self.y))


@dataclass(frozen=True)
class ControlTarget:
    """Latest filtered image error consumed by the fixed-rate controller."""

    error_x_px: float
    error_y_px: float
    timestamp: float
    locked: bool


@dataclass(frozen=True)
class FilteredPoint:
    accepted: bool
    outlier: bool
    x: float
    y: float
    locked: bool


class TimeBasedTargetFilter:
    """Frame-rate-independent point smoothing, validation, and lock timing."""

    def __init__(
        self,
        smoothing_tau_sec: float = 0.06,
        lock_time_sec: float = 0.05,
        reset_after_sec: float = 0.15,
        max_speed_px_sec: float = 8000.0,
        jump_allowance_px: float = 40.0,
    ):
        self.smoothing_tau_sec = max(1e-6, float(smoothing_tau_sec))
        self.lock_time_sec = max(0.0, float(lock_time_sec))
        self.reset_after_sec = max(0.0, float(reset_after_sec))
        self.max_speed_px_sec = max(0.0, float(max_speed_px_sec))
        self.jump_allowance_px = max(0.0, float(jump_allowance_px))
        self.reset()

    def configure(
        self,
        smoothing_tau_sec: float,
        lock_time_sec: float,
        max_speed_px_sec: float,
    ) -> None:
        self.smoothing_tau_sec = max(1e-6, float(smoothing_tau_sec))
        self.lock_time_sec = max(0.0, float(lock_time_sec))
        self.max_speed_px_sec = max(0.0, float(max_speed_px_sec))

    def reset(self) -> None:
        self._x = None
        self._y = None
        self._last_raw_x = None
        self._last_raw_y = None
        self._last_timestamp = None
        self._continuous_since = None

    def miss(self, timestamp: float) -> None:
        timestamp = float(timestamp)
        self._continuous_since = None
        if (
            self._last_timestamp is not None
            and timestamp - self._last_timestamp >= self.reset_after_sec
        ):
            self.reset()

    def update(self, x: float, y: float, timestamp: float) -> FilteredPoint:
        x, y, timestamp = float(x), float(y), float(timestamp)
        if not all(math.isfinite(value) for value in (x, y, timestamp)):
            return FilteredPoint(False, True, 0.0, 0.0, False)

        if self._last_timestamp is not None:
            dt = timestamp - self._last_timestamp
            if dt < 0.0:
                return FilteredPoint(False, True, self._x, self._y, False)
            if dt >= self.reset_after_sec:
                self.reset()

        if self._last_timestamp is None:
            self._x, self._y = x, y
            self._continuous_since = timestamp
        else:
            dt = max(0.0, timestamp - self._last_timestamp)
            jump = math.hypot(x - self._last_raw_x, y - self._last_raw_y)
            allowed_jump = self.jump_allowance_px + self.max_speed_px_sec * dt
            if jump > allowed_jump:
                self._continuous_since = None
                return FilteredPoint(False, True, self._x, self._y, False)
            alpha = 1.0 - math.exp(-dt / self.smoothing_tau_sec)
            self._x = (1.0 - alpha) * self._x + alpha * x
            self._y = (1.0 - alpha) * self._y + alpha * y
            if self._continuous_since is None:
                self._continuous_since = timestamp

        self._last_raw_x = x
        self._last_raw_y = y
        self._last_timestamp = timestamp
        locked = timestamp - self._continuous_since >= self.lock_time_sec
        return FilteredPoint(True, False, self._x, self._y, locked)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def degrees_to_position_ticks(degrees: float) -> int:
    """Map the displayed inclusive 0..360 range onto DYNAMIXEL 0..4095."""
    degrees = float(degrees)
    if not math.isfinite(degrees):
        raise ValueError("position degrees must be finite")
    bounded = clamp(degrees, 0.0, DISPLAY_DEGREES_MAX)
    return int(round((bounded / DISPLAY_DEGREES_MAX) * POSITION_TICK_MAX))


def position_ticks_to_degrees(ticks: int) -> float:
    """Map a single-turn DYNAMIXEL position back to displayed degrees."""
    bounded = int(clamp(int(ticks), 0, POSITION_TICK_MAX))
    return (bounded / POSITION_TICK_MAX) * DISPLAY_DEGREES_MAX


def sanitize_motion_profile(velocity: int, acceleration: int) -> Tuple[int, int]:
    """Return finite XL430 profile values satisfying acceleration <= velocity/2."""
    safe_velocity = max(2, int(velocity))
    safe_acceleration = max(1, int(acceleration))
    safe_acceleration = min(safe_acceleration, safe_velocity // 2)
    return safe_velocity, safe_acceleration


def tracking_delta_degrees(
    error_px: float,
    gain_deg_per_px: float,
    direction: int,
    dt: float,
    max_step_deg: float,
    reference_hz: float = CONTROL_REFERENCE_HZ,
) -> float:
    """Convert the legacy per-frame gain into a time-scaled position delta."""
    values = (
        float(error_px),
        float(gain_deg_per_px),
        float(direction),
        float(dt),
        float(max_step_deg),
        float(reference_hz),
    )
    if not all(math.isfinite(value) for value in values):
        return 0.0
    safe_dt = max(0.0, float(dt))
    desired_velocity = (
        float(direction) * float(gain_deg_per_px) * float(error_px) * reference_hz
    )
    delta = desired_velocity * safe_dt
    max_delta = max(0.0, float(max_step_deg)) * reference_hz * safe_dt
    return clamp(delta, -max_delta, max_delta)


def tracking_pd_delta_degrees(
    error_px: float,
    error_rate_px_s: float,
    proportional_gain_deg_per_px: float,
    damping_gain_deg_per_px: float,
    direction: int,
    dt: float,
    max_step_deg: float,
    reference_hz: float = CONTROL_REFERENCE_HZ,
) -> float:
    """Return a time-scaled PD outer-loop command with the legacy step cap.

    The proportional gain retains the existing per-reference-frame behavior.
    The damping gain converts the measured pixel-error rate directly to an
    opposing or assisting angular velocity. A closing error therefore brakes
    the axis before it crosses the target, while an opening error helps the
    axis begin following a moving target.
    """
    values = (
        float(error_px),
        float(error_rate_px_s),
        float(proportional_gain_deg_per_px),
        float(damping_gain_deg_per_px),
        float(direction),
        float(dt),
        float(max_step_deg),
        float(reference_hz),
    )
    if not all(math.isfinite(value) for value in values):
        return 0.0
    safe_dt = max(0.0, float(dt))
    desired_velocity = float(direction) * (
        float(proportional_gain_deg_per_px)
        * float(error_px)
        * float(reference_hz)
        + float(damping_gain_deg_per_px) * float(error_rate_px_s)
    )
    delta = desired_velocity * safe_dt
    max_delta = max(0.0, float(max_step_deg)) * reference_hz * safe_dt
    return clamp(delta, -max_delta, max_delta)


class FilteredDerivative:
    """Estimate a derivative from timestamped samples without frame-rate bias."""

    def __init__(self, smoothing_tau_sec: float, reset_after_sec: float):
        if smoothing_tau_sec < 0.0:
            raise ValueError("smoothing_tau_sec must be non-negative")
        if reset_after_sec <= 0.0:
            raise ValueError("reset_after_sec must be positive")
        self.smoothing_tau_sec = float(smoothing_tau_sec)
        self.reset_after_sec = float(reset_after_sec)
        self.reset()

    def reset(self) -> None:
        self._last_value = None
        self._last_timestamp = None
        self._rate = 0.0

    @property
    def rate(self) -> float:
        return self._rate

    def update(self, value: float, timestamp: float) -> float:
        value = float(value)
        timestamp = float(timestamp)
        if not (math.isfinite(value) and math.isfinite(timestamp)):
            self.reset()
            return 0.0

        if self._last_timestamp is None:
            self._last_value = value
            self._last_timestamp = timestamp
            self._rate = 0.0
            return self._rate

        dt = timestamp - self._last_timestamp
        if dt == 0.0:
            return self._rate
        if dt < 0.0 or dt > self.reset_after_sec:
            self._last_value = value
            self._last_timestamp = timestamp
            self._rate = 0.0
            return self._rate

        raw_rate = (value - self._last_value) / dt
        if self.smoothing_tau_sec == 0.0:
            alpha = 1.0
        else:
            alpha = 1.0 - math.exp(-dt / self.smoothing_tau_sec)
        self._rate += alpha * (raw_rate - self._rate)
        self._last_value = value
        self._last_timestamp = timestamp
        return self._rate


def within_reacquisition_radius(
    candidate_x: float,
    candidate_y: float,
    expected_position: Optional[Tuple[float, float]],
    max_distance_px: Optional[float],
) -> bool:
    """Return whether a candidate preserves the current target identity.

    A missing anchor or radius means this is a deliberate unrestricted
    acquisition. Invalid coordinates fail closed whenever a gate is active.
    The helper is detector-neutral so a future AI detector can reuse it.
    """
    if expected_position is None or max_distance_px is None:
        return True
    values = (
        float(candidate_x),
        float(candidate_y),
        float(expected_position[0]),
        float(expected_position[1]),
        float(max_distance_px),
    )
    if not all(math.isfinite(value) for value in values) or values[4] < 0.0:
        return False
    return math.hypot(values[0] - values[2], values[1] - values[3]) <= values[4]


class EveryNFrames:
    """Return ``True`` once every N calls, including the first call."""

    def __init__(self, every_n: int):
        if every_n < 1:
            raise ValueError("every_n must be at least 1")
        self.every_n = int(every_n)
        self._frames_until_due = 0

    def step(self) -> bool:
        if self._frames_until_due == 0:
            self._frames_until_due = self.every_n - 1
            return True
        self._frames_until_due -= 1
        return False

    def reset(self, due_immediately: bool = True) -> None:
        self._frames_until_due = 0 if due_immediately else self.every_n - 1

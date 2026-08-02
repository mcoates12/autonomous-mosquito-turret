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

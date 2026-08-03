"""Dependency-free validation helpers for the browser dashboard."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class ParameterSpec:
    kind: type
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: Optional[Tuple[Any, ...]] = None


PARAMETER_SPECS = {
    "display_mode": ParameterSpec(
        str, choices=("Left", "Right", "Side-by-side", "Combined")
    ),
    "track_source": ParameterSpec(str, choices=("Left", "Right")),
    "v_thresh": ParameterSpec(int, 0, 255),
    "s_thresh": ParameterSpec(int, 0, 255),
    "min_area": ParameterSpec(float, 0.0, 200.0),
    "max_area": ParameterSpec(float, 1.0, 5000.0),
    "peak_v_gate": ParameterSpec(int, 0, 255),
    "local_contrast_gate": ParameterSpec(int, 0, 255),
    "local_red_contrast_gate": ParameterSpec(int, 0, 255),
    "laser_edge_margin_px": ParameterSpec(int, 0, 480),
    "area_hi_gate": ParameterSpec(float, 1.0, 5000.0),
    "smoothing_tau_ms": ParameterSpec(float, 5.0, 500.0),
    "lock_time_ms": ParameterSpec(float, 0.0, 1000.0),
    "outlier_speed_px_s": ParameterSpec(float, 100.0, 20000.0),
    "laser_roi_half_size": ParameterSpec(int, 32, 960),
    "laser_reacquire_radius_px": ParameterSpec(int, 32, 1920),
    "manual_exposure": ParameterSpec(bool),
    "exposure_time_absolute": ParameterSpec(int, 1, 160),
    "camera_gain": ParameterSpec(int, 1, 40),
    "auto_white_balance": ParameterSpec(bool),
    "white_balance_temperature": ParameterSpec(int, 10, 10000),
    "low_latency_mode": ParameterSpec(bool),
    "deg_per_px_pan": ParameterSpec(float, 0.0001, 0.05),
    "deg_per_px_tilt": ParameterSpec(float, 0.0001, 0.05),
    "pan_damping_gain": ParameterSpec(float, 0.0, 0.05),
    "tilt_damping_gain": ParameterSpec(float, 0.0, 0.05),
    "max_step_deg": ParameterSpec(float, 0.1, 20.0),
    "deadband_px": ParameterSpec(int, 0, 200),
    "rate_hz": ParameterSpec(float, 5.0, 120.0),
    "profile_velocity": ParameterSpec(int, 2, 1500),
    "profile_acceleration": ParameterSpec(int, 1, 750),
    "pan_dir": ParameterSpec(int, choices=(-1, 1)),
    "tilt_dir": ParameterSpec(int, choices=(-1, 1)),
    "servos_enabled": ParameterSpec(bool),
}


def coerce_parameter(name: str, value: Any) -> Any:
    """Validate one JSON value and return its normalized Python value."""
    spec = PARAMETER_SPECS.get(name)
    if spec is None:
        raise ValueError(f"unknown parameter: {name}")

    if spec.kind is bool:
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
        normalized = value
    elif spec.kind is int:
        if type(value) is not int:
            raise ValueError(f"{name} must be an integer")
        normalized = value
    elif spec.kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        normalized = float(value)
    elif spec.kind is str:
        if not isinstance(value, str):
            raise ValueError(f"{name} must be text")
        normalized = value
    else:
        raise ValueError(f"unsupported parameter type for {name}")

    if spec.choices is not None and normalized not in spec.choices:
        allowed = ", ".join(str(item) for item in spec.choices)
        raise ValueError(f"{name} must be one of: {allowed}")
    if spec.minimum is not None and normalized < spec.minimum:
        raise ValueError(f"{name} must be at least {spec.minimum}")
    if spec.maximum is not None and normalized > spec.maximum:
        raise ValueError(f"{name} must be at most {spec.maximum}")
    return normalized


def apply_parameter_updates(store, updates: Any) -> dict:
    """Validate an update object completely before mutating the store."""
    if not isinstance(updates, dict) or not updates:
        raise ValueError("request must contain a non-empty JSON object")
    normalized = {
        name: coerce_parameter(name, value) for name, value in updates.items()
    }
    for name, value in normalized.items():
        store.set_attr(name, value)
    return normalized

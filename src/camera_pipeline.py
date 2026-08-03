"""GStreamer capture pipelines and camera-rate helpers."""

from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass(frozen=True)
class CapturePipeline:
    name: str
    pipeline: str


def rolling_rate_hz(timestamps: Sequence[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    elapsed = timestamps[-1] - timestamps[0]
    return (len(timestamps) - 1) / max(1e-6, elapsed)


def gst_nvidia_vic_bgr_pipeline(
    device: str, width: int, height: int, fps: int = 60
) -> str:
    """Capture UYVY into NVMM and use the Jetson VIC for color conversion."""
    return (
        f"nvv4l2camerasrc device={device} ! "
        "video/x-raw(memory:NVMM), format=UYVY, "
        f"width={width}, height={height}, framerate={fps}/1, "
        "interlace-mode=progressive ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false processing-deadline=0"
    )


def gst_v4l2_bgr_pipeline(
    device: str, width: int, height: int, fps: int = 60
) -> str:
    """Portable software conversion fallback for non-NVIDIA pipelines."""
    return (
        f"v4l2src device={device} io-mode=2 ! "
        "video/x-raw, format=UYVY, "
        f"width={width}, height={height}, framerate={fps}/1 ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false processing-deadline=0"
    )


def capture_pipeline_candidates(
    device: str, width: int, height: int, fps: int = 60
) -> Tuple[CapturePipeline, ...]:
    """Prefer Jetson VIC acceleration while retaining a tested fallback."""
    return (
        CapturePipeline(
            "nvidia-vic",
            gst_nvidia_vic_bgr_pipeline(device, width, height, fps),
        ),
        CapturePipeline(
            "software-videoconvert",
            gst_v4l2_bgr_pipeline(device, width, height, fps),
        ),
    )

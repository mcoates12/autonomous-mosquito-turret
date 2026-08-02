"""Small, dependency-free types shared by target detectors and the control loop."""

from dataclasses import dataclass
from typing import Optional, Tuple


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

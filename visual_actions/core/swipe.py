"""Swipe detection: a fast sideways travel of the wrist while a given shape is held.

A swipe is motion, not pose, so it runs per frame (like pinch) rather than through
the 250 ms token smoother. It fires once per stroke: when the wrist has travelled
`min_travel` of the frame width within the last `window_ns` with a peak speed above
`min_speed` (frame widths per second), and the majority of those frames held the
required shape. A cooldown stops one stroke from firing twice.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from .types import WRIST, HandFrame


class SwipeDirection(Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class SwipeEvent:
    t_ns: int
    direction: SwipeDirection
    shape: str
    travel: float  # fraction of frame width
    peak_speed: float  # frame widths per second


class SwipeDetector:
    def __init__(
        self,
        min_travel: float = 0.18,
        min_speed: float = 1.2,
        window_ns: int = 350_000_000,
        cooldown_ns: int = 450_000_000,
        max_vertical: float = 0.12,
    ) -> None:
        self.min_travel = min_travel
        self.min_speed = min_speed
        self.window_ns = window_ns
        self.cooldown_ns = cooldown_ns
        self.max_vertical = max_vertical
        self._hist: deque[tuple[int, float, float, str]] = deque()
        self._last_fire_ns: int = -(10**18)

    def reset(self) -> None:
        self._hist.clear()

    def update(self, hf: HandFrame, shape: str) -> SwipeEvent | None:
        w = hf.landmarks[WRIST]
        self._hist.append((hf.t_ns, w.x, w.y, shape))
        while self._hist and hf.t_ns - self._hist[0][0] > self.window_ns:
            self._hist.popleft()
        if len(self._hist) < 3 or hf.t_ns - self._last_fire_ns < self.cooldown_ns:
            return None
        t0, x0, y0, _ = self._hist[0]
        travel = w.x - x0
        if abs(travel) < self.min_travel or abs(w.y - y0) > self.max_vertical:
            return None
        # peak speed between consecutive samples, in frame widths per second
        peak = 0.0
        prev = self._hist[0]
        for cur in list(self._hist)[1:]:
            dt = (cur[0] - prev[0]) / 1e9
            if dt > 0:
                peak = max(peak, abs(cur[1] - prev[1]) / dt)
            prev = cur
        if peak < self.min_speed:
            return None
        shapes = [s for _, _, _, s in self._hist]
        held = max(set(shapes), key=shapes.count)
        if shapes.count(held) < len(shapes) * 0.7:
            return None
        self._last_fire_ns = hf.t_ns
        self._hist.clear()
        return SwipeEvent(hf.t_ns, SwipeDirection.RIGHT if travel > 0 else SwipeDirection.LEFT, held, abs(travel), peak)

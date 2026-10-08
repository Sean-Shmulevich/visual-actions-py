"""Per-frame pinch detection: thumb tip to index tip, with hysteresis and debounce.

Runs on every HandSeen frame (not through the token smoother) so drags stay low-latency.
Distances are in canonical units (wrist -> middle MCP = 1), so they are distance-invariant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .normalize import canonical
from .types import INDEX_TIP, MIDDLE_MCP, THUMB_TIP, WRIST, HandFrame


class PinchPhase(Enum):
    START = "start"
    MOVE = "move"
    END = "end"


@dataclass(frozen=True)
class PinchEvent:
    t_ns: int
    phase: PinchPhase
    x: float  # pinch point in the user frame, 0..1 (midpoint of thumb and index tips)
    y: float
    hand_scale: float  # wrist -> middle MCP length in frame units (depth proxy)
    distance: float  # thumb-index distance in canonical units


def pinch_distance(hf: HandFrame) -> float:
    """3-D distance between thumb tip and index tip in canonical units."""
    c = canonical(hf)
    d = c.pts[THUMB_TIP] - c.pts[INDEX_TIP]
    return float(math.sqrt(float(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)))


def pinch_point(hf: HandFrame) -> tuple[float, float]:
    t, i = hf.landmarks[THUMB_TIP], hf.landmarks[INDEX_TIP]
    return ((t.x + i.x) / 2, (t.y + i.y) / 2)


def hand_scale(hf: HandFrame) -> float:
    w, m = hf.landmarks[WRIST], hf.landmarks[MIDDLE_MCP]
    return math.hypot(m.x - w.x, m.y - w.y)


class PinchDetector:
    """Hysteresis: pinched when distance < on_threshold, released when > off_threshold.
    `debounce_frames` consecutive frames are needed to change state."""

    def __init__(self, on_threshold: float = 0.3, off_threshold: float = 0.5, debounce_frames: int = 2) -> None:
        if off_threshold <= on_threshold:
            raise ValueError("off_threshold must exceed on_threshold")
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.debounce_frames = debounce_frames
        self.pinched = False
        self._pending = 0
        self._last: tuple[int, float, float, float, float] | None = None

    def reset(self) -> int | None:
        """Forget state; returns the last timestamp if a pinch was in progress (caller emits END)."""
        was = self._last[0] if (self.pinched and self._last) else None
        self.pinched = False
        self._pending = 0
        self._last = None
        return was

    def update(self, hf: HandFrame) -> PinchEvent | None:
        d = pinch_distance(hf)
        x, y = pinch_point(hf)
        s = hand_scale(hf)
        self._last = (hf.t_ns, x, y, s, d)
        want = d < self.on_threshold if not self.pinched else not (d > self.off_threshold)
        if want != self.pinched:
            self._pending += 1
            if self._pending >= self.debounce_frames:
                self.pinched = want
                self._pending = 0
                return PinchEvent(hf.t_ns, PinchPhase.START if want else PinchPhase.END, x, y, s, d)
            return None  # a change is pending: hold position, no MOVE (an opening hand jumps the midpoint)
        self._pending = 0
        if self.pinched:
            return PinchEvent(hf.t_ns, PinchPhase.MOVE, x, y, s, d)
        return None

    def end_event(self, t_ns: int) -> PinchEvent | None:
        """Synthesize an END when the hand is lost mid-pinch."""
        if not self.pinched or self._last is None:
            return None
        _, x, y, s, d = self._last
        self.pinched = False
        self._pending = 0
        return PinchEvent(t_ns, PinchPhase.END, x, y, s, d)

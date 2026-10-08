"""Tier 0: a cheap check that keeps the tracker off an empty desk.

A skin mask alone opens on faces, wood and walls (99 % of frames in a live test), so
the gate is motion-driven with tracker feedback: it opens on motion, and stays open
while the tracker keeps seeing a hand, plus a hold time. A still desk closes it.
"""

from __future__ import annotations

from typing import Any, Protocol


class HandGate(Protocol):
    def open(self, frame_bgr: Any, t_ns: int) -> bool: ...

    def notify(self, t_ns: int, hand_seen: bool) -> None: ...


class AlwaysOpenGate:
    def open(self, frame_bgr: Any, t_ns: int) -> bool:
        return True

    def notify(self, t_ns: int, hand_seen: bool) -> None:
        pass


class MotionGate:
    """Opens when enough pixels change between downscaled grey frames; stays open for
    `hold_ns` after the last motion or the last tracked hand."""

    def __init__(
        self,
        motion_fraction: float = 0.004,
        diff_threshold: int = 24,
        hold_ns: int = 1_500_000_000,
        width: int = 160,
    ) -> None:
        self.motion_fraction = motion_fraction
        self.diff_threshold = diff_threshold
        self.hold_ns = hold_ns
        self.width = width
        self._prev: Any = None
        self._open_until_ns: int = 0
        self.last_motion_fraction: float = 0.0

    def open(self, frame_bgr: Any, t_ns: int) -> bool:
        import cv2

        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
        grey = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if self._prev is not None:
            diff = cv2.absdiff(grey, self._prev)
            moving = cv2.countNonZero(cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)[1])
            self.last_motion_fraction = moving / diff.size
            if self.last_motion_fraction >= self.motion_fraction:
                self._open_until_ns = t_ns + self.hold_ns
        self._prev = grey
        return t_ns < self._open_until_ns

    def notify(self, t_ns: int, hand_seen: bool) -> None:
        if hand_seen:
            self._open_until_ns = max(self._open_until_ns, t_ns + self.hold_ns)

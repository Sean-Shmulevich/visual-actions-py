"""Tier 0: a cheap check that keeps the tracker off an empty desk."""

from __future__ import annotations

from typing import Any, Protocol


class HandGate(Protocol):
    def open(self, frame_bgr: Any, t_ns: int) -> bool: ...


class AlwaysOpenGate:
    def open(self, frame_bgr: Any, t_ns: int) -> bool:
        return True


class MotionSkinGate:
    """Skin mask in YCrCb on a 160 px wide downscale, plus frame differencing, with hysteresis."""

    def __init__(
        self,
        min_blob_fraction: float = 0.02,
        hold_ns: int = 500_000_000,
        width: int = 160,
    ) -> None:
        self.min_blob_fraction = min_blob_fraction
        self.hold_ns = hold_ns
        self.width = width
        self._last_positive_ns: int | None = None

    def open(self, frame_bgr: Any, t_ns: int) -> bool:
        import cv2  # lazy: core stays importable without OpenCV
        import numpy as np

        h, w = frame_bgr.shape[:2]
        scale = self.width / w
        small = cv2.resize(frame_bgr, (self.width, int(h * scale)), interpolation=cv2.INTER_AREA)
        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, np.array((0, 133, 77)), np.array((255, 173, 127)))
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        largest = max((stats[i, cv2.CC_STAT_AREA] for i in range(1, n)), default=0)
        positive = largest / mask.size >= self.min_blob_fraction
        if positive:
            self._last_positive_ns = t_ns
        if self._last_positive_ns is None:
            return False
        return t_ns - self._last_positive_ns <= self.hold_ns

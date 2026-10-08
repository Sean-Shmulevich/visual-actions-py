"""Hand position in the camera frame -> screen point, plus smoothing.

The reach box is the part of the camera frame a seated user can comfortably cover;
it maps to the full screen. Optional depth normalization scales the box with hand
size so the same physical reach covers the screen from any distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReachBox:
    x0: float = 0.15
    x1: float = 0.85
    y0: float = 0.15
    y1: float = 0.85


class PointerMap:
    def __init__(
        self,
        screen_w: float,
        screen_h: float,
        box: ReachBox = ReachBox(),
        depth_gain: float = 0.0,
        ref_hand_scale: float = 0.12,
    ) -> None:
        self.screen_w, self.screen_h = screen_w, screen_h
        self.box = box
        self.depth_gain = depth_gain
        self.ref_hand_scale = ref_hand_scale

    def effective_box(self, hand_scale: float | None) -> ReachBox:
        """A farther (smaller) hand gets a smaller box around the same centre."""
        if not self.depth_gain or not hand_scale or hand_scale <= 0:
            return self.box
        k = (hand_scale / self.ref_hand_scale) ** self.depth_gain
        cx, cy = (self.box.x0 + self.box.x1) / 2, (self.box.y0 + self.box.y1) / 2
        hw, hh = (self.box.x1 - self.box.x0) / 2 * k, (self.box.y1 - self.box.y0) / 2 * k
        return ReachBox(cx - hw, cx + hw, cy - hh, cy + hh)

    def to_screen(self, x: float, y: float, hand_scale: float | None = None) -> tuple[float, float]:
        b = self.effective_box(hand_scale)
        u = (x - b.x0) / max(b.x1 - b.x0, 1e-6)
        v = (y - b.y0) / max(b.y1 - b.y0, 1e-6)
        u, v = min(1.0, max(0.0, u)), min(1.0, max(0.0, v))
        return (u * self.screen_w, v * self.screen_h)


class OneEuroFilter:
    """Casiez et al. 2012: low lag at speed, low jitter at rest. One instance per axis."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x, self._dx, self._t = None, 0.0, None

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or self._t is None:
            self._x, self._t = x, t
            return x
        dt = max(t - self._t, 1e-4)
        self._t = t
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x


class SmoothedPointer:
    def __init__(self, pmap: PointerMap, min_cutoff: float = 1.5, beta: float = 0.05) -> None:
        self.pmap = pmap
        self.fx = OneEuroFilter(min_cutoff, beta)
        self.fy = OneEuroFilter(min_cutoff, beta)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()

    def update(self, t_ns: int, x: float, y: float, hand_scale: float | None = None) -> tuple[float, float]:
        sx, sy = self.pmap.to_screen(x, y, hand_scale)
        t = t_ns / 1e9
        return (self.fx(sx, t), self.fy(sy, t))

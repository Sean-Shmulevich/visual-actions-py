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
    """Screen point from a hand point: reach-box map, one-frame glitch rejection, One Euro.

    Parameters measured on 325 recorded drags (2026-10-09). With screen pixels as units the
    One Euro speed term must be small: beta 0.05 let 270 px/s of tracker noise open the
    cutoff to ~15 Hz, so the filter passed the jitter through (still-hand step p90 4.9 px).
    min_cutoff 0.6 / beta 0.01 halves that (2.9 px) for ~17 px of lag at 600 px/s.
    Tracker glitches (a single frame 20..40 px away, then back) are not noise and no linear
    filter removes them without lag: a jump of `glitch_px` from a slow-moving hand is held
    for one frame; if the next frame comes back it is dropped, otherwise both are applied.
    """

    STILL_PX_S = 60.0  # slower than this over the recent frames counts as a still hand

    def __init__(self, pmap: PointerMap, min_cutoff: float = 0.6, beta: float = 0.01, glitch_px: float = 24.0) -> None:
        self.pmap = pmap
        self.fx = OneEuroFilter(min_cutoff, beta)
        self.fy = OneEuroFilter(min_cutoff, beta)
        self.glitch_px = glitch_px
        self._raw: list[tuple[float, float, float]] = []  # (t, sx, sy) recent raw screen points
        self._suspect: tuple[int, float, float] | None = None  # a jump held back for one frame
        self._out: tuple[float, float] | None = None

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()
        self._raw.clear()
        self._suspect = None
        self._out = None

    def _slow(self) -> bool:
        if len(self._raw) < 3:
            return False
        t0, x0, y0 = self._raw[0]
        t1, x1, y1 = self._raw[-1]
        dt = t1 - t0
        return dt > 0 and math.hypot(x1 - x0, y1 - y0) / dt < self.STILL_PX_S

    def _apply(self, t_ns: int, sx: float, sy: float) -> tuple[float, float]:
        t = t_ns / 1e9
        self._raw.append((t, sx, sy))
        del self._raw[:-5]
        self._out = (self.fx(sx, t), self.fy(sy, t))
        return self._out

    def update(self, t_ns: int, x: float, y: float, hand_scale: float | None = None) -> tuple[float, float]:
        sx, sy = self.pmap.to_screen(x, y, hand_scale)
        if self._suspect is not None:
            st_ns, px, py = self._suspect
            self._suspect = None
            _, lx, ly = self._raw[-1]
            if math.hypot(sx - lx, sy - ly) < self.glitch_px / 2:
                return self._apply(t_ns, sx, sy)  # the jump was a one-frame glitch: dropped
            self._apply(st_ns, px, py)  # a real move: the held frame goes in first
            return self._apply(t_ns, sx, sy)
        if self.glitch_px > 0 and self._raw and self._slow() and self._out is not None:
            _, lx, ly = self._raw[-1]
            if math.hypot(sx - lx, sy - ly) >= self.glitch_px:
                self._suspect = (t_ns, sx, sy)
                return self._out  # hold one frame
        return self._apply(t_ns, sx, sy)

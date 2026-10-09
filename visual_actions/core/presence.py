"""Hand presence with debounce and a fast-exit predictor.

Evidence from the 2026-10-08 16:55 session: 32 of 64 hand-lost events lasted under
0.4 s and most exactly one frame. Each one reset the pinch detector and paused the
drag. A hand is therefore only reported lost after `lost_frames` consecutive frames
without a usable hand (tracker found none, or the edge rule rejected it). A gate
closure means no motion and no hand for the gate's hold time, so it reports at once.

Fast exits (2026-10-08 evening). Across 189 min of sessions the debounce costs every
exit ~130 ms, and a hand flicked out of the frame (35 exits at >= 2 frame widths/s,
up to 4.8) had the tracker report a phantom clamped to the border for a few frames
first, so a pinch-drag jumped or released late. The predictor declares the hand lost
at once, reason "fast-exit", when either
- the wrist (left/right/top) or the pinch point (any edge) is moving at
  >= `fast_exit_speed` frame widths/s and its position extrapolated
  `fast_exit_lookahead_frames` ahead is outside the frame, or
- the hand touches a border and its landmark spread collapsed to under
  `fast_exit_collapse` of what it was a few frames ago, or its confidence fell under
  `fast_exit_min_confidence`: the phantom a tracker reports while the hand is gone.
The bottom edge keeps the wrist exemption: there the fingers and pinch point decide.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .types import INDEX_TIP, THUMB_TIP, WRIST, HandFrame

if TYPE_CHECKING:
    from .config import PresenceConfig


@dataclass(frozen=True)
class Presence:
    seen: HandFrame | None = None  # a usable hand this frame
    lost: tuple[str, str] | None = None  # (reason, detail) when the hand is now declared lost


def _pinch_point(hf: HandFrame) -> tuple[float, float]:
    t, i = hf.landmarks[THUMB_TIP], hf.landmarks[INDEX_TIP]
    return ((t.x + i.x) / 2, (t.y + i.y) / 2)


def _spread(hf: HandFrame) -> float:
    xs = [lm.x for lm in hf.landmarks]
    ys = [lm.y for lm in hf.landmarks]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


class PresenceFilter:
    EDGE_MARGIN = 0.02
    EDGE_MAX_OUTSIDE = 6
    MAX_DT_S = 0.1  # a stalled camera must not extrapolate across the stall
    BORDER_ZONE = 0.03  # 'touching' a border for the phantom rules
    HISTORY = 6  # frames of wrist/spread history kept for the predictor

    def __init__(
        self,
        lost_frames: int = 3,
        fast_exit: bool = True,
        fast_exit_speed: float = 2.0,
        fast_exit_lookahead_frames: float = 2.0,
        fast_exit_collapse: float = 0.5,
        fast_exit_min_confidence: float = 0.5,
        fast_exit_bottom: bool = False,
        fast_exit_reach: float = 0.1,
    ) -> None:
        self.lost_frames = lost_frames
        self.fast_exit = fast_exit
        self.fast_exit_speed = fast_exit_speed
        self.fast_exit_lookahead_frames = fast_exit_lookahead_frames
        self.fast_exit_collapse = fast_exit_collapse
        self.fast_exit_min_confidence = fast_exit_min_confidence
        self.fast_exit_bottom = fast_exit_bottom
        self.fast_exit_reach = fast_exit_reach
        self.present = False
        self._missing = 0
        self._last_reason: tuple[str, str] | None = None
        self._history: deque[HandFrame] = deque(maxlen=self.HISTORY)
        self._reentry_clear = False  # after a fast exit the hand must come back clear of the border

    @classmethod
    def from_config(cls, pc: PresenceConfig) -> PresenceFilter:
        return cls(
            lost_frames=pc.lost_frames,
            fast_exit=pc.fast_exit,
            fast_exit_speed=pc.fast_exit_speed,
            fast_exit_lookahead_frames=pc.fast_exit_lookahead_frames,
            fast_exit_collapse=pc.fast_exit_collapse,
            fast_exit_min_confidence=pc.fast_exit_min_confidence,
            fast_exit_bottom=pc.fast_exit_bottom,
            fast_exit_reach=pc.fast_exit_reach,
        )

    @classmethod
    def out_of_frame(cls, hf: HandFrame) -> str | None:
        """MediaPipe keeps reporting a hand clamped to the border after it leaves.

        The bottom edge is exempt: a hand reaching down naturally has its wrist below
        the frame while the fingers and the pinch are still in view (2026-10-08 feedback).
        A hand is off-screen when the wrist, the pinch point or most landmarks have left
        through the left, right or top.
        """
        m = cls.EDGE_MARGIN

        def out_lrt(x: float, y: float) -> bool:  # outside via left, right or top only
            return x < -m or x > 1 + m or y < -m

        w = hf.landmarks[WRIST]
        if out_lrt(w.x, w.y):
            return f"wrist outside ({w.x:.2f},{w.y:.2f})"
        px, py = _pinch_point(hf)
        # The pinch point (thumb/index midpoint) decides for the sides and the bottom only: a
        # finger pointing up sits at the top edge all the time (2026-10-09: the desktop flick
        # kept reading as hand lost), and the wrist rule already covers a hand leaving upward.
        if px < -m or px > 1 + m or py > 1 + m:
            return f"pinch point outside ({px:.2f},{py:.2f})"
        outside = sum(1 for lm in hf.landmarks if out_lrt(lm.x, lm.y))
        if outside >= cls.EDGE_MAX_OUTSIDE:
            xs = [lm.x for lm in hf.landmarks]
            ys = [lm.y for lm in hf.landmarks]
            return f"{outside} landmarks outside, x {min(xs):.2f}..{max(xs):.2f} y {min(ys):.2f}..{max(ys):.2f}"
        return None

    @classmethod
    def touching_border(cls, hf: HandFrame) -> bool:
        """Any landmark within BORDER_ZONE of the left, right or top edge (bottom exempt)."""
        z = cls.BORDER_ZONE
        return any(lm.x < z or lm.x > 1 - z or lm.y < z for lm in hf.landmarks)

    def _exiting(self, p0: tuple[float, float], p1: tuple[float, float], dt: float, look: float, bottom: bool, reach: float | None = None, top: bool = True) -> str | None:
        """p1 is within reach of an edge, moving toward it fast enough to be past it in `look` s."""
        vx, vy = (p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt
        v = math.hypot(vx, vy)
        if v < self.fast_exit_speed:
            return None
        m = self.EDGE_MARGIN
        reach = self.fast_exit_reach if reach is None else reach
        x, y = p1[0] + vx * look, p1[1] + vy * look
        near = (
            (x < -m and p1[0] < reach)
            or (x > 1 + m and p1[0] > 1 - reach)
            or (top and y < -m and p1[1] < reach)
            or (bottom and y > 1 + m and p1[1] > 1 - reach)
        )
        if not near:
            return None
        return f"at ({p1[0]:.2f},{p1[1]:.2f}) moving {v:.1f} fw/s, next ({x:.2f},{y:.2f})"

    def fast_exit_detail(self, hf: HandFrame) -> str | None:
        """The hand is in frame now but is about to be gone, or is already a phantom."""
        if not self.fast_exit or not self._history:
            return None
        prev = self._history[-1]
        dt = (hf.t_ns - prev.t_ns) / 1e9
        if 0 < dt <= self.MAX_DT_S:
            look = self.fast_exit_lookahead_frames * dt
            w0, w1 = prev.landmarks[WRIST], hf.landmarks[WRIST]
            why = self._exiting((w0.x, w0.y), (w1.x, w1.y), dt, look, bottom=False)
            if why:
                return f"wrist {why}"
            why = self._exiting(_pinch_point(prev), _pinch_point(hf), dt, look, bottom=self.fast_exit_bottom, top=False)
            if why:
                return f"pinch point {why}"
        if not self.touching_border(hf):
            return None
        xs = [lm.x for lm in hf.landmarks]
        ys = [lm.y for lm in hf.landmarks]
        if hf.confidence < self.fast_exit_min_confidence:
            return f"confidence {hf.confidence:.2f} at the border, x {min(xs):.2f}..{max(xs):.2f} y {min(ys):.2f}..{max(ys):.2f}"
        ref = max(_spread(h) for h in self._history)
        now = _spread(hf)
        if ref > 0 and now < ref * self.fast_exit_collapse:
            return f"spread collapsed {ref:.2f} -> {now:.2f} at the border, x {min(xs):.2f}..{max(xs):.2f} y {min(ys):.2f}..{max(ys):.2f}"
        return None

    def racing_out_detail(self, hand: HandFrame | None) -> str | None:
        """This frame has no usable hand. Was the last one racing toward the edge it is now past?

        The debounce exists for one-frame tracker glitches; a hand that was moving at
        `fast_exit_speed` toward an edge and is now outside it (or gone, with its
        extrapolated position outside) is not a glitch, so the loss is reported at once.
        """
        if not self._history:
            return None
        prev = self._history[-1]
        m = self.EDGE_MARGIN
        if hand is not None:  # rejected by the edge rule: the point is outside now, or about to be
            dt = (hand.t_ns - prev.t_ns) / 1e9
            if not 0 < dt <= self.MAX_DT_S:
                return None
            look = self.fast_exit_lookahead_frames * dt
            w0, w1 = prev.landmarks[WRIST], hand.landmarks[WRIST]
            for label, p0, p1, bottom in (
                ("wrist", (w0.x, w0.y), (w1.x, w1.y), False),
                ("pinch point", _pinch_point(prev), _pinch_point(hand), True),
            ):
                v = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / dt
                outside = p1[0] < -m or p1[0] > 1 + m or (label == "wrist" and p1[1] < -m) or (bottom and p1[1] > 1 + m)
                if v >= self.fast_exit_speed and outside:
                    return f"{label} left at {v:.1f} fw/s, now ({p1[0]:.2f},{p1[1]:.2f})"
                why = self._exiting(p0, p1, dt, look, bottom=bottom and self.fast_exit_bottom, reach=1.0, top=label == "wrist")
                if why:
                    return f"{label} {why}"
            return None
        if len(self._history) < 2:
            return None
        prev2 = self._history[-2]
        dt = (prev.t_ns - prev2.t_ns) / 1e9
        if not 0 < dt <= self.MAX_DT_S:
            return None
        look = (self.fast_exit_lookahead_frames + 1) * dt  # this frame is already one frame later
        w0, w1 = prev2.landmarks[WRIST], prev.landmarks[WRIST]
        why = self._exiting((w0.x, w0.y), (w1.x, w1.y), dt, look, bottom=False, reach=1.0)
        if why:
            return f"tracker lost the wrist {why}"
        why = self._exiting(_pinch_point(prev2), _pinch_point(prev), dt, look, bottom=self.fast_exit_bottom, reach=1.0, top=False)
        if why:
            return f"tracker lost the pinch point {why}"
        return None

    def update(self, hand: HandFrame | None, frame_index: int = 0) -> Presence:
        edge = self.out_of_frame(hand) if hand is not None else None
        if hand is not None and edge is None:
            if self.present:
                fast = self.fast_exit_detail(hand)
                if fast is not None:
                    self._reentry_clear = True
                    return self.declare_lost("fast-exit", fast)
            elif self._reentry_clear:
                if self.touching_border(hand):
                    return Presence()  # the phantom lingering at the border is not a return
                self._reentry_clear = False
            self._missing = 0
            self._last_reason = None
            self.present = True
            self._history.append(hand)
            return Presence(seen=hand)
        if not self.present:
            return Presence()
        if self.fast_exit:
            fast = self.racing_out_detail(hand)
            if fast is not None:
                self._reentry_clear = True
                return self.declare_lost("fast-exit", fast)
        self._missing += 1
        self._last_reason = ("edge", edge) if edge else ("tracker", f"tracker reported no hand (frame {frame_index})")
        if self._missing >= self.lost_frames:
            return self.declare_lost(*self._last_reason)
        return Presence()

    def declare_lost(self, reason: str, detail: str) -> Presence:
        """Report the hand lost now, skipping the debounce. No repeat while already lost."""
        if not self.present:
            return Presence()
        self.present = False
        self._missing = 0
        self._history.clear()
        return Presence(lost=(reason, detail))

    def gate_closed(self) -> Presence:
        """The gate closed: no motion and no hand for its hold time. Immediate."""
        return self.declare_lost("gate", "no motion and no tracked hand for the hold time")

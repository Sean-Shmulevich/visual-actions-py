"""Hand presence with debounce.

Evidence from the 2026-10-08 16:55 session: 32 of 64 hand-lost events lasted under
0.4 s and most exactly one frame. Each one reset the pinch detector and paused the
drag. A hand is therefore only reported lost after `lost_frames` consecutive frames
without a usable hand (tracker found none, or the edge rule rejected it). A gate
closure means no motion and no hand for the gate's hold time, so it reports at once.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import WRIST, HandFrame


@dataclass(frozen=True)
class Presence:
    seen: HandFrame | None = None  # a usable hand this frame
    lost: tuple[str, str] | None = None  # (reason, detail) when the hand is now declared lost


class PresenceFilter:
    EDGE_MARGIN = 0.02
    EDGE_MAX_OUTSIDE = 6

    def __init__(self, lost_frames: int = 3) -> None:
        self.lost_frames = lost_frames
        self.present = False
        self._missing = 0
        self._last_reason: tuple[str, str] | None = None

    @classmethod
    def out_of_frame(cls, hf: HandFrame) -> str | None:
        """MediaPipe keeps reporting a hand clamped to the border after it leaves."""
        m = cls.EDGE_MARGIN
        w = hf.landmarks[WRIST]
        if not (-m <= w.x <= 1 + m and -m <= w.y <= 1 + m):
            return f"wrist outside ({w.x:.2f},{w.y:.2f})"
        outside = sum(1 for lm in hf.landmarks if not (-m <= lm.x <= 1 + m and -m <= lm.y <= 1 + m))
        if outside >= cls.EDGE_MAX_OUTSIDE:
            xs = [lm.x for lm in hf.landmarks]
            ys = [lm.y for lm in hf.landmarks]
            return f"{outside} landmarks outside, x {min(xs):.2f}..{max(xs):.2f} y {min(ys):.2f}..{max(ys):.2f}"
        return None

    def update(self, hand: HandFrame | None, frame_index: int = 0) -> Presence:
        edge = self.out_of_frame(hand) if hand is not None else None
        if hand is not None and edge is None:
            self._missing = 0
            self._last_reason = None
            self.present = True
            return Presence(seen=hand)
        if not self.present:
            return Presence()
        self._missing += 1
        self._last_reason = ("edge", edge) if edge else ("tracker", f"tracker reported no hand (frame {frame_index})")
        if self._missing >= self.lost_frames:
            self.present = False
            self._missing = 0
            return Presence(lost=self._last_reason)
        return Presence()

    def gate_closed(self) -> Presence:
        """The gate closed: no motion and no hand for its hold time. Immediate."""
        if not self.present:
            return Presence()
        self.present = False
        self._missing = 0
        return Presence(lost=("gate", "no motion and no tracked hand for the hold time"))

"""Stick scrolling: the scroll hand is a joystick, not a slider.

The hand shape starts a scroll where it is; that spot is the anchor and stays put (the
overlay draws it as a sticky dot). Lifting the hand above the anchor scrolls up, dropping
it below scrolls down, and the speed follows how far from the anchor the hand is: nothing
inside a small dead zone, then a curve up to `max_lines_s` at `span` of the frame height.
The hand drives it per frame, so the scroll is continuous while the hand holds an offset.

Why not a slider (one scroll per step of travel): on the 2026-10-09 session the step
rarely fired and moved the page a few lines when it did; a joystick gives feedback the
whole time and lets a small held offset scroll a long page.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from .events import Bus
from .pointer import PointerMap


class ScrollPhase(Enum):
    SETTLE = "settle"  # the scroll shape is shown but not still yet: hold still to set the anchor
    START = "start"
    MOVE = "move"
    END = "end"


@dataclass(frozen=True)
class ScrollEvent:
    t_ns: int
    phase: ScrollPhase
    ax: float  # the anchor, in screen points
    ay: float
    x: float  # the hand now, in screen points
    y: float
    velocity: float = 0.0  # lines per second, positive = scrolling up (content moves down)
    lines: int = 0  # lines sent on this update (MOVE) or in total (END)


@dataclass
class ScrollController:
    bus: Bus
    scroll: Callable[[int, int], None]  # (dx, dy) in lines; the driver's scroll
    pmap: PointerMap  # hand point -> screen point, for the overlay only
    deadzone: float = 0.03  # offset (fraction of frame height) that does nothing
    span: float = 0.25  # offset at which the speed reaches max_lines_s
    max_lines_s: float = 90.0
    curve: float = 1.5  # speed = max * ((offset - deadzone) / (span - deadzone)) ** curve
    active: bool = False
    total_lines: int = 0
    _anchor: tuple[float, float] = (0.5, 0.5)
    _acc: float = 0.0
    _last_ns: int | None = None

    def velocity_for(self, dy: float) -> float:
        """Signed lines/s for a vertical offset `dy` (user frame, positive = hand lower)."""
        mag = abs(dy)
        if mag <= self.deadzone:
            return 0.0
        u = min(1.0, (mag - self.deadzone) / max(self.span - self.deadzone, 1e-6))
        speed = self.max_lines_s * u**self.curve
        return speed if dy < 0 else -speed  # hand up -> scroll up

    def settling(self, t_ns: int, x: float, y: float) -> None:
        """The shape is up but moving: the overlay shows where the anchor will be set."""
        sx, sy = self.pmap.to_screen(x, y)
        self.bus.publish(ScrollEvent(t_ns, ScrollPhase.SETTLE, sx, sy, sx, sy))

    def start(self, t_ns: int, x: float, y: float) -> None:
        self.active = True
        self.total_lines = 0
        self._anchor = (x, y)
        self._acc = 0.0
        self._last_ns = t_ns
        sx, sy = self.pmap.to_screen(x, y)
        self.bus.publish(ScrollEvent(t_ns, ScrollPhase.START, sx, sy, sx, sy))

    def update(self, t_ns: int, x: float, y: float) -> int:
        """Per frame. Returns the lines sent this frame."""
        if not self.active:
            return 0
        dt = 0.0 if self._last_ns is None else max(0.0, min(0.2, (t_ns - self._last_ns) / 1e9))
        self._last_ns = t_ns
        v = self.velocity_for(y - self._anchor[1])
        self._acc += v * dt
        lines = int(self._acc)  # truncates toward zero
        if lines:
            self._acc -= lines
            self.scroll(0, lines)
            self.total_lines += lines
        ax, ay = self.pmap.to_screen(*self._anchor)
        sx, sy = self.pmap.to_screen(x, y)
        self.bus.publish(ScrollEvent(t_ns, ScrollPhase.MOVE, ax, ay, sx, sy, v, lines))
        return lines

    def end(self, t_ns: int) -> None:
        if not self.active:
            return
        self.active = False
        ax, ay = self.pmap.to_screen(*self._anchor)
        self.bus.publish(ScrollEvent(t_ns, ScrollPhase.END, ax, ay, ax, ay, 0.0, self.total_lines))

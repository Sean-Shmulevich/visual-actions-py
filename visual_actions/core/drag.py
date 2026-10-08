"""Pinch-and-drag of windows. Pure logic against a small WindowMover interface so it
runs headless; the dispatcher binds it to the platform driver."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from .events import Bus
from .pinch import PinchEvent, PinchPhase
from .pointer import SmoothedPointer


class WindowMover(Protocol):
    def grab(self, x: float, y: float) -> Any | None:
        """Return an opaque handle for the window under screen point (x, y), or None."""
        ...

    def origin(self, handle: Any) -> tuple[float, float]:
        """Current top-left of the window in screen points."""
        ...

    def move(self, handle: Any, x: float, y: float) -> bool: ...

    def label(self, handle: Any) -> str: ...


class DragPhase(Enum):
    START = "start"
    MOVE = "move"
    END = "end"
    MISS = "miss"  # pinched over nothing


@dataclass(frozen=True)
class DragEvent:
    t_ns: int
    phase: DragPhase
    window: str
    x: float  # pointer in screen points
    y: float


class DragController:
    """Turns pinch events into window moves. One drag at a time."""

    def __init__(self, bus: Bus, mover: WindowMover, pointer: SmoothedPointer, gain: float = 1.0) -> None:
        self.bus = bus
        self.mover = mover
        self.pointer = pointer
        self.gain = gain
        self.handle: Any | None = None
        self._grab_pointer: tuple[float, float] | None = None
        self._grab_origin: tuple[float, float] | None = None
        self.moves = 0

    @property
    def dragging(self) -> bool:
        return self.handle is not None

    def on_pinch(self, ev: PinchEvent) -> bool:
        """Returns True if the event was consumed by a drag."""
        if ev.phase is PinchPhase.START:
            self.pointer.reset()
            sx, sy = self.pointer.update(ev.t_ns, ev.x, ev.y, ev.hand_scale)
            handle = self.mover.grab(sx, sy)
            if handle is None:
                self.bus.publish(DragEvent(ev.t_ns, DragPhase.MISS, "", sx, sy))
                return False
            self.handle = handle
            self._grab_pointer = (sx, sy)
            self._grab_origin = self.mover.origin(handle)
            self.moves = 0
            self.bus.publish(DragEvent(ev.t_ns, DragPhase.START, self.mover.label(handle), sx, sy))
            return True
        if self.handle is None:
            return False
        sx, sy = self.pointer.update(ev.t_ns, ev.x, ev.y, ev.hand_scale)
        if ev.phase is PinchPhase.MOVE:
            assert self._grab_pointer is not None and self._grab_origin is not None
            nx = self._grab_origin[0] + (sx - self._grab_pointer[0]) * self.gain
            ny = self._grab_origin[1] + (sy - self._grab_pointer[1]) * self.gain
            if self.mover.move(self.handle, nx, ny):
                self.moves += 1
            self.bus.publish(DragEvent(ev.t_ns, DragPhase.MOVE, self.mover.label(self.handle), sx, sy))
            return True
        self._end(ev.t_ns, sx, sy)
        return True

    def cancel(self, t_ns: int) -> None:
        if self.handle is not None:
            self._end(t_ns, *(self._grab_pointer or (0.0, 0.0)))

    def _end(self, t_ns: int, sx: float, sy: float) -> None:
        label = self.mover.label(self.handle)
        self.handle = None
        self._grab_pointer = None
        self._grab_origin = None
        self.bus.publish(DragEvent(t_ns, DragPhase.END, label, sx, sy))


class FakeWindows:
    """A headless WindowMover: rectangles in z-order (front first). Used by tests and replay."""

    def __init__(self, windows: list[tuple[str, float, float, float, float]] | None = None) -> None:
        # (label, x, y, w, h)
        self.windows: list[dict[str, Any]] = [
            {"label": lb, "x": x, "y": y, "w": w, "h": h}
            for lb, x, y, w, h in (windows or [("Front", 300, 200, 600, 400), ("Back", 100, 100, 900, 600)])
        ]
        self.moves: list[tuple[str, float, float]] = []

    def grab(self, x: float, y: float) -> dict[str, Any] | None:
        for w in self.windows:
            if w["x"] <= x <= w["x"] + w["w"] and w["y"] <= y <= w["y"] + w["h"]:
                return w
        return None

    def origin(self, handle: dict[str, Any]) -> tuple[float, float]:
        return (handle["x"], handle["y"])

    def move(self, handle: dict[str, Any], x: float, y: float) -> bool:
        handle["x"], handle["y"] = x, y
        self.moves.append((handle["label"], x, y))
        return True

    def label(self, handle: dict[str, Any]) -> str:
        return handle["label"]

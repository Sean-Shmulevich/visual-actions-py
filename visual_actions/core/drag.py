"""Pinch-and-drag of windows, with optional edge snapping. Pure logic against a small
WindowMover interface so it runs headless; the dispatcher binds it to the driver."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from .automation import Rect
from .events import Bus, SnapPreview
from .pinch import PinchEvent, PinchPhase
from .pointer import SmoothedPointer
from .snap import SnapEngine


class WindowMover(Protocol):
    def grab(self, x: float, y: float) -> Any | None:
        """Return an opaque handle for the window under screen point (x, y), or None."""
        ...

    def frame(self, handle: Any) -> Rect:
        """The window's frame at grab time."""
        ...

    def move(self, handle: Any, x: float, y: float) -> bool: ...

    def set_frame(self, handle: Any, rect: Rect) -> bool: ...

    def focus(self, handle: Any, x: float, y: float) -> bool:
        """Make the window active and focus the element under (x, y). Never a click."""
        ...

    def label(self, handle: Any) -> str: ...


class DragPhase(Enum):
    START = "start"
    MOVE = "move"
    END = "end"
    MISS = "miss"  # pinched over nothing
    PAUSE = "pause"  # hand lost mid-drag: window stays, waiting for the hand to come back
    RESUME = "resume"  # hand came back pinching: continue from where the window is


@dataclass(frozen=True)
class DragEvent:
    t_ns: int
    phase: DragPhase
    window: str
    x: float  # pointer in screen points
    y: float
    snapped: str | None = None  # zone name when a release snapped the window


class DragController:
    """Turns pinch events into window moves. One drag at a time."""

    def __init__(
        self,
        bus: Bus,
        mover: WindowMover,
        pointer: SmoothedPointer,
        gain: float = 1.0,
        snap: SnapEngine | None = None,
        focus_on_grab: bool = True,
    ) -> None:
        self.focus_on_grab = focus_on_grab
        self.bus = bus
        self.mover = mover
        self.pointer = pointer
        self.gain = gain
        self.snap = snap
        self.handle: Any | None = None
        self._grab_pointer: tuple[float, float] | None = None
        self._grab_origin: tuple[float, float] | None = None
        self._size: tuple[int, int] = (0, 0)
        self.moves = 0
        # label -> (frame we snapped it to, frame it had before). Lets a re-drag un-snap.
        self._snapped: dict[str, tuple[Rect, Rect]] = {}
        self._preview: str | None = None
        self.suspended = False
        self._pos: tuple[float, float] | None = None  # where we last put the window

    @property
    def dragging(self) -> bool:
        return self.handle is not None

    # -- pinch input ------------------------------------------------------------

    def on_pinch(self, ev: PinchEvent) -> bool:
        """Returns True if the event was consumed by a drag."""
        if self.suspended and self.handle is not None:
            if ev.phase is PinchPhase.END:
                return False  # a release while suspended is handled by the engine (drop)
            self._resume(ev)
            return True
        if ev.phase is PinchPhase.START:
            return self._start(ev)
        if self.handle is None:
            return False
        sx, sy = self.pointer.update(ev.t_ns, ev.x, ev.y, ev.hand_scale)
        if ev.phase is PinchPhase.MOVE:
            self._move(ev.t_ns, sx, sy)
            return True
        self._end(ev.t_ns, sx, sy)
        return True

    def suspend(self, t_ns: int) -> None:
        """Hand lost mid-drag: keep the window where it is, clear any snap preview, wait."""
        if self.handle is None or self.suspended:
            return
        self.suspended = True
        self._set_preview(t_ns, None)
        if self.snap is not None:
            self.snap.reset()
        self.bus.publish(DragEvent(t_ns, DragPhase.PAUSE, self.mover.label(self.handle), *(self._grab_pointer or (0.0, 0.0))))

    def _resume(self, ev: PinchEvent) -> None:
        """Re-anchor on the returning hand so the window continues from its current place."""
        self.suspended = False
        self.pointer.reset()
        sx, sy = self.pointer.update(ev.t_ns, ev.x, ev.y, ev.hand_scale)
        self._grab_pointer = (sx, sy)
        self._grab_origin = self._pos or self._grab_origin
        self.bus.publish(DragEvent(ev.t_ns, DragPhase.RESUME, self.mover.label(self.handle), sx, sy))

    def cancel(self, t_ns: int) -> None:
        """Drop in place, never snap (hand lost for good, or came back without a pinch)."""
        if self.handle is not None:
            self.suspended = False
            self._set_preview(t_ns, None)
            if self.snap is not None:
                self.snap.reset()
            self._finish(t_ns, *(self._grab_pointer or (0.0, 0.0)), snapped=None)

    # -- phases -------------------------------------------------------------------

    def _start(self, ev: PinchEvent) -> bool:
        self.pointer.reset()
        sx, sy = self.pointer.update(ev.t_ns, ev.x, ev.y, ev.hand_scale)
        handle = self.mover.grab(sx, sy)
        if handle is None:
            self.bus.publish(DragEvent(ev.t_ns, DragPhase.MISS, "", sx, sy))
            return False
        label = self.mover.label(handle)
        frame = self.mover.frame(handle)
        # Un-snap: a window we snapped earlier gets its old size back, placed so the
        # pointer keeps the same relative position inside it.
        prev = self._snapped.get(label)
        if prev is not None and self._same_rect(frame, prev[0]):
            old = prev[1]
            rel_x = (sx - frame.x) / max(frame.w, 1)
            rel_y = (sy - frame.y) / max(frame.h, 1)
            frame = Rect(int(sx - rel_x * old.w), int(sy - rel_y * old.h), old.w, old.h)
            self.mover.set_frame(handle, frame)
            del self._snapped[label]
        self.handle = handle
        self._grab_pointer = (sx, sy)
        self._grab_origin = (float(frame.x), float(frame.y))
        self._size = (frame.w, frame.h)
        self._pos = (float(frame.x), float(frame.y))
        self.suspended = False
        self.moves = 0
        if self.focus_on_grab:
            self.mover.focus(handle, sx, sy)  # the grabbed window becomes the active one
        if self.snap is not None:
            self.snap.reset()
        self.bus.publish(DragEvent(ev.t_ns, DragPhase.START, label, sx, sy))
        return True

    def _move(self, t_ns: int, sx: float, sy: float) -> None:
        assert self._grab_pointer is not None and self._grab_origin is not None
        nx = self._grab_origin[0] + (sx - self._grab_pointer[0]) * self.gain
        ny = self._grab_origin[1] + (sy - self._grab_pointer[1]) * self.gain
        if self.mover.move(self.handle, nx, ny):
            self.moves += 1
            self._pos = (nx, ny)
        if self.snap is not None:
            zone = self.snap.update(sx, sy, t_ns)
            self._set_preview(t_ns, zone.name if zone else None, zone.target if zone else None)
        self.bus.publish(DragEvent(t_ns, DragPhase.MOVE, self.mover.label(self.handle), sx, sy))

    def _end(self, t_ns: int, sx: float, sy: float) -> None:
        snapped: str | None = None
        if self.snap is not None and self.snap.active is not None:
            zone = self.snap.active
            label = self.mover.label(self.handle)
            before = self._current_frame(sx, sy)
            if self.mover.set_frame(self.handle, zone.target):
                self._snapped[label] = (zone.target, before)
                snapped = zone.name
            self.snap.reset()
        self._set_preview(t_ns, None)
        self._finish(t_ns, sx, sy, snapped)

    def _finish(self, t_ns: int, sx: float, sy: float, snapped: str | None) -> None:
        label = self.mover.label(self.handle)
        self.handle = None
        self.suspended = False
        self._grab_pointer = None
        self._grab_origin = None
        self._pos = None
        self.bus.publish(DragEvent(t_ns, DragPhase.END, label, sx, sy, snapped=snapped))

    # -- helpers ------------------------------------------------------------------

    def _current_frame(self, sx: float, sy: float) -> Rect:
        assert self._grab_pointer is not None and self._grab_origin is not None
        x = self._grab_origin[0] + (sx - self._grab_pointer[0]) * self.gain
        y = self._grab_origin[1] + (sy - self._grab_pointer[1]) * self.gain
        return Rect(int(x), int(y), self._size[0], self._size[1])

    def _set_preview(self, t_ns: int, zone: str | None, rect: Rect | None = None) -> None:
        if zone == self._preview:
            return
        self._preview = zone
        if zone is None or rect is None:
            self.bus.publish(SnapPreview(t_ns, None))
        else:
            self.bus.publish(SnapPreview(t_ns, zone, rect.x, rect.y, rect.w, rect.h))

    @staticmethod
    def _same_rect(a: Rect, b: Rect, tol: int = 4) -> bool:
        return abs(a.x - b.x) <= tol and abs(a.y - b.y) <= tol and abs(a.w - b.w) <= tol and abs(a.h - b.h) <= tol


class FakeWindows:
    """A headless WindowMover: rectangles in z-order (front first). Used by tests and replay."""

    def __init__(self, windows: list[tuple[str, float, float, float, float]] | None = None) -> None:
        # (label, x, y, w, h)
        self.windows: list[dict[str, Any]] = [
            {"label": lb, "x": x, "y": y, "w": w, "h": h}
            for lb, x, y, w, h in (windows or [("Front", 300, 200, 600, 400), ("Back", 100, 100, 900, 600)])
        ]
        self.moves: list[tuple[str, float, float]] = []
        self.focused: list[tuple[str, float, float]] = []

    def focus(self, handle: dict[str, Any], x: float, y: float) -> bool:
        self.focused.append((handle["label"], x, y))
        return True

    def grab(self, x: float, y: float) -> dict[str, Any] | None:
        for w in self.windows:
            if w["x"] <= x <= w["x"] + w["w"] and w["y"] <= y <= w["y"] + w["h"]:
                return w
        return None

    def frame(self, handle: dict[str, Any]) -> Rect:
        return Rect(int(handle["x"]), int(handle["y"]), int(handle["w"]), int(handle["h"]))

    def move(self, handle: dict[str, Any], x: float, y: float) -> bool:
        handle["x"], handle["y"] = x, y
        self.moves.append((handle["label"], x, y))
        return True

    def set_frame(self, handle: dict[str, Any], rect: Rect) -> bool:
        handle["x"], handle["y"], handle["w"], handle["h"] = rect.x, rect.y, rect.w, rect.h
        return True

    def label(self, handle: dict[str, Any]) -> str:
        return handle["label"]

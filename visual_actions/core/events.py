"""Typed synchronous pub/sub. Handlers run in order on the publishing thread."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .types import Action, HandFrame, Token


@dataclass(frozen=True)
class FrameCaptured:
    t_ns: int
    gate_open: bool
    frame: Any = None  # ndarray only when a preview wants pixels


@dataclass(frozen=True)
class HandSeen:
    hand_frame: HandFrame
    face_overlap: float = 0.0  # fraction of the hand box inside a face box (0 when no face detection)


@dataclass(frozen=True)
class PalmVetoed:
    """An open palm was ignored because the hand is on a face (face-touch veto)."""

    t_ns: int
    overlap: float
    spread: float


@dataclass(frozen=True)
class HandLost:
    t_ns: int
    reason: str = ""  # gate | tracker | edge
    detail: str = ""  # e.g. landmark bounds at the moment of loss


@dataclass(frozen=True)
class TokenEmitted:
    token: Token


@dataclass(frozen=True)
class ModeChanged:
    t_ns: int
    old: str
    new: str
    namespace: str | None = None
    deadline_ns: int | None = None


@dataclass(frozen=True)
class HoldProgress:
    t_ns: int
    fraction: float  # 0..1 of the leader hold, confidence-weighted
    rate: float = 0.0  # evidence per wall-clock second, in units of the full hold (for smooth extrapolation)


@dataclass(frozen=True)
class PointerMoved:
    """Where a pinch would land (ARMED) or where the drag is (DRAGGING), in screen points."""

    t_ns: int
    x: float
    y: float
    dragging: bool


class AdjustPhase(Enum):
    SETTLE = "settle"  # pinched but not still yet: hold still to set the anchor
    START = "start"  # settled: the anchor is set here
    MOVE = "move"  # a pinch frame after settling; `steps` fired on this frame (+ right, - left)
    END = "end"  # released, fist, or lost past the grace; `steps` is the total fired


@dataclass(frozen=True)
class AdjustEvent:
    """The media volume pinch (ADJUST), in screen points: the sticky anchor and the pinch point.
    The anchor moves one adjust_step per step fired, so the ring shows where the next step counts from."""

    t_ns: int
    phase: AdjustPhase
    ax: float  # the anchor, in screen points
    ay: float
    x: float  # the pinch point now, in screen points
    y: float
    steps: int = 0


@dataclass(frozen=True)
class SnapPreview:
    """The frame the window would snap to on release, or None when no zone is active."""

    t_ns: int
    zone: str | None
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0


@dataclass(frozen=True)
class ActionFired:
    t_ns: int
    action: Action
    ok: bool
    message: str = ""


@dataclass(frozen=True)
class Tick:
    t_ns: int


Event = FrameCaptured | HandSeen | HandLost | TokenEmitted | ModeChanged | HoldProgress | ActionFired | Tick
Handler = Callable[[Any], None]


class Bus:
    def __init__(self) -> None:
        self._handlers: dict[type, list[Handler]] = defaultdict(list)

    def subscribe(self, event_type: type, handler: Handler) -> None:
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: type, handler: Handler) -> None:
        self._handlers[event_type].remove(handler)

    def publish(self, event: object) -> None:
        for handler in list(self._handlers[type(event)]):
            handler(event)

"""Core value types. Everything here is frozen and hashable so it can cross threads and the bus."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Hand(Enum):
    LEFT = "left"
    RIGHT = "right"


# MediaPipe landmark indices, used everywhere by name.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
N_LANDMARKS = 21


@dataclass(frozen=True)
class Landmark:
    x: float  # 0..1 across the frame; in the user frame +x is the user's right
    y: float  # 0..1 down the frame
    z: float = 0.0  # relative depth, MediaPipe convention


@dataclass(frozen=True)
class HandFrame:
    t_ns: int
    hand: Hand
    landmarks: tuple[Landmark, ...]
    confidence: float

    def __post_init__(self) -> None:
        if len(self.landmarks) != N_LANDMARKS:
            raise ValueError(f"HandFrame needs {N_LANDMARKS} landmarks, got {len(self.landmarks)}")


@dataclass(frozen=True)
class Token:
    """A recognized, smoothed gesture. The mode engine consumes tokens, never frames."""

    t_ns: int
    name: str  # "open_palm", "fist", "h_left", "h_right", "none"
    confidence: float
    hand: Hand
    still: bool


class ActionKind(Enum):
    KEY = "key"
    SCROLL = "scroll"
    WINDOW = "window"
    MEDIA = "media"
    PLUGIN = "plugin"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    name: str  # label for the popup, e.g. "Cmd+Tab"
    args: tuple[tuple[str, str], ...] = ()

    def arg(self, key: str, default: str | None = None) -> str | None:
        for k, v in self.args:
            if k == key:
                return v
        return default


@dataclass(frozen=True)
class Binding:
    namespace: str
    gesture: str
    action: Action

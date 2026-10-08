"""The DesktopAutomation bridge interface. One driver per platform implements it."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol


class MediaVerb(Enum):
    PLAY_PAUSE = "play_pause"
    NEXT = "next"
    PREV = "prev"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"


@dataclass(frozen=True)
class WindowRef:
    frontmost: bool = True
    title: str | None = None


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class NativeResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""


class DesktopAutomation(Protocol):
    def press(self, chord: str) -> None: ...

    def scroll(self, dx: int, dy: int) -> None: ...

    def set_window_frame(self, target: WindowRef, frame: Rect) -> None: ...

    def media(self, verb: MediaVerb) -> None: ...

    def run_native(self, script_path: Path, timeout_s: float) -> NativeResult: ...

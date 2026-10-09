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
    MUTE = "mute"


@dataclass(frozen=True)
class WindowRef:
    frontmost: bool = True
    title: str | None = None


@dataclass(frozen=True)
class FocusResult:
    """What a focus request did. `detail` is for the session log: which app was active before,
    which was asked for, and what each mechanism returned, so a focus that did not stick can be
    diagnosed from a recording instead of guessed at."""

    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class WindowInfo:
    """An on-screen window. `frame` is in global screen points, origin top-left."""

    id: int
    pid: int
    app: str
    title: str
    frame: Rect
    layer: int = 0


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

    def volume(self) -> tuple[float, bool] | None:
        """System output volume as (level 0..1, muted), or None where it cannot be read."""
        ...

    def run_native(self, script_path: Path, timeout_s: float) -> NativeResult: ...

    def open(self, target: str) -> bool:
        """Open a URL or file with the system's default handler."""
        ...

    def focus_at(self, x: float, y: float) -> FocusResult:
        """Activate the app and window under the screen point and give keyboard focus to the
        element there (a terminal pane, a sidebar), without any mouse click."""
        ...

    # -- windows (screen points, origin top-left) ---------------------------

    def list_windows(self) -> list[WindowInfo]:
        """On-screen normal windows, front-most first."""
        ...

    def window_at(self, x: float, y: float) -> WindowInfo | None:
        """The front-most window containing the point, or None."""
        ...

    def move_window(self, win: WindowInfo, x: float, y: float) -> bool:
        """Set the window's top-left corner. False if it cannot be moved."""
        ...

    def screen_size(self) -> tuple[int, int]:
        """Main display size in points."""
        ...

    def resize_window(self, win: WindowInfo, w: float, h: float) -> bool:
        """Set the window's size. False if it cannot be resized."""
        ...

    def visible_frame(self) -> Rect:
        """Main display area windows may occupy (minus menu bar, dock, taskbar), top-left origin."""
        ...

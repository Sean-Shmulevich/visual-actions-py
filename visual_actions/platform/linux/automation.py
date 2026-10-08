"""Linux mock. Logs only. Real driver: xdotool on X11, ydotool on Wayland."""

from __future__ import annotations

import logging
from pathlib import Path

from ...core.automation import MediaVerb, NativeResult, Rect, WindowInfo, WindowRef

log = logging.getLogger(__name__)


class LinuxAutomation:
    def press(self, chord: str) -> None:
        log.info("mock press %s", chord)

    def scroll(self, dx: int, dy: int) -> None:
        log.info("mock scroll %s %s", dx, dy)

    def set_window_frame(self, target: WindowRef, frame: Rect) -> None:
        log.info("mock set_window_frame %s %s", target, frame)

    def media(self, verb: MediaVerb) -> None:
        log.info("mock media %s", verb)

    def run_native(self, script_path: Path, timeout_s: float) -> NativeResult:
        log.info("mock run_native %s", script_path)
        return NativeResult(ok=True, stdout="mock")

    def list_windows(self) -> list[WindowInfo]:
        return []

    def window_at(self, x: float, y: float) -> WindowInfo | None:
        return None

    def move_window(self, win: WindowInfo, x: float, y: float) -> bool:
        return False

    def screen_size(self) -> tuple[int, int]:
        return (1440, 900)

    def resize_window(self, win: WindowInfo, w: float, h: float) -> bool:
        return False

    def visible_frame(self) -> Rect:
        return Rect(0, 0, 1440, 900)

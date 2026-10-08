"""Windows stub. Logs and returns success. Real driver: SendInput + UI Automation."""

from __future__ import annotations

import logging
from pathlib import Path

from ...core.automation import MediaVerb, NativeResult, Rect, WindowRef

log = logging.getLogger(__name__)


class WindowsAutomation:
    def press(self, chord: str) -> None:
        log.info("stub press %s", chord)

    def scroll(self, dx: int, dy: int) -> None:
        log.info("stub scroll %s %s", dx, dy)

    def set_window_frame(self, target: WindowRef, frame: Rect) -> None:
        log.info("stub set_window_frame %s %s", target, frame)

    def media(self, verb: MediaVerb) -> None:
        log.info("stub media %s", verb)

    def run_native(self, script_path: Path, timeout_s: float) -> NativeResult:
        log.info("stub run_native %s", script_path)
        return NativeResult(ok=True, stdout="stub")

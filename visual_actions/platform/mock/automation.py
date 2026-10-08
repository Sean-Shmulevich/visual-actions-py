"""Records every call. Used by tests on every OS and by --dry runs."""

from __future__ import annotations

from pathlib import Path

from ...core.automation import MediaVerb, NativeResult, Rect, WindowRef


class MockAutomation:
    def __init__(self, echo: bool = False) -> None:
        self.calls: list[tuple] = []
        self.echo = echo

    def _rec(self, *call: object) -> None:
        self.calls.append(call)
        if self.echo:
            print("[mock automation]", *call)

    def press(self, chord: str) -> None:
        self._rec("press", chord)

    def scroll(self, dx: int, dy: int) -> None:
        self._rec("scroll", dx, dy)

    def set_window_frame(self, target: WindowRef, frame: Rect) -> None:
        self._rec("set_window_frame", target, frame)

    def media(self, verb: MediaVerb) -> None:
        self._rec("media", verb)

    def run_native(self, script_path: Path, timeout_s: float) -> NativeResult:
        self._rec("run_native", str(script_path), timeout_s)
        return NativeResult(ok=True, stdout="mock")

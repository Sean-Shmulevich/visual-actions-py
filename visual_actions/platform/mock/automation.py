"""Records every call. Used by tests on every OS and by --dry runs.

Holds a fake window list (front-most first) so window lookups and moves can be
tested headless. Default: two overlapping windows on a 1440x900 screen.
"""

from __future__ import annotations

from pathlib import Path

from ...core.automation import MediaVerb, NativeResult, Rect, WindowInfo, WindowRef


def default_windows() -> list[WindowInfo]:
    return [
        WindowInfo(id=1, pid=100, app="Front App", title="Front", frame=Rect(400, 200, 600, 400)),
        WindowInfo(id=2, pid=101, app="Back App", title="Back", frame=Rect(100, 100, 800, 600)),
    ]


class MockAutomation:
    def __init__(
        self,
        echo: bool = False,
        windows: list[WindowInfo] | None = None,
        screen: tuple[int, int] = (1440, 900),
        movable: bool = True,
    ) -> None:
        self.calls: list[tuple] = []
        self.echo = echo
        self.windows: list[WindowInfo] = list(windows) if windows is not None else default_windows()
        self.screen = screen
        self.movable = movable

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

    # -- windows -------------------------------------------------------------

    def list_windows(self) -> list[WindowInfo]:
        return list(self.windows)

    def window_at(self, x: float, y: float) -> WindowInfo | None:
        self._rec("window_at", x, y)
        for w in self.windows:
            f = w.frame
            if f.x <= x < f.x + f.w and f.y <= y < f.y + f.h:
                return w
        return None

    def move_window(self, win: WindowInfo, x: float, y: float) -> bool:
        self._rec("move_window", win.id, x, y)
        if not self.movable:
            return False
        for i, w in enumerate(self.windows):
            if w.id == win.id:
                self.windows[i] = WindowInfo(w.id, w.pid, w.app, w.title, Rect(int(x), int(y), w.frame.w, w.frame.h), w.layer)
                return True
        return False

    def screen_size(self) -> tuple[int, int]:
        return self.screen

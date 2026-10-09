"""Records every call. Used by tests on every OS and by --dry runs.

Holds a fake window list (front-most first) so window lookups and moves can be
tested headless. Default: two overlapping windows on a 1440x900 screen.
"""

from __future__ import annotations

from pathlib import Path

from ...core.automation import FocusResult, MediaVerb, NativeResult, Rect, WindowInfo, WindowRef


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
        self.volume_level = 0.5
        self.muted = False
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
        step = 1 / 16  # one hardware volume key step
        if verb is MediaVerb.VOLUME_UP:
            self.volume_level, self.muted = min(1.0, self.volume_level + step), False
        elif verb is MediaVerb.VOLUME_DOWN:
            self.volume_level = max(0.0, self.volume_level - step)
        elif verb is MediaVerb.MUTE:
            self.muted = not self.muted

    def volume(self) -> tuple[float, bool] | None:
        return self.volume_level, self.muted

    def run_native(self, script_path: Path, timeout_s: float) -> NativeResult:
        self._rec("run_native", str(script_path), timeout_s)
        return NativeResult(ok=True, stdout="mock")

    def focus_at(self, x: float, y: float) -> FocusResult:
        self._rec("focus_at", x, y)
        return FocusResult(True, "mock")

    def open(self, target: str) -> bool:
        self._rec("open", target)
        return True

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

    def resize_window(self, win: WindowInfo, w: float, h: float) -> bool:
        self._rec("resize_window", win.id, w, h)
        if not self.movable:
            return False
        for i, win_ in enumerate(self.windows):
            if win_.id == win.id:
                f = win_.frame
                self.windows[i] = WindowInfo(win_.id, win_.pid, win_.app, win_.title, Rect(f.x, f.y, int(w), int(h)), win_.layer)
                return True
        return False

    def visible_frame(self) -> Rect:
        w, h = self.screen
        return Rect(0, 25, w, h - 25)  # a 25 pt menu bar, like macOS

"""macOS window lookup (Quartz window list) and window moves (Accessibility API).

Coordinates everywhere are global screen points with the origin at the top-left of
the main display, which is what both CGWindowList bounds and AX positions use.
"""

from __future__ import annotations

import logging
import os
import time

from ...core.automation import Rect, WindowInfo

log = logging.getLogger(__name__)

MATCH_TOLERANCE_PT = 2.0
TINY_PT = 24  # windows with no title and both sides under this are menu bar extras


def match_ax_window(target: WindowInfo, candidates: list[tuple[int, str, Rect]]) -> int | None:
    """Pick the AX window matching a CG window. Pure, for tests.

    `candidates` are (index, title, frame) for each AX window of the owning app.
    Frame match (position and size within 2 pt) wins; otherwise a unique title match.
    """
    for i, _title, frame in candidates:
        if (
            abs(frame.x - target.frame.x) <= MATCH_TOLERANCE_PT
            and abs(frame.y - target.frame.y) <= MATCH_TOLERANCE_PT
            and abs(frame.w - target.frame.w) <= MATCH_TOLERANCE_PT
            and abs(frame.h - target.frame.h) <= MATCH_TOLERANCE_PT
        ):
            return i
    if target.title:
        by_title = [i for i, title, _ in candidates if title == target.title]
        if len(by_title) == 1:
            return by_title[0]
    return None


def pick_window_at(windows: list[WindowInfo], x: float, y: float, own_pid: int) -> WindowInfo | None:
    """Front-most window (list order) containing the point. Pure, for tests."""
    for w in windows:
        if w.pid == own_pid:
            continue
        if not w.title and w.frame.w < TINY_PT and w.frame.h < TINY_PT:
            continue
        f = w.frame
        if f.x <= x < f.x + f.w and f.y <= y < f.y + f.h:
            return w
    return None


class MacWindows:
    def __init__(self) -> None:
        import Quartz

        self._q = Quartz
        self._own_pid = os.getpid()
        self._ax_cache: dict[int, object] = {}  # window id -> AXUIElement
        self._ax_cache_pid: dict[int, int] = {}
        self.last_move_ms: float = 0.0

    # -- listing --------------------------------------------------------------

    def list_windows(self) -> list[WindowInfo]:
        q = self._q
        opts = q.kCGWindowListOptionOnScreenOnly | q.kCGWindowListExcludeDesktopElements
        infos = q.CGWindowListCopyWindowInfo(opts, q.kCGNullWindowID) or []
        out: list[WindowInfo] = []
        for d in infos:
            if d.get("kCGWindowLayer", 0) != 0 or d.get("kCGWindowAlpha", 1.0) <= 0:
                continue
            app = d.get("kCGWindowOwnerName") or ""
            if not app:
                continue
            b = d.get("kCGWindowBounds") or {}
            out.append(
                WindowInfo(
                    id=int(d.get("kCGWindowNumber", 0)),
                    pid=int(d.get("kCGWindowOwnerPID", 0)),
                    app=str(app),
                    title=str(d.get("kCGWindowName") or ""),
                    frame=Rect(int(b.get("X", 0)), int(b.get("Y", 0)), int(b.get("Width", 0)), int(b.get("Height", 0))),
                    layer=0,
                )
            )
        return out  # CGWindowList is front-to-back already

    def window_at(self, x: float, y: float) -> WindowInfo | None:
        return pick_window_at(self.list_windows(), x, y, self._own_pid)

    def screen_size(self) -> tuple[int, int]:
        q = self._q
        did = q.CGMainDisplayID()
        return int(q.CGDisplayPixelsWide(did)), int(q.CGDisplayPixelsHigh(did))

    # -- moving ---------------------------------------------------------------

    def _ax_element(self, win: WindowInfo) -> object | None:
        from ApplicationServices import (
            AXUIElementCopyAttributeValue,
            AXUIElementCreateApplication,
            AXValueGetValue,
            kAXErrorSuccess,
            kAXPositionAttribute,
            kAXSizeAttribute,
            kAXTitleAttribute,
            kAXValueCGPointType,
            kAXValueCGSizeType,
            kAXWindowsAttribute,
        )

        cached = self._ax_cache.get(win.id)
        if cached is not None and self._ax_cache_pid.get(win.id) == win.pid:
            return cached
        app = AXUIElementCreateApplication(win.pid)
        err, ax_windows = AXUIElementCopyAttributeValue(app, kAXWindowsAttribute, None)
        if err != kAXErrorSuccess or not ax_windows:
            log.warning("AX windows unavailable for pid %s (err %s); is Accessibility granted?", win.pid, err)
            return None
        candidates: list[tuple[int, str, Rect]] = []
        for i, el in enumerate(ax_windows):
            _, pos_v = AXUIElementCopyAttributeValue(el, kAXPositionAttribute, None)
            _, size_v = AXUIElementCopyAttributeValue(el, kAXSizeAttribute, None)
            _, title = AXUIElementCopyAttributeValue(el, kAXTitleAttribute, None)
            if pos_v is None or size_v is None:
                continue
            ok_p, pos = AXValueGetValue(pos_v, kAXValueCGPointType, None)
            ok_s, size = AXValueGetValue(size_v, kAXValueCGSizeType, None)
            if not (ok_p and ok_s):
                continue
            candidates.append((i, str(title or ""), Rect(int(pos.x), int(pos.y), int(size.width), int(size.height))))
        idx = match_ax_window(win, candidates)
        if idx is None:
            log.warning("no AX window matches CG window %s (%s: %r)", win.id, win.app, win.title)
            return None
        el = ax_windows[idx]
        self._ax_cache[win.id] = el
        self._ax_cache_pid[win.id] = win.pid
        return el

    def move_window(self, win: WindowInfo, x: float, y: float) -> bool:
        from ApplicationServices import (
            AXUIElementSetAttributeValue,
            AXValueCreate,
            kAXErrorSuccess,
            kAXPositionAttribute,
            kAXValueCGPointType,
        )
        from Quartz import CGPoint

        t0 = time.perf_counter()
        el = self._ax_element(win)
        if el is None:
            return False
        value = AXValueCreate(kAXValueCGPointType, CGPoint(x, y))
        err = AXUIElementSetAttributeValue(el, kAXPositionAttribute, value)
        self.last_move_ms = (time.perf_counter() - t0) * 1000
        if err != kAXErrorSuccess:
            log.warning("AX move failed for %s (err %s)", win.app, err)
            self._ax_cache.pop(win.id, None)  # the element may be stale; re-resolve next time
            return False
        return True

    def forget(self, win: WindowInfo) -> None:
        self._ax_cache.pop(win.id, None)
        self._ax_cache_pid.pop(win.id, None)

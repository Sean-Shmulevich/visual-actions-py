"""Spike: focus_at on the window at screen centre, without any mouse event.

Reports the frontmost app before/after, the AX element that holds focus afterwards
(role + title), the cost, then restores the previously frontmost app.
"""

import subprocess
import time

from AppKit import NSApplicationActivateIgnoringOtherApps, NSWorkspace
from ApplicationServices import (
    AXUIElementCopyAttributeValue,
    AXUIElementCreateApplication,
    kAXFocusedUIElementAttribute,
    kAXRoleAttribute,
    kAXSubroleAttribute,
    kAXTitleAttribute,
)

from visual_actions.platform.macos.automation import MacAutomation


def frontmost() -> str:
    return subprocess.run(
        ["osascript", "-e", 'tell application "System Events" to get name of first application process whose frontmost is true'],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def focused_element(pid: int) -> str:
    app = AXUIElementCreateApplication(pid)
    err, el = AXUIElementCopyAttributeValue(app, kAXFocusedUIElementAttribute, None)
    if err != 0 or el is None:
        return f"(no focused element, err {err})"
    parts = []
    for attr in (kAXRoleAttribute, kAXSubroleAttribute, kAXTitleAttribute):
        e, v = AXUIElementCopyAttributeValue(el, attr, None)
        if e == 0 and v:
            parts.append(str(v))
    return " / ".join(parts) or "(untitled element)"


def main() -> int:
    before_app = NSWorkspace.sharedWorkspace().frontmostApplication()
    before = frontmost()
    a = MacAutomation()
    sw, sh = a.screen_size()
    x, y = sw / 2, sh / 2
    win = a.window_at(x, y)
    print(f"window at centre: {win.app!r} {win.title!r} pid={win.pid}" if win else "no window at centre")
    t0 = time.perf_counter()
    ok = a.focus_at(x, y)
    ms = (time.perf_counter() - t0) * 1000
    time.sleep(0.4)
    after = frontmost()
    print(f"focus_at ok={ok} in {ms:.1f} ms; frontmost {before!r} -> {after!r}")
    if win:
        print("focused element:", focused_element(win.pid))
    if before_app is not None:
        before_app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
        time.sleep(0.4)
        print("restored frontmost:", frontmost())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

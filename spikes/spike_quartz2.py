"""Spike 2b: Cmd+Tab variants. The app switcher wants a real Command key-down event,
not just a flag on the Tab event. Try the explicit sequence on two taps."""

import sys
import time

from AppKit import NSWorkspace
from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventPost,
    CGEventSetFlags,
    kCGEventFlagMaskCommand,
    kCGHIDEventTap,
    kCGSessionEventTap,
)

KEY_TAB = 0x30
KEY_CMD = 0x37


def frontmost() -> str:
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return app.localizedName() if app else "?"


def post(tap, keycode, down, flags=0):
    ev = CGEventCreateKeyboardEvent(None, keycode, down)
    CGEventSetFlags(ev, flags)
    CGEventPost(tap, ev)


def cmd_tab(tap) -> None:
    post(tap, KEY_CMD, True, kCGEventFlagMaskCommand)
    time.sleep(0.05)
    post(tap, KEY_TAB, True, kCGEventFlagMaskCommand)
    time.sleep(0.05)
    post(tap, KEY_TAB, False, kCGEventFlagMaskCommand)
    time.sleep(0.15)
    post(tap, KEY_CMD, False, 0)


def main() -> int:
    for name, tap in (("HID", kCGHIDEventTap), ("Session", kCGSessionEventTap)):
        before = frontmost()
        cmd_tab(tap)
        time.sleep(0.8)
        after = frontmost()
        print(f"{name} tap: {before} -> {after}  {'PASS' if after != before else 'FAIL'}")
        if after != before:
            return 0
        time.sleep(0.5)
    return 3


if __name__ == "__main__":
    sys.exit(main())

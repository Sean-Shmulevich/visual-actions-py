"""Spike 2: can this process post Cmd+Tab through Quartz?

Checks the Accessibility permission, then posts Cmd+Tab. If it works, the frontmost
app changes. Pass --dry to only check the permission.
"""

import sys
import time

from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventPost,
    CGEventSetFlags,
    kCGEventFlagMaskCommand,
    kCGHIDEventTap,
)
from AppKit import NSWorkspace

KEY_TAB = 0x30


def frontmost() -> str:
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return app.localizedName() if app else "?"


def press(keycode: int, flags: int) -> None:
    down = CGEventCreateKeyboardEvent(None, keycode, True)
    up = CGEventCreateKeyboardEvent(None, keycode, False)
    CGEventSetFlags(down, flags)
    CGEventSetFlags(up, flags)
    CGEventPost(kCGHIDEventTap, down)
    time.sleep(0.02)
    CGEventPost(kCGHIDEventTap, up)


def main() -> int:
    trusted = AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
    print(f"accessibility trusted: {trusted}")
    if not trusted:
        print("Grant Accessibility to the terminal app in System Settings > Privacy & Security, then rerun.")
        return 2
    if "--dry" in sys.argv:
        return 0
    before = frontmost()
    print(f"frontmost before: {before}")
    press(KEY_TAB, kCGEventFlagMaskCommand)
    time.sleep(0.6)
    after = frontmost()
    print(f"frontmost after:  {after}")
    ok = after != before
    print("VERDICT:", "PASS" if ok else "FAIL (front app did not change)")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())

"""macOS driver: Quartz events for keys and scroll, osascript/python/sh for native scripts.

Spike-verified: the app switcher needs real modifier key-down/up events around the
main key, not just flags on the main key. `press` always does that.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from ...core.automation import MediaVerb, NativeResult, Rect, WindowInfo, WindowRef
from ...core.chords import parse_chord
from .windows import MacWindows

MOD_KEYCODES = {"cmd": 0x37, "shift": 0x38, "alt": 0x3A, "ctrl": 0x3B, "fn": 0x3F}
KEYCODES = {
    "a": 0x00, "s": 0x01, "d": 0x02, "f": 0x03, "h": 0x04, "g": 0x05, "z": 0x06, "x": 0x07,
    "c": 0x08, "v": 0x09, "b": 0x0B, "q": 0x0C, "w": 0x0D, "e": 0x0E, "r": 0x0F, "y": 0x10,
    "t": 0x11, "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15, "6": 0x16, "5": 0x17, "9": 0x19,
    "7": 0x1A, "8": 0x1C, "0": 0x1D, "o": 0x1F, "u": 0x20, "i": 0x22, "p": 0x23, "l": 0x25,
    "j": 0x26, "k": 0x28, "n": 0x2D, "m": 0x2E,
    "return": 0x24, "enter": 0x24, "tab": 0x30, "space": 0x31, "delete": 0x33, "backspace": 0x33,
    "esc": 0x35, "escape": 0x35, "left": 0x7B, "right": 0x7C, "down": 0x7D, "up": 0x7E,
    "f1": 0x7A, "f2": 0x78, "f3": 0x63, "f4": 0x76, "f5": 0x60, "f6": 0x61, "f7": 0x62,
    "f8": 0x64, "f9": 0x65, "f10": 0x6D, "f11": 0x67, "f12": 0x6F,
    "[": 0x21, "]": 0x1E, "-": 0x1B, "=": 0x18, ";": 0x29, "'": 0x27, ",": 0x2B, ".": 0x2F, "/": 0x2C, "`": 0x32,
}


class MacAutomation:
    def __init__(self) -> None:
        import Quartz

        self._q = Quartz
        self.windows = MacWindows()
        self._flag = {
            "cmd": Quartz.kCGEventFlagMaskCommand,
            "shift": Quartz.kCGEventFlagMaskShift,
            "alt": Quartz.kCGEventFlagMaskAlternate,
            "ctrl": Quartz.kCGEventFlagMaskControl,
            "fn": Quartz.kCGEventFlagMaskSecondaryFn,
        }

    def _post(self, keycode: int, down: bool, flags: int) -> None:
        ev = self._q.CGEventCreateKeyboardEvent(None, keycode, down)
        self._q.CGEventSetFlags(ev, flags)
        self._q.CGEventPost(self._q.kCGHIDEventTap, ev)

    def press(self, chord: str) -> None:
        c = parse_chord(chord)
        if c.key not in KEYCODES:
            raise ValueError(f"no macOS keycode for {c.key!r}")
        flags = 0
        for m in c.modifiers:
            flags |= self._flag[m]
        for m in c.modifiers:
            self._post(MOD_KEYCODES[m], True, flags)
            time.sleep(0.03)
        self._post(KEYCODES[c.key], True, flags)
        time.sleep(0.03)
        self._post(KEYCODES[c.key], False, flags)
        time.sleep(0.12 if c.modifiers else 0.0)
        for m in reversed(c.modifiers):
            self._post(MOD_KEYCODES[m], False, 0)

    def scroll(self, dx: int, dy: int) -> None:
        ev = self._q.CGEventCreateScrollWheelEvent(None, self._q.kCGScrollEventUnitLine, 2, dy, dx)
        self._q.CGEventPost(self._q.kCGHIDEventTap, ev)

    def set_window_frame(self, target: WindowRef, frame: Rect) -> None:
        raise NotImplementedError("AX window management arrives in a later milestone")

    def media(self, verb: MediaVerb) -> None:
        script = {
            MediaVerb.PLAY_PAUSE: 'tell application "Music" to playpause',
            MediaVerb.NEXT: 'tell application "Music" to next track',
            MediaVerb.PREV: 'tell application "Music" to previous track',
            MediaVerb.VOLUME_UP: "set volume output volume ((output volume of (get volume settings)) + 10)",
            MediaVerb.VOLUME_DOWN: "set volume output volume ((output volume of (get volume settings)) - 10)",
        }[verb]
        subprocess.run(["osascript", "-e", script], check=True, timeout=5, capture_output=True)

    def run_native(self, script_path: Path, timeout_s: float) -> NativeResult:
        ext = script_path.suffix.lower()
        if ext in (".applescript", ".scpt"):
            cmd = ["osascript", str(script_path)]
        elif ext == ".py":
            cmd = [sys.executable, str(script_path)]
        elif ext == ".sh":
            cmd = ["sh", str(script_path)]
        else:
            return NativeResult(ok=False, stderr=f"unsupported script type {ext}")
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
        except subprocess.TimeoutExpired:
            return NativeResult(ok=False, stderr=f"timed out after {timeout_s}s")
        return NativeResult(ok=p.returncode == 0, stdout=p.stdout.strip(), stderr=p.stderr.strip())

    # -- windows -------------------------------------------------------------

    def list_windows(self) -> list[WindowInfo]:
        return self.windows.list_windows()

    def window_at(self, x: float, y: float) -> WindowInfo | None:
        return self.windows.window_at(x, y)

    def move_window(self, win: WindowInfo, x: float, y: float) -> bool:
        return self.windows.move_window(win, x, y)

    def screen_size(self) -> tuple[int, int]:
        return self.windows.screen_size()

    def resize_window(self, win: WindowInfo, w: float, h: float) -> bool:
        return self.windows.resize_window(win, w, h)

    def visible_frame(self) -> Rect:
        return self.windows.visible_frame()

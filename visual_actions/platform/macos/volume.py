"""System output volume via CoreAudio (ctypes, no extra dependency).

Reads the default output device's virtual main volume ('vmvc', the value the menu-bar
slider and the volume keys move) and its mute flag. The default device is looked up
on every read, so switching to headphones or AirPlay is followed. Verified 2026-10-08:
'vmvc' read 0.1875 while `get volume settings` reported output volume 19.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import struct

_SYSTEM_OBJECT = 1


def _fourcc(s: str) -> int:
    return int(struct.unpack(">I", s.encode())[0])


class _Address(ctypes.Structure):
    _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("element", ctypes.c_uint32)]


class CoreAudioVolume:
    def __init__(self) -> None:
        path = ctypes.util.find_library("CoreAudio")
        if path is None:
            raise OSError("CoreAudio framework not found")
        self._ca = ctypes.CDLL(path)
        f = self._ca.AudioObjectGetPropertyData
        f.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(_Address),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]
        f.restype = ctypes.c_int32

    def _get(self, obj: int, selector: str, scope: str, ctype: type) -> int | float | None:
        addr = _Address(_fourcc(selector), _fourcc(scope), 0)
        value = ctype()
        size = ctypes.c_uint32(ctypes.sizeof(value))
        err = self._ca.AudioObjectGetPropertyData(obj, ctypes.byref(addr), 0, None, ctypes.byref(size), ctypes.byref(value))
        return None if err != 0 else value.value

    def read(self) -> tuple[float, bool] | None:
        """(level 0..1, muted) for the default output device, or None if it has no volume."""
        device = self._get(_SYSTEM_OBJECT, "dOut", "glob", ctypes.c_uint32)
        if not device:
            return None
        level = self._get(int(device), "vmvc", "outp", ctypes.c_float)
        if level is None:
            return None
        muted = self._get(int(device), "mute", "outp", ctypes.c_uint32)
        return max(0.0, min(1.0, float(level))), bool(muted)

"""A camera that yields black frames at ~30 fps. Keeps the live loop runnable with no device."""

from __future__ import annotations

import time
from typing import Any


class SyntheticCamera:
    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width, self.height = width, height
        self._open = False

    def open(self) -> None:
        self._open = True

    def read(self) -> tuple[int, Any] | None:
        import numpy as np

        if not self._open:
            return None
        time.sleep(1 / 30)
        return time.monotonic_ns(), np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def close(self) -> None:
        self._open = False

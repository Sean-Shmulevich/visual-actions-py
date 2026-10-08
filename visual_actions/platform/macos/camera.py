from __future__ import annotations

import sys
import time
from typing import Any

from ...core.config import CameraConfig


class OpenCVCamera:
    """OpenCV VideoCapture: AVFoundation on macOS, Media Foundation on Windows."""

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self._cap: Any = None

    def open(self) -> None:
        import cv2

        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.cfg.index, backend)
        if not self._cap.isOpened():
            raise RuntimeError(f"camera {self.cfg.index} failed to open (permission denied or no device)")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)

    def read(self) -> tuple[int, Any] | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok:
            return None
        return time.monotonic_ns(), frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

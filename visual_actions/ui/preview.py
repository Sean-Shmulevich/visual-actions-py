"""Debug preview: the live camera feed with the hand skeleton drawn on it.

The capture thread drops annotated frames into `latest`; the main thread shows them
(cv2 windows must be driven from the main thread on macOS). Recording is untouched:
the session sink receives the raw frame before any drawing.
"""

from __future__ import annotations

import threading
from typing import Any

from ..core.types import HandFrame

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


class DebugPreview:
    def __init__(self, title: str = "visual-actions camera") -> None:
        self.title = title
        self._lock = threading.Lock()
        self.latest: Any = None
        self._opened = False
        self.label = ""

    def submit(self, frame_bgr: Any, hands: list[HandFrame], note: str = "") -> None:
        """Capture thread: draw and hand over a copy."""
        import cv2

        img = cv2.flip(frame_bgr.copy(), 1)  # selfie view, like the recorder preview
        h, w = img.shape[:2]
        for hf in hands:
            pts = [(int((1 - lm.x) * w), int(lm.y * h)) for lm in hf.landmarks]
            for a, b in CONNECTIONS:
                cv2.line(img, pts[a], pts[b], (0, 255, 0), 2)
            for p in pts:
                cv2.circle(img, p, 3, (0, 0, 255), -1)
        text = f"{self.label}  {note}".strip()
        if text:
            cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        with self._lock:
            self.latest = img

    def show(self) -> None:
        """Main thread, on each tick."""
        import cv2

        with self._lock:
            img = self.latest
            self.latest = None
        if img is None:
            return
        if not self._opened:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            self._opened = True
        cv2.imshow(self.title, img)
        cv2.waitKey(1)

    def close(self) -> None:
        if self._opened:
            import cv2

            cv2.destroyWindow(self.title)
            cv2.waitKey(1)
            self._opened = False

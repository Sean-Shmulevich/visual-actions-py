"""Face boxes for the face-touch veto. Runs MediaPipe's BlazeFace (short range) every
few gated frames on the capture thread; ~1-2 ms per call on CPU. If the model file is
missing, detection is off and every hand reports zero overlap."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..core.types import HandFrame

log = logging.getLogger(__name__)


def hand_face_overlap(hf: HandFrame, faces: list[tuple[float, float, float, float]]) -> float:
    """Fraction of the hand's bounding box (raw frame, 0..1) that lies inside the nearest face box."""
    if not faces:
        return 0.0
    xs = [lm.x for lm in hf.landmarks]
    ys = [lm.y for lm in hf.landmarks]
    hx0, hy0, hx1, hy1 = min(xs), min(ys), max(xs), max(ys)
    area = max((hx1 - hx0) * (hy1 - hy0), 1e-9)
    best = 0.0
    for fx0, fy0, fx1, fy1 in faces:
        ix = max(0.0, min(hx1, fx1) - max(hx0, fx0))
        iy = max(0.0, min(hy1, fy1) - max(hy0, fy0))
        best = max(best, ix * iy / area)
    return best


class FaceTracker:
    def __init__(self, model_path: Path | None, every_n: int = 5) -> None:
        self.every_n = every_n
        self._n = 0
        self.faces: list[tuple[float, float, float, float]] = []
        self._det: Any = None
        if model_path is not None and model_path.exists():
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            self._mp = mp
            self._det = vision.FaceDetector.create_from_options(
                vision.FaceDetectorOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=vision.RunningMode.IMAGE,
                    min_detection_confidence=0.5,
                )
            )
        else:
            log.warning("face model not found (%s): face-touch veto disabled", model_path)

    @property
    def enabled(self) -> bool:
        return self._det is not None

    def update(self, frame_bgr: Any) -> list[tuple[float, float, float, float]]:
        """Refresh face boxes every `every_n` calls; returns the current boxes in 0..1 raw-frame coords."""
        if self._det is None:
            return []
        self._n += 1
        if (self._n - 1) % self.every_n != 0:
            return self.faces
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._det.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        self.faces = [
            (d.bounding_box.origin_x / w, d.bounding_box.origin_y / h, (d.bounding_box.origin_x + d.bounding_box.width) / w, (d.bounding_box.origin_y + d.bounding_box.height) / h)
            for d in res.detections
        ]
        return self.faces

    def close(self) -> None:
        if self._det is not None:
            self._det.close()
            self._det = None

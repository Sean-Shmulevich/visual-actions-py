"""Tier 1: frame -> hand landmarks in the raw camera frame."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .types import Hand, HandFrame, Landmark


class HandTracker(Protocol):
    def track(self, frame_bgr: Any, t_ns: int) -> list[HandFrame]: ...

    def close(self) -> None: ...


class MediaPipeTracker:
    """Wraps the MediaPipe Tasks HandLandmarker in VIDEO mode.

    Handedness: MediaPipe assumes a selfie-style input, so on the raw AVFoundation
    frame the label already matches the user's real hand (verified in the spike).
    """

    def __init__(self, model_path: Path, num_hands: int = 1, min_confidence: float = 0.6) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opts)
        self._t0_ns: int | None = None

    def track(self, frame_bgr: Any, t_ns: int) -> list[HandFrame]:
        import cv2

        if self._t0_ns is None:
            self._t0_ns = t_ns
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts_ms = (t_ns - self._t0_ns) // 1_000_000
        res = self._landmarker.detect_for_video(img, ts_ms)
        out: list[HandFrame] = []
        for i, lms in enumerate(res.hand_landmarks):
            cat = res.handedness[i][0] if res.handedness and res.handedness[i] else None
            hand = Hand.LEFT if cat is not None and cat.category_name == "Left" else Hand.RIGHT
            conf = float(cat.score) if cat is not None else 1.0
            out.append(
                HandFrame(
                    t_ns=t_ns,
                    hand=hand,
                    landmarks=tuple(Landmark(lm.x, lm.y, lm.z) for lm in lms),
                    confidence=conf,
                )
            )
        return out

    def close(self) -> None:
        self._landmarker.close()

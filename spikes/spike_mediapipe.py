"""Spike 1: can MediaPipe give usable hand landmarks at 25 fps on this Mac?

Runs the camera for N seconds, reports capture fps, tracker ms, and how many frames
had a hand. No GUI. Hold a hand up to the camera while it runs.
"""

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
MODEL = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


def main() -> int:
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("camera: FAILED to open (permission denied or no device)")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = vision.HandLandmarker.create_from_options(opts)

    frames = 0
    hands = 0
    track_ms: list[float] = []
    t_start = time.perf_counter()
    last_hand = None
    while time.perf_counter() - t_start < SECONDS:
        ok, frame = cap.read()
        if not ok:
            print("camera: read failed")
            break
        frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.perf_counter() - t_start) * 1000)
        t0 = time.perf_counter()
        res = landmarker.detect_for_video(img, ts_ms)
        track_ms.append((time.perf_counter() - t0) * 1000)
        if res.hand_landmarks:
            hands += 1
            lm = res.hand_landmarks[0]
            handed = res.handedness[0][0].category_name if res.handedness else "?"
            last_hand = (handed, lm[8].x, lm[8].y, lm[4].visibility if hasattr(lm[4], "visibility") else None)

    elapsed = time.perf_counter() - t_start
    cap.release()
    track_ms.sort()
    p50 = track_ms[len(track_ms) // 2] if track_ms else 0
    p95 = track_ms[int(len(track_ms) * 0.95)] if track_ms else 0
    print(f"frames={frames} elapsed={elapsed:.1f}s fps={frames / elapsed:.1f}")
    print(f"tracker ms p50={p50:.1f} p95={p95:.1f}")
    print(f"frames with hand={hands} ({100 * hands / max(frames, 1):.0f}%)")
    if last_hand:
        handed, ix, iy, vis = last_hand
        print(f"last hand: handedness={handed} index_tip=({ix:.2f},{iy:.2f}) thumb_visibility={vis}")
    verdict = frames / elapsed >= 25 and p95 < 30
    print("VERDICT:", "PASS" if verdict else "FAIL", "(need fps>=25 and tracker p95<30ms)")
    return 0 if verdict else 2


if __name__ == "__main__":
    sys.exit(main())

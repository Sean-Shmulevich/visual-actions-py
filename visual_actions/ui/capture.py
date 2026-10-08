"""The capture thread: camera -> gate -> tracker -> queue of HandSeen/HandLost."""

from __future__ import annotations

import queue
import threading

from ..core.camera import CameraSource
from ..core.events import HandLost, HandSeen
from ..core.gate import AlwaysOpenGate, HandGate, MotionGate
from ..core.tracker import MediaPipeTracker
from ..paths import model_path


class CaptureThread:
    def __init__(self, camera: CameraSource, q: queue.Queue, use_gate: bool = True) -> None:
        self.camera = camera
        self.q = q
        self.gate: HandGate = MotionGate() if use_gate else AlwaysOpenGate()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self.error: str | None = None
        self.frames = 0
        self.tracked = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        tracker = None
        try:
            tracker = MediaPipeTracker(model_path())
            self.camera.open()
            seen = False
            while not self._stop.is_set():
                r = self.camera.read()
                if r is None:
                    continue
                t_ns, frame = r
                self.frames += 1
                if not self.gate.open(frame, t_ns):
                    if seen:
                        self._put(HandLost(t_ns))
                        seen = False
                    continue
                hands = tracker.track(frame, t_ns)
                self.tracked += 1
                self.gate.notify(t_ns, bool(hands))
                if hands:
                    self._put(HandSeen(hands[0]))
                    seen = True
                elif seen:
                    self._put(HandLost(t_ns))
                    seen = False
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI through .error
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.camera.close()
            if tracker is not None:
                tracker.close()

    def _put(self, ev: object) -> None:
        try:
            self.q.put_nowait(ev)
        except queue.Full:
            pass  # drop rather than queue stale frames

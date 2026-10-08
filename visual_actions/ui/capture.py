"""The capture thread: camera -> gate -> tracker -> queue of HandSeen/HandLost."""

from __future__ import annotations

import queue
import threading

from ..core.camera import CameraSource
from ..core.events import HandLost, HandSeen
from ..core.gate import AlwaysOpenGate, HandGate, MotionGate
from ..core.tracker import MediaPipeTracker
from ..core.types import WRIST, HandFrame
from ..paths import model_path


class CaptureThread:
    # Off-screen rule: MediaPipe keeps reporting a hand for a while after it leaves the frame,
    # with landmarks clamped to the border. A hand whose wrist is outside, or with this many
    # landmarks outside the frame (with a small margin), counts as lost.
    EDGE_MARGIN = 0.02
    EDGE_MAX_OUTSIDE = 6

    @classmethod
    def out_of_frame(cls, hf: HandFrame) -> str | None:
        m = cls.EDGE_MARGIN
        w = hf.landmarks[WRIST]
        if not (-m <= w.x <= 1 + m and -m <= w.y <= 1 + m):
            return f"wrist outside ({w.x:.2f},{w.y:.2f})"
        outside = sum(1 for lm in hf.landmarks if not (-m <= lm.x <= 1 + m and -m <= lm.y <= 1 + m))
        if outside >= cls.EDGE_MAX_OUTSIDE:
            xs = [lm.x for lm in hf.landmarks]
            ys = [lm.y for lm in hf.landmarks]
            return f"{outside} landmarks outside, x {min(xs):.2f}..{max(xs):.2f} y {min(ys):.2f}..{max(ys):.2f}"
        return None

    def __init__(self, camera: CameraSource, q: queue.Queue, use_gate: bool = True, sink: object | None = None) -> None:
        self.camera = camera
        self.q = q
        self.sink = sink  # SessionRecorder-like: write_frame / write_hand / write_lost
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
                if self.sink is not None:
                    self.sink.write_frame(frame, t_ns)
                if not self.gate.open(frame, t_ns):
                    if seen:
                        self._lost(t_ns, "gate", "no motion and no tracked hand for the hold time")
                        seen = False
                    continue
                hands = tracker.track(frame, t_ns)
                self.tracked += 1
                edge = self.out_of_frame(hands[0]) if hands else None
                self.gate.notify(t_ns, bool(hands) and edge is None)
                if hands and edge is None:
                    self._put(HandSeen(hands[0]))
                    if self.sink is not None:
                        self.sink.write_hand(hands[0])
                    seen = True
                elif seen:
                    self._lost(t_ns, "edge" if edge else "tracker", edge or f"tracker reported no hand (frame {self.frames})")
                    seen = False
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI through .error
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.camera.close()
            if tracker is not None:
                tracker.close()

    def _lost(self, t_ns: int, reason: str, detail: str) -> None:
        self._put(HandLost(t_ns, reason, detail))
        if self.sink is not None:
            self.sink.write_lost(t_ns)

    def _put(self, ev: object) -> None:
        try:
            self.q.put_nowait(ev)
        except queue.Full:
            pass  # drop rather than queue stale frames

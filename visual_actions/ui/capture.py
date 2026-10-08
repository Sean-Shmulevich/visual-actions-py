"""The capture thread: camera -> gate -> tracker -> queue of HandSeen/HandLost."""

from __future__ import annotations

import queue
import threading

from ..core.camera import CameraSource
from ..core.events import HandLost, HandSeen
from ..core.gate import AlwaysOpenGate, HandGate, MotionGate
from ..core.presence import PresenceFilter
from ..core.tracker import MediaPipeTracker
from ..paths import face_model_path, model_path
from .face import FaceTracker, hand_face_overlap


class CaptureThread:
    out_of_frame = staticmethod(PresenceFilter.out_of_frame)  # kept for callers/tests

    def __init__(
        self,
        camera: CameraSource,
        q: queue.Queue,
        use_gate: bool = True,
        sink: object | None = None,
        face_veto: bool = True,
        preview: object | None = None,
        presence: PresenceFilter | None = None,
    ) -> None:
        self.face_veto = face_veto
        self.preview = preview  # DebugPreview: annotated frames for the main thread to show
        self.camera = camera
        self.q = q
        self.sink = sink  # SessionRecorder-like: write_frame / write_hand / write_lost
        self.presence = presence or PresenceFilter()
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
        faces = FaceTracker(face_model_path() if self.face_veto else None)
        try:
            tracker = MediaPipeTracker(model_path())
            self.camera.open()
            while not self._stop.is_set():
                r = self.camera.read()
                if r is None:
                    continue
                t_ns, frame = r
                self.frames += 1
                if self.sink is not None:
                    self.sink.write_frame(frame, t_ns)
                if not self.gate.open(frame, t_ns):
                    if self.preview is not None:
                        self.preview.submit(frame, [], "gate closed")
                    pr = self.presence.gate_closed()
                    if pr.lost:
                        self._lost(t_ns, *pr.lost)
                    continue
                hands = tracker.track(frame, t_ns)
                self.tracked += 1
                if self.preview is not None:
                    self.preview.submit(frame, hands, "gate open")
                pr = self.presence.update(hands[0] if hands else None, self.frames)
                self.gate.notify(t_ns, pr.seen is not None)
                if pr.seen is not None:
                    overlap = hand_face_overlap(pr.seen, faces.update(frame)) if faces.enabled else 0.0
                    self._put(HandSeen(pr.seen, overlap))
                    if self.sink is not None:
                        self.sink.write_hand(pr.seen)
                elif pr.lost:
                    self._lost(t_ns, *pr.lost)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI through .error
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.camera.close()
            faces.close()
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

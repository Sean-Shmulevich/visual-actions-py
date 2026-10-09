"""Session recording: camera video, a timestamped event log, and the raw landmarks.

One folder per run under the data dir:
    sessions/<YYYYmmdd-HHMMSS>/video-001.mp4  camera frames, elapsed time + mode burned in,
                                              in segments of `segment_s` so a crash or a hard
                                              quit loses at most the last segment
    sessions/<YYYYmmdd-HHMMSS>/events.log     "elapsed  wall-clock  kind  details", one line each
    sessions/<YYYYmmdd-HHMMSS>/landmarks.jsonl raw HandFrames (same format as datasets/) plus
                                              "face": the hand/face overlap the capture thread saw
    sessions/<YYYYmmdd-HHMMSS>/config.toml    the resolved Config the session ran with
    sessions/<YYYYmmdd-HHMMSS>/meta.json      git hash, app version, host, screen, model path

Frames are written from the capture thread; events from the main thread. The writer
is guarded by a lock. Video is 640x480 mp4v, roughly 1-2 MB per minute. config.toml and
meta.json make a replay reproduce the live run: same keys, same model, same veto input.
"""

from __future__ import annotations

import atexit
import json
import platform
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .core.config import Config, save_config
from .core.drag import DragEvent, DragPhase
from .core.events import (
    ActionFired,
    Bus,
    HandLost,
    HandSeen,
    ModeChanged,
    PalmVetoed,
    SnapPreview,
    TokenEmitted,
)
from .core.recorder import Recorder
from .core.types import HandFrame
from .paths import REPO_ROOT


def git_short_hash(repo: Path = REPO_ROOT) -> str | None:
    """`git rev-parse --short HEAD` of the repo, or None when git or the repo is not there."""
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=2, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None


def app_version() -> str:
    try:
        from importlib.metadata import version

        return version("visual-actions")
    except Exception:  # noqa: BLE001 - not installed as a distribution
        return "unknown"


def session_meta(**extra: Any) -> dict[str, Any]:
    """What a replay needs to know about the run besides the config: every value best-effort,
    so a missing git or model never stops a session from starting."""
    meta: dict[str, Any] = {
        "git": git_short_hash(),
        "version": app_version(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    meta.update(extra)
    return meta


class SessionRecorder:
    def __init__(
        self,
        root: Path,
        bus: Bus | None = None,
        fps: float = 30.0,
        log_tokens: bool = True,
        segment_s: float = 120.0,
        config: Config | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """`config` is snapshotted to <session>/config.toml and `meta` (screen size, model path,
        whatever the caller knows) joins the generic session_meta() in <session>/meta.json."""
        self.segment_s = segment_s
        self._segment = 0
        self._segment_started_ns = 0
        self.dir = root / f"{datetime.now():%Y%m%d-%H%M%S}"  # noqa: DTZ005 - local time for a folder name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.log_tokens = log_tokens
        self.t0 = time.monotonic_ns()
        self.started_wall = datetime.now()  # noqa: DTZ005
        self._lock = threading.Lock()
        self._video: Any = None
        self._size: tuple[int, int] | None = None
        self.frames = 0
        self.mode = "idle"
        self._log = (self.dir / "events.log").open("w", encoding="utf-8", buffering=1)
        self._landmarks = Recorder(self.dir / "landmarks.jsonl")
        self._index = (self.dir / "video.idx").open("w", encoding="utf-8", buffering=1)
        self._segment_frames = 0
        self.log("session", f"started {self.started_wall:%Y-%m-%d %H:%M:%S} t0_ns={self.t0}")  # t0_ns joins events.log to landmarks.jsonl
        self.meta = session_meta(started=self.started_wall.isoformat(timespec="seconds"), **(meta or {}))
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str) + "\n", encoding="utf-8")
        if config is not None:
            save_config(config, self.dir / "config.toml")
        self._closed = False
        atexit.register(self.close)
        if bus is not None:
            bus.subscribe(ModeChanged, self._on_mode)
            bus.subscribe(ActionFired, self._on_action)
            bus.subscribe(DragEvent, self._on_drag)
            bus.subscribe(SnapPreview, self._on_snap)
            bus.subscribe(HandLost, self._on_hand_lost)
            bus.subscribe(HandSeen, self._on_hand_seen)
            bus.subscribe(PalmVetoed, lambda e: self.log("veto", f"palm on face: overlap={e.overlap:.2f} spread={e.spread:.2f}", e.t_ns))
            if log_tokens:
                bus.subscribe(TokenEmitted, self._on_token)
        self._hand_present = False
        self._lost_at_ns: int | None = None

    # -- capture thread -----------------------------------------------------------

    def write_frame(self, frame: Any, t_ns: int) -> None:
        import cv2

        with self._lock:
            if self._video is not None and (t_ns - self._segment_started_ns) / 1e9 >= self.segment_s:
                self._video.release()  # close the segment so its index is written; the next frame opens a new one
                self._video = None
            if self._video is None:
                h, w = frame.shape[:2]
                self._size = (w, h)
                self._segment += 1
                self._segment_started_ns = t_ns
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
                self._video = cv2.VideoWriter(str(self.dir / f"video-{self._segment:03d}.mp4"), fourcc, self.fps, (w, h))
                self._segment_frames = 0
            # frame index: "segment frame t_ns" per written frame, so a clip for an event can be cut at the
            # exact video frame (the writer's fixed fps drifts from the camera's real cadence)
            self._index.write(f"{self._segment} {self._segment_frames} {t_ns}\n")
            self._segment_frames += 1
            img = cv2.flip(frame, 1)  # selfie view, same as the preview
            stamp = f"{self.elapsed_s(t_ns):8.2f}s  {self.mode}"
            cv2.putText(img, stamp, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
            cv2.putText(img, stamp, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
            self._video.write(img)
            self.frames += 1

    def write_hand(self, hf: HandFrame, face_overlap: float | None = None) -> None:
        """`face_overlap` is the hand/face box fraction the capture thread computed for the palm
        veto; recorded as "face" so a replay sees the veto the live run saw."""
        with self._lock:
            self._landmarks.write(hf, {"face": round(face_overlap, 3)} if face_overlap is not None else None)

    def write_lost(self, t_ns: int) -> None:
        with self._lock:
            self._landmarks.write_lost(t_ns)

    # -- main thread ----------------------------------------------------------------

    def elapsed_s(self, t_ns: int | None = None) -> float:
        return ((t_ns if t_ns is not None else time.monotonic_ns()) - self.t0) / 1e9

    def log(self, kind: str, details: str, t_ns: int | None = None) -> None:
        line = f"{self.elapsed_s(t_ns):9.3f}  {datetime.now():%H:%M:%S.%f}  {kind:7s} {details}\n"  # noqa: DTZ005
        with self._lock:
            self._log.write(line)

    def mark_wrong(self, action_fired: bool) -> str:
        """The menu bar's "Last action was wrong": a `human` line the intent labellers read as a
        weight-1.0 misfire vote on the last fire, or on the last arm when nothing fired since it.
        Returns the text written."""
        text = "last action wrong" if action_fired else "last arm wrong"
        self.log("human", text)
        return text

    def _on_mode(self, ev: ModeChanged) -> None:
        self.mode = ev.new
        self.log("mode", f"{ev.old} -> {ev.new}" + (f" [{ev.namespace}]" if ev.namespace else ""), ev.t_ns)

    def _on_token(self, ev: TokenEmitted) -> None:
        t = ev.token
        self.log("token", f"{t.name} conf={t.confidence:.2f} still={t.still}", t.t_ns)

    def _on_action(self, ev: ActionFired) -> None:
        self.log("action", f"{ev.action.name} {'ok' if ev.ok else 'FAILED ' + ev.message}", ev.t_ns)

    def _on_drag(self, ev: DragEvent) -> None:
        if ev.phase is DragPhase.MOVE:
            return
        extra = f" snapped={ev.snapped}" if ev.snapped else ""
        if ev.focus is not None:
            extra += f" focus[{ev.focus}]"
        self.log("drag", f"{ev.phase.value} {ev.window} @({ev.x:.0f},{ev.y:.0f}){extra}", ev.t_ns)

    def _on_hand_lost(self, ev: HandLost) -> None:
        self._hand_present = False
        self._lost_at_ns = ev.t_ns
        self.log("hand", f"lost [{ev.reason}] {ev.detail} (mode {self.mode})", ev.t_ns)

    def _on_hand_seen(self, ev: HandSeen) -> None:
        if not self._hand_present:
            gap = f" after {(ev.hand_frame.t_ns - self._lost_at_ns) / 1e9:.2f}s away" if self._lost_at_ns else ""
            hf = ev.hand_frame
            self.log("hand", f"seen{gap} conf={hf.confidence:.2f} wrist=({hf.landmarks[0].x:.2f},{hf.landmarks[0].y:.2f}) (mode {self.mode})", hf.t_ns)
        self._hand_present = True

    def _on_snap(self, ev: SnapPreview) -> None:
        self.log("snap", f"preview {ev.zone}" if ev.zone else "preview cleared", ev.t_ns)

    def close(self) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                return {"dir": str(self.dir), "frames": self.frames, "seconds": round(self.elapsed_s(), 1), "landmark_frames": self._landmarks.count, "segments": self._segment}
            self._closed = True
            summary = {"dir": str(self.dir), "frames": self.frames, "seconds": round(self.elapsed_s(), 1), "landmark_frames": self._landmarks.count, "segments": self._segment}
            self._log.write(json.dumps({"summary": summary}) + "\n")
            self._log.close()
            self._landmarks.close()
            self._index.close()
            if self._video is not None:
                self._video.release()
                self._video = None
        return summary

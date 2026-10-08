import json
from pathlib import Path

import numpy as np

from visual_actions.core.drag import DragEvent, DragPhase
from visual_actions.core.events import ActionFired, Bus, ModeChanged, TokenEmitted
from visual_actions.core.recorder import read_session
from visual_actions.core.types import Action, ActionKind, Hand, HandFrame, Token
from visual_actions.session import SessionRecorder
from visual_actions.tools.synth import hand_frame


def test_session_writes_video_log_and_landmarks(tmp_path: Path):
    bus = Bus()
    s = SessionRecorder(tmp_path, bus, fps=30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(10):
        s.write_frame(frame, s.t0 + i * 33_000_000)
    s.write_hand(hand_frame("open_palm", s.t0 + 1))
    s.write_lost(s.t0 + 2)
    bus.publish(ModeChanged(s.t0 + 3, "idle", "holding"))
    bus.publish(TokenEmitted(Token(s.t0 + 4, "open_palm", 0.93, Hand.RIGHT, True)))
    bus.publish(ActionFired(s.t0 + 5, Action(ActionKind.KEY, "Cmd+Tab"), True))
    bus.publish(ActionFired(s.t0 + 6, Action(ActionKind.KEY, "Nope"), False, "boom"))
    bus.publish(DragEvent(s.t0 + 7, DragPhase.MOVE, "W", 1, 1))  # not logged
    bus.publish(DragEvent(s.t0 + 8, DragPhase.END, "App: W", 10, 20, snapped="left"))
    summary = s.close()

    assert summary["frames"] == 10 and summary["landmark_frames"] == 1
    assert (s.dir / "video.mp4").stat().st_size > 0
    log = (s.dir / "events.log").read_text().splitlines()
    kinds = [line.split()[2] for line in log[:-1]]
    assert kinds == ["session", "mode", "token", "action", "action", "drag"]
    assert "Cmd+Tab ok" in log[3] and "FAILED boom" in log[4] and "snapped=left" in log[5]
    assert json.loads(log[-1])["summary"]["frames"] == 10
    recs = list(read_session(s.dir / "landmarks.jsonl"))
    assert isinstance(recs[0], HandFrame) and isinstance(recs[1], int)


def test_session_without_bus_only_records_frames(tmp_path: Path):
    s = SessionRecorder(tmp_path)
    s.write_frame(np.zeros((48, 64, 3), dtype=np.uint8), s.t0)
    summary = s.close()
    assert summary["frames"] == 1 and (s.dir / "events.log").exists()

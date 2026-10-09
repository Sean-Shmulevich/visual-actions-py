import json
from pathlib import Path

import numpy as np

from visual_actions.core.config import default_config, load_config
from visual_actions.core.drag import DragEvent, DragPhase
from visual_actions.core.events import ActionFired, Bus, ModeChanged, TokenEmitted
from visual_actions.core.recorder import Recorder, from_json, read_records, read_session, to_json
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

    assert summary["frames"] == 10 and summary["landmark_frames"] == 1 and summary["segments"] == 1
    assert (s.dir / "video-001.mp4").stat().st_size > 0
    log = (s.dir / "events.log").read_text().splitlines()
    kinds = [line.split()[2] for line in log[:-1]]
    assert kinds == ["session", "mode", "token", "action", "action", "drag"]
    assert "Cmd+Tab ok" in log[3] and "FAILED boom" in log[4] and "snapped=left" in log[5]
    assert json.loads(log[-1])["summary"]["frames"] == 10
    recs = list(read_session(s.dir / "landmarks.jsonl"))
    assert isinstance(recs[0], HandFrame) and isinstance(recs[1], int)


def test_video_is_segmented_and_each_segment_is_readable(tmp_path: Path):
    import cv2

    s = SessionRecorder(tmp_path, fps=30, segment_s=0.5)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    for i in range(40):  # 1.33 s at 30 fps -> 3 segments
        s.write_frame(frame, s.t0 + i * 33_000_000)
    s.close()
    segs = sorted(s.dir.glob("video-*.mp4"))
    assert len(segs) == 3
    for seg in segs:
        c = cv2.VideoCapture(str(seg))
        assert c.isOpened() and c.read()[0]


def test_close_is_idempotent(tmp_path: Path):
    s = SessionRecorder(tmp_path)
    a = s.close()
    b = s.close()
    assert a["frames"] == b["frames"] == 0


def test_session_without_bus_only_records_frames(tmp_path: Path):
    s = SessionRecorder(tmp_path)
    s.write_frame(np.zeros((48, 64, 3), dtype=np.uint8), s.t0)
    summary = s.close()
    assert summary["frames"] == 1 and (s.dir / "events.log").exists()


def test_session_snapshots_config_and_meta(tmp_path: Path):
    cfg = default_config()
    cfg.timing.leader_hold_s = 0.77
    cfg.recognizer.model = None
    s = SessionRecorder(tmp_path, config=cfg, meta={"screen": [1440, 900], "model_path": "/tmp/gestures.joblib"})
    s.close()
    back = load_config(s.dir / "config.toml")
    assert back.timing.leader_hold_s == 0.77 and back.leader.profile == cfg.leader.profile
    meta = json.loads((s.dir / "meta.json").read_text())
    assert meta["screen"] == [1440, 900] and meta["model_path"] == "/tmp/gestures.joblib"
    assert {"git", "version", "hostname", "platform", "python", "started"} <= meta.keys()
    assert meta["git"] is None or (3 <= len(meta["git"]) <= 40)  # best-effort: None without git


def test_session_without_config_still_writes_meta(tmp_path: Path):
    s = SessionRecorder(tmp_path)
    s.close()
    assert (s.dir / "meta.json").exists() and not (s.dir / "config.toml").exists()


def test_face_overlap_is_recorded_per_landmark_frame(tmp_path: Path):
    s = SessionRecorder(tmp_path)
    s.write_hand(hand_frame("open_palm", s.t0 + 1), 0.7512)
    s.write_hand(hand_frame("open_palm", s.t0 + 2))  # no face tracker: no key
    s.close()
    lines = [json.loads(line) for line in (s.dir / "landmarks.jsonl").read_text().splitlines()]
    assert lines[0]["face"] == 0.751 and "face" not in lines[1]
    recs = list(read_records(s.dir / "landmarks.jsonl"))
    assert isinstance(recs[0][0], HandFrame) and recs[0][1] == {"face": 0.751} and recs[1][1] == {}
    assert len(list(read_session(s.dir / "landmarks.jsonl"))) == 2  # the plain reader ignores it


def test_recorder_round_trips_extra_keys(tmp_path: Path):
    hf = hand_frame("fist", 123)
    d = to_json(hf, {"face": 0.5})
    assert d["face"] == 0.5 and from_json(d) == hf
    rec = Recorder(tmp_path / "x.jsonl")
    rec.write(hf, {"face": 0.25})
    rec.write(hf)
    rec.write_lost(456)
    rec.close()
    out = list(read_records(tmp_path / "x.jsonl"))
    assert [e for _, e in out] == [{"face": 0.25}, {}, {}]
    assert out[2][0] == 456

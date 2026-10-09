"""Whole pipeline on recorded sessions with mock automation and fake time."""

from pathlib import Path

import pytest

from visual_actions.core.config import default_config
from visual_actions.core.events import Bus, HandSeen, PalmVetoed
from visual_actions.core.recorder import Recorder
from visual_actions.tools.replay import replay, replay_full
from visual_actions.tools.synth import hand_frame, write_session

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    out = tmp_path_factory.mktemp("sessions")
    write_session(out / "h_left.jsonl", [("open_palm", 2.5), ("h_left", 0.8), ("lost", 0.5)])
    write_session(out / "h_right.jsonl", [("open_palm", 2.5), ("h_right", 0.8), ("lost", 0.5)])
    write_session(out / "timeout.jsonl", [("open_palm", 2.5), ("none", 6.0), ("lost", 0.5)])
    write_session(out / "no_leader.jsonl", [("h_left", 2.0), ("lost", 0.5)])
    write_session(out / "escape_fist.jsonl", [("open_palm", 2.5), ("fist", 1.5), ("h_left", 0.8), ("lost", 0.5)])
    return out


def test_committed_fixture_fires_cmd_tab():
    fired = replay(FIXTURES / "h_left.jsonl", default_config())
    assert [f.action.name for f in fired] == ["Cmd+Tab"]
    assert fired[0].ok


def test_h_left_fires_cmd_tab(fixtures):
    fired = replay(fixtures / "h_left.jsonl")
    assert [f.action.name for f in fired] == ["Cmd+Tab"]


def test_h_right_fires_cmd_shift_tab(fixtures):
    fired = replay(fixtures / "h_right.jsonl")
    assert [f.action.name for f in fired] == ["Cmd+Shift+Tab"]


def test_timeout_fires_nothing(fixtures):
    assert replay(fixtures / "timeout.jsonl") == []


def test_no_leader_fires_nothing(fixtures):
    assert replay(fixtures / "no_leader.jsonl") == []


def test_fist_escape_cancels_the_window(fixtures):
    assert replay(fixtures / "escape_fist.jsonl") == []


def test_even_a_bound_fist_cancels(fixtures):
    cfg = default_config()
    cfg.namespaces["window"].bindings.append({"gesture": "fist", "action": {"kind": "key", "name": "Fist", "chord": "ctrl+down"}})
    assert replay(fixtures / "escape_fist.jsonl", cfg) == []


def test_replay_publishes_the_recorded_face_overlap(tmp_path: Path):
    from .test_face_veto import bunched_palm

    rec = Recorder(tmp_path / "face.jsonl")
    for i in range(60):  # a hand resting on a cheek: bunched fingers, 75 % inside the face box
        rec.write(bunched_palm(i * 33_000_000, mirror_to_raw=True), {"face": 0.75})
    rec.write(hand_frame("open_palm", 60 * 33_000_000))  # a plain dataset line: overlap defaults to 0
    rec.close()
    seen, vetoes = [], []
    bus = Bus()
    bus.subscribe(HandSeen, seen.append)
    bus.subscribe(PalmVetoed, vetoes.append)
    cfg = default_config()
    cfg.recognizer.model = None
    r = replay_full(tmp_path / "face.jsonl", cfg, bus=bus)
    assert [s.face_overlap for s in seen[:2]] == [0.75, 0.75] and seen[-1].face_overlap == 0.0
    assert vetoes and vetoes[0].overlap == 0.75
    assert not any(m.new == "holding" for m in r.modes)

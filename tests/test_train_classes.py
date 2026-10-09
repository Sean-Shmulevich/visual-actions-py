"""datasets/ may hold scratch folders; only gesture classes train."""

from pathlib import Path

import pytest

from visual_actions.tools.synth import write_session
from visual_actions.tools.train import CLASSES, class_dirs, load


def test_unknown_directories_are_ignored_with_a_warning(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    for name in ("open_palm", "fist", "_ideas", "three_up"):
        write_session(tmp_path / name / "take.jsonl", [("open_palm" if name != "fist" else "fist", 0.2)])
    assert [p.name for p in class_dirs(tmp_path)] == ["fist", "open_palm"]
    assert "_ideas, three_up" in capsys.readouterr().err
    _, y, _ = load(tmp_path)
    assert set(y) == {"fist", "open_palm"}


def test_class_list_covers_the_recognizer_tokens():
    assert {"none", "open_palm", "fist", "h_left", "h_right", "point_up", "two_up", "middle_up", "thumbs_up", "thumbs_down", "pinch"} == set(CLASSES)

"""Class mapping, image import with a fake tracker, and the HaGRID annotation path.
No network, no MediaPipe."""

from __future__ import annotations

from pathlib import Path

from visual_actions.core.recorder import read_session
from visual_actions.core.types import N_LANDMARKS, Hand, HandFrame, Landmark
from visual_actions.tools.import_public import (
    hagrid_record_to_frames,
    import_hagrid_annotations,
    import_images,
    map_class,
)


def _hf(x: float = 0.5, conf: float = 0.9, hand: Hand = Hand.RIGHT) -> HandFrame:
    return HandFrame(0, hand, tuple(Landmark(x, 0.5, 0.0) for _ in range(N_LANDMARKS)), conf)


class FakeTracker:
    """Returns hands based on the image path name: 'two' -> 2 hands, 'none' -> 0,
    'weak' -> one hand at 0.3 confidence, else one good hand."""

    def __init__(self) -> None:
        self.calls = 0

    def track(self, frame, t_ns):
        self.calls += 1
        name = str(frame)
        if "two" in name:
            return [_hf(), _hf()]
        if "nohand" in name:
            return []
        if "weak" in name:
            return [_hf(conf=0.3)]
        return [_hf()]

    def close(self) -> None:
        pass


def test_map_class_known_and_unknown():
    assert map_class("ok") == "pinch"
    assert map_class("Palm") == "open_palm"
    assert map_class("stop") == "open_palm"
    assert map_class("one") == "point_up"
    assert map_class("peace") == "two_up"
    assert map_class("like") == "none"
    assert map_class("something_new") == "none"


def _make_images(root: Path, layout: dict[str, list[str]]) -> None:
    for cls, names in layout.items():
        d = root / cls
        d.mkdir(parents=True)
        for n in names:
            (d / f"{n}.jpg").write_bytes(b"x")


def test_import_images_filters_and_writes_per_class(tmp_path: Path):
    root = tmp_path / "ds"
    _make_images(root, {"ok": ["a", "b", "two_c", "nohand_d", "weak_e"], "palm": ["a", "b", "c"], "like": ["a"]})
    out = tmp_path / "datasets"
    tracker = FakeTracker()
    report = import_images(root, "fake", out, tracker, reader=lambda p: p)
    assert tracker.calls == 9
    assert report["counts"]["pinch"] == 2
    assert report["counts"]["open_palm"] == 3
    assert report["counts"]["none"] == 1
    assert report["counts"]["no_pinch"] == 2  # matched to the pinch count
    assert report["skipped"] == {"multi_hand": 1, "no_hand": 1, "low_confidence": 1}
    pinch = list(read_session(out / "pinch" / "public-fake.jsonl"))
    assert len(pinch) == 2 and all(isinstance(r, HandFrame) for r in pinch)
    assert (out / "no_pinch" / "public-fake.jsonl").exists()
    assert (out / "open_palm" / "public-fake.jsonl").exists()


def test_import_images_per_class_cap(tmp_path: Path):
    root = tmp_path / "ds"
    _make_images(root, {"fist": ["a", "b", "c", "d"]})
    report = import_images(root, "cap", tmp_path / "datasets", FakeTracker(), per_class=2, reader=lambda p: p)
    assert report["counts"] == {"fist": 2}


def _hagrid_record(label: str, n_hands: int = 1, n_points: int = N_LANDMARKS) -> dict:
    return {
        "bboxes": [[0.1, 0.1, 0.2, 0.2]] * n_hands,
        "labels": [label] * n_hands,
        "hand_landmarks": [[[0.1 * i / 21, 0.2] for i in range(n_points)] for _ in range(n_hands)],
        "user_id": "u",
    }


def test_hagrid_record_to_frames_handles_missing_landmarks():
    frames = hagrid_record_to_frames("k", _hagrid_record("ok"), 7)
    assert len(frames) == 1 and frames[0][0] == "ok" and frames[0][1].t_ns == 7
    assert frames[0][1].landmarks[0].z == 0.0
    assert hagrid_record_to_frames("k", _hagrid_record("ok", n_points=5), 0) == []
    assert hagrid_record_to_frames("k", {"labels": ["ok"], "hand_landmarks": [[]]}, 0) == []


def test_import_hagrid_annotations_caps_and_pairs_pinch(tmp_path: Path):
    data = {f"ok{i}": _hagrid_record("ok") for i in range(10)}
    data.update({f"palm{i}": _hagrid_record("palm") for i in range(4)})
    data.update({f"like{i}": _hagrid_record("like") for i in range(4)})
    data["two"] = _hagrid_record("ok", n_hands=2)
    report = import_hagrid_annotations([("ok.json", data)], "test", tmp_path / "datasets", per_class=5)
    assert report["counts"]["pinch"] == 5
    assert report["counts"]["open_palm"] == 4
    assert report["counts"]["none"] == 4
    assert report["counts"]["no_pinch"] == 5
    assert report["seen"]["pinch"] == 10
    assert report["skipped"] == {"multi_hand": 1}
    assert (tmp_path / "datasets" / "pinch" / "public-test.jsonl").exists()

"""Labelled moments -> datasets/, the intent dataset, and labelling-function accuracy."""

import io
import json
from pathlib import Path

from visual_actions.core.recorder import read_session
from visual_actions.core.types import HandFrame
from visual_actions.intent.export import export_labels, labelfn_accuracy
from visual_actions.intent.schema import HumanLabel, Intent, Motion, Segment, SegmentKind, Tag, Verdict, write_jsonl
from visual_actions.tools.train import CLASSES

T0_NS = 5_000_000_000
FPS = 10


def landmark_line(t_ns: int) -> str:
    return json.dumps({"t_ns": t_ns, "hand": "right", "confidence": 0.99, "landmarks": [[0.5, 0.5, 0.0]] * 21})


def write_session(tmp_path: Path, segments: list[Segment], human: list[HumanLabel] = (), tags: list[Tag] = (), seconds: float = 60.0) -> Path:
    d = tmp_path / "20261009-100000"
    d.mkdir()
    (d / "events.log").write_text(f"    0.000  10:00:00.000000  session started 2026-10-09 10:00:00 t0_ns={T0_NS}\n   {seconds:.3f}  10:01:00.000000  hand    lost [edge] (mode idle)\n")
    lines = [landmark_line(T0_NS + int(k * 1e9 / FPS)) for k in range(int(seconds * FPS))]
    lines.insert(5, json.dumps({"lost": T0_NS + 1}))  # a lost marker the exporter must skip
    (d / "landmarks.jsonl").write_text("\n".join(lines) + "\n")
    write_jsonl(d / "intent" / "segments.jsonl", segments)
    if human:
        write_jsonl(d / "intent" / "human.jsonl", human)
    if tags:
        write_jsonl(d / "intent" / "tags.jsonl", tags)
    return d


def seg(i: int, kind: SegmentKind, t0: float, t1: float, gesture: str | None = "h_left", votes: dict[str, str] | None = None) -> Segment:
    return Segment(f"s/{kind.value}/{i:04d}", "s", kind, t0, t1, engine_gesture=gesture, labelfn_votes=votes or {}, weak_label=None)


def human(s: Segment, verdict: Verdict, gesture: str | None = None, intent: Intent = Intent.COMMAND) -> HumanLabel:
    return HumanLabel(s.segment_id, verdict, intent, true_gesture=gesture, motion=Motion.STILL, at="2026-10-09T10:05:00+00:00")


def frames_of(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_export_writes_classes_sources_weights_and_the_fire_window(tmp_path: Path):
    s_fire = seg(0, SegmentKind.FIRE, 8.5, 11.5)  # engine event at 10.0 s
    s_mis = seg(1, SegmentKind.FIRE, 18.5, 21.5, gesture="h_right")
    s_hand = seg(2, SegmentKind.HAND, 30.0, 34.0, gesture=None)
    s_missed = seg(3, SegmentKind.HAND, 40.0, 42.0, gesture=None)
    s_amb = seg(4, SegmentKind.FIRE, 48.5, 51.5)
    s_fist = seg(5, SegmentKind.FIRE, 52.5, 55.5, gesture="fist")
    labels = [
        human(s_fire, Verdict.INTENDED),  # class = engine gesture h_left
        human(s_mis, Verdict.MISFIRE, intent=Intent.INCIDENTAL),  # class none
        human(s_hand, Verdict.NO_EVENT, intent=Intent.INCIDENTAL),  # class none, whole span
        human(s_missed, Verdict.MISSED, gesture="two_up"),  # class two_up, whole span
        human(s_amb, Verdict.AMBIGUOUS, intent=Intent.UNSURE),  # nothing
        human(s_fist, Verdict.INTENDED, gesture="fist"),  # a human may label fist
    ]
    d = write_session(tmp_path, [s_fire, s_mis, s_hand, s_missed, s_amb, s_fist], labels)
    datasets = tmp_path / "datasets"
    summary = export_labels(d, datasets, stamp="20261009-120000")

    assert set(summary.files) == {"h_left", "none", "two_up", "fist"} and set(summary.files) <= CLASSES
    assert summary.files["h_left"] == datasets / "h_left" / "review-20261009-100000-20261009-120000.jsonl"
    hl = frames_of(summary.files["h_left"])
    # ±0.5 s around the fire at 10.0 s: 10 fps -> 11 frames (9.5 .. 10.5 inclusive), not the 3 s span
    assert len(hl) == 11 and all(f["source"] == "human" and f["weight"] == 1.0 for f in hl)
    assert all(9.5 <= (f["t_ns"] - T0_NS) / 1e9 <= 10.5 for f in hl)
    none = frames_of(summary.files["none"])
    assert len(none) == 11 + 41  # misfire window + whole 4 s hand span
    assert len(frames_of(summary.files["two_up"])) == 21 and len(frames_of(summary.files["fist"])) == 11
    assert any("ambiguous" in why for why in summary.skipped)
    # the trainer's loader reads the extra keys away
    frames = list(read_session(summary.files["h_left"]))
    assert len(frames) == 11 and all(isinstance(f, HandFrame) for f in frames)

    rows = [json.loads(line) for line in summary.intent_file.read_text().splitlines()]
    assert [r["segment_id"] for r in rows] == [s.segment_id for s in (s_fire, s_mis, s_hand, s_missed, s_amb, s_fist)]
    r0 = rows[0]
    assert r0 == {"segment_id": "s/fire/0000", "kind": "fire", "t0": 8.5, "t1": 11.5, "intent": "command", "motion": "still", "verdict": "intended", "source": "human", "weight": 1.0, "frame_count": 11}
    assert rows[4]["verdict"] == "ambiguous" and rows[4]["intent"] == "unsure"
    assert summary.intent_file == datasets / "intent" / "20261009-100000.jsonl"


def test_judge_labels_are_weighted_and_never_fist(tmp_path: Path):
    s_judge = seg(0, SegmentKind.FIRE, 8.5, 11.5, gesture="open_palm")
    s_judge_fist = seg(1, SegmentKind.FIRE, 18.5, 21.5, gesture="fist")
    s_judge_unsure = seg(2, SegmentKind.FIRE, 28.5, 31.5)
    s_judge_low = seg(3, SegmentKind.FIRE, 38.5, 41.5)
    s_both = seg(4, SegmentKind.FIRE, 48.5, 51.5)
    tags = [
        Tag(s_judge.segment_id, Verdict.INTENDED, Intent.COMMAND, Motion.STILL, "open_palm", confidence=0.9, needs_human=False),
        Tag(s_judge_fist.segment_id, Verdict.INTENDED, Intent.COMMAND, Motion.STILL, "fist", confidence=0.95, needs_human=False),
        Tag(s_judge_unsure.segment_id, Verdict.MISFIRE, Intent.INCIDENTAL, Motion.STILL, None, confidence=0.9, needs_human=True),
        Tag(s_judge_low.segment_id, Verdict.MISFIRE, Intent.INCIDENTAL, Motion.STILL, None, confidence=0.4, needs_human=False),
        Tag(s_both.segment_id, Verdict.MISFIRE, Intent.INCIDENTAL, Motion.STILL, None, confidence=0.9, needs_human=False),
    ]
    d = write_session(tmp_path, [s_judge, s_judge_fist, s_judge_unsure, s_judge_low, s_both], [human(s_both, Verdict.INTENDED)], tags)
    datasets = tmp_path / "datasets"

    only_human = export_labels(d, datasets, stamp="a")
    assert set(only_human.files) == {"h_left"} and only_human.segments == 1

    both = export_labels(d, datasets, stamp="b", include_judge=True)
    assert set(both.files) == {"h_left", "open_palm"} and "fist" not in both.files
    assert not (datasets / "fist").exists()
    assert any("fist from a judge" in why for why in both.skipped)
    op = frames_of(both.files["open_palm"])
    assert all(f["source"] == "judge" and f["weight"] == 0.9 for f in op)
    hl = frames_of(both.files["h_left"])
    assert all(f["source"] == "human" and f["weight"] == 1.0 for f in hl)  # the human label wins over the judge's misfire
    rows = {json.loads(l)["segment_id"]: json.loads(l) for l in both.intent_file.read_text().splitlines()}
    assert set(rows) == {s_judge.segment_id, s_judge_fist.segment_id, s_both.segment_id}  # unsure and low-confidence tags stay out
    assert rows[s_judge.segment_id]["source"] == "judge" and rows[s_judge.segment_id]["weight"] == 0.9
    assert rows[s_both.segment_id]["source"] == "human"


def test_export_without_landmarks_writes_no_frames_but_says_why(tmp_path: Path):
    s = seg(0, SegmentKind.FIRE, 8.5, 11.5)
    d = tmp_path / "s"
    write_jsonl(d / "intent" / "segments.jsonl", [s])
    write_jsonl(d / "intent" / "human.jsonl", [human(s, Verdict.INTENDED)])
    summary = export_labels(d, tmp_path / "datasets", stamp="x")
    assert summary.files == {} and summary.segments == 1
    assert any("no t0_ns" in why for why in summary.skipped) and summary.intent_file is not None


def test_labelfn_accuracy_table(tmp_path: Path):
    fires = [
        seg(0, SegmentKind.FIRE, 8.5, 11.5, votes={"fist_after_fire": "misfire", "clean_command": "intended"}),  # fist fn right
        seg(1, SegmentKind.FIRE, 18.5, 21.5, votes={"fist_after_fire": "misfire"}),  # fist fn wrong
        seg(2, SegmentKind.FIRE, 28.5, 31.5, votes={"clean_command": "intended"}),  # clean right
        seg(3, SegmentKind.FIRE, 38.5, 41.5, votes={}),  # a misfire nobody caught
        seg(4, SegmentKind.FIRE, 48.5, 51.5, votes={"clean_command": "intended"}),  # ambiguous: not scored
    ]
    arm = seg(0, SegmentKind.ARM, 1.0, 5.0, gesture="open_palm", votes={"arm_then_command": "intended"})
    labels = [
        human(fires[0], Verdict.MISFIRE, intent=Intent.INCIDENTAL),
        human(fires[1], Verdict.INTENDED),
        human(fires[2], Verdict.INTENDED),
        human(fires[3], Verdict.MISFIRE, intent=Intent.INCIDENTAL),
        human(fires[4], Verdict.AMBIGUOUS, intent=Intent.UNSURE),
        human(arm, Verdict.INTENDED),
    ]
    d = write_session(tmp_path, fires + [arm], labels)
    out = io.StringIO()
    scores = labelfn_accuracy([d], out=out)
    fist = scores["fist_after_fire"]
    assert (fist.votes, fist.correct, fist.truth, fist.labelled) == (2, 1, 2, 4)
    assert fist.precision == 0.5 and fist.recall == 0.5 and fist.coverage == 0.5
    clean = scores["clean_command"]
    assert clean.label == "intended" and (clean.votes, clean.correct, clean.truth) == (2, 1, 2)
    assert clean.precision == 0.5 and clean.recall == 0.5
    assert scores["arm_then_command"].precision == 1.0 and scores["arm_then_command"].labelled == 1
    assert scores["immediate_redo"].votes == 0 and scores["immediate_redo"].precision is None
    table = out.getvalue()
    assert "fist_after_fire" in table and "precision" in table and "arm_then_command" in table


def test_lf_accuracy_without_labels_prints_a_note(tmp_path: Path):
    out = io.StringIO()
    assert labelfn_accuracy([tmp_path], out=out) == {}
    assert "no human labels" in out.getvalue()

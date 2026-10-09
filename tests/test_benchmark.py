"""tools/benchmark: engine counts on synthetic sessions, the classifier protocol on a synthetic
datasets/ tree, label agreement, the JSON / markdown writers and `compare`. No real data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visual_actions.core.config import default_config
from visual_actions.intent.events import Event
from visual_actions.intent.schema import (
    HumanLabel,
    Intent,
    Motion,
    Segment,
    SegmentKind,
    Verdict,
    write_jsonl,
)
from visual_actions.learn.evaluate import replay_events
from visual_actions.tools import benchmark
from visual_actions.tools.benchmark import (
    MISFIRE_FNS,
    EngineCounts,
    classifier_benchmark,
    compare,
    engine_benchmark,
    labels_benchmark,
    main,
    metric_rows,
    render_markdown,
    run_benchmark,
    select_sessions,
    session_counts,
    write_outputs,
)
from visual_actions.tools.synth import write_drag_session, write_session

FIRE_LEFT = [("open_palm", 2.5), ("h_left", 0.8), ("lost", 0.5)]
FIRE_THEN_FIST = [("open_palm", 2.5), ("h_left", 0.8), ("fist", 1.5), ("lost", 0.5)]
PALM_BROKEN = [("open_palm", 0.8), ("none", 0.5), ("lost", 0.5)]
TIMEOUT = [("open_palm", 2.5), ("none", 6.5), ("lost", 0.5)]


def write_events_log(path: Path, events: list[Event], closed: bool = True) -> None:
    """events.log in session.py's line format, from replayed events."""
    lines = [f"{0.0:9.3f}  00:00:00.000000  {'session':7s} started 2026-10-09 00:00:00 t0_ns=0"]
    lines += [f"{e.t:9.3f}  00:00:00.000000  {e.kind:7s} {e.text}" for e in events if e.kind != "session"]
    if closed:
        lines.append(json.dumps({"summary": {"frames": 0}}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_session(sessions: Path, stamp: str, segments: list[tuple[str, float]] | None, closed: bool = True) -> Path:
    """A synthetic session: landmarks from synth, events.log from the default pipeline's replay
    (the mock automation's default windows catch the drag). `segments` None = a pinch-drag."""
    d = sessions / stamp
    d.mkdir(parents=True, exist_ok=True)
    if segments is None:
        write_drag_session(d / "landmarks.jsonl")
    else:
        write_session(d / "landmarks.jsonl", segments)
    write_events_log(d / "events.log", replay_events(d, default_config()), closed=closed)
    return d


def make_datasets(root: Path, classes: tuple[str, ...] = ("open_palm", "fist", "h_left", "none")) -> Path:
    for cls in classes:
        for stamp in ("20261008-000001", "20261009-000001"):
            write_session(root / cls / f"{stamp}.jsonl", [(cls, 1.0)])
    write_session(root / "open_palm" / "public-fake.jsonl", [("open_palm", 1.0)])
    write_session(root / "fist" / "synth-fake.jsonl", [("fist", 1.0)])
    return root


# -- engine counts -----------------------------------------------------------------------------


def test_fire_counts_and_replay_equals_live(tmp_path: Path):
    d = make_session(tmp_path, "20261009-100000", FIRE_LEFT)
    c = session_counts(d, default_config())
    r = c["replay"]
    assert r["fires"] == 1 and r["arms"] == 1 and r["weak_misfire_fires"] == 0 and r["weak_intended_fires"] == 1
    assert r["misfire_rate"] == 0.0 and r["broken_hold_ratio"] == 0.0 and r["releases_per_drag"] is None
    assert r["hand_losses"] == 1 and r["seconds"] > 0 and r["hand_losses_per_min"] == pytest.approx(60 / r["seconds"], rel=0.01)
    # the log was written from the same replay, so the live view counts the same
    assert c["live"] == r


def test_fist_after_fire_is_a_weak_misfire(tmp_path: Path):
    d = make_session(tmp_path, "20261009-100000", FIRE_THEN_FIST)
    r = session_counts(d, default_config())["replay"]
    assert r["fires"] == 1 and r["weak_misfire_fires"] == 1 and r["misfire_rate"] == 1.0
    assert r["misfire_votes"]["fist_after_fire"] == 1 and set(r["misfire_votes"]) == set(MISFIRE_FNS)
    assert r["weak_intended_fires"] == 0


def test_broken_hold_and_empty_arm(tmp_path: Path):
    r = session_counts(make_session(tmp_path, "20261009-100000", PALM_BROKEN), default_config())["replay"]
    assert r["broken_holds"] == 1 and r["arms"] == 0 and r["broken_hold_ratio"] == 1.0
    r = session_counts(make_session(tmp_path, "20261009-110000", TIMEOUT), default_config())["replay"]
    assert r["arms"] == 1 and r["empty_arms"] == 1 and r["fires"] == 0 and r["misfire_rate"] is None


def test_drag_session_counts_releases(tmp_path: Path):
    r = session_counts(make_session(tmp_path, "20261009-100000", None), default_config())["replay"]
    assert r["drags"] == 1 and r["drag_ends"] == 1 and r["regrabs"] == 0 and r["releases_per_drag"] == 1.0
    assert r["fist_cancelled_drags"] == 0 and r["fires"] == 0


def test_regrab_flaps_counted_from_drag_lines():
    mk = lambda t, kind, text: Event(t, "", kind, text)
    events = [
        mk(0.0, "session", "started t0_ns=0"),
        mk(1.0, "hand", "seen conf=0.9 wrist=(0.5,0.5) (mode idle)"),
        mk(2.0, "drag", "start App: W @(100,100)"),
        mk(2.5, "drag", "resume App: W @(110,100)"),  # a release re-grabbed within the grace: a flap
        mk(3.0, "drag", "pause App: W @(120,100)"),
        mk(3.2, "drag", "resume App: W @(120,100)"),  # back after a hand loss: not a flap
        mk(4.0, "drag", "end App: W @(130,100)"),
        mk(5.0, "hand", "lost [tracker] (mode idle)"),
        mk(6.0, "hand", "seen conf=0.9 wrist=(0.5,0.5) (mode idle)"),
        mk(6.5, "hand", "lost [edge] (mode idle)"),
    ]
    c = benchmark.engine_counts([], events)
    assert c.regrabs == 1 and c.drag_ends == 1 and c.hand_losses == 2 and c.seconds == 6.5
    assert c.rates()["hand_losses_per_min"] == pytest.approx(2 / (6.5 / 60), rel=0.01)


def test_engine_counts_add_and_totals(tmp_path: Path):
    a = make_session(tmp_path, "20261009-100000", FIRE_LEFT)
    b = make_session(tmp_path, "20261009-110000", FIRE_THEN_FIST)
    e = engine_benchmark([a, b], default_config())
    assert e["sessions"] == [a.name, b.name] and set(e["per_session"]) == {a.name, b.name}
    tot = e["total"]["replay"]
    assert tot["sessions"] == 2 and tot["fires"] == 2 and tot["weak_misfire_fires"] == 1 and tot["misfire_rate"] == 0.5
    assert tot["misfire_votes"]["fist_after_fire"] == 1
    x = EngineCounts(fires=2, weak_misfire_fires=1)
    x.add(EngineCounts(fires=1, misfire_votes={"fist_after_fire": 1}))
    assert x.fires == 3 and x.misfire_votes["fist_after_fire"] == 1 and x.rates()["misfire_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_select_sessions_newest_closed_with_landmarks(tmp_path: Path):
    s = tmp_path / "sessions"
    make_session(s, "20261008-100000", FIRE_LEFT)
    make_session(s, "20261009-100000", FIRE_LEFT)
    make_session(s, "20261009-120000", FIRE_LEFT, closed=False)  # still being written
    (s / "20261009-130000").mkdir()  # no files at all
    (s / "notes.txt").write_text("x")
    assert [p.name for p in select_sessions(s, 6)] == ["20261009-100000", "20261008-100000"]
    assert [p.name for p in select_sessions(s, 1)] == ["20261009-100000"]
    assert [p.name for p in select_sessions(s, 0)] == ["20261009-100000", "20261008-100000"]
    assert select_sessions(tmp_path / "missing", 6) == []


# -- classifier --------------------------------------------------------------------------------


def test_classifier_protocol_holds_out_newest_user_session_per_class(tmp_path: Path):
    c = classifier_benchmark(make_datasets(tmp_path / "datasets"))
    assert c["status"] == "ok" and c["frames"] == 300
    assert c["by_source"] == {"public": 30, "synth": 30, "user": 240}
    assert c["by_class"] == {"fist": 90, "h_left": 60, "none": 60, "open_palm": 90}
    h = c["heldout"]
    assert h["groups"] == [f"{cls}/20261009-000001.jsonl" for cls in ("fist", "h_left", "none", "open_palm")]
    assert h["frames"] == 120 and h["train_frames"] == 180
    assert h["accuracy"] == 1.0 and h["fist_recall"] == 1.0
    assert set(h["per_class"]) == {"fist", "h_left", "none", "open_palm"} and h["per_class"]["fist"]["n"] == 30
    assert h["top_confusions"] == [] and h["confusion"]["fist"] == {"fist": 30}
    files = {f["file"]: f for f in c["files"]}
    assert files["open_palm/public-fake.jsonl"]["source"] == "public" and len(files["fist/synth-fake.jsonl"]["sha256"]) == 16


def test_classifier_without_a_second_session_has_no_heldout(tmp_path: Path):
    ds = tmp_path / "datasets"
    for cls in ("open_palm", "fist"):
        write_session(ds / cls / "20261008-000001.jsonl", [(cls, 1.0)])
    c = classifier_benchmark(ds)
    assert "heldout" not in c and c["status"].startswith("only one session")
    (tmp_path / "empty").mkdir()
    assert classifier_benchmark(tmp_path / "empty")["status"] == "no data"


# -- labels ------------------------------------------------------------------------------------


def write_labelled_session(root: Path, stamp: str = "20261009-100000") -> Path:
    d = root / stamp
    (d / "intent").mkdir(parents=True)
    segs = [
        Segment(f"{stamp}/fire/0000", stamp, SegmentKind.FIRE, 8.5, 11.5, engine_gesture="h_left", labelfn_votes={"fist_after_fire": "misfire"}),
        Segment(f"{stamp}/fire/0001", stamp, SegmentKind.FIRE, 18.5, 21.5, engine_gesture="h_left", labelfn_votes={"clean_command": "intended"}),
        Segment(f"{stamp}/arm/0000", stamp, SegmentKind.ARM, 30.0, 35.0, engine_gesture="open_palm", labelfn_votes={"arm_then_command": "intended"}),
    ]
    write_jsonl(d / "intent" / "segments.jsonl", segs)
    human = [
        HumanLabel(segs[0].segment_id, Verdict.MISFIRE, Intent.INCIDENTAL, motion=Motion.STILL),
        HumanLabel(segs[1].segment_id, Verdict.MISFIRE, Intent.INCIDENTAL, motion=Motion.STILL),  # clean_command was wrong here
        HumanLabel(segs[2].segment_id, Verdict.INTENDED, Intent.COMMAND, motion=Motion.STILL),
    ]
    write_jsonl(d / "intent" / "human.jsonl", human)
    return d


def test_labels_na_without_human_labels(tmp_path: Path):
    d = make_session(tmp_path, "20261009-100000", FIRE_LEFT)
    (d / "intent").mkdir()
    (d / "intent" / "tags.jsonl").write_text('{"segment_id": "x"}\n')
    lab = labels_benchmark([d])
    assert lab["status"] == "n/a" and lab["tags"] == 1 and lab["human_labels"] == 0 and lab["sessions_with_tags"] == [d.name]
    assert labels_benchmark([])["status"] == "n/a"


def test_labels_precision_recall_against_human_labels(tmp_path: Path):
    lab = labels_benchmark([write_labelled_session(tmp_path)])
    assert lab["status"] == "ok" and lab["human_labels"] == 3 and lab["sessions_with_human"] == ["20261009-100000"]
    fns = lab["functions"]
    assert fns["fist_after_fire"]["precision"] == 1.0 and fns["fist_after_fire"]["recall"] == 0.5  # 1 of the 2 misfires
    assert fns["clean_command"]["precision"] == 0.0 and fns["clean_command"]["votes"] == 1
    assert fns["arm_then_command"]["precision"] == 1.0 and fns["arm_then_command"]["kind"] == "arm"
    assert "labelling function" in lab["table"]


# -- the run, the writers, compare ---------------------------------------------------------------


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    sessions = tmp_path / "sessions"
    make_session(sessions, "20261008-100000", FIRE_THEN_FIST)
    make_session(sessions, "20261009-100000", FIRE_LEFT)
    make_session(sessions, "20261009-110000", None)
    make_session(sessions, "20261009-120000", FIRE_LEFT, closed=False)
    make_datasets(tmp_path / "datasets")
    return {"sessions": sessions, "datasets": tmp_path / "datasets", "config": tmp_path / "no-config.toml", "out": tmp_path / "bench"}


def test_run_benchmark_and_writers(world: dict[str, Path]):
    r = run_benchmark("t", 2, world["sessions"], world["datasets"], None, world["config"])
    assert r["label"] == "t" and r["sessions"] == ["20261009-110000", "20261009-100000"] and r["sessions_requested"] == 2
    assert r["model"] == {"path": None, "exists": False, "sha256": None} and r["config"]["profile"] == "strict" and r["config"]["exists"] is False
    assert r["classifier"]["heldout"]["accuracy"] == 1.0 and r["labels"]["status"] == "n/a"
    assert r["engine"]["total"]["replay"]["drags"] == 1 and r["engine"]["total"]["replay"]["fires"] == 1
    js, md = write_outputs(r, world["out"])
    assert js.name == f"{r['date']}-t.json" and md.name == f"{r['date']}-t.md"
    assert json.loads(js.read_text()) == json.loads(json.dumps(r))  # JSON-clean, round-trips
    text = md.read_text()
    assert "# Benchmark t" in text and "| metric | t |" in text and "| metric | t replay | t live |" in text
    assert "| held-out accuracy | 1.0 |" in text and "| drags | 1 | 1 |" in text and "| 20261009-110000 |" in text
    assert "n/a: no human labels" in text
    rows = {(s, m): v for s, m, v in metric_rows(r)}
    assert rows[("engine replay", "fires")] == 1 and rows[("classifier", "frames user")] == 240 and rows[("labels", "status")] == "n/a"


def test_markdown_shows_labelled_agreement_when_present(tmp_path: Path):
    r = {"label": "x", "date": "2026-10-09", "sessions": [], "classifier": {"status": "no data"}, "engine": {}, "labels": labels_benchmark([write_labelled_session(tmp_path)])}
    text = render_markdown(r)
    assert "| precision fist_after_fire | 1.0 |" in text and "| recall fist_after_fire | 0.5 |" in text and "| held-out accuracy | n/a |" in text


def test_compare_prints_side_by_side_with_deltas(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    before = {
        "label": "before", "date": "2026-10-09", "git": "aaa", "sessions": ["s1"], "model": {"path": "m.joblib", "sha256": "1"}, "config": {"profile": "strict"},
        "classifier": {"status": "ok", "frames": 10, "by_source": {"user": 10}, "by_class": {"fist": 10}, "heldout": {"accuracy": 0.9, "fist_recall": 0.8, "frames": 5, "per_class": {"fist": {"accuracy": 0.8}}}},
        "engine": {"sessions": ["s1"], "total": {"replay": {"fires": 10, "weak_misfire_fires": 3, "misfire_rate": 0.3}, "live": {"fires": 9}}},
        "labels": {"status": "n/a"},
    }
    after = json.loads(json.dumps(before))
    after.update(label="after", git="bbb")
    after["classifier"]["heldout"]["accuracy"] = 0.95
    after["engine"]["total"]["replay"].update(fires=12, weak_misfire_fires=1, misfire_rate=0.0833)
    after["labels"] = {"status": "ok", "human_labels": 4, "functions": {"fist_after_fire": {"precision": 1.0, "recall": 0.5}}}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(before))
    b.write_text(json.dumps(after))
    assert main(["compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert out == compare(before, after)
    assert "| metric | before | after | delta |" in out
    assert "| held-out accuracy | 0.9 | 0.95 | +0.05 |" in out
    assert "| fires | 10 | 12 | +2 |" in out and "| weak-misfire fires | 3 | 1 | -2 |" in out
    assert "| misfire rate | 0.3 | 0.0833 | -0.2167 |" in out
    assert "| fist recall | 0.8 | 0.8 | 0 |" in out
    assert "| status | n/a | ok | changed |" in out
    assert "| precision fist_after_fire | n/a | 1.0 |  |" in out  # a metric only one side has
    assert "| **engine replay** |" in out and "git `aaa`" in out and "git `bbb`" in out


def test_cli_run_writes_both_files(world: dict[str, Path], capsys: pytest.CaptureFixture[str]):
    argv = ["--label", "cli", "--sessions", "1", "--out", str(world["out"]), "--sessions-dir", str(world["sessions"]), "--datasets", str(world["datasets"]), "--model", "none", "--config", str(world["config"])]
    assert main(argv) == 0
    out = capsys.readouterr().out
    files = sorted(p.name for p in world["out"].iterdir())
    assert len(files) == 2 and files[0].endswith("-cli.json") and files[1].endswith("-cli.md")
    r = json.loads((world["out"] / files[0]).read_text())
    assert r["sessions"] == ["20261009-110000"] and r["model"]["path"] is None and out.startswith("# Benchmark cli")

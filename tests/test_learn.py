"""The nightly learning job: sessions -> labels -> candidate -> gate -> pointer."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from visual_actions import paths
from visual_actions.core.config import Config, default_config
from visual_actions.intent.events import Event, parse_events
from visual_actions.intent.jev import FakeJev
from visual_actions.intent.schema import (
    HumanLabel,
    Intent,
    Motion,
    SegmentKind,
    Verdict,
    write_jsonl,
)
from visual_actions.intent.segments import segment_session
from visual_actions.intent.tagger import FakeTagger
from visual_actions.learn import evaluate, schedule, versions
from visual_actions.learn.__main__ import main as learn_main
from visual_actions.learn.evaluate import (
    EngineMetrics,
    FrameMetrics,
    Scores,
    decide,
    engine_metrics,
    replay_events,
)
from visual_actions.learn.job import Judges, load_state, run
from visual_actions.learn.trainer import load_dataset, review_session, source_of, train_candidate
from visual_actions.tools.synth import write_session

FIRE_LEFT = [("open_palm", 2.5), ("h_left", 0.8), ("lost", 0.5)]
FIRE_RIGHT = [("open_palm", 2.5), ("h_right", 0.8), ("lost", 0.5)]
FIRE_THEN_FIST = [("open_palm", 2.5), ("h_left", 0.8), ("fist", 1.5), ("lost", 0.5)]
FIST_ONLY = [("fist", 3.0), ("lost", 0.5)]
PALM_BROKEN = [("open_palm", 0.5), ("none", 0.5), ("lost", 0.5)]


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "data"
    monkeypatch.setattr(paths, "data_dir", lambda: d)
    return d


def config(**learn: object) -> Config:
    cfg = default_config()
    cfg.learn.min_new_frames = 1
    cfg.learn.holdout_sessions = 2
    for k, v in learn.items():
        setattr(cfg.learn, k, v)
    return cfg


def write_events_log(path: Path, events: list[Event], t0_ns: int = 0, closed: bool = True) -> None:
    """events.log in session.py's line format, from replayed events."""
    lines = [f"{0.0:9.3f}  00:00:00.000000  {'session':7s} started 2026-10-09 00:00:00 t0_ns={t0_ns}"]
    for e in events:
        if e.kind == "session":
            continue
        lines.append(f"{e.t:9.3f}  00:00:00.000000  {e.kind:7s} {e.text}")
    if closed:
        lines.append(json.dumps({"summary": {"frames": 0}}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_session(sessions: Path, stamp: str, segments: list[tuple[str, float]], cfg: Config | None = None, closed: bool = True) -> Path:
    d = sessions / stamp
    d.mkdir(parents=True, exist_ok=True)
    write_session(d / "landmarks.jsonl", segments)
    events = replay_events(d, cfg or default_config())
    write_events_log(d / "events.log", events, closed=closed)
    return d


def label_all(session: Path) -> int:
    """Segments plus a human label for each: fires intended (class = the engine gesture), a
    held fist with the engine idle = fist, broken holds and idle hands otherwise = none."""
    segs = segment_session(session.name, parse_events(session / "events.log"))
    write_jsonl(session / "intent" / "segments.jsonl", segs)
    labels = []
    for s in segs:
        if s.kind is SegmentKind.FIRE:
            labels.append(HumanLabel(s.segment_id, Verdict.INTENDED, Intent.COMMAND, s.engine_gesture, Motion.STILL))
        elif s.kind is SegmentKind.HAND:
            if "fist" in session.name:  # the idle hand in those sessions is a held fist
                labels.append(HumanLabel(s.segment_id, Verdict.INTENDED, Intent.COMMAND, "fist", Motion.STILL))
        elif s.kind is SegmentKind.BROKEN_HOLD:
            labels.append(HumanLabel(s.segment_id, Verdict.NO_EVENT, Intent.INCIDENTAL, None, Motion.STILL))
    write_jsonl(session / "intent" / "human.jsonl", labels)
    return len(labels)


def seed_datasets(datasets: Path) -> None:
    """A small user-recorded base so the candidate has every class."""
    for cls in ("open_palm", "fist", "h_left", "h_right", "none", "point_up"):
        write_session(datasets / cls / "20261001-000000.jsonl", [(cls, 1.0)])
    write_session(datasets / "none" / "public-hagrid.jsonl", [("none", 0.5)])


def day_one(data_dir: Path, cfg: Config) -> None:
    sessions = data_dir / "sessions"
    seed_datasets(data_dir / "datasets")
    for stamp, segs in (("20261008-100000", FIRE_LEFT), ("20261008-110000", FIRE_RIGHT + PALM_BROKEN), ("20261008-120000", FIRE_THEN_FIST)):
        label_all(make_session(sessions, stamp, segs, cfg))
    # the fist session gets fist labels (its name carries the marker label_all looks for)
    label_all(make_session(sessions, "20261008-130000-fist", FIST_ONLY, cfg))


def day_two(data_dir: Path, cfg: Config) -> None:
    for stamp, segs in (("20261009-100000", FIRE_LEFT + FIRE_RIGHT), ("20261009-110000-fist", FIST_ONLY + FIRE_LEFT)):
        label_all(make_session(data_dir / "sessions", stamp, segs, cfg))


# -- replay -> events -> metrics ---------------------------------------------------------------


def test_replay_events_match_the_session_log_format_and_segment(tmp_path: Path):
    d = make_session(tmp_path, "20261009-100000", FIRE_LEFT)
    events = replay_events(d, default_config())
    kinds = {e.kind for e in events}
    assert {"session", "token", "mode", "action", "hand"} <= kinds
    fire = [e for e in events if e.kind == "action"]
    assert len(fire) == 1 and fire[0].action_name == "Previous tab" and fire[0].action_ok
    # the written log parses back to the same events, and segments see the fire and the arm
    parsed = parse_events(d / "events.log")
    assert [(e.t, e.kind, e.text) for e in parsed if e.kind != "session"] == [(e.t, e.kind, e.text) for e in events if e.kind != "session"]
    m = engine_metrics(segment_session(d.name, parsed))
    assert m.fires == 1 and m.arms == 1 and m.weak_intended_fires == 1 and m.weak_misfire_fires == 0


def test_engine_metrics_count_misfire_votes_broken_holds_and_empty_arms(tmp_path: Path):
    cfg = default_config()
    fist = make_session(tmp_path, "20261009-100000", FIRE_THEN_FIST, cfg)
    m = engine_metrics(segment_session(fist.name, parse_events(fist / "events.log")))
    assert m.fires == 1 and m.weak_misfire_fires == 1  # fist_after_fire
    broken = make_session(tmp_path, "20261009-110000", PALM_BROKEN, cfg)
    m = engine_metrics(segment_session(broken.name, parse_events(broken / "events.log")))
    assert m.broken_holds == 1 and m.fires == 0
    timeout = make_session(tmp_path, "20261009-120000", [("open_palm", 2.5), ("none", 6.0), ("lost", 0.5)], cfg)
    m = engine_metrics(segment_session(timeout.name, parse_events(timeout / "events.log")))
    assert m.arms == 1 and m.empty_arms == 1


def test_select_holdout_skips_the_training_day_and_unreplayable_sessions(tmp_path: Path):
    a = make_session(tmp_path, "20261007-100000", FIRE_LEFT)
    b = make_session(tmp_path, "20261008-100000", FIRE_LEFT)
    c = make_session(tmp_path, "20261009-100000", FIRE_LEFT)
    (tmp_path / "20261006-100000").mkdir()  # no landmarks
    out = evaluate.select_holdout([a, b, c, tmp_path / "20261006-100000"], date(2026, 10, 9), 5)
    assert out == [b, a]
    assert evaluate.select_holdout([a, b, c], date(2026, 10, 9), 1) == [b]
    assert evaluate.session_closed(a)
    make_session(tmp_path, "20261009-200000", FIRE_LEFT, closed=False)
    assert not evaluate.session_closed(tmp_path / "20261009-200000")


# -- the gate ------------------------------------------------------------------------------------


def scores(acc: float | None = 0.9, fist: float | None = 1.0, misfire: int = 2, intended: int = 10, sessions: int = 2) -> Scores:
    return Scores("m", FrameMetrics(frames=100, accuracy=acc, fist_frames=10, fist_recall=fist), EngineMetrics(sessions=sessions, fires=12, weak_misfire_fires=misfire, weak_intended_fires=intended))


def test_gate_promotes_only_when_every_rule_passes():
    cfg = default_config().learn
    champ = scores()
    assert decide(champ, scores(), cfg).promote
    assert decide(champ, scores(acc=0.895), cfg).promote  # within the 0.01 margin
    d = decide(champ, scores(acc=0.88), cfg)
    assert not d.promote and [r.name for r in d.rules if not r.passed] == ["accuracy"]
    d = decide(champ, scores(misfire=3), cfg)
    assert not d.promote and [r.name for r in d.rules if not r.passed] == ["misfires"]
    assert decide(champ, scores(intended=10), cfg).promote
    d = decide(champ, scores(intended=9), cfg)  # 9 < 0.95 * 10
    assert not d.promote and [r.name for r in d.rules if not r.passed] == ["intended"]
    d = decide(champ, scores(fist=0.99), cfg)
    assert not d.promote and [r.name for r in d.rules if not r.passed] == ["fist_recall"]
    d = decide(champ, scores(acc=0.5, fist=0.5, misfire=9, intended=0), cfg)
    assert [r.name for r in d.rules if not r.passed] == ["accuracy", "misfires", "intended", "fist_recall"]


def test_gate_without_labelled_frames_measures_what_it_can_and_refuses_without_sessions():
    cfg = default_config().learn
    d = decide(scores(acc=None, fist=None), scores(acc=None, fist=None), cfg)
    assert d.promote and all("not measured" in r.detail for r in d.rules if r.name in ("accuracy", "fist_recall"))
    d = decide(scores(), scores(sessions=0), cfg)
    assert not d.promote and d.rules[0].name == "holdout"


# -- the trainer ---------------------------------------------------------------------------------


def test_loader_weights_sources_holds_out_sessions_and_refuses_judge_fist(tmp_path: Path):
    ds = tmp_path / "datasets"
    write_session(ds / "h_left" / "20261001-000000.jsonl", [("h_left", 0.5)])
    write_session(ds / "h_left" / "public-hagrid.jsonl", [("h_left", 0.5)])
    write_session(ds / "none" / "synth3d-a.jsonl", [("none", 0.5)])
    # review exports: a human line (weight 1.0), a judge line (weight 0.8), and a judge fist
    rev = ds / "h_left" / "review-20261008-100000-20261009-020000.jsonl"
    rows = [json.loads(line) for line in (ds / "h_left" / "20261001-000000.jsonl").read_text().splitlines()[:2]]
    rev.write_text("\n".join(json.dumps({**r, "source": s, "weight": w}) for r, (s, w) in zip(rows, (("human", 1.0), ("judge", 0.8)), strict=True)) + "\n")
    fist = ds / "fist" / "review-20261008-100000-20261009-020000.jsonl"
    fist.parent.mkdir()
    fist.write_text(json.dumps({**rows[0], "source": "judge", "weight": 0.9}) + "\n" + json.dumps({**rows[1], "source": "human", "weight": 1.0}) + "\n")
    assert source_of(rev) == "review" and review_session(rev) == "20261008-100000" and source_of(ds / "none" / "synth3d-a.jsonl") == "synth"

    train, hold = load_dataset(ds, user_weight=2.0)
    assert len(hold) == 0
    by_src = {s: set(train.w[train.sources == s].tolist()) for s in set(train.sources.tolist())}
    assert by_src["user"] == {2.0} and by_src["public"] == {1.0} and by_src["synth"] == {1.0}
    assert by_src["human"] == {2.0} and by_src["judge"] == {1.6}
    assert train.rejected_fist_judge == 1 and int((train.y == "fist").sum()) == 1

    train, hold = load_dataset(ds, user_weight=2.0, holdout_sessions={"20261008-100000"})
    assert len(hold) == 3 and set(hold.groups.tolist()) == {"20261008-100000"} and hold.rejected_fist_judge == 1
    assert "judge" not in set(train.sources.tolist())


def test_train_candidate_writes_a_deterministic_model_and_manifest(tmp_path: Path):
    ds = tmp_path / "datasets"
    seed_datasets(ds)
    cfg = config(location="desk").learn
    c1 = train_candidate(ds, tmp_path / "user", cfg, date(2026, 10, 9), hostname="mac")
    assert c1 is not None and c1.path.exists() and c1.manifest_path == tmp_path / "user" / f"{c1.name}.manifest.json"
    m = json.loads(c1.manifest_path.read_text())
    assert m["date"] == "2026-10-09" and m["hostname"] == "mac" and m["location"] == "desk" and m["decision"] == "pending"
    assert {f["file"] for f in m["datasets"]} >= {"open_palm/20261001-000000.jsonl", "none/public-hagrid.jsonl"}
    assert all(len(f["sha256"]) == 16 for f in m["datasets"])
    assert m["frames"]["by_source"] == {"public": 15, "user": 180} and m["frames"]["by_class"]["fist"] == 30
    assert c1.name.startswith("gestures-2026-10-09-") and c1.manifest["train_accuracy_weighted"] > 0.95
    c2 = train_candidate(ds, tmp_path / "user2", cfg, date(2026, 10, 9), hostname="mac")
    assert c2 is not None and c2.name == c1.name and c2.path.read_bytes() == c1.path.read_bytes()
    assert evaluate.frame_metrics(c1.path, c1.train).accuracy == c1.manifest["train_accuracy_weighted"]


# -- the job -------------------------------------------------------------------------------------


def test_job_processes_sessions_once_trains_and_promotes_then_is_idempotent(data_dir: Path, capsys: pytest.CaptureFixture[str]):
    cfg = config()
    day_one(data_dir, cfg)
    user_dir = data_dir / "models" / "user"

    # night 1: no earlier day to hold out -> the candidate cannot be compared, nothing promoted
    r1 = run(date(2026, 10, 8), judges=False, config=cfg, log=lambda s: None)
    assert [s.stamp for s in r1.sessions] == ["20261008-100000", "20261008-110000", "20261008-120000", "20261008-130000-fist"]
    assert r1.new_frames > 0 and r1.status == "discarded" and r1.holdout == [] and r1.gate["rules"][0]["name"] == "holdout"
    assert versions.current_name(user_dir) is None and len(versions.list_versions(user_dir)) == 1
    state = load_state(data_dir / "learn" / "state.json")
    assert set(state["processed"]) == {s.stamp for s in r1.sessions}
    reviews = sorted(p.name for p in (data_dir / "datasets").glob("*/review-*.jsonl"))
    assert any("review-20261008-100000-" in n for n in reviews) and any(n.startswith("review-20261008-130000-fist") for n in reviews)
    assert (data_dir / "datasets" / "fist").glob("review-*") and (data_dir / "learn" / "reports" / "2026-10-08.md").exists()

    # night 2: the 10-09 sessions are new; the two newest 10-08 sessions are held out and replayed
    day_two(data_dir, cfg)
    r2 = run(date(2026, 10, 9), judges=False, config=cfg, log=print)
    out = capsys.readouterr().out
    assert [s.stamp for s in r2.sessions] == ["20261009-100000", "20261009-110000-fist"]
    assert r2.holdout == ["20261008-130000-fist", "20261008-120000"]
    assert r2.scores["candidate"]["frame"]["frames"] > 0 and r2.scores["candidate"]["frame"]["fist_frames"] > 0
    assert r2.scores["candidate"]["engine"]["sessions"] == 2 and r2.scores["champion"]["engine"]["fires"] == 1
    assert "gate accuracy" in out and r2.status == "promoted", r2.message
    assert versions.current_name(user_dir) == f"{r2.candidate}.joblib" and (user_dir / "current.joblib").is_symlink()
    assert paths.live_model_path() == user_dir / "current.joblib"
    manifest = json.loads((user_dir / f"{r2.candidate}.manifest.json").read_text())
    assert manifest["decision"]["promote"] is True and manifest["holdout_sessions"] == sorted(r2.holdout) and "candidate" in manifest["metrics"]
    last = json.loads((data_dir / "learn" / "last.json").read_text())
    assert last["status"] == "promoted" and last["candidate"] == r2.candidate
    assert last["decision"] == "promoted" and last["model_version"] == r2.candidate and set(last["misfires"]) == {"champion", "candidate"}
    from visual_actions.ui.learning_menu import format_status

    assert format_status(last).startswith("Learning: 2026-10-09 promoted")
    md = (data_dir / "learn" / "reports" / "2026-10-09.md").read_text()
    assert "## Gate" in md and "PASS accuracy" in md and r2.candidate in md

    # night 2 again: nothing new, nothing trained, pointer untouched
    r3 = run(date(2026, 10, 9), judges=False, config=cfg, log=lambda s: None)
    assert r3.sessions == [] and r3.status == "skipped" and versions.current_name(user_dir) == f"{r2.candidate}.joblib"
    assert set(load_state(data_dir / "learn" / "state.json")["processed"]) == set(state["processed"]) | {"20261009-100000", "20261009-110000-fist"}


def test_job_skips_below_min_new_frames_and_leaves_an_open_session_for_tomorrow(data_dir: Path):
    cfg = config(min_new_frames=10_000)
    seed_datasets(data_dir / "datasets")
    label_all(make_session(data_dir / "sessions", "20261009-100000", FIRE_LEFT, cfg))
    make_session(data_dir / "sessions", "20261009-110000", FIRE_LEFT, cfg, closed=False)  # the app is still recording it
    r = run(date(2026, 10, 9), judges=False, config=cfg, log=lambda s: None)
    assert r.status == "skipped" and "min_new_frames" in r.message and r.pending == ["20261009-110000"]
    assert not (data_dir / "models" / "user").exists()
    assert set(load_state(data_dir / "learn" / "state.json")["processed"]) == {"20261009-100000"}
    r = run(date(2026, 10, 9), judges=False, config=config(min_new_frames=10_000), log=lambda s: None, force=True)
    assert r.status in ("discarded", "promoted") and r.candidate


def test_dry_run_writes_segments_and_reports_but_no_datasets_or_models(data_dir: Path, capsys: pytest.CaptureFixture[str]):
    cfg = config()
    day_one(data_dir, cfg)
    day_two(data_dir, cfg)
    for s in (data_dir / "sessions").iterdir():
        if s.name.startswith("20261009"):
            (s / "intent" / "segments.jsonl").unlink()  # the job must cut them itself; the labels are keyed by id and stay
    before = sorted(p.relative_to(data_dir) for p in (data_dir / "datasets").rglob("*.jsonl"))
    assert learn_main(["run", "--dry-run", "--no-judges", "--day", "2026-10-09"]) == 0
    out = capsys.readouterr().out
    assert "would be promoted (dry run)" in out or "discarded" in out
    assert sorted(p.relative_to(data_dir) for p in (data_dir / "datasets").rglob("*.jsonl")) == before
    assert not (data_dir / "models").exists() and not (data_dir / "learn" / "state.json").exists()
    assert (data_dir / "sessions" / "20261009-100000" / "intent" / "segments.jsonl").exists()
    assert (data_dir / "learn" / "reports" / "2026-10-09-dry-run.md").exists() and (data_dir / "learn" / "last-dry-run.json").exists()
    assert not (data_dir / "learn" / "last.json").exists()


def test_judge_passes_are_best_effort_and_judge_labels_export_without_fist(data_dir: Path):
    cfg = config()
    sessions = data_dir / "sessions"
    seed_datasets(data_dir / "datasets")
    make_session(sessions, "20261009-100000", FIRE_LEFT + FIST_ONLY, cfg)

    class Boom:
        def judge(self, *a: object) -> None:
            raise RuntimeError("no network")

    def fist_tag(segment):  # the tagger claims every idle hand is a deliberate fist: refused at export
        from visual_actions.intent.tagger import Answer

        g = "fist" if segment.kind is SegmentKind.HAND else segment.engine_gesture
        return Answer(Verdict.INTENDED, Intent.COMMAND, Motion.STILL, g, [], 0.95, "fake")

    judges = Judges(cosmos=Boom(), jev=FakeJev(), tagger=FakeTagger(fist_tag))
    lines: list[str] = []
    r = run(date(2026, 10, 9), config=cfg, log=lines.append, judge_clients=judges)
    s = r.sessions[0]
    assert s.judges["cosmos"].startswith("failed:") and s.judges["jev"].endswith("new verdicts") and s.judges["tag"].endswith("new tags")
    d = sessions / "20261009-100000" / "intent"
    assert (d / "jev.jsonl").exists() and (d / "tags.jsonl").exists() and not (d / "cosmos.jsonl").exists()
    assert s.exported > 0 and "fist" not in s.frames
    assert any("fist from a judge" in why for why in s.skipped)
    rev = next((data_dir / "datasets" / next(iter(s.frames))).glob("review-*"))
    assert all(json.loads(line)["source"] == "judge" for line in rev.read_text().splitlines())


def test_judges_from_env_skip_every_pass_without_credentials_and_never_use_the_fake_tagger(monkeypatch: pytest.MonkeyPatch):
    from visual_actions.intent import tagger as tagmod

    for k in ("NVIDIA_API_KEY", "TYPESAFE_API_KEY", "JEV_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(tagmod, "make_tagger", lambda: FakeTagger())
    lines: list[str] = []
    j = Judges.from_env(lines.append)
    assert j.cosmos is None and j.jev is None and j.tagger is None
    assert any("canned tags" in line for line in lines)
    monkeypatch.setattr(tagmod, "make_tagger", lambda: (_ for _ in ()).throw(RuntimeError("no codex")))
    assert Judges.from_env(lines.append).tagger is None and "no codex" in lines[-1]


# -- the pointer ----------------------------------------------------------------------------------


def test_promote_rollback_and_prune(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(paths, "data_dir", lambda: tmp_path)
    user = tmp_path / "models" / "user"
    user.mkdir(parents=True)
    names = [f"gestures-2026-10-0{i}-{i:08x}" for i in range(1, 8)]
    for n in names:
        (user / f"{n}.joblib").write_bytes(n.encode())
        (user / f"{n}.manifest.json").write_text("{}")
    assert paths.live_model_path() is None
    versions.promote(user, user / f"{names[0]}.joblib")
    versions.promote(user, user / f"{names[2]}.joblib")
    assert versions.current_name(user) == f"{names[2]}.joblib" and (user / "current.joblib").read_bytes() == names[2].encode()
    assert paths.live_model_path() == user / "current.joblib"
    assert versions.champion_path(user, tmp_path / "models" / "gestures.joblib") == user / f"{names[2]}.joblib"
    assert versions.rollback(user) == user / f"{names[0]}.joblib" and versions.current_name(user) == f"{names[0]}.joblib"
    assert versions.rollback(user) is None and versions.current_name(user) is None
    assert paths.live_model_path() is None  # back to rules: no shipped model in this tmp dir
    (tmp_path / "models" / "gestures.joblib").write_bytes(b"shipped")
    assert paths.live_model_path() == tmp_path / "models" / "gestures.joblib"
    assert versions.champion_path(user, tmp_path / "models" / "gestures.joblib") == tmp_path / "models" / "gestures.joblib"

    versions.promote(user, user / f"{names[1]}.joblib")  # the oldest-but-one is current: pruning must keep it
    removed = versions.prune(user, keep=3)
    assert [p.name for p in removed] == [f"{names[0]}.joblib", f"{names[2]}.joblib", f"{names[3]}.joblib", f"{names[4]}.joblib"]
    assert [p.name for p in versions.list_versions(user)] == [f"{names[1]}.joblib", f"{names[5]}.joblib", f"{names[6]}.joblib"]
    assert versions.current_name(user) == f"{names[1]}.joblib" and not (user / f"{names[0]}.manifest.json").exists()
    with pytest.raises(ValueError):
        versions.promote(user, tmp_path / "elsewhere.joblib")


def test_rollback_and_status_through_the_cli(data_dir: Path, capsys: pytest.CaptureFixture[str]):
    user = data_dir / "models" / "user"
    user.mkdir(parents=True)
    (user / "gestures-2026-10-09-abcdef01.joblib").write_bytes(b"x")
    versions.promote(user, user / "gestures-2026-10-09-abcdef01.joblib")
    assert learn_main(["status"]) == 0
    out = capsys.readouterr().out
    assert "current.joblib" in out and "gestures-2026-10-09-abcdef01" in out and "last run: none" in out
    assert learn_main(["rollback"]) == 0
    assert "was gestures-2026-10-09-abcdef01" in capsys.readouterr().out and versions.current_name(user) is None


# -- launchd --------------------------------------------------------------------------------------


def test_plist_content(tmp_path: Path):
    xml = schedule.plist_xml(2, python="/repo/.venv/bin/python3", cwd=Path("/repo"), log_dir=tmp_path / "logs")
    assert "<string>com.visual-actions.learn</string>" in xml
    assert "<string>/repo/.venv/bin/python3</string>" in xml and "<string>visual_actions.learn</string>" in xml and "<string>run</string>" in xml
    assert "<key>Hour</key>\n        <integer>2</integer>" in xml and "<integer>0</integer>" in xml
    assert f"<string>{tmp_path / 'logs' / 'learn.log'}</string>" in xml and "<string>/repo</string>" in xml
    import plistlib

    d = plistlib.loads(xml.encode())
    assert d["ProgramArguments"] == ["/repo/.venv/bin/python3", "-m", "visual_actions.learn", "run"] and d["StartCalendarInterval"] == {"Hour": 2, "Minute": 0}
    with pytest.raises(ValueError):
        schedule.plist_xml(24)


def test_install_uninstall_status_call_launchctl_through_the_runner(tmp_path: Path):
    calls: list[list[str]] = []

    class R:
        returncode = 0
        stdout = "\tstate = waiting\n\tlast exit code = 0\n"

    def runner(cmd: list[str], **kw: object) -> R:
        calls.append(cmd)
        return R()

    plist = tmp_path / "LaunchAgents" / "com.visual-actions.learn.plist"
    assert schedule.install(3, runner=runner, path=plist, python="/py", log_dir=tmp_path / "logs") == plist
    assert plist.exists() and "<integer>3</integer>" in plist.read_text() and (tmp_path / "logs").is_dir()
    assert [c[:2] for c in calls] == [["launchctl", "bootout"], ["launchctl", "bootstrap"]] and calls[1][-1] == str(plist)
    st = schedule.status(runner=runner, path=plist)
    assert st["installed"] and st["loaded"] and st["state"] == "waiting" and st["last_exit_code"] == "0"
    assert schedule.uninstall(runner=runner, path=plist) and not plist.exists() and calls[-1][:2] == ["launchctl", "bootout"]
    assert not schedule.uninstall(runner=runner, path=plist)
    assert schedule.status(runner=runner, path=plist) == {"plist": str(plist), "installed": False, "loaded": False}

"""Score a candidate classifier against the champion, and decide.

Two views, both on sessions the candidate never trained on:

(a) frame accuracy: the held-out sessions' review exports (human and judge labelled frames),
    weighted by each line's weight, per class, with fist recall kept apart (the fist is the
    cancel; it must never regress);
(b) engine metrics: each held-out session's landmarks replayed through the whole pipeline on
    fake time (tools/replay.replay_full, mock automation) once per model. The bus events are
    turned into intent.events.Event records in the exact shape session.py logs, so
    intent.segments and intent.labelfns run unchanged over the replay and count fires,
    weak-misfire and weak-intended fires, arms, empty arms, broken holds and fist-cancelled
    drags.

The gate (`decide`) promotes only when the candidate is at least as good on every rule in
LearnConfig: accuracy within `accuracy_margin` of the champion, no more weak-misfire fires,
at least `intended_keep` of the weak-intended fires, and fist recall not below the champion.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import Config, LearnConfig
from ..core.drag import DragEvent, DragPhase
from ..core.events import (
    ActionFired,
    Bus,
    HandLost,
    HandSeen,
    ModeChanged,
    PalmVetoed,
    TokenEmitted,
)
from ..core.recognizer import RuleRecognizer
from ..core.recorder import read_records
from ..core.types import HandFrame
from ..intent.events import Event, parse_events, session_t0_ns
from ..intent.schema import Segment, SegmentKind
from ..intent.segments import segment_session
from ..tools.replay import replay_full
from .trainer import Dataset

FIST = "fist"
MISFIRE_VOTES = ("fist_after_fire", "token_flip_around_fire", "immediate_redo")
EMPTY_ARM_OUTCOMES = ("timeout", "cancelled_by_fist", "lost")


# -- sessions ------------------------------------------------------------------------------


def session_date(stamp: str) -> date | None:
    try:
        return date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
    except ValueError:
        return None


def session_closed(session_dir: Path) -> bool:
    """The recorder wrote its summary line: the session is not being written any more."""
    log = session_dir / "events.log"
    if not log.exists():
        return False
    try:
        with log.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    return '{"summary"' in tail


def select_holdout(sessions: list[Path], day: date, n: int) -> list[Path]:
    """The `n` newest replayable sessions from days other than `day`, newest first."""
    out: list[Path] = []
    for s in sorted(sessions, key=lambda p: p.name, reverse=True):
        d = session_date(s.name)
        if d is None or d == day or not (s / "landmarks.jsonl").exists() or not (s / "events.log").exists():
            continue
        out.append(s)
        if len(out) >= n:
            break
    return out


# -- replay -> events ---------------------------------------------------------------------


def model_config(config: Config, model_path: Path | None) -> Config:
    cfg = copy.deepcopy(config)
    cfg.recognizer.model = str(model_path) if model_path else None
    return cfg


def replay_events(session_dir: Path, config: Config) -> list[Event]:
    """Replay the session's landmarks through the pipeline built from `config` and return the
    events the session recorder would have logged, on the session's elapsed clock."""
    session_dir = Path(session_dir)
    landmarks = session_dir / "landmarks.jsonl"
    log_path = session_dir / "events.log"
    t0_ns = session_t0_ns(parse_events(log_path), landmarks) if log_path.exists() else None
    if t0_ns is None:
        first = next((r for r, _ in read_records(landmarks) if isinstance(r, HandFrame)), None)
        t0_ns = first.t_ns if first is not None else 0

    out: list[Event] = [Event(0.0, "", "session", f"replay t0_ns={t0_ns}")]
    state = {"mode": "idle", "present": False, "lost_at": None}

    def add(kind: str, text: str, t_ns: int) -> None:
        out.append(Event(round((t_ns - t0_ns) / 1e9, 3), "", kind, text))

    def on_mode(e: ModeChanged) -> None:
        state["mode"] = e.new
        add("mode", f"{e.old} -> {e.new}" + (f" [{e.namespace}]" if e.namespace else ""), e.t_ns)

    def on_token(e: TokenEmitted) -> None:
        t = e.token
        add("token", f"{t.name} conf={t.confidence:.2f} still={t.still}", t.t_ns)

    def on_action(e: ActionFired) -> None:
        add("action", f"{e.action.name} {'ok' if e.ok else 'FAILED ' + e.message}", e.t_ns)

    def on_drag(e: DragEvent) -> None:
        if e.phase is DragPhase.MOVE:
            return
        extra = f" snapped={e.snapped}" if e.snapped else ""
        if e.focus is not None:
            extra += f" focus[{e.focus}]"
        add("drag", f"{e.phase.value} {e.window} @({e.x:.0f},{e.y:.0f}){extra}", e.t_ns)

    def on_lost(e: HandLost) -> None:
        state["present"] = False
        state["lost_at"] = e.t_ns
        add("hand", f"lost [{e.reason}] {e.detail} (mode {state['mode']})", e.t_ns)

    def on_seen(e: HandSeen) -> None:
        if not state["present"]:
            hf = e.hand_frame
            lost_at = state["lost_at"]
            gap = f" after {(hf.t_ns - lost_at) / 1e9:.2f}s away" if lost_at else ""
            add("hand", f"seen{gap} conf={hf.confidence:.2f} wrist=({hf.landmarks[0].x:.2f},{hf.landmarks[0].y:.2f}) (mode {state['mode']})", hf.t_ns)
        state["present"] = True

    def on_veto(e: PalmVetoed) -> None:
        add("veto", f"palm on face: overlap={e.overlap:.2f} spread={e.spread:.2f}", e.t_ns)

    bus = Bus()
    bus.subscribe(ModeChanged, on_mode)
    bus.subscribe(TokenEmitted, on_token)
    bus.subscribe(ActionFired, on_action)
    bus.subscribe(DragEvent, on_drag)
    bus.subscribe(HandLost, on_lost)
    bus.subscribe(HandSeen, on_seen)
    bus.subscribe(PalmVetoed, on_veto)
    replay_full(landmarks, config, bus=bus)
    out.sort(key=lambda e: e.t)  # the session line first; the bus delivers in time order anyway
    return out


# -- engine metrics -----------------------------------------------------------------------


@dataclass
class EngineMetrics:
    sessions: int = 0
    fires: int = 0
    weak_misfire_fires: int = 0  # fist_after_fire / token_flip_around_fire / immediate_redo voted misfire
    weak_intended_fires: int = 0  # combined weak label = intended
    arms: int = 0
    empty_arms: int = 0  # armed, then timeout / fist / hand lost with no command
    broken_holds: int = 0
    drags: int = 0
    fist_cancelled_drags: int = 0

    def add(self, other: EngineMetrics) -> None:
        for k, v in asdict(other).items():
            setattr(self, k, getattr(self, k) + v)


def engine_metrics(segments: list[Segment]) -> EngineMetrics:
    m = EngineMetrics(sessions=1)
    for s in segments:
        if s.kind is SegmentKind.FIRE:
            m.fires += 1
            if any(s.labelfn_votes.get(v) == "misfire" for v in MISFIRE_VOTES):
                m.weak_misfire_fires += 1
            if s.weak_label == "intended":
                m.weak_intended_fires += 1
        elif s.kind is SegmentKind.ARM:
            m.arms += 1
            if s.engine_outcome in EMPTY_ARM_OUTCOMES:
                m.empty_arms += 1
        elif s.kind is SegmentKind.BROKEN_HOLD:
            m.broken_holds += 1
        elif s.kind is SegmentKind.DRAG:
            m.drags += 1
            if s.engine_outcome == "dropped_by_fist":
                m.fist_cancelled_drags += 1
    return m


def replay_metrics(sessions: list[Path], config: Config, model_path: Path | None, log: Callable[[str], None] | None = None) -> EngineMetrics:
    cfg = model_config(config, model_path)
    total = EngineMetrics()
    for s in sessions:
        events = replay_events(s, cfg)
        m = engine_metrics(segment_session(s.name, events))
        if log:
            log(f"  replay {s.name} [{model_path.name if model_path else 'rules'}]: fires={m.fires} misfire={m.weak_misfire_fires} intended={m.weak_intended_fires} arms={m.arms} empty={m.empty_arms} broken={m.broken_holds}")
        total.add(m)
    return total


# -- frame metrics ------------------------------------------------------------------------


@dataclass
class ClassScore:
    n: int = 0
    weight: float = 0.0
    accuracy: float | None = None  # weighted


@dataclass
class FrameMetrics:
    frames: int = 0
    weight: float = 0.0
    accuracy: float | None = None  # weighted, over every held-out frame
    fist_frames: int = 0
    fist_recall: float | None = None  # weighted recall on held-out fist frames
    per_class: dict[str, ClassScore] = field(default_factory=dict)
    per_session: dict[str, float] = field(default_factory=dict)


def classify_frames(model_path: Path | None, ds: Dataset) -> np.ndarray:
    """Predicted class per held-out frame: the joblib pipeline, or the rules when there is no model."""
    if len(ds) == 0:
        return np.array([], dtype=str)
    if model_path is not None:
        import joblib

        return np.asarray(joblib.load(model_path).predict(ds.x)).astype(str)
    rules = RuleRecognizer()
    return np.array([rules.classify(hf)[0] for hf in ds.frames], dtype=str)


def frame_metrics(model_path: Path | None, ds: Dataset) -> FrameMetrics:
    fm = FrameMetrics(frames=len(ds), weight=float(ds.w.sum()) if len(ds) else 0.0)
    if len(ds) == 0:
        return fm
    pred = classify_frames(model_path, ds)
    hit = (pred == ds.y).astype(float)
    fm.accuracy = round(float(np.average(hit, weights=ds.w)), 4)
    for cls in sorted(set(ds.y.tolist())):
        sel = ds.y == cls
        fm.per_class[cls] = ClassScore(int(sel.sum()), round(float(ds.w[sel].sum()), 2), round(float(np.average(hit[sel], weights=ds.w[sel])), 4))
    for g in sorted(set(ds.groups.tolist())):
        sel = ds.groups == g
        fm.per_session[g] = round(float(np.average(hit[sel], weights=ds.w[sel])), 4)
    fist = ds.y == FIST
    fm.fist_frames = int(fist.sum())
    if fm.fist_frames:
        fm.fist_recall = round(float(np.average(hit[fist], weights=ds.w[fist])), 4)
    return fm


# -- the gate -----------------------------------------------------------------------------


@dataclass
class Scores:
    model: str  # file name, or "rules"
    frame: FrameMetrics
    engine: EngineMetrics


@dataclass
class Rule:
    name: str
    passed: bool
    detail: str


@dataclass
class Decision:
    promote: bool
    rules: list[Rule]
    reason: str


def decide(champion: Scores, candidate: Scores, cfg: LearnConfig) -> Decision:
    """The promotion gate. Every rule must pass; a rule with nothing to measure passes with
    a note, except that no held-out session at all is a refusal."""
    rules: list[Rule] = []
    c, k = champion, candidate
    if k.engine.sessions == 0:
        return Decision(False, [Rule("holdout", False, "no held-out session to replay")], "no held-out sessions: nothing to compare against")
    if k.frame.accuracy is None or c.frame.accuracy is None:
        rules.append(Rule("accuracy", True, "no held-out labelled frames (not measured)"))
    else:
        ok = k.frame.accuracy >= c.frame.accuracy - cfg.accuracy_margin
        rules.append(Rule("accuracy", ok, f"candidate {k.frame.accuracy:.4f} vs champion {c.frame.accuracy:.4f} (margin {cfg.accuracy_margin})"))
    ok = k.engine.weak_misfire_fires <= c.engine.weak_misfire_fires
    rules.append(Rule("misfires", ok, f"weak-misfire fires {k.engine.weak_misfire_fires} vs {c.engine.weak_misfire_fires}"))
    need = cfg.intended_keep * c.engine.weak_intended_fires
    ok = k.engine.weak_intended_fires >= need
    rules.append(Rule("intended", ok, f"weak-intended fires {k.engine.weak_intended_fires} vs {c.engine.weak_intended_fires} (need >= {need:.1f})"))
    if k.frame.fist_recall is None or c.frame.fist_recall is None:
        rules.append(Rule("fist_recall", True, "no held-out fist frames (not measured)"))
    else:
        ok = k.frame.fist_recall >= c.frame.fist_recall
        rules.append(Rule("fist_recall", ok, f"fist recall {k.frame.fist_recall:.4f} vs {c.frame.fist_recall:.4f}"))
    failed = [r.name for r in rules if not r.passed]
    return Decision(not failed, rules, "promote: every gate passed" if not failed else "discard: failed " + ", ".join(failed))


def score(model_path: Path | None, holdout: Dataset, sessions: list[Path], config: Config, log: Callable[[str], None] | None = None) -> Scores:
    return Scores(model_path.name if model_path else "rules", frame_metrics(model_path, holdout), replay_metrics(sessions, config, model_path, log))


def scores_dict(s: Scores) -> dict[str, Any]:
    return {"model": s.model, "frame": asdict(s.frame), "engine": asdict(s.engine)}

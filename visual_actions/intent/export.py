"""Labelled moments -> training data.

Two outputs per session:

  datasets/<class>/review-<session>-<stamp>.jsonl   landmark frames in the recorder format, one
      file per gesture class, each line carrying "source" ("human" | "judge") and "weight" on
      top of the recorder keys (tools/train.py's loader reads t_ns/hand/landmarks/confidence
      and ignores the rest). Class = the true gesture for intended / missed moments, "none"
      for misfires and no-event moments (an unintended shape is a none example), nothing for
      ambiguous ones. Fist frames come from human labels only: fist is the cancel.
  datasets/intent/<session>.jsonl   one line per labelled segment for the intent gate.

Human labels are taken as they are. A judge tag (tags.jsonl, needs_human false) is taken only
under JudgeRules, which are deliberately stricter than the human path: the first Cosmos pass
showed that exporting every confident tag as a half-second of frames under one class sweeps
transition and pinch frames into the wrong classes and weighs a judge like a person.
  - confidence >= min_conf (0.85), never a drag (a drag's frames are pinches in motion);
  - intended: the tag's true_gesture must equal the segment's engine_gesture and the Cosmos
    verdict, when there is one, must say a still command; frames come from +-window_s (0.25 s)
    around the engine event of a fire / arm segment, not the half second a human label gets;
  - misfire / no_event -> none: Cosmos must call the hand incidental or dead, or the tag must
    be >= none_min_conf (0.9);
  - missed -> gesture: confidence >= missed_min_conf (0.9) and a Cosmos command verdict;
  - weight = weight (0.3) x confidence; fist never.

`labelfn_accuracy` scores each labelling function against the human labels, for reweighting.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.recorder import read_session, to_json
from ..core.types import HandFrame
from .events import parse_events, session_t0_ns
from .labelfns import ARM_FNS, FIRE_FNS
from .schema import CosmosVerdict, HumanLabel, Intent, Motion, Segment, SegmentKind, Tag, Verdict, by_id, read_jsonl
from .segments import CONTEXT_AFTER_S

FIRE_WINDOW_S = 0.5  # frames this close to the engine event of a fire segment are the shape (human labels)
ARM_EVENT_OFFSET_S = 0.5 + 0.75  # an arm segment opens 0.5 s before the hold; the leader is held by 0.75 s in
JUDGE_MIN_CONF = 0.85
NONE = "none"
FIST = "fist"


@dataclass(frozen=True)
class JudgeRules:
    """When a judge tag becomes training frames (see the module docstring)."""

    min_conf: float = JUDGE_MIN_CONF  # below this a tag is not a label at all
    none_min_conf: float = 0.9  # misfire / no_event -> none without a Cosmos incidental / dead verdict
    missed_min_conf: float = 0.9  # missed -> gesture (with a Cosmos command verdict)
    weight: float = 0.3  # x tag confidence; a human row weighs 1.0
    window_s: float = 0.25  # +- around the engine event of a fire / arm segment
    none_rows: bool = True  # export misfire / no_event tags as none at all
    missed_rows: bool = True  # export missed tags as their gesture at all


def _classes() -> frozenset[str]:
    from ..tools.train import CLASSES  # the allow-list lives with the trainer

    return CLASSES


@dataclass
class Labelled:
    """One segment with the label chosen for export."""

    segment: Segment
    verdict: Verdict
    intent: str
    motion: str | None
    true_gesture: str | None
    source: str  # human | judge
    weight: float
    window_s: float = FIRE_WINDOW_S  # +- around the engine event (fire; arm too for judge rows)
    arm_window: bool = False  # window an arm segment around its hold as well (judge rows)

    @property
    def gesture_class(self) -> str | None:
        if self.verdict in (Verdict.INTENDED, Verdict.MISSED):
            if self.true_gesture:
                return self.true_gesture
            return self.segment.engine_gesture if self.verdict is Verdict.INTENDED else None
        if self.verdict in (Verdict.MISFIRE, Verdict.NO_EVENT):
            return NONE
        return None  # ambiguous

    def span(self) -> tuple[float, float]:
        s = self.segment
        t_event: float | None = None
        if s.kind is SegmentKind.FIRE:
            t_event = s.t1 - CONTEXT_AFTER_S
        elif s.kind is SegmentKind.ARM and self.arm_window:
            t_event = s.t0 + ARM_EVENT_OFFSET_S
        if t_event is None:
            return s.t0, s.t1
        return max(s.t0, t_event - self.window_s), min(s.t1, t_event + self.window_s)


@dataclass
class ExportSummary:
    session: str
    files: dict[str, Path] = field(default_factory=dict)  # class -> file
    frames: dict[str, int] = field(default_factory=dict)  # class -> frames written
    intent_file: Path | None = None
    segments: int = 0
    skipped: list[str] = field(default_factory=list)  # "<segment_id>: why"


def judge_label(seg: Segment, t: Tag, cosmos: CosmosVerdict | None, rules: JudgeRules) -> Labelled | str:
    """The Labelled row a judge tag becomes under `rules`, or the reason it is refused."""
    if t.needs_human:
        return "judge: needs a human"
    if t.confidence < rules.min_conf:
        return f"judge: confidence {t.confidence:.2f} < {rules.min_conf}"
    if seg.kind is SegmentKind.DRAG:
        return "judge: drag frames are pinches in motion, humans only"
    if t.true_gesture == FIST:
        return "fist from a judge, humans only"
    v = t.verdict
    if v is Verdict.INTENDED:
        if not t.true_gesture or t.true_gesture != seg.engine_gesture:
            return f"judge: intended but true_gesture {t.true_gesture!r} != engine {seg.engine_gesture!r}"
        if cosmos is not None and not (cosmos.intent is Intent.COMMAND and cosmos.motion is Motion.STILL):
            return f"judge: intended but cosmos saw {cosmos.intent.value} / {cosmos.motion.value}"
    elif v in (Verdict.MISFIRE, Verdict.NO_EVENT):
        if not rules.none_rows:
            return f"judge: {v.value} -> none rows are off"
        incidental = cosmos is not None and cosmos.intent in (Intent.INCIDENTAL, Intent.DEAD)
        if not incidental and t.confidence < rules.none_min_conf:
            return f"judge: {v.value} -> none needs cosmos incidental / dead or confidence >= {rules.none_min_conf}"
    elif v is Verdict.MISSED:
        if not rules.missed_rows:
            return "judge: missed rows are off"
        if t.confidence < rules.missed_min_conf:
            return f"judge: missed needs confidence >= {rules.missed_min_conf}"
        if cosmos is None or cosmos.intent is not Intent.COMMAND:
            return "judge: missed needs a cosmos command verdict"
    weight = round(rules.weight * t.confidence, 3)
    return Labelled(seg, v, t.intent.value, t.motion.value, t.true_gesture, "judge", weight, window_s=rules.window_s, arm_window=True)


def collect_labels(
    session_dir: Path,
    include_judge: bool = False,
    judge_min_conf: float | None = None,
    rules: JudgeRules | None = None,
    skipped: list[str] | None = None,
) -> list[Labelled]:
    """Human labels, plus the judge tags that pass `rules` (humans win a segment). `judge_min_conf`
    overrides rules.min_conf (the learn config carries it); refused judge rows are explained
    into `skipped` when given."""
    rules = rules or JudgeRules()
    if judge_min_conf is not None:
        rules = replace(rules, min_conf=judge_min_conf)
    d = session_dir / "intent"
    segments = by_id(read_jsonl(d / "segments.jsonl", Segment))
    tags = by_id(read_jsonl(d / "tags.jsonl", Tag))
    human = by_id(read_jsonl(d / "human.jsonl", HumanLabel))
    out: list[Labelled] = []
    for sid, h in human.items():
        seg = segments.get(sid)
        if seg is None:
            continue
        out.append(Labelled(seg, h.verdict, h.intent.value, h.motion.value if h.motion else None, h.true_gesture, "human", 1.0))
    if include_judge:
        cosmos = by_id(read_jsonl(d / "cosmos.jsonl", CosmosVerdict))
        for sid, t in tags.items():
            seg = segments.get(sid)
            if seg is None or sid in human:
                continue
            lab = judge_label(seg, t, cosmos.get(sid), rules)
            if isinstance(lab, str):
                if skipped is not None:
                    skipped.append(f"{sid}: {lab}")
                continue
            out.append(lab)
    out.sort(key=lambda x: x.segment.t0)
    return out


def export_labels(
    session_dir: Path,
    datasets_dir: Path,
    out_prefix: str = "review",
    include_judge: bool = False,
    judge_min_conf: float | None = None,
    stamp: str | None = None,
    rules: JudgeRules | None = None,
) -> ExportSummary:
    session_dir, datasets_dir = Path(session_dir), Path(datasets_dir)
    session = session_dir.name
    summary = ExportSummary(session)
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    classes = _classes()
    labelled = collect_labels(session_dir, include_judge, judge_min_conf, rules, summary.skipped)

    events_path, landmarks = session_dir / "events.log", session_dir / "landmarks.jsonl"
    t0_ns = session_t0_ns(parse_events(events_path), landmarks) if events_path.exists() else None
    if t0_ns is None:
        summary.skipped.append("session: no t0_ns (events.log / landmarks.jsonl missing), no frames exported")
    all_frames = list(read_session(landmarks)) if t0_ns is not None and landmarks.exists() else []
    hand_frames = [f for f in all_frames if isinstance(f, HandFrame)]

    handles: dict[str, Any] = {}
    intent_rows: list[dict[str, Any]] = []
    try:
        for lab in labelled:
            cls = lab.gesture_class
            t0, t1 = lab.span()
            frames: list[HandFrame] = []
            if t0_ns is not None:
                lo, hi = t0_ns + int(t0 * 1e9), t0_ns + int(t1 * 1e9)
                frames = [f for f in hand_frames if lo <= f.t_ns <= hi]
            intent_rows.append(
                {
                    "segment_id": lab.segment.segment_id,
                    "kind": lab.segment.kind.value,
                    "t0": lab.segment.t0,
                    "t1": lab.segment.t1,
                    "intent": lab.intent,
                    "motion": lab.motion,
                    "verdict": lab.verdict.value,
                    "source": lab.source,
                    "weight": lab.weight,
                    "frame_count": len(frames),
                }
            )
            if cls is None:
                summary.skipped.append(f"{lab.segment.segment_id}: {lab.verdict.value}, no class")
                continue
            if cls not in classes:
                summary.skipped.append(f"{lab.segment.segment_id}: '{cls}' is not a dataset class")
                continue
            if cls == FIST and lab.source != "human":
                summary.skipped.append(f"{lab.segment.segment_id}: fist from a judge, humans only")
                continue
            if not frames:
                summary.skipped.append(f"{lab.segment.segment_id}: no landmark frames in {t0:.2f}-{t1:.2f}s")
                continue
            if cls not in handles:
                path = datasets_dir / cls / f"{out_prefix}-{session}-{stamp}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                handles[cls] = path.open("w", encoding="utf-8")
                summary.files[cls] = path
            f = handles[cls]
            for fr in frames:
                f.write(json.dumps({**to_json(fr), "source": lab.source, "weight": lab.weight}) + "\n")
            summary.frames[cls] = summary.frames.get(cls, 0) + len(frames)
    finally:
        for f in handles.values():
            f.close()

    if intent_rows:
        intent_path = datasets_dir / "intent" / f"{session}.jsonl"
        intent_path.parent.mkdir(parents=True, exist_ok=True)
        with intent_path.open("w", encoding="utf-8") as f:
            for row in intent_rows:
                f.write(json.dumps(row) + "\n")
        summary.intent_file = intent_path
    summary.segments = len(intent_rows)
    return summary


def print_summary(s: ExportSummary) -> None:
    print(f"{s.session}: {s.segments} labelled segment(s)" + (f" -> {s.intent_file}" if s.intent_file else ""))
    for cls, path in sorted(s.files.items()):
        print(f"  {cls:<12} {s.frames.get(cls, 0):>6} frames  {path}")
    for why in s.skipped:
        print(f"  skipped {why}")


# -- labelling function accuracy ----------------------------------------------------------


@dataclass
class FnScore:
    name: str
    label: str  # the label the function votes
    kind: str  # fire | arm
    votes: int = 0  # times it fired on a human-labelled segment of its kind
    correct: int = 0  # ... and the human agreed
    truth: int = 0  # human labels equal to `label` among segments of its kind
    labelled: int = 0  # human-labelled segments of its kind

    @property
    def precision(self) -> float | None:
        return self.correct / self.votes if self.votes else None

    @property
    def recall(self) -> float | None:
        return self.correct / self.truth if self.truth else None

    @property
    def coverage(self) -> float:
        return self.votes / self.labelled if self.labelled else 0.0


def _fn_kinds() -> dict[str, str]:
    kinds = {name: "fire" for name in FIRE_FNS}
    kinds.update({name: "arm" for name in ARM_FNS})
    return kinds


def labelfn_accuracy(session_dirs: Iterable[Path], out: Any = sys.stdout) -> dict[str, FnScore]:
    """Score every labelling function against the human labels of the given sessions, print a
    table, return the scores by function name."""
    kinds = _fn_kinds()
    scores: dict[str, FnScore] = {}
    scored = {Verdict.INTENDED.value, Verdict.MISFIRE.value, Verdict.MISSED.value}
    for sd in session_dirs:
        d = Path(sd) / "intent"
        segments = by_id(read_jsonl(d / "segments.jsonl", Segment))
        for sid, h in by_id(read_jsonl(d / "human.jsonl", HumanLabel)).items():
            seg = segments.get(sid)
            if seg is None or h.verdict.value not in scored:
                continue
            seg_kind = "fire" if seg.kind is SegmentKind.FIRE else "arm" if seg.kind is SegmentKind.ARM else None
            if seg_kind is None:
                continue
            truth = h.verdict.value
            for name, fn_kind in kinds.items():
                if fn_kind != seg_kind:
                    continue
                vote = seg.labelfn_votes.get(name)
                label = vote or _fn_label(name)
                sc = scores.setdefault(name, FnScore(name, label, fn_kind))
                sc.labelled += 1
                if truth == sc.label:
                    sc.truth += 1
                if vote is not None:
                    sc.votes += 1
                    if vote == truth:
                        sc.correct += 1
    rows = sorted(scores.values(), key=lambda s: (s.kind, s.name))
    fmt = lambda v: "   —  " if v is None else f"{v:6.2f}"  # noqa: E731
    print(f"{'labelling function':<26}{'kind':<6}{'label':>9}{'votes/n':>9}{'precision':>10}{'recall':>8}{'coverage':>9}", file=out)
    for s in rows:
        print(f"{s.name:<26}{s.kind:<6}{s.label:>9}{s.votes:>5}/{s.labelled:<3}{fmt(s.precision):>10}{fmt(s.recall):>8}{s.coverage:>9.2f}", file=out)
    if not rows:
        print("no human labels on fire / arm segments yet", file=out)
    return scores


def _fn_label(name: str) -> str:
    """The label a function votes, by name: the intended ones say so, the rest vote misfire."""
    return "intended" if name in ("clean_command", "arm_then_command") else "misfire"

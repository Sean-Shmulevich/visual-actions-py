"""Records shared by every pass of the intent pipeline. Plain dataclasses, JSONL on disk.

A Segment is a span of one session. Each pass attaches its verdict to the segment's id:
cosmos.jsonl, jev.jsonl, tags.jsonl, human.jsonl all key on `segment_id`, so a pass can
be re-run or skipped and the join is by id, never by position.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar


class SegmentKind(StrEnum):
    ARM = "arm"  # a leader hold that reached ARMED (whatever came after)
    FIRE = "fire"  # a command fired (key chord, media verb, open)
    DRAG = "drag"  # a pinch-drag from grab to release
    REPEAT = "repeat"  # a slide-to-repeat window
    ADJUST = "adjust"  # a media pinch-adjust (volume)
    BROKEN_HOLD = "broken_hold"  # a hold that never armed
    HAND = "hand"  # hand in view, no engine activity: incidental or an unrecognised attempt
    DEAD = "dead"  # no hand for a while: dead footage


class Intent(StrEnum):
    """What the person was doing, as a judge sees it."""

    COMMAND = "command"  # a deliberate gesture at the camera
    INCIDENTAL = "incidental"  # hand in view doing something else: typing, face, drink, talking
    DEAD = "dead"  # nobody / no hand
    UNSURE = "unsure"


class Motion(StrEnum):
    STILL = "still"
    MOVING = "moving"  # hand travels but the shape is the point (e.g. carrying a pinch)
    SWIPE = "swipe"  # a stroke across the frame is the gesture
    SLIDE = "slide"  # a sideways step of a held shape (fingerspelling repeat)
    PINCH_DRAG = "pinch_drag"


class Verdict(StrEnum):
    """The final tag of an engine event, against the person's intent."""

    INTENDED = "intended"  # the engine did what the person wanted
    MISFIRE = "misfire"  # the engine acted, the person did not mean it
    MISSED = "missed"  # the person tried, the engine did not act (or acted late / wrong)
    NO_EVENT = "no_event"  # nothing to judge: incidental or dead footage, engine stayed quiet
    AMBIGUOUS = "ambiguous"


@dataclass
class Segment:
    segment_id: str  # "<session>/<kind>/<index>"
    session: str  # session folder name
    kind: SegmentKind
    t0: float  # elapsed seconds in the session (the events.log clock)
    t1: float
    # What the engine thought. Gesture names are the engine's own; judges never see them.
    engine_gesture: str | None = None  # token that armed / fired / was slid
    engine_action: str | None = None  # action name that fired
    engine_outcome: str | None = None  # "fired", "cancelled_by_fist", "timeout", "dropped", "lost", ...
    engine_namespace: str | None = None
    mean_confidence: float | None = None  # of the tokens that drove the event
    hand_present_fraction: float | None = None  # frames with a hand / expected frames in the span
    labelfn_votes: dict[str, str] = field(default_factory=dict)  # labelling function -> vote
    weak_label: str | None = None  # combined vote: intended | misfire | missed | unsure
    weak_weight: float = 0.0
    notes: str = ""


@dataclass
class CosmosVerdict:
    """Pass 1: what the video model saw. No engine vocabulary in here."""

    segment_id: str
    person_present: bool
    hand_present: bool
    attention_to_screen: str  # "yes" | "partly" | "no" | "unknown"
    arm_raised_toward_camera: bool
    face_touched: bool
    intent: Intent
    motion: Motion
    hand_description: str  # physical description: which fingers are out, orientation, travel
    reasoning: str
    confidence: float  # 0..1
    sub_spans: list[dict[str, Any]] = field(default_factory=list)  # [{t0, t1, intent}] when the clip splits
    model: str = ""
    raw: str = ""  # the model's text, for audit


@dataclass
class JevVerdict:
    """Pass 2a: calibrated atomic answers over the text state."""

    segment_id: str
    answers: dict[str, float]  # question id -> probability the statement is true
    choice: str | None = None  # the Choice question's pick over the gesture set (+ "none")
    choice_probs: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    model: str = ""


@dataclass
class Tag:
    """Pass 2b: the reasoning model's tag, after seeing everything above."""

    segment_id: str
    verdict: Verdict
    intent: Intent
    motion: Motion
    true_gesture: str | None  # engine vocabulary, mapped from the physical description; None = none/unknown
    tags: list[str] = field(default_factory=list)  # free tags: "face_touch", "typing", "fast_exit", ...
    confidence: float = 0.0
    needs_human: bool = True
    reason: str = ""
    model: str = ""


@dataclass
class HumanLabel:
    segment_id: str
    verdict: Verdict
    intent: Intent
    true_gesture: str | None = None
    motion: Motion | None = None
    note: str = ""
    at: str = ""  # ISO timestamp


T = TypeVar("T")


def _coerce(cls: type[T], d: dict[str, Any]) -> T:
    kw: dict[str, Any] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        if f.name not in d:
            continue
        v = d[f.name]
        t = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", "")
        if v is not None:
            if "SegmentKind" in str(t):
                v = SegmentKind(v)
            elif "Verdict" in str(t) and "Cosmos" not in str(t) and "Jev" not in str(t):
                v = Verdict(v)
            elif "Intent" in str(t):
                v = Intent(v)
            elif "Motion" in str(t):
                v = Motion(v)
        kw[f.name] = v
    return cls(**kw)


def write_jsonl(path: Path, records: Iterable[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_jsonl(path: Path, cls: type[T]) -> Iterator[T]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield _coerce(cls, json.loads(line))


def by_id(records: Iterable[Any]) -> dict[str, Any]:
    """Last record per segment_id wins, so an appended re-run overrides."""
    out: dict[str, Any] = {}
    for r in records:
        out[r.segment_id] = r
    return out

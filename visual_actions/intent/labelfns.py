"""Weak labels from the event log alone (Snorkel-style labelling functions).

Each function looks at one engine event in its context and votes `intended`, `misfire`,
`missed` or abstains (None). Votes are combined by fixed weights here; once human labels
exist, `export.py` can fit the weights (a label model) from agreement with them.

Repeatable actions (slide-to-repeat, volume steps) are excluded from the "same action
again" functions: repeating them is the point.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .events import Event

FIST_AFTER_S = 3.0
REDO_WINDOW_S = 4.0
HUMAN_WRONG_S = 10.0
LOW_CONF = 0.7
CLEAN_CONF = 0.9
REPEATABLE_ACTIONS = ("Previous tab", "Next tab", "Volume up", "Volume down")


@dataclass(frozen=True)
class Vote:
    name: str
    label: str  # intended | misfire | missed
    weight: float


Fn = Callable[[list[Event], int], str | None]


def _window(events: list[Event], i: int, before_s: float, after_s: float) -> list[Event]:
    t = events[i].t
    return [e for e in events if t - before_s <= e.t <= t + after_s]


def _after(events: list[Event], i: int, after_s: float) -> list[Event]:
    """Events strictly after index i (same-timestamp lines before it are not 'after')."""
    t = events[i].t
    out = []
    for e in events[i + 1 :]:
        if e.t > t + after_s:
            break
        out.append(e)
    return out


def _last_token_before(events: list[Event], i: int) -> Event | None:
    for j in range(i - 1, max(-1, i - 8), -1):
        if events[j].kind == "token":
            return events[j]
    return None


# -- fires ---------------------------------------------------------------------------------


def fist_after_fire(events: list[Event], i: int) -> str | None:
    """A fist within 3 s of a command is the person cancelling the menu, most often after a
    command they did not want (13 % of non-volume fires in the 2026-10-08 sessions)."""
    e = events[i]
    for x in _after(events, i, FIST_AFTER_S):
        if x.kind == "token" and x.token == "fist":
            return "misfire"
        if x.kind == "action":
            return None  # another command came first: the menu was wanted
    return None


def immediate_redo(events: list[Event], i: int) -> str | None:
    """The same non-repeatable command again within 4 s: the first one landed wrong (or did
    not land where expected). Weak."""
    e = events[i]
    if e.action_name in REPEATABLE_ACTIONS:
        return None
    for x in _after(events, i, REDO_WINDOW_S):
        if x.kind == "action" and x.action_name == e.action_name:
            return "misfire"
    return None


def low_confidence_trigger(events: list[Event], i: int) -> str | None:
    t = _last_token_before(events, i)
    if t is not None and t.confidence is not None and t.confidence < LOW_CONF:
        return "misfire"
    return None


def token_flip_around_fire(events: list[Event], i: int) -> str | None:
    """The shape changed within half a second on either side of the fire: a transition was
    read as a command."""
    trig = _last_token_before(events, i)
    if trig is None:
        return None
    for x in _window(events, i, 0.5, 0.5):
        if x.kind == "token" and x.token not in (trig.token, "none") and x is not trig:
            return "misfire"
    return None


def face_veto_near(events: list[Event], i: int) -> str | None:
    for x in _window(events, i, 1.0, 1.0):
        if x.kind == "veto":
            return "misfire"
    return None


def clean_command(events: list[Event], i: int) -> str | None:
    """Confident trigger, no fist and no redo for 5 s: the person got what they asked for."""
    e = events[i]
    trig = _last_token_before(events, i)
    if trig is None or trig.confidence is None or trig.confidence < CLEAN_CONF:
        return None
    for x in _after(events, i, 5.0):
        if x.kind == "token" and x.token == "fist":
            return None
        if x.kind == "action" and x.action_name == e.action_name and e.action_name not in REPEATABLE_ACTIONS:
            return None
    return "intended"


# -- the human marker (fires and arms) -----------------------------------------------------


def human_said_wrong(events: list[Event], i: int) -> str | None:
    """The person pressed "that was wrong" in the menu bar: a `human` line (`last action
    wrong`, or `last arm wrong` when nothing fired) within 10 s of the event and before any
    later fire. The strongest weak label there is, so it outweighs every heuristic."""
    for x in _after(events, i, HUMAN_WRONG_S):
        if x.kind == "human" and "wrong" in x.text:
            return "misfire"
        if x.kind == "action":
            return None  # a later command; the marker, if any, is about that one
    return None


# -- arms ----------------------------------------------------------------------------------


def arm_outcome(events: list[Event], i: int, horizon_s: float = 15.0) -> tuple[str, float]:
    """For a `holding -> armed` event: (outcome, t_end). Outcomes: command, drag, adjust,
    cancelled_by_fist, timeout, lost."""
    t = events[i].t
    for x in events[i + 1 :]:
        if x.t - t > horizon_s:
            break
        if x.kind == "action":
            return "command", x.t
        if x.kind == "mode" and x.mode_from == "armed" and x.mode_to in ("dragging", "adjust", "repeat"):
            return {"dragging": "drag", "adjust": "adjust", "repeat": "command"}[x.mode_to], x.t
        if x.kind == "mode" and x.mode_from == "armed" and x.mode_to == "idle":
            prev = events[events.index(x) - 1]
            if prev.kind == "token" and prev.token == "fist":
                return "cancelled_by_fist", x.t
            if prev.kind == "hand" and not prev.hand_seen:
                return "lost", x.t
            return "timeout", x.t
    return "timeout", t + horizon_s


def arm_then_fist(events: list[Event], i: int) -> str | None:
    return "misfire" if arm_outcome(events, i)[0] == "cancelled_by_fist" else None


def arm_timed_out(events: list[Event], i: int) -> str | None:
    return "misfire" if arm_outcome(events, i)[0] == "timeout" else None


def arm_then_command(events: list[Event], i: int) -> str | None:
    return "intended" if arm_outcome(events, i)[0] in ("command", "drag", "adjust") else None


FIRE_FNS: dict[str, tuple[Fn, float]] = {
    "fist_after_fire": (fist_after_fire, 0.6),
    "immediate_redo": (immediate_redo, 0.3),
    "low_confidence_trigger": (low_confidence_trigger, 0.3),
    "token_flip_around_fire": (token_flip_around_fire, 0.4),
    "face_veto_near": (face_veto_near, 0.5),
    "clean_command": (clean_command, 0.7),
    "human_said_wrong": (human_said_wrong, 1.0),
}

ARM_FNS: dict[str, tuple[Fn, float]] = {
    "arm_then_fist": (arm_then_fist, 0.6),
    "arm_timed_out": (arm_timed_out, 0.4),
    "arm_then_command": (arm_then_command, 0.7),
    "human_said_wrong": (human_said_wrong, 1.0),
}


def vote(events: list[Event], i: int, fns: dict[str, tuple[Fn, float]]) -> tuple[dict[str, str], str | None, float]:
    """Run the functions; return (votes by name, combined label, net weight in [0, 1])."""
    votes: dict[str, str] = {}
    mass: dict[str, float] = {}
    for name, (fn, w) in fns.items():
        label = fn(events, i)
        if label is not None:
            votes[name] = label
            mass[label] = mass.get(label, 0.0) + w
    if not mass:
        return votes, None, 0.0
    best = max(mass, key=lambda k: mass[k])
    total = sum(mass.values())
    margin = (mass[best] - (total - mass[best])) / total
    if margin <= 0:
        return votes, "unsure", 0.0
    return votes, best, min(1.0, margin)

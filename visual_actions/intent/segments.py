"""Cut a session into candidate moments.

Engine events become one segment each (arm, fire, drag, repeat, adjust, broken hold) with a
little context on both sides. The rest of the footage is split into `hand` stretches
(hand in view, engine idle: incidental use or attempts the engine never saw) and `dead`
stretches (no hand). Dead stretches are long and boring, so only one short sample per
stretch is kept for the judges; the whole stretch is still recorded in `notes`.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import labelfns
from .events import Event
from .schema import Segment, SegmentKind

CONTEXT_BEFORE_S = 1.5
CONTEXT_AFTER_S = 1.5
HAND_MIN_S = 2.0  # shorter idle hand stretches are transitions, not moments
HAND_CHUNK_S = 6.0
DEAD_MIN_S = 8.0
DEAD_SAMPLE_S = 4.0


@dataclass
class _Presence:
    """Hand presence over the session as (t, present) steps, from the hand lines."""

    steps: list[tuple[float, bool]]

    @classmethod
    def from_events(cls, events: list[Event]) -> _Presence:
        steps = [(0.0, False)]
        for e in events:
            if e.kind == "hand":
                steps.append((e.t, bool(e.hand_seen)))
        return cls(steps)

    def fraction(self, t0: float, t1: float) -> float:
        if t1 <= t0:
            return 0.0
        present = 0.0
        for k, (t, p) in enumerate(self.steps):
            end = self.steps[k + 1][0] if k + 1 < len(self.steps) else float("inf")
            lo, hi = max(t, t0), min(end, t1)
            if p and hi > lo:
                present += hi - lo
        return present / (t1 - t0)

    def stretches(self, present: bool, end_t: float) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for k, (t, p) in enumerate(self.steps):
            if p != present:
                continue
            end = self.steps[k + 1][0] if k + 1 < len(self.steps) else end_t
            if end > t:
                out.append((t, end))
        return out


def _mean_conf(events: list[Event], gesture: str | None, t0: float, t1: float) -> float | None:
    confs = [e.confidence for e in events if e.kind == "token" and t0 <= e.t <= t1 and (gesture is None or e.token == gesture) and e.confidence is not None]
    return sum(confs) / len(confs) if confs else None


def _busy(events: list[Event], t0: float, t1: float) -> bool:
    """Any engine activity (mode other than idle, drag, action) inside the span?"""
    for e in events:
        if e.t < t0:
            continue
        if e.t > t1:
            break
        if e.kind in ("action", "drag") or (e.kind == "mode" and e.mode_to != "idle"):
            return True
    return False


def segment_session(session: str, events: list[Event]) -> list[Segment]:
    out: list[Segment] = []
    counters: dict[SegmentKind, int] = {}
    end_t = events[-1].t if events else 0.0
    presence = _Presence.from_events(events)

    def new(kind: SegmentKind, t0: float, t1: float, **kw: object) -> Segment:
        n = counters.get(kind, 0)
        counters[kind] = n + 1
        t0, t1 = max(0.0, t0), min(end_t, t1)
        seg = Segment(f"{session}/{kind.value}/{n:04d}", session, kind, round(t0, 3), round(t1, 3), **kw)  # type: ignore[arg-type]
        seg.hand_present_fraction = round(presence.fraction(t0, t1), 3)
        return seg

    hold_start: float | None = None
    hold_leader: str | None = None
    for i, e in enumerate(events):
        if e.kind == "mode" and e.mode_from == "idle" and e.mode_to == "holding":
            hold_start = e.t
            prev = events[i - 1] if i > 0 else None
            hold_leader = prev.token if prev is not None and prev.kind == "token" else None
        elif e.kind == "mode" and e.mode_from == "holding" and e.mode_to == "idle":
            t0 = hold_start if hold_start is not None else e.t - 1.0
            prev = events[i - 1]
            why = "fist" if prev.kind == "token" and prev.token == "fist" else "lost" if prev.kind == "hand" else "flicker"
            out.append(new(SegmentKind.BROKEN_HOLD, t0 - 0.5, e.t + 0.5, engine_gesture=hold_leader, engine_outcome=why, mean_confidence=_mean_conf(events, hold_leader, t0, e.t)))
            hold_start = None
        elif e.kind == "mode" and e.mode_from == "holding" and e.mode_to == "armed":
            outcome, t_end = labelfns.arm_outcome(events, i)
            votes, label, weight = labelfns.vote(events, i, labelfns.ARM_FNS)
            t0 = hold_start if hold_start is not None else e.t - 1.5
            out.append(new(SegmentKind.ARM, t0 - 0.5, min(t_end, e.t + 6.0) + 0.5, engine_gesture=hold_leader, engine_outcome=outcome, engine_namespace=e.namespace, mean_confidence=_mean_conf(events, hold_leader, t0, e.t), labelfn_votes=votes, weak_label=label, weak_weight=weight))
            hold_start = None
        elif e.kind == "action":
            trig = labelfns._last_token_before(events, i)
            votes, label, weight = labelfns.vote(events, i, labelfns.FIRE_FNS)
            outcome = "cancelled_by_fist" if votes.get("fist_after_fire") else "repeated" if votes.get("immediate_redo") else "accepted"
            ns = next((x.namespace for x in reversed(events[:i]) if x.kind == "mode" and x.namespace), None)
            out.append(new(SegmentKind.FIRE, e.t - CONTEXT_BEFORE_S, e.t + CONTEXT_AFTER_S, engine_gesture=trig.token if trig else None, engine_action=e.action_name, engine_outcome=outcome if e.action_ok else "failed", engine_namespace=ns, mean_confidence=trig.confidence if trig else None, labelfn_votes=votes, weak_label=label, weak_weight=weight))
        elif e.kind == "drag" and e.drag_phase == "start":
            pauses = 0
            t_end, outcome = e.t + 10.0, "unfinished"
            for x in events[i + 1 :]:
                if x.kind == "drag" and x.drag_phase == "pause":
                    pauses += 1
                if x.kind == "drag" and x.drag_phase == "end":
                    t_end = x.t
                    nxt = events[events.index(x) + 1] if events.index(x) + 1 < len(events) else None
                    outcome = "dropped_by_fist" if nxt is not None and nxt.kind == "mode" and nxt.mode_to == "idle" else "released"
                    if "snapped=" in x.text:
                        outcome = "snapped"
                    break
            out.append(new(SegmentKind.DRAG, e.t - 1.0, t_end + 0.5, engine_gesture="pinch", engine_outcome=outcome, notes=f"pauses={pauses} window={e.text.split(' ', 1)[1] if ' ' in e.text else ''}"))
        elif e.kind == "mode" and e.mode_from == "armed" and e.mode_to in ("repeat", "adjust"):
            kind = SegmentKind.REPEAT if e.mode_to == "repeat" else SegmentKind.ADJUST
            fires = 0
            t_end = e.t + 5.0
            for x in events[i + 1 :]:
                if x.kind == "action":
                    fires += 1
                if x.kind == "mode" and x.mode_from == e.mode_to and x.mode_to != e.mode_to:
                    t_end = x.t
                    break
            trig = labelfns._last_token_before(events, i)
            out.append(new(kind, e.t - 1.0, t_end + 0.5, engine_gesture=trig.token if trig else None, engine_outcome=f"fires={fires}"))

    # hand stretches with no engine activity, and dead stretches
    for t0, t1 in presence.stretches(True, end_t):
        if t1 - t0 < HAND_MIN_S:
            continue
        t = t0
        while t < t1:
            c1 = min(t + HAND_CHUNK_S, t1)
            if c1 - t >= HAND_MIN_S and not _busy(events, t, c1):
                out.append(new(SegmentKind.HAND, t, c1))
            t = c1
    for t0, t1 in presence.stretches(False, end_t):
        if t1 - t0 < DEAD_MIN_S:
            continue
        mid = (t0 + t1) / 2
        out.append(new(SegmentKind.DEAD, mid - DEAD_SAMPLE_S / 2, mid + DEAD_SAMPLE_S / 2, notes=f"stretch {t0:.1f}-{t1:.1f}s"))
    out.sort(key=lambda s: s.t0)
    return out

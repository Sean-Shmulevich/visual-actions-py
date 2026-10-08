"""The leader-flow state machine. Time only ever arrives on tokens and Tick, so it
replays on fake time.

Each namespace has a leader gesture (root mode): open_palm -> "window", two_up (the peace
sign) -> "media". The leader that starts the hold picks the namespace.

IDLE    --leader & still-----------------> HOLDING(namespace of that leader, evidence 0)
IDLE    --the shape shown when the engine last went idle--> ignored until the hand changes
                                           shape or is lost (no re-arm from a held leader)
HOLDING --leader token-------------------> evidence += dt * rate(confidence)
                                           rate = clamp((c - 0.5) / 0.5, -1, 1) * gain
HOLDING --token != leader----------------> hold pauses; IDLE after hold_break_tokens in a row
HOLDING --evidence >= quick_min & confident BOUND gesture (or pinch)--> ARMED at once, token handled there
HOLDING --tick, evidence >= leader_hold--> ARMED(namespace, deadline)
HOLDING --another root mode's leader------> HOLDING restarts in that namespace (never a quick command)
ARMED   --leader token-------------------> ignored until leader_release_tokens other tokens in a row,
                                           or a brief hand loss (drop the hand, show it again)
ARMED   --bound token--------------------> fire_mass += confidence; >= fire_evidence -> fire -> IDLE
ARMED   --different bound token----------> fire_mass restarts with that gesture
ARMED   --tick >= deadline---------------> IDLE (timeout)
ARMED   --pinch START over a window------> DRAGGING (drag namespace only, "window") (window grabbed under the mapped pointer)
DRAGGING--pinch MOVE---------------------> window follows the hand (DragController)
DRAGGING--pinch END--------------------> IDLE
DRAGGING--hand lost--------------------> DRAGGING, suspended (window stays, red frame)
suspended--pinch back within grace-----> DRAGGING resumed from the window's current place
suspended--hand back unpinched / grace-> IDLE (dropped in place)
ARMED   --repeatable action fires------> REPEAT(deadline = now + repeat_window)
REPEAT  --same shape, wrist slid sideways >= repeat_slide--> fire again, deadline refreshed
REPEAT  --deadline-----------------------> IDLE
ARMED   --pinch START, namespace binds pinch_right/pinch_left--> ADJUST(anchor = pinch x)
ADJUST  --until the pinch holds still adjust_settle--> nothing (transition pinches move, never settle)
ADJUST  --pinch MOVE, user's right/left by adjust_step--> fire pinch_right / pinch_left, anchor moves one step
ADJUST  --pinch END----------------------> IDLE
ANY     --fist token (conf >= 0.6)-------> IDLE   (always; a drag is dropped in place)
ANY     --hand lost escape_lost_s--------> IDLE

Confidence therefore accelerates or decelerates both phases: a sure palm arms
early, a hesitant one stalls; a sure gesture fires on one token, a weak one needs
several in a row, a very weak stream never fires.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .bindings import Bindings
from .drag import DragController
from .events import Bus, HoldProgress, ModeChanged
from .pinch import PinchEvent, PinchPhase
from .recognizer import FIST, OPEN_PALM
from .types import Action, Token

IDLE, HOLDING, ARMED, DRAGGING, REPEAT, ADJUST = "idle", "holding", "armed", "dragging", "repeat", "adjust"
PINCH_RIGHT, PINCH_LEFT = "pinch_right", "pinch_left"  # binding names for sideways pinch travel (user's right / left)


@dataclass(frozen=True)
class Timing:
    leader_hold_ns: int = 2_000_000_000
    command_timeout_ns: int = 5_000_000_000
    escape_fist_ns: int = 1_000_000_000
    escape_lost_ns: int = 1_500_000_000
    confidence_gain: float = 1.5  # fill rate at confidence 1.0, relative to wall clock
    drag_lost_grace_ns: int = 2_500_000_000  # hand lost mid-drag: wait this long for it to come back
    repeat_window_ns: int = 1_500_000_000  # slide-to-repeat window after a repeatable action
    hold_break_tokens: int = 2  # consecutive non-palm tokens tolerated during a hold (classifier flicker)
    quick_command_min_hold_ns: int = 300_000_000  # a clear palm this long, then a confident bound gesture, arms + fires at once
    quick_command_min_confidence: float = 0.85
    resume_grace_ns: int = 600_000_000  # hand back mid-suspension: wait this long for the pinch before dropping
    repeat_slide: float = 0.10  # sideways wrist travel (fraction of frame width) that counts as a slide
    leader_min_confidence: float = 0.8  # palm tokens below this neither start nor fill the hold
    adjust_step: float = 0.05  # ADJUST: sideways pinch travel (fraction of frame width) per pinch_right / pinch_left
    adjust_settle_ns: int = 250_000_000  # ADJUST: the pinch must hold still this long before travel counts (shape changes pinch briefly)
    adjust_settle_travel: float = 0.04  # ADJUST: pinch-point drift (frame fraction) that restarts the settle
    leader_release_tokens: int = 3  # ARMED: other tokens in a row before the leader shape may fire (a moving peace misreads for 2)


def hold_rate(confidence: float, gain: float) -> float:
    return max(-1.0, min(1.0, (confidence - 0.5) / 0.5)) * gain


class ModeEngine:
    def __init__(
        self,
        bus: Bus,
        bindings: Bindings,
        timing: Timing,
        fire: Callable[[Action, int], None],
        default_namespace: str = "window",
        fire_evidence: float = 0.9,
        min_token_confidence: float = 0.3,
        drag: DragController | None = None,
        fist_min_confidence: float = 0.6,
        leaders: dict[str, str] | None = None,
        drag_namespace: str = "window",
    ) -> None:
        self.fist_min_confidence = fist_min_confidence
        # leader gesture -> namespace it opens; open_palm alone keeps the single-mode behavior
        self.leaders = dict(leaders) if leaders else {OPEN_PALM: default_namespace}
        self.drag_namespace = drag_namespace
        self.drag = drag
        self.bus = bus
        self.bindings = bindings
        self.timing = timing
        self.fire = fire
        self.default_namespace = default_namespace
        self.fire_evidence = fire_evidence
        self.min_token_confidence = min_token_confidence
        self.state = IDLE
        self.namespace: str | None = None
        self.hold_evidence_ns: float = 0.0
        self._last_palm_ns: int | None = None
        self._deadline_ns: int | None = None
        self._fist_since_ns: int | None = None
        self._lost_since_ns: int | None = None
        self._fire_gesture: str | None = None
        self._fire_mass: float = 0.0
        self._hold_rate: float = 0.0
        self._hold_misses = 0
        self._returned_at_ns: int | None = None  # first token after the hand came back mid-suspension
        self._repeat_action: Action | None = None
        self._repeat_gesture: str | None = None
        self._repeat_anchor_x: float = 0.5
        self.repeat_count = 0
        self.leader: str | None = None  # the gesture that started the current hold
        self._leader_released = False  # ARMED: the hand has shown another shape since arming
        self._release_tokens = 0
        self._last_token: str | None = None  # last shape seen while the hand is present
        self._idle_block: str | None = None  # shape held when the engine went idle; cannot lead until it changes
        self._adjust_anchor_x: float = 0.5
        self._adjust_since_ns = 0
        self._adjust_settle_xy = (0.5, 0.5)
        self._adjust_settled = False

    def projected_hold_ns(self, t_ns: int) -> float:
        """Evidence extrapolated to t_ns at the last token's rate, so arming and the ring are smooth."""
        if self.state != HOLDING or self._last_palm_ns is None or self._hold_rate <= 0:
            return self.hold_evidence_ns
        return self.hold_evidence_ns + (t_ns - self._last_palm_ns) * self._hold_rate

    # -- inputs -------------------------------------------------------------

    @property
    def drag_allowed(self) -> bool:
        return self.drag is not None and self.namespace == self.drag_namespace

    def on_token(self, tok: Token) -> None:
        self._lost_since_ns = None
        self._last_token = tok.name
        if tok.name == FIST and tok.confidence >= self.fist_min_confidence and self.state != IDLE:
            # A fist always ends whatever is happening: hold, armed window, repeat, or a drag (dropped in place).
            if self.state == DRAGGING and self.drag is not None:
                self.drag.cancel(tok.t_ns)
            self._go(IDLE, tok.t_ns)
            return
        if self.state == ADJUST:
            return  # per-frame pinch events own this state
        if self.state == DRAGGING:
            if self.drag is not None and self.drag.suspended:
                # the hand is back: give the pinch detector a moment before deciding it let go
                if self._returned_at_ns is None:
                    self._returned_at_ns = tok.t_ns
                elif tok.t_ns - self._returned_at_ns >= self.timing.resume_grace_ns:
                    self.drag.cancel(tok.t_ns)
                    self._go(IDLE, tok.t_ns)
            return  # per-frame pinch events own this state; tokens are ignored
        if self.state == IDLE:
            if self._idle_block is not None:
                if tok.name == self._idle_block:
                    return  # still the shape from before going idle: not a fresh leader
                self._idle_block = None
            self._start_hold(tok)
        elif self.state == HOLDING:
            if tok.name != self.leader and self._start_hold(tok):
                return  # another root mode's leader switches the hold to that mode, e.g. palm then peace
            if tok.name != self.leader and tok.name not in self.leaders and self._quick_command(tok):
                return
            if tok.name != self.leader:
                # classifier flicker: a brief non-palm token pauses the hold instead of killing it
                self._hold_misses += 1
                if self._hold_misses >= self.timing.hold_break_tokens:
                    self._go(IDLE, tok.t_ns)
                else:
                    self._last_palm_ns = tok.t_ns  # no evidence for the gap
                    self._hold_rate = 0.0
                return
            self._hold_misses = 0
            dt = 0 if self._last_palm_ns is None else tok.t_ns - self._last_palm_ns
            self._last_palm_ns = tok.t_ns
            sure = tok.still and tok.confidence >= self.timing.leader_min_confidence
            rate = hold_rate(tok.confidence, self.timing.confidence_gain) if sure else 0.0
            self.hold_evidence_ns = max(0.0, self.hold_evidence_ns + dt * rate)
            self._hold_rate = rate
            hold = self.timing.leader_hold_ns
            self.bus.publish(HoldProgress(tok.t_ns, min(1.0, self.hold_evidence_ns / hold), rate * 1e9 / hold))
        elif self.state == ARMED:
            if tok.name == self.leader and not self._leader_released:
                self._release_tokens = 0
                return  # the leader still held from arming never fires a command
            if tok.name != self.leader and not self._leader_released:
                # a misread while the hand moves is not a release: leader_release_tokens in a row are
                self._release_tokens += 1
                self._leader_released = self._release_tokens >= self.timing.leader_release_tokens
            action = self.bindings.lookup(self.namespace or self.default_namespace, tok.name)
            if action is None or tok.confidence < self.min_token_confidence:
                return
            if tok.name != self._fire_gesture:
                self._fire_gesture, self._fire_mass = tok.name, 0.0
            self._fire_mass += tok.confidence
            if self._fire_mass >= self.fire_evidence:
                if action.arg("repeat") in ("true", "True", "1"):
                    # fingerspelling-style slide: same shape, move sideways, fires again
                    self._repeat_action, self._repeat_gesture = action, tok.name
                    self._repeat_anchor_x = tok.x
                    self.repeat_count = 1
                    self._deadline_ns = tok.t_ns + self.timing.repeat_window_ns
                    self._go(REPEAT, tok.t_ns)
                else:
                    self._go(IDLE, tok.t_ns)
                self.fire(action, tok.t_ns)
        elif self.state == REPEAT:
            if tok.name != self._repeat_gesture or tok.confidence < self.min_token_confidence:
                return  # only the same shape counts; anything else is ignored until the window closes
            if abs(tok.x - self._repeat_anchor_x) >= self.timing.repeat_slide and self._repeat_action is not None:
                self._repeat_anchor_x = tok.x
                self.repeat_count += 1
                self._deadline_ns = tok.t_ns + self.timing.repeat_window_ns
                self.bus.publish(ModeChanged(tok.t_ns, REPEAT, REPEAT, self.namespace, self._deadline_ns))
                self.fire(self._repeat_action, tok.t_ns)
            elif tok.still:
                self._repeat_anchor_x = tok.x  # a still hand re-anchors, so slow drift never adds up to a slide

    def on_pinch(self, ev: PinchEvent) -> None:
        """Per-frame pinch input. In the window namespace a pinch while ARMED grabs the
        window under the hand and DRAGGING follows it. In a namespace that binds
        pinch_right / pinch_left (media), a pinch enters ADJUST and sideways travel fires them."""
        if self.state == ADJUST:
            self._lost_since_ns = None
            self._adjust(ev)
            return
        pinch_ok = self.drag_allowed or self._adjust_bound()
        if self.state in (HOLDING, ARMED) and not pinch_ok:
            return
        self._lost_since_ns = None
        if self.state == HOLDING and ev.phase is PinchPhase.START and self.hold_evidence_ns >= self.timing.quick_command_min_hold_ns:
            # leader then pinch in one motion: arm and grab (or adjust)
            self._arm(ev.t_ns)
        if self.state == ARMED and ev.phase is PinchPhase.START:
            if not self.drag_allowed:
                self._adjust_anchor_x = ev.x
                self._adjust_since_ns, self._adjust_settle_xy = ev.t_ns, (ev.x, ev.y)
                self._adjust_settled = False
                self._go(ADJUST, ev.t_ns)
            elif self.drag is not None and self.drag.on_pinch(ev):
                self._go(DRAGGING, ev.t_ns)
            return  # a miss keeps the window armed
        if self.state == DRAGGING and self.drag is not None:
            if self.drag.suspended and ev.phase is PinchPhase.END:
                self.drag.cancel(ev.t_ns)  # came back and let go: drop
                self._go(IDLE, ev.t_ns)
                return
            if self.drag.suspended:
                self._returned_at_ns = None  # pinch is back: resume
            self.drag.on_pinch(ev)
            if ev.phase is PinchPhase.END:
                self._go(IDLE, ev.t_ns)

    def _adjust_bound(self) -> bool:
        ns = self.namespace or self.default_namespace
        return self.bindings.lookup(ns, PINCH_RIGHT) is not None or self.bindings.lookup(ns, PINCH_LEFT) is not None

    def _adjust(self, ev: PinchEvent) -> None:
        """Pinched hand travels sideways in the user frame (+x is the user's right): one
        pinch_right / pinch_left per adjust_step of frame width. Releasing ends it."""
        if ev.phase is PinchPhase.END:
            self._go(IDLE, ev.t_ns)
            return
        if not self._adjust_settled:
            # "pinch, hold, then move": a pinch crossed during a shape change is moving, never settles
            if abs(ev.x - self._adjust_settle_xy[0]) + abs(ev.y - self._adjust_settle_xy[1]) > self.timing.adjust_settle_travel:
                self._adjust_since_ns, self._adjust_settle_xy = ev.t_ns, (ev.x, ev.y)
                return
            if ev.t_ns - self._adjust_since_ns < self.timing.adjust_settle_ns:
                return
            self._adjust_settled = True
            self._adjust_anchor_x = ev.x
            self.bus.publish(ModeChanged(ev.t_ns, ADJUST, ADJUST, self.namespace, None))  # settled: the overlay says "move"
        ns = self.namespace or self.default_namespace
        step = self.timing.adjust_step
        while ev.x - self._adjust_anchor_x >= step:
            self._adjust_anchor_x += step
            self._adjust_fire(ns, PINCH_RIGHT, ev.t_ns)
        while self._adjust_anchor_x - ev.x >= step:
            self._adjust_anchor_x -= step
            self._adjust_fire(ns, PINCH_LEFT, ev.t_ns)

    def _adjust_fire(self, ns: str, gesture: str, t_ns: int) -> None:
        action = self.bindings.lookup(ns, gesture)
        if action is not None:
            self.repeat_count += 1
            self.fire(action, t_ns)

    def _quick_command(self, tok: Token) -> bool:
        """Palm-then-gesture in one motion: the user's natural rhythm is ~0.4 s of palm before
        the command (2026-10-08 sessions). A clear palm held past quick_command_min_hold
        followed by a confident BOUND gesture arms immediately and feeds the token to ARMED."""
        ns = self.namespace or self.default_namespace
        if (
            self.hold_evidence_ns < self.timing.quick_command_min_hold_ns
            or tok.confidence < self.timing.quick_command_min_confidence
            or self.bindings.lookup(ns, tok.name) is None
        ):
            return False
        self._arm(tok.t_ns)
        self.on_token(tok)
        return True

    def _start_hold(self, tok: Token) -> bool:
        ns = self.leaders.get(tok.name)
        if ns is None or not tok.still or tok.confidence < self.timing.leader_min_confidence:
            return False
        self.leader, self.namespace = tok.name, ns
        self.hold_evidence_ns = 0.0
        self._hold_rate = 0.0
        self._last_palm_ns = tok.t_ns
        self._hold_misses = 0
        self._go(HOLDING, tok.t_ns)
        return True

    def _arm(self, t_ns: int) -> None:
        self.namespace = self.namespace or self.default_namespace
        self._deadline_ns = t_ns + self.timing.command_timeout_ns
        self._fire_gesture, self._fire_mass = None, 0.0
        self._leader_released, self._release_tokens = False, 0
        self._go(ARMED, t_ns)

    def on_hand_lost(self, t_ns: int) -> None:
        if self._lost_since_ns is None:
            self._lost_since_ns = t_ns
        self._fist_since_ns = None
        self._last_token = None
        self._idle_block = None  # the hand left: whatever it shows next is a fresh start
        if self.state == ARMED:
            self._leader_released = True  # came back within escape_lost: the leader shape is now a command
        if self.state == DRAGGING and self.drag is not None:
            self._returned_at_ns = None
            self.drag.suspend(t_ns)  # window stays; the drag resumes if the hand returns pinching

    def on_tick(self, t_ns: int) -> None:
        if self.state == DRAGGING:
            if (
                self.drag is not None
                and self.drag.suspended
                and self._lost_since_ns is not None
                and t_ns - self._lost_since_ns >= self.timing.drag_lost_grace_ns
            ):
                self.drag.cancel(t_ns)
                self._go(IDLE, t_ns)
            return
        if (
            self._lost_since_ns is not None
            and self.state != IDLE
            and t_ns - self._lost_since_ns >= self.timing.escape_lost_ns
        ):
            self._go(IDLE, t_ns)
            return
        if self.state == HOLDING and self.projected_hold_ns(t_ns) >= self.timing.leader_hold_ns:
            self._arm(t_ns)
        elif self.state in (ARMED, REPEAT) and self._deadline_ns is not None and t_ns >= self._deadline_ns:
            self._go(IDLE, t_ns)

    # -- internals ----------------------------------------------------------

    def _go(self, new: str, t_ns: int) -> None:
        old = self.state
        self.state = new
        if new == IDLE:
            if old not in (IDLE, HOLDING) and self._lost_since_ns is None and self._last_token != FIST:
                # e.g. a peace sign still held after "Next tab" fired, or after the media
                # window timed out, must not start a new hold by itself
                self._idle_block = self._last_token
            self.leader = None
            self._leader_released = False
            self.namespace = None
            self.hold_evidence_ns = 0.0
            self._last_palm_ns = None
            self._deadline_ns = None
            self._fist_since_ns = None
            self._lost_since_ns = None
            self._fire_gesture, self._fire_mass = None, 0.0
            self._hold_rate = 0.0
            self._repeat_action, self._repeat_gesture, self.repeat_count = None, None, 0
            self._returned_at_ns = None
        self.bus.publish(
            ModeChanged(t_ns=t_ns, old=old, new=new, namespace=self.namespace, deadline_ns=self._deadline_ns)
        )

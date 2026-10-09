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
ARMED   --bound token--------------------> fire_mass += confidence; >= fire_evidence -> fire -> ARMED
                                           (chain_commands: fresh command_timeout, so commands chain
                                           without the leader; the gesture that fired must be released
                                           first: leader_release_tokens other tokens or a brief hand loss.
                                           chain_commands off: -> IDLE)
ARMED   --different bound token----------> fire_mass restarts with that gesture
ARMED   --tick >= deadline---------------> IDLE (timeout)
ARMED   --pinch START over a window------> DRAGGING (drag namespace only, "window") (window grabbed under the mapped pointer)
DRAGGING--pinch MOVE---------------------> window follows the hand (DragController)
DRAGGING--pinch END--------------------> ARMED (chaining) / IDLE
DRAGGING--hand lost--------------------> DRAGGING, suspended (window stays, red frame)
suspended--pinch back within grace-----> DRAGGING resumed from the window's current place
suspended--hand back unpinched / grace-> IDLE (dropped in place)
ARMED   --slide shape (bindings "<shape>:left|right|up|down")--> the wrist is anchored once the shape
                                           is still; a travel of repeat_slide (or the binding's own "step")
                                           along the dominant axis fires the action of that direction.
                                           Continuous (scroll): re-anchors at each step, so travel keeps
                                           firing. Flick ("flick": true, desktops): the anchor stays, the
                                           return stroke fires nothing, and the next stroke counts only
                                           once the hand is back near the anchor: out-and-back is one
                                           command. A still hand re-anchors; the shape itself never fires.
ARMED   --repeatable action fires------> REPEAT(deadline = now + repeat_window)
REPEAT  --same shape, wrist slid sideways >= repeat_slide--> fire again, deadline refreshed
REPEAT  --deadline-----------------------> ARMED (chaining) / IDLE
REPEAT  --confident other bound gesture--> ARMED, and that gesture is handled at once
ARMED   --pinch START, namespace binds pinch_right/pinch_left--> ADJUST(anchor = pinch x)
ADJUST  --until the pinch holds still adjust_settle--> nothing (transition pinches move, never settle)
ADJUST  --pinch MOVE, user's right/left by adjust_step--> fire pinch_right / pinch_left, anchor moves one step
ADJUST  --pinch END----------------------> ARMED (chaining) / IDLE
ADJUST  --hand lost----------------------> the anchor is kept for ADJUST_LOST_GRACE; a pinch back within it
                                           continues from the anchor (no re-settle); longer -> ARMED (fresh timeout)
ADJUST  --pinch frames-------------------> AdjustEvent SETTLE / START / MOVE / END for the cursor overlay
ARMED   --scroll shape (a binding of kind scroll with mode "stick"), still--> SCROLL(anchor = wrist)
SCROLL  --hand frames---------------------> the ScrollController scrolls by the wrist's offset from the anchor
SCROLL  --two tokens of another shape----> ARMED (fresh timeout)
SCROLL  --hand lost / fist---------------> scroll ends; ARMED (lost, keeps its deadline) / IDLE (fist)
ANY     --fist token (conf >= 0.6)-------> IDLE   (always; a drag is dropped in place)
HOLDING --hand lost--------------------> IDLE at once (hold_lost_grace 0, STRICT: the palm is continuous),
                                           else the hold pauses and drops after hold_lost_grace
ARMED/REPEAT --hand lost (keep_armed_on_lost)--> the window keeps its own deadline; IDLE when it
                                           passes, or when the hand has been gone 2 x escape_lost
                                           with nothing pending (no half-recognized gesture, no slide window)
ARMED/REPEAT --hand lost (keep_armed_on_lost off)--> IDLE after escape_lost

Confidence therefore accelerates or decelerates both phases: a sure palm arms
early, a hesitant one stalls; a sure gesture fires on one token, a weak one needs
several in a row, a very weak stream never fires.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .bindings import Bindings
from .drag import DragController
from .events import AdjustEvent, AdjustPhase, Bus, HoldProgress, ModeChanged
from .pinch import PinchEvent, PinchPhase
from .recognizer import FIST, OPEN_PALM
from .scroll import ScrollController
from .types import Action, Token

IDLE, HOLDING, ARMED, DRAGGING, REPEAT, ADJUST, SCROLL = "idle", "holding", "armed", "dragging", "repeat", "adjust", "scroll"
PINCH_RIGHT, PINCH_LEFT = "pinch_right", "pinch_left"  # binding names for sideways pinch travel (user's right / left)


@dataclass(frozen=True)
class Timing:
    # Defaults mirror TimingConfig (the STRICT profile); tests/test_strict_profile.py pins the two equal.
    leader_hold_ns: int = 1_500_000_000
    command_timeout_ns: int = 5_000_000_000
    escape_fist_ns: int = 1_000_000_000
    escape_lost_ns: int = 1_500_000_000
    confidence_gain: float = 1.0  # fill rate at confidence 1.0, relative to wall clock; FAST uses 1.5
    drag_lost_grace_ns: int = 2_500_000_000  # hand lost mid-drag: wait this long for it to come back
    repeat_window_ns: int = 1_500_000_000  # slide-to-repeat window after a repeatable action
    hold_break_tokens: int = 1  # consecutive non-palm tokens tolerated during a hold (1 = none); FAST sets 2
    hold_reset_on_move: bool = True  # moving palm resets evidence (else pauses); FAST sets False
    quick_command: bool = False  # FAST sets True
    quick_command_min_hold_ns: int = 300_000_000  # a clear palm this long, then a confident bound gesture, arms + fires at once
    quick_command_min_confidence: float = 0.85
    resume_grace_ns: int = 600_000_000  # hand back mid-suspension: wait this long for the pinch before dropping
    repeat_slide: float = 0.10  # sideways wrist travel (fraction of frame width) that counts as a slide
    leader_min_confidence: float = 0.9  # palm tokens below this neither start nor fill the hold; FAST 0.8
    adjust_step: float = 0.05  # ADJUST: sideways pinch travel (fraction of frame width) per pinch_right / pinch_left
    adjust_range_steps: int = 16  # ADJUST: steps from the anchor to the frame edge when the anchor is central (16 = the full volume range)
    adjust_step_min: float = 0.02  # ADJUST: the step never shrinks under this when the anchor sits near an edge
    adjust_settle_ns: int = 250_000_000  # ADJUST: the pinch must hold still this long before travel counts (shape changes pinch briefly)
    adjust_settle_travel: float = 0.04  # ADJUST: pinch-point drift (frame fraction) that restarts the settle
    chain_commands: bool = True  # stay ARMED after a command (timeout restarts) so commands chain without the leader
    leader_release_tokens: int = 3  # ARMED: other tokens in a row before the leader shape may fire (a moving peace misreads for 2)
    keep_armed_on_lost: bool = True  # ARMED/REPEAT survive a hand loss until their own deadline (see on_tick); off: escape_lost drops them
    hold_lost_grace_ns: int = 0  # HOLDING: a loss shorter than this pauses the hold instead of resetting it; 0 = reset at once; FAST 0.3 s


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
        scroll: ScrollController | None = None,
        to_screen: Callable[[float, float], tuple[float, float]] | None = None,
    ) -> None:
        self.scroll = scroll
        # user-frame point -> screen point for the ADJUST overlay (the pipeline's PointerMap); identity when absent
        self.to_screen: Callable[[float, float], tuple[float, float]] = to_screen if to_screen is not None else (lambda x, y: (x, y))
        self._adjust_lost_ns: int | None = None  # ADJUST: when the hand was lost; the anchor waits ADJUST_LOST_GRACE_NS
        self._adjust_anchor_y = 0.5
        self._scroll_shape: str | None = None
        self._scroll_gap = 0
        self._scroll_lost_ns: int | None = None  # SCROLL: when the hand was lost; the scroll waits scroll_lost_grace
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
        self._fired_block: str | None = None  # chaining: the gesture that just fired cannot fire again until released
        self._fired_gap = 0
        self._adjust_since_ns = 0
        self._adjust_settle_xy = (0.5, 0.5)
        self._adjust_settled = False
        self._adjust_step = 0.05
        self._refresh_on_return = False  # ARMED kept across a loss: the returning hand's leader shape renews the deadline
        self._slide_shape: str | None = None  # ARMED: the slide shape being tracked, and where its wrist was anchored
        self._slide_anchor = (0.5, 0.5)
        self._slide_gap = 0  # other tokens seen since the slide shape; SLIDE_GAP_TOKENS of them drop the anchor
        self._slide_out: str | None = None  # flick slides: the direction fired, until the hand is back at the anchor
        self._slide_last = (0.5, 0.5)  # the slide shape's last wrist position (a loss mid-stroke completes the stroke)

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
            if self.state == SCROLL and self.scroll is not None:
                self.scroll.end(tok.t_ns)
            if self.state == ADJUST:
                self._publish_adjust(AdjustPhase.END, tok.t_ns, self._adjust_anchor_x, self._adjust_anchor_y, self.repeat_count)
            self._go(IDLE, tok.t_ns)
            return
        if self.state == SCROLL:
            self._scroll_lost_ns = None
            if tok.name == self._scroll_shape:
                self._scroll_gap = 0
            else:
                self._scroll_gap += 1
                if self._scroll_gap >= 2:
                    self._end_scroll(tok.t_ns)
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
            if not tok.still and self.timing.hold_reset_on_move:
                self.hold_evidence_ns = 0.0  # strict: the palm must be still for the whole hold
            sure = tok.still and tok.confidence >= self.timing.leader_min_confidence
            rate = hold_rate(tok.confidence, self.timing.confidence_gain) if sure else 0.0
            self.hold_evidence_ns = max(0.0, self.hold_evidence_ns + dt * rate)
            self._hold_rate = rate
            hold = self.timing.leader_hold_ns
            self.bus.publish(HoldProgress(tok.t_ns, min(1.0, self.hold_evidence_ns / hold), rate * 1e9 / hold))
        elif self.state == ARMED:
            if (
                self._refresh_on_return
                and tok.name == self.leader
                and tok.still
                and tok.confidence >= self.timing.leader_min_confidence
            ):
                # the hand came back and shows the leader again: it wants the menu, so the window
                # it kept through the loss gets a full timeout instead of expiring under the palm
                self._refresh_on_return = False
                self._deadline_ns = tok.t_ns + self.timing.command_timeout_ns
                self.bus.publish(ModeChanged(tok.t_ns, ARMED, ARMED, self.namespace, self._deadline_ns))
                return
            if self._slide(tok):
                return
            if self._start_scroll(tok):
                return
            if tok.name == self.leader and not self._leader_released:
                self._release_tokens = 0
                return  # the leader still held from arming never fires a command
            if tok.name != self.leader and not self._leader_released:
                # a misread while the hand moves is not a release: leader_release_tokens in a row are
                self._release_tokens += 1
                self._leader_released = self._release_tokens >= self.timing.leader_release_tokens
            if self._fired_block is not None:
                if tok.name == self._fired_block:
                    if self._fired_gap < self.timing.leader_release_tokens:
                        self._fired_gap = 0
                        return  # still the shape that just fired: holding it is not a second command
                    self._fired_block = None
                else:
                    self._fired_gap += 1
            ns = self.namespace or self.default_namespace
            if self._slide_shape is not None:
                # a flicker (none, the model's other guess) between two slide tokens keeps the anchor
                self._slide_gap += 1
                if self._slide_gap >= self.SLIDE_GAP_TOKENS:
                    self._end_slide(tok.t_ns)
            action = self.bindings.lookup(ns, tok.name)
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
                    self._after_command(tok.t_ns, tok.name)
                self.fire(action, tok.t_ns)
        elif self.state == REPEAT:
            if (
                self.timing.chain_commands
                and tok.name != self._repeat_gesture
                and tok.confidence >= self.timing.quick_command_min_confidence
                and self.bindings.lookup(self.namespace or self.default_namespace, tok.name) is not None
            ):
                # chaining: a confident different command ends the slide window and is handled at once
                self._after_command(tok.t_ns, self._repeat_gesture)
                self.on_token(tok)
                return
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
            self._adjust_lost_ns = None  # the pinch is back (a fresh START after a loss continues from the anchor)
            self._adjust(ev)
            return
        pinch_ok = self.drag_allowed or self._adjust_bound()
        if self.state in (HOLDING, ARMED) and not pinch_ok:
            return
        self._lost_since_ns = None
        if self.state == HOLDING and ev.phase is PinchPhase.START and self.timing.quick_command and self.hold_evidence_ns >= self.timing.quick_command_min_hold_ns:
            # leader then pinch in one motion: arm and grab (or adjust)
            self._arm(ev.t_ns)
        if self.state == ARMED and ev.phase is PinchPhase.START:
            if not self.drag_allowed:
                self._adjust_anchor_x, self._adjust_anchor_y = ev.x, ev.y
                self._adjust_since_ns, self._adjust_settle_xy = ev.t_ns, (ev.x, ev.y)
                self._adjust_settled = False
                self._adjust_lost_ns = None
                self._go(ADJUST, ev.t_ns)
                self._publish_adjust(AdjustPhase.SETTLE, ev.t_ns, ev.x, ev.y)
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
            if not self.drag.dragging:
                self._after_command(ev.t_ns, None)  # dropped: grab another window, or anything else

    DIRECTIONS = ("left", "right", "up", "down")
    SLIDE_GAP_TOKENS = 3  # tokens of another shape that end a slide (the shape flickers while the hand moves)
    SLIDE_JUMP = 3.0  # a single-token travel over this many steps is the hand arriving, not a slide

    def slide_actions(self, ns: str, shape: str) -> dict[str, Action] | None:
        """direction -> action when `shape` is bound as a slide in `ns` ("<shape>:left" etc.), else None."""
        out = {d: a for d in self.DIRECTIONS if (a := self.bindings.lookup(ns, f"{shape}:{d}")) is not None}
        return out or None

    def _slide(self, tok: Token) -> bool:
        """ARMED: a slide shape is tracked by wrist travel; True when the token was a slide shape
        (handled here, whether or not it fired). The shape alone never fires."""
        ns = self.namespace or self.default_namespace
        slide = self.slide_actions(ns, tok.name)
        if slide is None:
            return False
        if tok.confidence < self.min_token_confidence:
            return True
        self._slide_gap = 0
        if self._slide_shape != tok.name:
            # anchored where the shape first appears, no wait: the sticky dot shows the spot, a
            # jump of SLIDE_JUMP steps in one token re-anchors without firing (the hand arriving)
            self._slide_shape, self._slide_anchor = tok.name, (tok.x, tok.y)
            self._slide_out = None
            self._slide_last = (tok.x, tok.y)
            self._publish_adjust(AdjustPhase.START, tok.t_ns, tok.x, tok.y)
            return True
        self._slide_last = (tok.x, tok.y)
        self._publish_slide_move(tok.t_ns, tok.x, tok.y)
        dx, dy = tok.x - self._slide_anchor[0], tok.y - self._slide_anchor[1]  # user frame: +x right, +y down
        step = self.timing.repeat_slide
        for a in slide.values():
            try:
                step = float(a.arg("step") or step)
                break
            except ValueError:
                pass
        flick = any(a.arg("flick") in ("true", "True", "1") for a in slide.values())
        travel = max(abs(dx), abs(dy))
        if self._slide_out is not None:
            # flick: the stroke fired on the way out; the way back to the anchor is the same gesture,
            # not the opposite command, and nothing fires again until the hand is back near the anchor
            if travel <= step / 2:
                self._slide_out = None
            return True
        if travel >= self.SLIDE_JUMP * step:
            self._slide_anchor = (tok.x, tok.y)  # a jump: re-anchor, never fire
        elif travel >= step:
            direction = ("right" if dx > 0 else "left") if abs(dx) >= abs(dy) else ("down" if dy > 0 else "up")
            if flick:
                self._slide_out = direction  # the anchor stays: that is where the hand returns to
            else:
                self._slide_anchor = (tok.x, tok.y)  # continuous: each step from the last fires again
            action = slide.get(direction)
            if action is not None:
                self._deadline_ns = tok.t_ns + self.timing.command_timeout_ns
                self.repeat_count += 1
                self.bus.publish(ModeChanged(tok.t_ns, ARMED, ARMED, self.namespace, self._deadline_ns))
                self.fire(action, tok.t_ns)
        elif tok.still:
            self._slide_anchor = (tok.x, tok.y)
        return True

    def _adjust_bound(self) -> bool:
        ns = self.namespace or self.default_namespace
        return self.bindings.lookup(ns, PINCH_RIGHT) is not None or self.bindings.lookup(ns, PINCH_LEFT) is not None

    def _adjust(self, ev: PinchEvent) -> None:
        """Pinched hand travels sideways in the user frame (+x is the user's right): one
        pinch_right / pinch_left per adjust_step of frame width. Releasing ends it."""
        if ev.phase is PinchPhase.END:
            self._end_adjust(ev.t_ns)
            return
        if not self._adjust_settled:
            # "pinch, hold, then move": a pinch crossed during a shape change is moving, never settles
            if abs(ev.x - self._adjust_settle_xy[0]) + abs(ev.y - self._adjust_settle_xy[1]) > self.timing.adjust_settle_travel:
                self._adjust_since_ns, self._adjust_settle_xy = ev.t_ns, (ev.x, ev.y)
                self._publish_adjust(AdjustPhase.SETTLE, ev.t_ns, ev.x, ev.y)
                return
            if ev.t_ns - self._adjust_since_ns < self.timing.adjust_settle_ns:
                self._publish_adjust(AdjustPhase.SETTLE, ev.t_ns, ev.x, ev.y)
                return
            self._adjust_settled = True
            # The step scales to the room the anchor has: from the centre, the frame edge is the
            # full volume range (adjust_range_steps steps); off-centre the step never drops under
            # adjust_step_min, so a dot set near an edge simply covers less of the range.
            room = min(ev.x, 1.0 - ev.x)
            self._adjust_step = max(self.timing.adjust_step_min, min(self.timing.adjust_step, room / self.timing.adjust_range_steps))
            self._adjust_anchor_x, self._adjust_anchor_y = ev.x, ev.y
            self.bus.publish(ModeChanged(ev.t_ns, ADJUST, ADJUST, self.namespace, None))  # settled: the overlay says "move"
            self._publish_adjust(AdjustPhase.START, ev.t_ns, ev.x, ev.y)
            return
        ns = self.namespace or self.default_namespace
        step = self._adjust_step
        steps = 0
        while ev.x - self._adjust_anchor_x >= step:
            self._adjust_anchor_x += step
            self._adjust_fire(ns, PINCH_RIGHT, ev.t_ns)
            steps += 1
        while self._adjust_anchor_x - ev.x >= step:
            self._adjust_anchor_x -= step
            self._adjust_fire(ns, PINCH_LEFT, ev.t_ns)
            steps -= 1
        self._publish_adjust(AdjustPhase.MOVE, ev.t_ns, ev.x, ev.y, steps)

    ADJUST_LOST_GRACE_NS = 1_000_000_000

    def _publish_adjust(self, phase: AdjustPhase, t_ns: int, x: float, y: float, steps: int = 0) -> None:
        """The overlay's view of the pinch: anchor and pinch point in screen points (SETTLE: both at the pinch)."""
        ax, ay = (x, y) if phase is AdjustPhase.SETTLE else (self._adjust_anchor_x, self._adjust_anchor_y)
        sax, say = self.to_screen(ax, ay)
        sx, sy = self.to_screen(x, y)
        self.bus.publish(AdjustEvent(t_ns, phase, sax, say, sx, sy, steps))

    def _end_adjust(self, t_ns: int) -> None:
        """Released, or lost past the grace: END for the overlay, then the menu stays open (chaining)."""
        self._publish_adjust(AdjustPhase.END, t_ns, self._adjust_anchor_x, self._adjust_anchor_y, self.repeat_count)
        self._adjust_lost_ns = None
        self._after_command(t_ns, None)

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
            not self.timing.quick_command
            or self.hold_evidence_ns < self.timing.quick_command_min_hold_ns
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

    def _after_command(self, t_ns: int, gesture: str | None) -> None:
        """A command finished. Chaining keeps the menu open with a fresh timeout; the
        gesture that fired must be released before it can fire again."""
        if not self.timing.chain_commands:
            self._go(IDLE, t_ns)
            return
        self._deadline_ns = t_ns + self.timing.command_timeout_ns
        self._fire_gesture, self._fire_mass = None, 0.0
        self._repeat_action, self._repeat_gesture, self.repeat_count = None, None, 0
        self._fired_block, self._fired_gap = gesture, 0
        self._go(ARMED, t_ns)

    def _arm(self, t_ns: int) -> None:
        self.namespace = self.namespace or self.default_namespace
        self._deadline_ns = t_ns + self.timing.command_timeout_ns
        self._fire_gesture, self._fire_mass = None, 0.0
        self._leader_released, self._release_tokens = False, 0
        self._fired_block, self._fired_gap = None, 0
        self._refresh_on_return = False
        self._slide_shape = None
        self._go(ARMED, t_ns)

    def _start_scroll(self, tok: Token) -> bool:
        """ARMED: a still scroll shape (binding of kind scroll, mode stick) starts a stick scroll."""
        if self.scroll is None:
            return False
        action = self.bindings.lookup(self.namespace or self.default_namespace, tok.name)
        if action is None or action.kind.value != "scroll" or action.arg("mode") != "stick":
            return False
        if tok.confidence < self.min_token_confidence:
            return True
        if not tok.still:
            self.scroll.settling(tok.t_ns, tok.x, tok.y)  # the shape, moving: hold still to set the anchor
            return True
        self._scroll_shape, self._scroll_gap = tok.name, 0
        self.scroll.start(tok.t_ns, tok.x, tok.y)
        self._go(SCROLL, tok.t_ns)
        return True

    def _complete_slide_on_loss(self, t_ns: int) -> None:
        """The flick carried the hand out of the frame: that completes the stroke. The anchor is
        kept, so the hand coming back toward it is the return, never the opposite switch."""
        ns = self.namespace or self.default_namespace
        slide = self.slide_actions(ns, self._slide_shape or "")
        if slide is None or self._slide_out is not None:
            return
        dx, dy = self._slide_last[0] - self._slide_anchor[0], self._slide_last[1] - self._slide_anchor[1]
        if max(abs(dx), abs(dy)) < self.timing.repeat_slide / 2:
            return
        direction = ("right" if dx > 0 else "left") if abs(dx) >= abs(dy) else ("down" if dy > 0 else "up")
        self._slide_out = direction
        action = slide.get(direction)
        if action is not None:
            self.repeat_count += 1
            self.fire(action, t_ns)

    def _publish_slide_move(self, t_ns: int, x: float, y: float) -> None:
        sax, say = self.to_screen(*self._slide_anchor)
        sx, sy = self.to_screen(x, y)
        self.bus.publish(AdjustEvent(t_ns, AdjustPhase.MOVE, sax, say, sx, sy, 0))

    def _end_slide(self, t_ns: int) -> None:
        if self._slide_shape is not None:
            ax, ay = self.to_screen(*self._slide_anchor)
            self.bus.publish(AdjustEvent(t_ns, AdjustPhase.END, ax, ay, ax, ay, 0))
        self._slide_shape = None
        self._slide_out = None

    def _end_scroll(self, t_ns: int) -> None:
        if self.scroll is not None:
            self.scroll.end(t_ns)
        self._scroll_shape = None
        self._after_command(t_ns, None)

    def on_hand_position(self, t_ns: int, x: float, y: float) -> None:
        """Per frame while scrolling: the wrist in the user frame."""
        if self.state == SCROLL and self.scroll is not None:
            self._scroll_lost_ns = None
            self.scroll.update(t_ns, x, y)

    def on_release_at_loss(self, t_ns: int) -> None:
        """The pinch was opening when the hand left the frame and it has not come straight
        back: the person released and pulled away. The window is dropped where it is (no snap:
        the pointer never reached a zone) and the menu stays open, as after any release."""
        if self.state == DRAGGING and self.drag is not None:
            self.drag.cancel(t_ns)
            self._after_command(t_ns, None)

    def on_hand_seen(self, t_ns: int) -> None:
        """A frame with a hand, before any token. A paused hold stops its grace clock here, since
        the next palm token can be a smoothing window (250 ms) away; the other states keep
        counting until a token or pinch arrives, as before."""
        if self.state == HOLDING:
            self._lost_since_ns = None

    def on_hand_lost(self, t_ns: int) -> None:
        if self._lost_since_ns is None:
            self._lost_since_ns = t_ns
        self._fist_since_ns = None
        self._last_token = None
        self._idle_block = None  # the hand left: whatever it shows next is a fresh start
        if self._slide_shape is not None and self.state == ARMED:
            self._complete_slide_on_loss(t_ns)  # the anchor stays through the loss
        if self.state == SCROLL:
            # the scroll hand sits near the top and left edges, where the tracker drops it for a
            # moment: the anchor is kept and the scroll resumes when the hand is back
            self._scroll_lost_ns = t_ns
            return
        if self.state == ADJUST:
            # the pinch is dropped by the tracker for a moment: the anchor is kept and the pinch
            # resumes from it without settling again when it is back within ADJUST_LOST_GRACE_NS
            self._adjust_lost_ns = t_ns
            return
        if self.state == ARMED:
            self._leader_released = True  # came back within escape_lost: the leader shape is now a command
            self._fired_block = None  # and so is the shape that fired last
        if self.state in (ARMED, REPEAT) and self.timing.keep_armed_on_lost:
            self._refresh_on_return = True
        if self.state == HOLDING:
            if self.timing.hold_lost_grace_ns <= 0:
                self._go(IDLE, t_ns)  # strict: the palm must be continuous
                return
            self._hold_rate = 0.0  # pause: nothing is projected across the gap
            self._last_palm_ns = None  # and the gap earns no evidence when the palm is back
        if self.state == DRAGGING and self.drag is not None:
            self._returned_at_ns = None
            self.drag.suspend(t_ns)  # window stays; the drag resumes if the hand returns pinching
            if not self.drag.dragging:
                self._after_command(t_ns, None)  # a pending release was committed instead

    SCROLL_LOST_GRACE_NS = 1_000_000_000

    def on_tick(self, t_ns: int) -> None:
        if self.state == SCROLL:
            if self._scroll_lost_ns is not None and t_ns - self._scroll_lost_ns >= self.SCROLL_LOST_GRACE_NS:
                self._scroll_lost_ns = None
                self._end_scroll(t_ns)
            return
        if self.state == ADJUST:
            if self._adjust_lost_ns is not None and t_ns - self._adjust_lost_ns >= self.ADJUST_LOST_GRACE_NS:
                self._end_adjust(t_ns)  # gone for good: the menu stays open, as after a release
            return
        if self.state == DRAGGING:
            if self.drag is not None and self.drag.on_tick(t_ns):
                self._after_command(t_ns, None)  # a release outlived its re-grab grace
                return
            if (
                self.drag is not None
                and self.drag.suspended
                and self._lost_since_ns is not None
                and t_ns - self._lost_since_ns >= self.timing.drag_lost_grace_ns
            ):
                self.drag.cancel(t_ns)
                self._go(IDLE, t_ns)
            return
        if self._lost_since_ns is not None and self.state != IDLE and self._lost_escape(t_ns - self._lost_since_ns):
            self._go(IDLE, t_ns)
            return
        if self.state == HOLDING and self.projected_hold_ns(t_ns) >= self.timing.leader_hold_ns:
            self._arm(t_ns)
        elif self.state == REPEAT and self._deadline_ns is not None and t_ns >= self._deadline_ns:
            self._after_command(t_ns, self._repeat_gesture)  # slide window over: back to the menu
        elif self.state == ARMED and self._deadline_ns is not None and t_ns >= self._deadline_ns:
            self._go(IDLE, t_ns)

    # -- internals ----------------------------------------------------------

    def _lost_escape(self, gone_ns: int) -> bool:
        """Hand gone this long: give up the current state?
        A paused hold lasts only its grace.
        An armed or repeat window keeps its own deadline (on_tick checks it) and is only
        abandoned once the hand is clearly gone with nothing half-done; the command deadline
        bounds the wait either way."""
        t = self.timing
        if self.state == HOLDING:
            return gone_ns >= min(t.escape_lost_ns, t.hold_lost_grace_ns)
        if self.state in (ARMED, REPEAT) and t.keep_armed_on_lost:
            pending = self.state == REPEAT or self._fire_mass > 0
            return not pending and gone_ns >= 2 * t.escape_lost_ns
        return gone_ns >= t.escape_lost_ns

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
            self._fired_block, self._fired_gap = None, 0
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
            self._refresh_on_return = False
            self._slide_shape = None
            self._slide_out = None
            self._scroll_shape = None
            self._adjust_lost_ns = None
        self.bus.publish(
            ModeChanged(t_ns=t_ns, old=old, new=new, namespace=self.namespace, deadline_ns=self._deadline_ns)
        )

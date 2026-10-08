"""The leader-flow state machine. Time only ever arrives on tokens and Tick, so it
replays on fake time.

IDLE    --open_palm & still--------------> HOLDING (evidence 0)
HOLDING --open_palm token----------------> evidence += dt * rate(confidence)
                                           rate = clamp((c - 0.5) / 0.5, -1, 1) * gain
HOLDING --token != open_palm-------------> IDLE
HOLDING --tick, evidence >= leader_hold--> ARMED(namespace, deadline)
ARMED   --bound token--------------------> fire_mass += confidence; >= fire_evidence -> fire -> IDLE
ARMED   --different bound token----------> fire_mass restarts with that gesture
ARMED   --tick >= deadline---------------> IDLE (timeout)
ANY     --fist held escape_fist_s--------> IDLE   (only while fist is unbound in the namespace)
ANY     --hand lost escape_lost_s--------> IDLE

Confidence therefore accelerates or decelerates both phases: a sure palm arms
early, a hesitant one stalls; a sure gesture fires on one token, a weak one needs
several in a row, a very weak stream never fires.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .bindings import Bindings
from .events import Bus, HoldProgress, ModeChanged
from .recognizer import FIST, OPEN_PALM
from .types import Action, Token

IDLE, HOLDING, ARMED = "idle", "holding", "armed"


@dataclass(frozen=True)
class Timing:
    leader_hold_ns: int = 2_000_000_000
    command_timeout_ns: int = 5_000_000_000
    escape_fist_ns: int = 1_000_000_000
    escape_lost_ns: int = 1_500_000_000
    confidence_gain: float = 1.5  # fill rate at confidence 1.0, relative to wall clock
    leader_min_confidence: float = 0.8  # palm tokens below this neither start nor fill the hold


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
    ) -> None:
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

    def projected_hold_ns(self, t_ns: int) -> float:
        """Evidence extrapolated to t_ns at the last token's rate, so arming and the ring are smooth."""
        if self.state != HOLDING or self._last_palm_ns is None or self._hold_rate <= 0:
            return self.hold_evidence_ns
        return self.hold_evidence_ns + (t_ns - self._last_palm_ns) * self._hold_rate

    # -- inputs -------------------------------------------------------------

    def on_token(self, tok: Token) -> None:
        self._lost_since_ns = None
        fist_is_escape = self.bindings.lookup(self.namespace or self.default_namespace, FIST) is None
        if tok.name == FIST and fist_is_escape:
            if self._fist_since_ns is None:
                self._fist_since_ns = tok.t_ns
            elif tok.t_ns - self._fist_since_ns >= self.timing.escape_fist_ns and self.state != IDLE:
                self._go(IDLE, tok.t_ns)
                return
        else:
            self._fist_since_ns = None

        if self.state == IDLE:
            if tok.name == OPEN_PALM and tok.still and tok.confidence >= self.timing.leader_min_confidence:
                self.hold_evidence_ns = 0.0
                self._last_palm_ns = tok.t_ns
                self._go(HOLDING, tok.t_ns)
        elif self.state == HOLDING:
            if tok.name != OPEN_PALM:
                self._go(IDLE, tok.t_ns)
                return
            dt = 0 if self._last_palm_ns is None else tok.t_ns - self._last_palm_ns
            self._last_palm_ns = tok.t_ns
            sure = tok.still and tok.confidence >= self.timing.leader_min_confidence
            rate = hold_rate(tok.confidence, self.timing.confidence_gain) if sure else 0.0
            self.hold_evidence_ns = max(0.0, self.hold_evidence_ns + dt * rate)
            self._hold_rate = rate
            hold = self.timing.leader_hold_ns
            self.bus.publish(HoldProgress(tok.t_ns, min(1.0, self.hold_evidence_ns / hold), rate * 1e9 / hold))
        elif self.state == ARMED:
            if tok.name == OPEN_PALM:
                return  # the leader itself never fires a command
            action = self.bindings.lookup(self.namespace or self.default_namespace, tok.name)
            if action is None or tok.confidence < self.min_token_confidence:
                return
            if tok.name != self._fire_gesture:
                self._fire_gesture, self._fire_mass = tok.name, 0.0
            self._fire_mass += tok.confidence
            if self._fire_mass >= self.fire_evidence:
                self._go(IDLE, tok.t_ns)
                self.fire(action, tok.t_ns)

    def on_hand_lost(self, t_ns: int) -> None:
        if self._lost_since_ns is None:
            self._lost_since_ns = t_ns
        self._fist_since_ns = None

    def on_tick(self, t_ns: int) -> None:
        if (
            self._lost_since_ns is not None
            and self.state != IDLE
            and t_ns - self._lost_since_ns >= self.timing.escape_lost_ns
        ):
            self._go(IDLE, t_ns)
            return
        if self.state == HOLDING and self.projected_hold_ns(t_ns) >= self.timing.leader_hold_ns:
            self.namespace = self.default_namespace
            self._deadline_ns = t_ns + self.timing.command_timeout_ns
            self._fire_gesture, self._fire_mass = None, 0.0
            self._go(ARMED, t_ns)
        elif self.state == ARMED and self._deadline_ns is not None and t_ns >= self._deadline_ns:
            self._go(IDLE, t_ns)

    # -- internals ----------------------------------------------------------

    def _go(self, new: str, t_ns: int) -> None:
        old = self.state
        self.state = new
        if new == IDLE:
            self.namespace = None
            self.hold_evidence_ns = 0.0
            self._last_palm_ns = None
            self._deadline_ns = None
            self._fist_since_ns = None
            self._lost_since_ns = None
            self._fire_gesture, self._fire_mass = None, 0.0
            self._hold_rate = 0.0
        self.bus.publish(
            ModeChanged(t_ns=t_ns, old=old, new=new, namespace=self.namespace, deadline_ns=self._deadline_ns)
        )

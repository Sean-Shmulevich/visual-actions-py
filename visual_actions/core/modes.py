"""The leader-flow state machine. Time only ever arrives on Tick, so it replays on fake time.

IDLE    --open_palm & still--------------> HOLDING
HOLDING --token != open_palm or !still---> IDLE
HOLDING --tick, held >= leader_hold------> ARMED(namespace, deadline)
ARMED   --token in bindings[namespace]---> fire -> IDLE
ARMED   --tick >= deadline---------------> IDLE (timeout)
ANY     --fist held escape_fist_s--------> IDLE
ANY     --hand lost escape_lost_s--------> IDLE
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .bindings import Bindings
from .events import Bus, ModeChanged
from .recognizer import FIST, OPEN_PALM
from .types import Action, Token

IDLE, HOLDING, ARMED = "idle", "holding", "armed"


@dataclass(frozen=True)
class Timing:
    leader_hold_ns: int = 2_000_000_000
    command_timeout_ns: int = 5_000_000_000
    escape_fist_ns: int = 1_000_000_000
    escape_lost_ns: int = 1_500_000_000


class ModeEngine:
    def __init__(
        self,
        bus: Bus,
        bindings: Bindings,
        timing: Timing,
        fire: Callable[[Action, int], None],
        default_namespace: str = "window",
    ) -> None:
        self.bus = bus
        self.bindings = bindings
        self.timing = timing
        self.fire = fire
        self.default_namespace = default_namespace
        self.state = IDLE
        self.namespace: str | None = None
        self._hold_start_ns: int | None = None
        self._deadline_ns: int | None = None
        self._fist_since_ns: int | None = None
        self._lost_since_ns: int | None = None

    # -- inputs -------------------------------------------------------------

    def on_token(self, tok: Token) -> None:
        self._lost_since_ns = None
        if tok.name == FIST:
            if self._fist_since_ns is None:
                self._fist_since_ns = tok.t_ns
            elif tok.t_ns - self._fist_since_ns >= self.timing.escape_fist_ns and self.state != IDLE:
                self._go(IDLE, tok.t_ns)
                return
        else:
            self._fist_since_ns = None

        if self.state == IDLE:
            if tok.name == OPEN_PALM and tok.still:
                self._hold_start_ns = tok.t_ns
                self._go(HOLDING, tok.t_ns)
        elif self.state == HOLDING:
            if tok.name != OPEN_PALM or not tok.still:
                self._go(IDLE, tok.t_ns)
        elif self.state == ARMED:
            if tok.name == OPEN_PALM:
                return  # the leader itself never fires a command
            action = self.bindings.lookup(self.namespace or self.default_namespace, tok.name)
            if action is not None:
                self._go(IDLE, tok.t_ns)
                self.fire(action, tok.t_ns)

    def on_hand_lost(self, t_ns: int) -> None:
        if self._lost_since_ns is None:
            self._lost_since_ns = t_ns
        self._fist_since_ns = None

    def on_tick(self, t_ns: int) -> None:
        if self._lost_since_ns is not None and self.state != IDLE:
            if t_ns - self._lost_since_ns >= self.timing.escape_lost_ns:
                self._go(IDLE, t_ns)
                return
        if self.state == HOLDING and self._hold_start_ns is not None:
            if t_ns - self._hold_start_ns >= self.timing.leader_hold_ns:
                self.namespace = self.default_namespace
                self._deadline_ns = t_ns + self.timing.command_timeout_ns
                self._go(ARMED, t_ns)
        elif self.state == ARMED and self._deadline_ns is not None:
            if t_ns >= self._deadline_ns:
                self._go(IDLE, t_ns)

    # -- internals ----------------------------------------------------------

    def _go(self, new: str, t_ns: int) -> None:
        old = self.state
        self.state = new
        if new == IDLE:
            self.namespace = None
            self._hold_start_ns = None
            self._deadline_ns = None
            self._fist_since_ns = None
            self._lost_since_ns = None
        self.bus.publish(
            ModeChanged(t_ns=t_ns, old=old, new=new, namespace=self.namespace, deadline_ns=self._deadline_ns)
        )

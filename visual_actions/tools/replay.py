"""Replay a JSONL session through the full pipeline on fake time with mock automation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..core.config import Config, default_config
from ..core.dispatcher import Dispatcher
from ..core.drag import DragEvent
from ..core.events import ActionFired, Bus, HandLost, HandSeen, ModeChanged, Tick, TokenEmitted
from ..core.pipeline import Pipeline
from ..core.presence import PresenceFilter
from ..core.recorder import read_session
from ..core.types import HandFrame
from ..platform.mock.automation import MockAutomation


@dataclass
class ReplayResult:
    fired: list[ActionFired]
    drags: list[DragEvent]
    automation: MockAutomation
    modes: list[ModeChanged]


def replay(path: Path, config: Config | None = None, verbose: bool = False, tail_s: float = 1.0) -> list[ActionFired]:
    return replay_full(path, config, verbose, tail_s).fired


def replay_full(
    path: Path,
    config: Config | None = None,
    verbose: bool = False,
    tail_s: float = 1.0,
    automation: MockAutomation | None = None,
    presence: PresenceFilter | None = None,
) -> ReplayResult:
    """With `presence`, every recorded frame is re-filtered through it (a recording holds the
    frames that passed presence at the time, plus its lost markers): a fast exit the filter
    now predicts becomes a HandLost before the recorded marker, and the marker is skipped
    if the filter already declared the hand lost."""
    cfg = config or default_config()
    bus = Bus()
    automation = automation or MockAutomation()
    Pipeline(bus, cfg, Dispatcher(bus, automation))
    fired: list[ActionFired] = []
    drags: list[DragEvent] = []
    modes: list[ModeChanged] = []
    bus.subscribe(ActionFired, fired.append)
    bus.subscribe(DragEvent, drags.append)
    bus.subscribe(ModeChanged, modes.append)
    if verbose:
        bus.subscribe(DragEvent, lambda e: print(f"{e.t_ns / 1e9:7.3f}s drag  {e.phase.value:5s} {e.window} @({e.x:.0f},{e.y:.0f})"))
    if verbose:
        bus.subscribe(TokenEmitted, lambda e: print(f"{e.token.t_ns / 1e9:7.3f}s token {e.token.name:10s} conf={e.token.confidence:.2f} still={e.token.still}"))
        bus.subscribe(ModeChanged, lambda e: print(f"{e.t_ns / 1e9:7.3f}s mode  {e.old} -> {e.new}"))
        bus.subscribe(ActionFired, lambda e: print(f"{e.t_ns / 1e9:7.3f}s ACTION {e.action.name} ok={e.ok} {e.message}"))

    tick_ns = cfg.timing.tick_ms * 1_000_000
    clock: int | None = None
    records = list(read_session(path))
    result = ReplayResult(fired, drags, automation, modes)
    if not records:
        return result
    last_t = max(r if isinstance(r, int) else r.t_ns for r in records)
    end_t = last_t + int(tail_s * 1e9)

    def advance(to_ns: int) -> None:
        nonlocal clock
        if clock is None:
            clock = to_ns
            bus.publish(Tick(clock))
            return
        while clock + tick_ns <= to_ns:
            clock += tick_ns
            bus.publish(Tick(clock))

    for r in records:
        if isinstance(r, HandFrame):
            advance(r.t_ns)
            if presence is None:
                bus.publish(HandSeen(r))
                continue
            pr = presence.update(r)
            if pr.seen is not None:
                bus.publish(HandSeen(pr.seen))
            elif pr.lost is not None:
                bus.publish(HandLost(r.t_ns, *pr.lost))
        else:
            advance(r)
            if presence is None:
                bus.publish(HandLost(r))
            else:
                pr = presence.declare_lost("recorded", "lost marker in the recording")
                if pr.lost is not None:
                    bus.publish(HandLost(r, *pr.lost))
    advance(end_t)
    return result

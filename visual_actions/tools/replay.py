"""Replay a JSONL session through the full pipeline on fake time with mock automation."""

from __future__ import annotations

from pathlib import Path

from ..core.config import Config, default_config
from ..core.dispatcher import Dispatcher
from ..core.events import ActionFired, Bus, HandLost, HandSeen, ModeChanged, Tick, TokenEmitted
from ..core.pipeline import Pipeline
from ..core.recorder import read_session
from ..core.types import HandFrame
from ..platform.mock.automation import MockAutomation


def replay(path: Path, config: Config | None = None, verbose: bool = False, tail_s: float = 1.0) -> list[ActionFired]:
    cfg = config or default_config()
    bus = Bus()
    automation = MockAutomation()
    Pipeline(bus, cfg, Dispatcher(bus, automation))
    fired: list[ActionFired] = []
    bus.subscribe(ActionFired, fired.append)
    if verbose:
        bus.subscribe(TokenEmitted, lambda e: print(f"{e.token.t_ns / 1e9:7.3f}s token {e.token.name:10s} conf={e.token.confidence:.2f} still={e.token.still}"))
        bus.subscribe(ModeChanged, lambda e: print(f"{e.t_ns / 1e9:7.3f}s mode  {e.old} -> {e.new}"))
        bus.subscribe(ActionFired, lambda e: print(f"{e.t_ns / 1e9:7.3f}s ACTION {e.action.name} ok={e.ok} {e.message}"))

    tick_ns = cfg.timing.tick_ms * 1_000_000
    clock: int | None = None
    records = list(read_session(path))
    if not records:
        return fired
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
            bus.publish(HandSeen(r))
        else:
            advance(r)
            bus.publish(HandLost(r))
    advance(end_t)
    return fired

"""Headless robustness evaluation.

    uv run python -m visual_actions.tools.evaluate leader     # palm start / arm / false-start stats on every dataset and session

`leader` replays every recorded JSONL (datasets/<class>/* and sessions/*/landmarks.jsonl)
through the real pipeline with mock automation and counts mode transitions:
  starts  = idle -> holding        arms   = holding -> armed
  breaks  = holding -> idle        fires  = actions fired
On a non-palm class, every start is a false start and every arm a false arm.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from ..core.config import Config, default_config
from ..core.recorder import read_session
from ..core.types import HandFrame
from ..paths import datasets_dir, sessions_dir
from .replay import replay_full


@dataclass
class LeaderStats:
    frames: int = 0
    starts: int = 0
    arms: int = 0
    breaks: int = 0
    fires: int = 0
    drags: int = 0
    time_to_arm_s: list[float] = field(default_factory=list)

    def add(self, other: LeaderStats) -> None:
        self.frames += other.frames
        self.starts += other.starts
        self.arms += other.arms
        self.breaks += other.breaks
        self.fires += other.fires
        self.drags += other.drags
        self.time_to_arm_s += other.time_to_arm_s


def leader_stats(path: Path, cfg: Config) -> LeaderStats:
    st = LeaderStats(frames=sum(1 for r in read_session(path) if isinstance(r, HandFrame)))
    r = replay_full(path, cfg, tail_s=0.3)
    start_t: int | None = None
    for m in r.modes:
        if m.old == "idle" and m.new == "holding":
            st.starts += 1
            start_t = m.t_ns
        elif m.old == "holding" and m.new == "armed":
            st.arms += 1
            if start_t is not None:
                st.time_to_arm_s.append((m.t_ns - start_t) / 1e9)
        elif m.old == "holding" and m.new == "idle":
            st.breaks += 1
    st.fires = len(r.fired)
    st.drags = sum(1 for d in r.drags if d.phase.value == "start")
    return st


def run_leader(cfg: Config, datasets: Path, sessions: Path, include_public: bool) -> dict[str, LeaderStats]:
    out: dict[str, LeaderStats] = {}
    if datasets.exists():
        for cls_dir in sorted(p for p in datasets.iterdir() if p.is_dir()):
            for f in sorted(cls_dir.glob("*.jsonl")):
                if not include_public and f.name.startswith(("public-", "synth")):
                    continue
                out.setdefault(cls_dir.name, LeaderStats()).add(leader_stats(f, cfg))
    if sessions.exists():
        for f in sorted(sessions.glob("*/landmarks.jsonl")):
            out.setdefault("sessions", LeaderStats()).add(leader_stats(f, cfg))
    return out


def print_leader(stats: dict[str, LeaderStats]) -> None:
    print(f"{'class':10s} {'frames':>7s} {'starts':>7s} {'arms':>6s} {'breaks':>7s} {'fires':>6s} {'drags':>6s}  {'arm time p50':>12s}")
    for cls, s in stats.items():
        tta = f"{sorted(s.time_to_arm_s)[len(s.time_to_arm_s) // 2]:.2f}s" if s.time_to_arm_s else "-"
        print(f"{cls:10s} {s.frames:7d} {s.starts:7d} {s.arms:6d} {s.breaks:7d} {s.fires:6d} {s.drags:6d}  {tta:>12s}")
    false_starts = sum(s.starts for c, s in stats.items() if c not in ("open_palm", "sessions"))
    false_arms = sum(s.arms for c, s in stats.items() if c not in ("open_palm", "sessions"))
    frames = sum(s.frames for c, s in stats.items() if c not in ("open_palm", "sessions"))
    if frames:
        print(f"\nfalse starts on non-palm data: {false_starts} ({false_starts / frames * 1800:.2f} per minute), false arms: {false_arms}")
    palm = stats.get("open_palm")
    if palm and palm.starts:
        print(f"palm data: {palm.arms}/{palm.starts} starts reached armed ({100 * palm.arms / palm.starts:.0f}%), {palm.breaks} breaks")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["leader"])
    ap.add_argument("--datasets", type=Path, default=datasets_dir())
    ap.add_argument("--sessions", type=Path, default=sessions_dir())
    ap.add_argument("--public", action="store_true", help="include public/synthetic dataset files")
    ap.add_argument("--model", type=Path, default=None, help="classifier joblib (default: rules + the trained model if present)")
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()
    cfg = default_config()
    from ..paths import models_dir

    trained = args.model or (models_dir() / "gestures.joblib")
    if trained.exists():
        cfg.recognizer.model = str(trained)
    print_leader(run_leader(cfg, args.datasets, args.sessions, args.public))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

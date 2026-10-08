"""Headless robustness evaluation.

    uv run python -m visual_actions.tools.evaluate leader     # palm start / arm / false-start stats on every dataset and session
    uv run python -m visual_actions.tools.evaluate swipe      # three_up vs blade swipe fire rate under noise and rotation (synthetic)

`leader` replays every recorded JSONL (datasets/<class>/* and sessions/*/landmarks.jsonl)
through the real pipeline with mock automation and counts mode transitions:
  starts  = idle -> holding        arms   = holding -> armed
  breaks  = holding -> idle        fires  = actions fired
On a non-palm class, every start is a false start and every arm a false arm.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..core.config import Config, default_config
from ..core.recorder import Recorder, read_session
from ..core.types import WRIST, HandFrame, Landmark
from ..paths import datasets_dir, sessions_dir
from .replay import replay_full
from .synth import hand_frame


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


# -- swipe robustness (synthetic) -------------------------------------------------------


def _rotate(hf: HandFrame, deg: float) -> HandFrame:
    """Rotate landmarks about the wrist in the image plane (user frame)."""
    a = math.radians(deg)
    w = hf.landmarks[WRIST]
    out = []
    for lm in hf.landmarks:
        dx, dy = lm.x - w.x, lm.y - w.y
        out.append(Landmark(w.x + dx * math.cos(a) - dy * math.sin(a), w.y + dx * math.sin(a) + dy * math.cos(a), lm.z))
    return HandFrame(hf.t_ns, hf.hand, tuple(out), hf.confidence)


def write_swipe_session(path: Path, shape: str, noise: float, rot_deg: float, direction: int, rng: random.Random) -> None:
    rec = Recorder(path)
    t = 0.0
    for _ in range(45):  # palm 1.5 s
        rec.write(hand_frame("open_palm", int(t * 1e9), center=(0.5, 0.5), jitter=0.0005, rng=rng))
        t += 1 / 30
    for _ in range(12):  # shape forms and holds 0.4 s
        hf = hand_frame(shape, int(t * 1e9), center=(0.5, 0.5), jitter=noise, rng=rng, mirror_to_raw=False)
        rec.write(_mirror(_rotate(hf, rng.uniform(-rot_deg, rot_deg))))
        t += 1 / 30
    for i in range(7):  # the stroke: 0.3 of the frame in ~230 ms
        x = 0.5 + direction * 0.3 * i / 6
        hf = hand_frame(shape, int(t * 1e9), center=(x, 0.5), jitter=noise, rng=rng, mirror_to_raw=False)
        rec.write(_mirror(_rotate(hf, rng.uniform(-rot_deg, rot_deg))))
        t += 1 / 30
    rec.write_lost(int(t * 1e9))
    rec.close()


def _mirror(hf: HandFrame) -> HandFrame:
    return HandFrame(hf.t_ns, hf.hand, tuple(Landmark(1.0 - lm.x, lm.y, lm.z) for lm in hf.landmarks), hf.confidence)


def run_swipe(cfg: Config, tmp: Path, trials: int = 30, seed: int = 0) -> None:
    rng = random.Random(seed)
    cfg.recognizer.model = None  # rules only: the model has no blade data yet; this compares the shapes' geometry
    print(f"{'shape':9s} {'noise':>6s} {'rot':>5s}  {'fired':>6s} {'wrong-dir':>9s} {'none':>5s}")
    for shape in ("three_up", "blade"):
        for noise in (0.002, 0.006, 0.012):
            for rot in (0.0, 15.0, 30.0):
                counts: Counter[str] = Counter()
                for i in range(trials):
                    direction = 1 if i % 2 else -1
                    p = tmp / f"{shape}_{i}.jsonl"
                    write_swipe_session(p, shape, noise, rot, direction, rng)
                    r = replay_full(p, cfg, tail_s=0.3)
                    if not r.fired:
                        counts["none"] += 1
                    else:
                        want = "Desktop right" if direction > 0 else "Desktop left"  # user-right swipe -> "Desktop left"? see note
                        # swipe to the user's right = direction RIGHT -> binding *_swipe_right -> "Desktop left"
                        want = "Desktop left" if direction > 0 else "Desktop right"
                        counts["fired" if r.fired[0].action.name == want else "wrong-dir"] += 1
                print(f"{shape:9s} {noise:6.3f} {rot:5.0f}  {counts['fired']:6d} {counts['wrong-dir']:9d} {counts['none']:5d}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["leader", "swipe"])
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
    if args.what == "leader":
        print_leader(run_leader(cfg, args.datasets, args.sessions, args.public))
    else:
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            run_swipe(cfg, Path(d), trials=args.trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

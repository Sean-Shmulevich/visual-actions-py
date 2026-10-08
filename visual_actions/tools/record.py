"""Record a labelled landmark dataset from the camera.

    uv run python -m visual_actions.tools.record --label h_left --seconds 30
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from ..core.config import default_config
from ..core.recorder import Recorder
from ..core.tracker import MediaPipeTracker
from ..paths import datasets_dir, model_path
from ..platform import factory


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = default_config()
    services = factory.create(cfg.camera, dry_run=True)
    tracker = MediaPipeTracker(model_path())
    out = args.out or datasets_dir() / args.label / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    rec = Recorder(out)
    services.camera.open()
    print(f"recording '{args.label}' for {args.seconds:.0f}s to {out}")
    t_end = time.monotonic() + args.seconds
    seen = False
    try:
        while time.monotonic() < t_end:
            r = services.camera.read()
            if r is None:
                continue
            t_ns, frame = r
            hands = tracker.track(frame, t_ns)
            if hands:
                rec.write(hands[0])
                seen = True
            elif seen:
                rec.write_lost(t_ns)
                seen = False
    finally:
        rec.close()
        services.camera.close()
        tracker.close()
    print(f"wrote {rec.count} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

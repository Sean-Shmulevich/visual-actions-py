"""Calibrate the drag reach box and pinch thresholds from recorded data, no camera needed.

    uv run python -m visual_actions.tools.calibrate            # print suggestions
    uv run python -m visual_actions.tools.calibrate --write    # save them to the config file

Reach box: the 5th..95th percentile of wrist positions across every recorded session,
padded a little, so the screen maps to where the hand actually goes. Pinch thresholds:
if datasets/pinch and other classes exist, pick on/off from the two distributions of
thumb-index distance (on at the pinch 90th percentile, off halfway to the non-pinch
10th percentile), with sane floors and a hysteresis gap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..core.config import load_config, save_config
from ..core.normalize import to_user_frame
from ..core.pinch import pinch_distance
from ..core.recorder import read_session
from ..core.types import WRIST, HandFrame
from ..paths import config_path, datasets_dir


def frames(datasets: Path, classes: set[str] | None = None, mirror: bool = True, user_only: bool = True) -> dict[str, list[HandFrame]]:
    """User recordings only by default: public datasets have other people's hands, framing and distance."""
    out: dict[str, list[HandFrame]] = {}
    for d in sorted(p for p in datasets.iterdir() if p.is_dir()):
        if classes and d.name not in classes:
            continue
        for f in d.glob("*.jsonl"):
            if user_only and f.name.startswith(("public-", "synth")) and d.name != "pinch":
                continue
            out.setdefault(d.name, []).extend(to_user_frame(r, mirror) for r in read_session(f) if isinstance(r, HandFrame))
    return out


def reach_box(hfs: list[HandFrame], pad: float = 0.03) -> tuple[float, float, float, float]:
    xs = np.array([h.landmarks[WRIST].x for h in hfs])
    ys = np.array([h.landmarks[WRIST].y for h in hfs])
    x0, x1 = np.percentile(xs, [5, 95])
    y0, y1 = np.percentile(ys, [5, 95])
    x0, x1 = max(0.0, x0 - pad), min(1.0, x1 + pad)
    y0, y1 = max(0.0, y0 - pad), min(1.0, y1 + pad)
    return (float(x0), float(x1), float(y0), float(y1))


def typical_hand_scale(hfs: list[HandFrame]) -> float:
    """Median wrist-to-middle-knuckle length in frame units: the user's normal seating distance."""
    from ..core.pinch import hand_scale

    return float(np.median([hand_scale(h) for h in hfs]))


def pinch_thresholds(pinch: list[HandFrame], other: list[HandFrame]) -> tuple[float, float, dict[str, float]]:
    dp = np.array([pinch_distance(h) for h in pinch])
    do = np.array([pinch_distance(h) for h in other])
    p90 = float(np.percentile(dp, 90))
    o10 = float(np.percentile(do, 10))
    on = min(max(p90, 0.15), 0.45)
    off = max(on + 0.12, min((on + o10) / 2, 0.8))
    stats = {"pinch_p50": float(np.median(dp)), "pinch_p90": p90, "other_p10": o10, "other_p50": float(np.median(do))}
    return on, off, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=Path, default=datasets_dir())
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--depth-gain", type=float, default=None, help="also set drag.depth_gain (0 = speed follows distance, the default; 1 = normalized)")
    args = ap.parse_args()

    by_class = frames(args.datasets)
    if not by_class:
        print(f"no datasets under {args.datasets}")
        return 1
    everything = [h for hfs in by_class.values() for h in hfs]
    x0, x1, y0, y1 = reach_box(everything)
    user_frames = [h for c, hfs in by_class.items() for h in hfs]  # datasets are the user's own sessions by default
    ref_scale = typical_hand_scale(user_frames)
    print(f"{len(everything)} frames across {len(by_class)} classes")
    print(f"reach box: x {x0:.2f}..{x1:.2f}  y {y0:.2f}..{y1:.2f}")
    print(f"typical hand scale (seating distance): {ref_scale:.3f} of frame width (default 0.12)")

    on = off = None
    if "pinch" in by_class:
        other = [h for c, hfs in by_class.items() if c not in ("pinch",) for h in hfs]
        on, off, stats = pinch_thresholds(by_class["pinch"], other)
        print(f"pinch thresholds: on {on:.2f} off {off:.2f}   ({', '.join(f'{k}={v:.2f}' for k, v in stats.items())})")
    else:
        print("no datasets/pinch: keeping default pinch thresholds")

    if args.write:
        path = args.config or config_path()
        cfg = load_config(path)
        cfg.drag.box_x0, cfg.drag.box_x1, cfg.drag.box_y0, cfg.drag.box_y1 = x0, x1, y0, y1
        cfg.drag.ref_hand_scale = round(ref_scale, 4)
        if args.depth_gain is not None:
            cfg.drag.depth_gain = args.depth_gain
        if on is not None and off is not None:
            cfg.drag.pinch_on, cfg.drag.pinch_off = on, off
        save_config(cfg, path)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Record labelled landmark datasets from the camera.

One label:
    uv run python -m visual_actions.tools.record --label h_left --seconds 30

Guided session, all v0.1 poses in a row with countdowns and spoken prompts:
    uv run python -m visual_actions.tools.record --guided --seconds 30
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.config import default_config
from ..core.recorder import Recorder
from ..core.tracker import MediaPipeTracker
from ..core.types import HandFrame
from ..paths import datasets_dir, model_path
from ..platform import factory

GUIDED = [
    ("open_palm", "Open palm facing the camera, fingers spread. Hold it still, then move it around slowly."),
    ("fist", "Closed fist. Turn it slowly: knuckles to the camera, then thumb side, then palm side."),
    ("h_left", "Index and middle finger together pointing to your left, thumb tucked, ring and pinky folded."),
    ("h_right", "Same H sign, but pointing to your right."),
    ("none", "Anything else: relax the hand, point one finger, wave, scratch your head, rest it on the desk."),
]

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def say(text: str) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["say", "-r", "200", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(text, flush=True)


class Preview:
    def __init__(self, title: str = "visual-actions record") -> None:
        import cv2

        self.cv2 = cv2
        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 640, 480)

    def show(self, frame: Any, hands: list[HandFrame], label: str, remaining: float, count: int, state: str) -> None:
        cv2 = self.cv2
        img = cv2.flip(frame, 1)  # selfie view
        h, w = img.shape[:2]
        for hf in hands:
            pts = [(int((1 - lm.x) * w), int(lm.y * h)) for lm in hf.landmarks]
            for a, b in CONNECTIONS:
                cv2.line(img, pts[a], pts[b], (0, 255, 0), 2)
            for p in pts:
                cv2.circle(img, p, 3, (0, 0, 255), -1)
        color = (0, 200, 0) if hands else (0, 0, 255)
        cv2.putText(img, f"{state}: {label}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(img, f"{remaining:4.1f}s left   {count} frames", (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if not hands:
            cv2.putText(img, "NO HAND SEEN", (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow(self.title, img)
        cv2.waitKey(1)

    def close(self) -> None:
        self.cv2.destroyAllWindows()
        self.cv2.waitKey(1)


def record_one(camera, tracker, preview: Preview | None, label: str, seconds: float, countdown: float, out: Path) -> int:
    # countdown with the preview live so the user can get in frame
    t_go = time.monotonic() + countdown
    last_said = None
    while time.monotonic() < t_go:
        r = camera.read()
        if r is None:
            continue
        t_ns, frame = r
        hands = tracker.track(frame, t_ns)
        left = t_go - time.monotonic()
        n = int(left) + 1
        if n != last_said and n <= 3:
            say(str(n))
            last_said = n
        if preview:
            preview.show(frame, hands, label, left, 0, "get ready")
    say("go")
    rec = Recorder(out)
    seen = False
    t_end = time.monotonic() + seconds
    try:
        while time.monotonic() < t_end:
            r = camera.read()
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
            if preview:
                preview.show(frame, hands, label, t_end - time.monotonic(), rec.count, "RECORDING")
    finally:
        rec.close()
    say(f"done, {rec.count} frames")
    return rec.count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--guided", action="store_true", help="record every v0.1 pose in sequence")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--countdown", type=float, default=5.0)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="file (single label) or directory (guided)")
    args = ap.parse_args()
    if not args.label and not args.guided:
        ap.error("need --label or --guided")

    cfg = default_config()
    services = factory.create(cfg.camera, dry_run=True)
    tracker = MediaPipeTracker(model_path())
    services.camera.open()
    preview = None if args.no_preview else Preview()
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"  # noqa: DTZ005 - local time is what a user expects in a filename
    try:
        if args.guided:
            base = args.out or datasets_dir()
            say(f"Guided recording. {len(GUIDED)} poses, {args.seconds:.0f} seconds each. Watch the preview window.")
            time.sleep(2)
            for label, instructions in GUIDED:
                seconds = args.seconds * (2 if label == "none" else 1)  # 'none' needs variety
                say(f"Next: {label.replace('_', ' ')}, {seconds:.0f} seconds. {instructions}")
                time.sleep(3)
                record_one(services.camera, tracker, preview, label, seconds, args.countdown, base / label / f"{stamp}.jsonl")
                time.sleep(1)
            say("All done.")
        else:
            out = args.out or datasets_dir() / args.label / f"{stamp}.jsonl"
            record_one(services.camera, tracker, preview, args.label, args.seconds, args.countdown, out)
            print(f"wrote {out}")
    finally:
        if preview:
            preview.close()
        services.camera.close()
        tracker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

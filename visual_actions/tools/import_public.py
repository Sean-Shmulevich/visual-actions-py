"""Harvest landmark training data from public hand-gesture datasets.

Two sources:

1. Image folders, one sub-folder per dataset class, run through our own MediaPipe
   landmarker in IMAGE mode:

       uv run python -m visual_actions.tools.import_public images \\
           --root "~/Library/Application Support/visual-actions/public/hagrid_hf_subset" --name hagrid_hf

2. HaGRIDv2 `annotations.zip` (719 MB, no images needed): per-image JSON with
   `labels` and MediaPipe `hand_landmarks` (21 x [x, y], normalized):

       uv run python -m visual_actions.tools.import_public hagrid-annotations \\
           --zip "~/Library/Application Support/visual-actions/public/hagrid_v2_annotations/annotations.zip" \\
           --per-class 1500

Both write `datasets/<our class>/public-<name>.jsonl` next to the webcam sessions
so `tools/train.py` picks them up as extra sessions, plus a `pinch` / `no_pinch`
pair for the pinch detector.

Caveats: HaGRID is third-person, FullHD, mostly frontal and well lit, so landmarks
are cleaner than a laptop webcam. The annotation landmarks have no z (written as
0.0). Handedness is as MediaPipe (or the annotators' MediaPipe run) reported it.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import time
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol

from ..core.recorder import Recorder
from ..core.types import N_LANDMARKS, Hand, HandFrame, Landmark

# dataset class name -> our class. Anything missing maps to "none".
CLASS_MAP: dict[str, str] = {
    "ok": "pinch",
    "palm": "open_palm",
    "stop": "open_palm",
    "fist": "fist",
    "one": "point_up",
    "point": "none",  # HaGRID "point" aims at the camera, not up
    "peace": "two_up",
    "two_up": "two_up",
    # explicit "none" families, listed so the mapping is reviewable
    "like": "thumbs_up",
    "dislike": "thumbs_down",
    "call": "none",
    "rock": "none",
    "mute": "none",
    "three": "none",
    "three2": "none",
    "three3": "none",
    "four": "none",
    "grabbing": "none",
    "grip": "none",
    "no_gesture": "none",
    "stop_inverted": "none",
    "peace_inverted": "none",
    "two_up_inverted": "none",
    "middle_finger": "none",
    "little_finger": "none",
    "thumb_index": "none",
    "thumb_index2": "none",
    "hand_heart": "none",
    "hand_heart2": "none",
    "holy": "none",
    "take_picture": "none",
    "three_gun": "none",
    "timeout": "none",
    "xsign": "none",
}
NONE = "none"
PINCH = "pinch"
NO_PINCH = "no_pinch"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def map_class(dataset_class: str) -> str:
    return CLASS_MAP.get(dataset_class.strip().lower(), NONE)


class Tracker(Protocol):
    def track(self, frame_bgr: Any, t_ns: int) -> list[HandFrame]: ...

    def close(self) -> None: ...


# -- writers ---------------------------------------------------------------------


class ClassWriter:
    """One JSONL per (our class) under datasets/<class>/public-<name>.jsonl, plus the
    pinch / no_pinch pair. no_pinch is sampled from every non-pinch frame at the end."""

    def __init__(self, datasets_dir: Path, name: str, seed: int = 0) -> None:
        self.datasets_dir = datasets_dir
        self.name = name
        self.counts: Counter[str] = Counter()
        self._writers: dict[str, Recorder] = {}
        self._non_pinch: list[HandFrame] = []
        self._rng = random.Random(seed)

    def _writer(self, cls: str) -> Recorder:
        if cls not in self._writers:
            self._writers[cls] = Recorder(self.datasets_dir / cls / f"public-{self.name}.jsonl")
        return self._writers[cls]

    def write(self, cls: str, hf: HandFrame) -> None:
        self._writer(cls).write(hf)
        self.counts[cls] += 1
        if cls != PINCH:
            self._non_pinch.append(hf)

    def close(self) -> dict[str, int]:
        n_pinch = self.counts.get(PINCH, 0)
        if n_pinch and self._non_pinch:
            k = min(len(self._non_pinch), n_pinch)
            for hf in self._rng.sample(self._non_pinch, k):
                self._writer(NO_PINCH).write(hf)
                self.counts[NO_PINCH] += 1
        for w in self._writers.values():
            w.close()
        self._writers.clear()
        return dict(self.counts)


# -- source 1: image folders ---------------------------------------------------


def iter_image_folders(root: Path, per_class: int | None = None) -> Iterator[tuple[str, Path]]:
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if per_class is not None:
            files = files[:per_class]
        for f in files:
            yield cls_dir.name, f


def import_images(
    root: Path,
    name: str,
    datasets_dir: Path,
    tracker: Tracker,
    per_class: int | None = None,
    min_confidence: float = 0.6,
    reader: Any = None,
) -> dict[str, Any]:
    """Run the tracker on every image. Keeps frames with exactly one hand at or above
    min_confidence. Returns counts and skip reasons."""
    if reader is None:
        import cv2

        reader = cv2.imread
    writer = ClassWriter(datasets_dir, name)
    skipped: Counter[str] = Counter()
    t0 = time.perf_counter()
    for i, (dataset_cls, path) in enumerate(iter_image_folders(root, per_class)):
        frame = reader(str(path))
        if frame is None:
            skipped["unreadable"] += 1
            continue
        hands = tracker.track(frame, i)
        if len(hands) != 1:
            skipped["no_hand" if not hands else "multi_hand"] += 1
            continue
        hf = hands[0]
        if hf.confidence < min_confidence:
            skipped["low_confidence"] += 1
            continue
        writer.write(map_class(dataset_cls), hf)
    counts = writer.close()
    return {"counts": counts, "skipped": dict(skipped), "seconds": round(time.perf_counter() - t0, 1)}


# -- source 2: HaGRIDv2 annotations -------------------------------------------


def hagrid_record_to_frames(key: str, rec: dict[str, Any], t_ns: int) -> list[tuple[str, HandFrame]]:
    """One (dataset label, HandFrame) per hand that has 21 landmarks."""
    labels = rec.get("labels") or []
    lms_per_hand = rec.get("hand_landmarks") or []
    out: list[tuple[str, HandFrame]] = []
    for label, lms in zip(labels, lms_per_hand):
        if not lms or len(lms) != N_LANDMARKS:
            continue
        pts = tuple(Landmark(float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0) for p in lms)
        # HaGRID annotations carry no handedness; the trainer ignores it (see normalize.py).
        out.append((label, HandFrame(t_ns=t_ns, hand=Hand.RIGHT, landmarks=pts, confidence=1.0)))
    return out


def iter_hagrid_annotation_files(zip_path: Path, splits: Iterable[str] = ("val",)) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yields (file name, parsed JSON) for every <split>/<class>.json in the archive.
    The archive holds train/val/test (4.2 GB of JSON in all); val alone has ~3,000
    records per class, which is plenty. `.ipynb_checkpoints` duplicates are skipped."""
    wanted = {f"annotations/{s}/" for s in splits}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = info.filename
            if not name.lower().endswith(".json") or ".ipynb_checkpoints" in name:
                continue
            if not any(name.startswith(w) for w in wanted):
                continue
            with z.open(info) as f:
                yield name, json.load(io.TextIOWrapper(f, encoding="utf-8"))


def import_hagrid_annotations(
    records: Iterable[tuple[str, dict[str, Any]]],
    name: str,
    datasets_dir: Path,
    per_class: int = 1500,
    seed: int = 0,
    only_single_hand: bool = True,
) -> dict[str, Any]:
    """`records` = (file name, {image_id: annotation}) pairs. Takes up to per_class
    hands per OUR class via reservoir sampling so every source file contributes."""
    rng = random.Random(seed)
    reservoir: dict[str, list[HandFrame]] = defaultdict(list)
    seen: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    t0 = time.perf_counter()
    t_ns = 0
    for _fname, data in records:
        for key, rec in data.items():
            frames = hagrid_record_to_frames(key, rec, t_ns)
            t_ns += 1
            if not frames:
                skipped["no_landmarks"] += 1
                continue
            if only_single_hand and len(frames) > 1:
                skipped["multi_hand"] += 1
                continue
            for label, hf in frames:
                cls = map_class(label)
                seen[cls] += 1
                bucket = reservoir[cls]
                if len(bucket) < per_class:
                    bucket.append(hf)
                else:
                    j = rng.randrange(seen[cls])
                    if j < per_class:
                        bucket[j] = hf
    writer = ClassWriter(datasets_dir, name, seed=seed)
    for cls, frames in reservoir.items():
        for hf in frames:
            writer.write(cls, hf)
    counts = writer.close()
    return {
        "counts": counts,
        "seen": dict(seen),
        "skipped": dict(skipped),
        "seconds": round(time.perf_counter() - t0, 1),
    }


# -- CLI -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from ..paths import datasets_dir, model_path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("images", help="image folders, one sub-folder per dataset class")
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--name", required=True, help="suffix for public-<name>.jsonl")
    a.add_argument("--per-class", type=int, default=None)
    a.add_argument("--out", type=Path, default=datasets_dir())
    b = sub.add_parser("hagrid-annotations", help="HaGRIDv2 annotations.zip with landmarks")
    b.add_argument("--zip", type=Path, required=True)
    b.add_argument("--name", default="hagrid_v2")
    b.add_argument("--per-class", type=int, default=1500)
    b.add_argument("--splits", nargs="*", default=["val"], help="val, test, train (train is ~3 GB of JSON)")
    b.add_argument("--out", type=Path, default=datasets_dir())
    args = ap.parse_args(argv)

    if args.cmd == "images":
        from ..core.tracker import ImageTracker

        tracker = ImageTracker(model_path(), num_hands=2)
        try:
            report = import_images(args.root.expanduser(), args.name, args.out, tracker, args.per_class)
        finally:
            tracker.close()
    else:
        report = import_hagrid_annotations(
            iter_hagrid_annotation_files(args.zip.expanduser(), args.splits), args.name, args.out, args.per_class
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

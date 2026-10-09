"""Train a per-user candidate classifier from datasets/.

Same features and pipeline as tools/train.py (normalize.features, StandardScaler + logistic
regression), but weighted for THIS user: frames the user recorded and the review exports of
their sessions carry `weight` (default 1.0) x `learn.user_weight`; public (HaGRID) and
synthetic frames weigh 1.0 and act as a prior. Fist frames are accepted from humans only, so
the cancel gesture can never be taught by a judge (export.py refuses too; checked again here).

Output: models/user/gestures-<date>-<shorthash>.joblib and, next to it, the manifest
gestures-<date>-<shorthash>.manifest.json: when, where (hostname, camera/location tag), what
(every dataset file with its hash, frame counts by class and source), how good (metrics)
and what was decided. Deterministic for a given set of files (random_state=0).
"""

from __future__ import annotations

import hashlib
import json
import socket
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import LearnConfig
from ..core.normalize import features, to_user_frame
from ..core.recorder import read_records
from ..core.types import HandFrame
from ..tools.train import CLASSES, class_dirs

FIST = "fist"
REVIEW_PREFIX = "review-"
SOURCE_USER, SOURCE_PUBLIC, SOURCE_SYNTH, SOURCE_HUMAN, SOURCE_JUDGE = "user", "public", "synth", "human", "judge"


def source_of(path: Path) -> str:
    """Where a dataset file came from, by name: public-* (imported), synth* (generated),
    review-* (labelled session exports; each line says human or judge), else the user's recorder."""
    n = path.name
    if n.startswith("public-"):
        return SOURCE_PUBLIC
    if n.startswith("synth"):
        return SOURCE_SYNTH
    if n.startswith(REVIEW_PREFIX):
        return "review"
    return SOURCE_USER


def review_session(path: Path) -> str | None:
    """`review-<session>-<YYYYmmdd-HHMMSS>.jsonl` -> the session stamp, else None."""
    if not path.name.startswith(REVIEW_PREFIX):
        return None
    return path.stem[len(REVIEW_PREFIX) :].rsplit("-", 2)[0]


def file_sha256(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


@dataclass
class FileInfo:
    file: str  # "<class>/<name>"
    sha256: str
    frames: int
    source: str  # user | public | synth | review
    cls: str
    session: str | None = None  # for review files


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    groups: np.ndarray  # session key per frame: review session stamp, else the file
    sources: np.ndarray  # user | public | synth | human | judge
    files: list[FileInfo] = field(default_factory=list)
    rejected_fist_judge: int = 0  # judge-labelled fist rows refused
    frames: list[HandFrame] = field(default_factory=list)  # user-frame HandFrames, for rule-based scoring

    def __len__(self) -> int:
        return len(self.y)

    def counts(self) -> dict[str, dict[str, int]]:
        return {
            "by_class": dict(sorted(Counter(self.y.tolist()).items())),
            "by_source": dict(sorted(Counter(self.sources.tolist()).items())),
        }

    @staticmethod
    def empty() -> Dataset:
        return Dataset(np.zeros((0, 0)), np.array([], dtype=str), np.array([]), np.array([], dtype=str), np.array([], dtype=str))


class _Builder:
    def __init__(self) -> None:
        self.xs: list[np.ndarray] = []
        self.ys: list[str] = []
        self.ws: list[float] = []
        self.gs: list[str] = []
        self.ss: list[str] = []
        self.files: list[FileInfo] = []
        self.rejected_fist = 0
        self.frames: list[HandFrame] = []

    def build(self) -> Dataset:
        if not self.ys:
            d = Dataset.empty()
            d.files, d.rejected_fist_judge = self.files, self.rejected_fist
            return d
        return Dataset(np.array(self.xs), np.array(self.ys), np.array(self.ws), np.array(self.gs), np.array(self.ss), self.files, self.rejected_fist, self.frames)


def load_dataset(
    datasets: Path,
    user_weight: float = 2.0,
    mirror: bool = True,
    holdout_sessions: frozenset[str] | set[str] = frozenset(),
    classes: frozenset[str] = CLASSES,
) -> tuple[Dataset, Dataset]:
    """(train, holdout): every class file under `datasets`, weighted per source; the review
    files of `holdout_sessions` go to the second dataset (frame-accuracy evaluation) and never
    train. Only the keys the recorder wrote are features; `weight` and `source` ride along."""
    train, hold = _Builder(), _Builder()
    for label_dir in class_dirs(datasets, classes):
        cls = label_dir.name
        for f in sorted(label_dir.glob("*.jsonl")):
            src = source_of(f)
            session = review_session(f)
            held = session is not None and session in holdout_sessions
            b = hold if held else train
            group = session or f"{cls}/{f.name}"
            n = 0
            for rec, extra in read_records(f):
                if not isinstance(rec, HandFrame):
                    continue
                line_src = str(extra.get("source", src)) if src == "review" else src
                if cls == FIST and line_src == SOURCE_JUDGE:
                    b.rejected_fist += 1
                    continue
                w = float(extra.get("weight", 1.0))
                if src in (SOURCE_PUBLIC, SOURCE_SYNTH):
                    w = 1.0
                else:
                    w *= user_weight
                uf = to_user_frame(rec, mirror)
                b.frames.append(uf)
                b.xs.append(features(uf))
                b.ys.append(cls)
                b.ws.append(w)
                b.gs.append(group)
                b.ss.append(line_src)
                n += 1
            b.files.append(FileInfo(f"{cls}/{f.name}", file_sha256(f), n, src, cls, session))
    return train.build(), hold.build()


def build_pipeline() -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=0))


def fit(ds: Dataset) -> Any:
    m = build_pipeline()
    m.fit(ds.x, ds.y, logisticregression__sample_weight=ds.w)
    return m


def short_hash(files: list[FileInfo], cfg: LearnConfig) -> str:
    h = hashlib.sha256()
    for fi in sorted(files, key=lambda f: f.file):
        h.update(f"{fi.file}:{fi.sha256}\n".encode())
    h.update(f"user_weight={cfg.user_weight}".encode())
    return h.hexdigest()[:8]


@dataclass
class Candidate:
    name: str  # gestures-<date>-<hash>
    path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    model: Any
    train: Dataset
    holdout: Dataset


def manifest_path_for(model_path: Path) -> Path:
    return model_path.with_name(model_path.stem + ".manifest.json")


def write_manifest(c: Candidate) -> None:
    c.manifest_path.write_text(json.dumps(c.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_candidate(
    datasets: Path,
    out_dir: Path,
    cfg: LearnConfig,
    day: date,
    holdout_sessions: frozenset[str] | set[str] = frozenset(),
    mirror: bool = True,
    hostname: str | None = None,
) -> Candidate | None:
    """Fit the candidate on every non-held-out dataset file and write it with its manifest
    under `out_dir`. None when there is no training data."""
    train, hold = load_dataset(datasets, cfg.user_weight, mirror, holdout_sessions)
    if len(train) == 0 or len(set(train.y.tolist())) < 2:
        return None
    model = fit(train)
    name = f"gestures-{day.isoformat()}-{short_hash(train.files, cfg)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.joblib"
    import joblib

    joblib.dump(model, path)
    train_acc = float(np.average(model.predict(train.x) == train.y, weights=train.w))
    manifest: dict[str, Any] = {
        "name": name,
        "date": day.isoformat(),
        "trained_at": datetime.now().isoformat(timespec="seconds"),  # noqa: DTZ005 - local, like the session stamps
        "hostname": hostname or socket.gethostname(),
        "location": cfg.location,
        "user_weight": cfg.user_weight,
        "model": "logreg(C=1.0, random_state=0) over StandardScaler, features=normalize.features",
        "datasets": [fi.__dict__ for fi in train.files],
        "holdout_sessions": sorted(holdout_sessions),
        "holdout_files": [fi.__dict__ for fi in hold.files],
        "frames": {**train.counts(), "total": len(train), "rejected_fist_judge": train.rejected_fist_judge},
        "train_accuracy_weighted": round(train_acc, 4),
        "metrics": {},
        "decision": "pending",
    }
    c = Candidate(name, path, manifest_path_for(path), manifest, model, train, hold)
    write_manifest(c)
    return c

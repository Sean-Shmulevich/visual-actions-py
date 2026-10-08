"""Train the tier 2 classifier from recorded datasets.

    uv run python -m visual_actions.tools.train            # datasets dir -> models/gestures.joblib
    uv run python -m visual_actions.tools.train --datasets tests/fixtures/real --out /tmp/m.joblib

Hold-out is by session file, never by frame: frames within one recording are correlated.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from ..core.normalize import features, to_user_frame
from ..core.recorder import read_session
from ..core.types import HandFrame
from ..paths import datasets_dir, models_dir

EXCLUDE = {"no_pinch", "three_up", "blade"}  # swipe shapes live on the swipe-desktops branch  # derived from the other classes by import_public; not a gesture


def load(datasets: Path, mirror: bool = True, exclude: set[str] = EXCLUDE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, groups = [], [], []
    for label_dir in sorted(p for p in datasets.iterdir() if p.is_dir() and p.name not in exclude):
        for i, f in enumerate(sorted(label_dir.glob("*.jsonl"))):
            for r in read_session(f):
                if isinstance(r, HandFrame):
                    xs.append(features(to_user_frame(r, mirror)))
                    ys.append(label_dir.name)
                    groups.append(f"{label_dir.name}/{f.name}")
    # flat layout fallback: datasets/<label>.jsonl
    for f in sorted(datasets.glob("*.jsonl")):
        for r in read_session(f):
            if isinstance(r, HandFrame):
                xs.append(features(to_user_frame(r, mirror)))
                ys.append(f.stem)
                groups.append(f.name)
    return np.array(xs), np.array(ys), np.array(groups)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=Path, default=datasets_dir())
    ap.add_argument("--out", type=Path, default=models_dir() / "gestures.joblib")
    ap.add_argument("--model", choices=["logreg", "mlp"], default="logreg")
    args = ap.parse_args()

    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x, y, g = load(args.datasets)
    if len(x) == 0:
        print(f"no data under {args.datasets}")
        return 1
    print(f"{len(x)} frames, classes: {dict(Counter(y))}")
    print(f"sessions: {len(set(g))}")

    def build():
        clf = (
            LogisticRegression(max_iter=2000, C=1.0)
            if args.model == "logreg"
            else MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=0)
        )
        return make_pipeline(StandardScaler(), clf)

    # Held-out evaluation: the newest session of EVERY class is the test set, so each
    # class appears on both sides and the score reflects a new day, not new frames.
    sessions_per_class: dict[str, list[str]] = {}
    for grp in sorted(set(g)):
        sessions_per_class.setdefault(grp.split("/")[0], []).append(grp)
    # Public datasets only ever train; the test set is each class's newest USER session.
    user_sessions = {c: [g for g in s if "/public-" not in g and "/synth" not in g] for c, s in sessions_per_class.items()}
    testable = {c: s for c, s in user_sessions.items() if len(s) >= 1 and len(sessions_per_class[c]) >= 2}
    if testable:
        test_groups = {s[-1] for s in testable.values()}
        te = np.array([grp in test_groups for grp in g])
        tr = ~te
        m = build().fit(x[tr], y[tr])
        pred = m.predict(x[te])
        labels = sorted(set(y))
        print(f"held-out = newest user session of {sorted(testable)} (public/synth files train only):")
        print(classification_report(y[te], pred, labels=labels, zero_division=0))  # pyright: ignore[reportArgumentType]
        print("confusion (rows=true, cols=pred):", labels)
        print(confusion_matrix(y[te], pred, labels=labels))
    else:
        print("only one session per class: no held-out score yet, record a second session on another day")

    final = build().fit(x, y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, args.out)
    train_acc = float((final.predict(x) == y).mean())
    print(f"saved {args.out} (train accuracy {train_acc:.3f}, {args.model})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

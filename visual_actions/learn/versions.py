"""The per-user model pointer: models/user/current.joblib -> gestures-<date>-<hash>.joblib.

The pointer is a relative symlink switched with os.replace, so the live app (which opens
the file by this path on every rebuild) either sees the old version or the new one, never a
half-written file: a version file is written once and never touched again. versions.json
beside it remembers the promotion order for `rollback`. Nothing here writes
models/gestures.joblib, the shipped model, which stays the fallback.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

CURRENT = "current.joblib"
VERSIONS = "versions.json"
PREFIX = "gestures-"


def _read(user_dir: Path) -> dict[str, Any]:
    p = user_dir / VERSIONS
    if not p.exists():
        return {"current": None, "history": []}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"current": None, "history": []}
    d.setdefault("current", None)
    d.setdefault("history", [])
    return d


def _write(user_dir: Path, data: dict[str, Any]) -> None:
    user_dir.mkdir(parents=True, exist_ok=True)
    tmp = user_dir / (VERSIONS + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, user_dir / VERSIONS)


def _point(user_dir: Path, name: str | None) -> None:
    """Atomically make current.joblib point at `name` (None removes the pointer)."""
    user_dir.mkdir(parents=True, exist_ok=True)
    link = user_dir / CURRENT
    if name is None:
        if link.is_symlink() or link.exists():
            link.unlink()
        return
    tmp = user_dir / (CURRENT + ".tmp")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(name)  # relative: the models dir can move with the data dir
    os.replace(tmp, link)


def current_name(user_dir: Path) -> str | None:
    """The version file current.joblib points at, or None (no pointer, or it dangles)."""
    link = user_dir / CURRENT
    if not link.is_symlink():
        return None
    target = os.readlink(link)
    return Path(target).name if (user_dir / target).exists() else None


def current_path(user_dir: Path) -> Path | None:
    name = current_name(user_dir)
    return user_dir / name if name else None


def champion_path(user_dir: Path, shipped: Path) -> Path | None:
    """What the live app runs: the promoted user model, else the shipped one, else rules (None)."""
    return current_path(user_dir) or (shipped if shipped.exists() else None)


def list_versions(user_dir: Path) -> list[Path]:
    """Version files, oldest first (the name sorts by date, then hash)."""
    if not user_dir.exists():
        return []
    return sorted(p for p in user_dir.glob(f"{PREFIX}*.joblib") if not p.is_symlink())


def promote(user_dir: Path, model_path: Path) -> None:
    """Point current.joblib at `model_path` (a file inside user_dir) and record it."""
    model_path = Path(model_path)
    if model_path.parent.resolve() != user_dir.resolve():
        raise ValueError(f"{model_path} is not under {user_dir}")
    data = _read(user_dir)
    _point(user_dir, model_path.name)
    data["history"] = [n for n in data["history"] if n != model_path.name] + [model_path.name]
    data["current"] = model_path.name
    data["promoted_at"] = datetime.now().isoformat(timespec="seconds")  # noqa: DTZ005
    _write(user_dir, data)


def rollback(user_dir: Path) -> Path | None:
    """Repoint current.joblib at the version promoted before the current one. With no earlier
    version the pointer is removed and the shipped model is live again. Returns the new target."""
    data = _read(user_dir)
    history: list[str] = [n for n in data["history"] if (user_dir / n).exists()]
    cur = current_name(user_dir)
    if cur in history:
        history = history[: history.index(cur)]
    prev = history[-1] if history else None
    _point(user_dir, prev)
    data["history"] = history
    data["current"] = prev
    data["rolled_back_at"] = datetime.now().isoformat(timespec="seconds")  # noqa: DTZ005
    _write(user_dir, data)
    return user_dir / prev if prev else None


def prune(user_dir: Path, keep: int = 5) -> list[Path]:
    """Delete the oldest versions (and their manifests) beyond `keep`, never the current one.
    Returns what was removed."""
    versions = list_versions(user_dir)
    cur = current_name(user_dir)
    removed: list[Path] = []
    excess = len(versions) - keep
    for p in versions:
        if excess <= 0:
            break
        if p.name == cur:
            continue
        p.unlink()
        manifest = p.with_name(p.stem + ".manifest.json")
        if manifest.exists():
            manifest.unlink()
        removed.append(p)
        excess -= 1
    if removed:
        data = _read(user_dir)
        gone = {p.name for p in removed}
        data["history"] = [n for n in data["history"] if n not in gone]
        _write(user_dir, data)
    return removed

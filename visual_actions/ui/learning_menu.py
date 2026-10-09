"""The Learning submenu's view-model: what it shows about the nightly job, without rumps.

The job itself lives in visual_actions.learn (imported lazily: the menu degrades to disabled
"not available" items when the package or its files are missing). Its state is
learn/last.json under the data dir and learn/reports/<date>.md; this module reads them
defensively, since the job owns their exact shape.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any

from ..paths import REPO_ROOT, data_dir

DEFAULT_HOUR = 2


def learn_dir() -> Path:
    return data_dir() / "learn"


def last_json_path() -> Path:
    return learn_dir() / "last.json"


def reports_dir() -> Path:
    return learn_dir() / "reports"


def run_log_path() -> Path:
    return learn_dir() / "run.log"


def run_command() -> list[str]:
    """`python -m visual_actions.learn run` with this interpreter; run it with cwd=REPO_ROOT."""
    return [sys.executable, "-m", "visual_actions.learn", "run"]


def repo_root() -> Path:
    return REPO_ROOT


def schedule_module() -> ModuleType | None:
    """visual_actions.learn.schedule (install(hour) / uninstall() / status()), or None when absent."""
    try:
        from ..learn import schedule
    except Exception:  # noqa: BLE001 - missing package, missing file, or an import error inside it
        return None
    return schedule


def learn_available() -> bool:
    """Whether `python -m visual_actions.learn` exists to run (the package's __main__)."""
    return (REPO_ROOT / "visual_actions" / "learn" / "__main__.py").is_file()


def learn_settings(cfg: object) -> tuple[bool, int]:
    """(enabled, hour) from the config's [learn] section, with defaults when it is not there yet."""
    section = getattr(cfg, "learn", None)
    enabled = bool(getattr(section, "enabled", True))
    hour = getattr(section, "hour", DEFAULT_HOUR)
    try:
        hour = int(hour) % 24
    except (TypeError, ValueError):
        hour = DEFAULT_HOUR
    return enabled, hour


def nightly_title(hour: int) -> str:
    return f"Improve nightly at {hour % 24:02d}:00"


def schedule_installed(status: object) -> bool:
    """Read the schedule's status() answer whatever its shape: a bool, a dict, an object, a string."""
    if isinstance(status, bool):
        return status
    if isinstance(status, dict):
        for key in ("installed", "loaded", "enabled", "active"):
            if key in status:
                return bool(status[key])
        return False
    if isinstance(status, str):
        s = status.lower()
        return ("installed" in s or "loaded" in s) and "not " not in s and "un" + "installed" not in s
    for key in ("installed", "loaded", "enabled", "active"):
        if hasattr(status, key):
            return bool(getattr(status, key))
    return bool(status)


def load_last(path: Path | None = None) -> dict[str, Any] | None:
    p = path or last_json_path()
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def latest_report(folder: Path | None = None) -> Path | None:
    d = folder or reports_dir()
    try:
        reports = sorted(p for p in d.glob("*.md") if p.is_file())
    except OSError:
        return None
    return reports[-1] if reports else None


# -- the status line -----------------------------------------------------------------


def _first(d: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _numbers(last: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("numbers", "metrics", "evaluation", "gate"):
        v = last.get(key)
        if isinstance(v, dict):
            out.update(v)
    out.update({k: v for k, v in last.items() if not isinstance(v, dict)})
    return out


def _fmt_num(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.2f}" if v < 10 else f"{v:.0f}"
    return str(v)


def misfire_summary(last: dict[str, Any]) -> str | None:
    """"misfires 12 -> 8" from whatever the job recorded: a before/after pair under a
    misfire key, a champion/candidate pair, or a single number."""
    nums = _numbers(last)
    for key in ("misfires", "misfire", "misfire_rate", "misfires_per_hour"):
        v = nums.get(key)
        if v is None:
            continue
        if isinstance(v, dict):
            before = _first(v, "before", "champion", "live", "old")
            after = _first(v, "after", "candidate", "new")
            if before is not None and after is not None:
                return f"misfires {_fmt_num(before)} -> {_fmt_num(after)}"
            single = _first(v, "count", "value", "n")
            return f"misfires {_fmt_num(single)}" if single is not None else None
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return f"misfires {_fmt_num(v[0])} -> {_fmt_num(v[1])}"
        if isinstance(v, (int, float)):
            return f"misfires {_fmt_num(v)}"
    before, after = nums.get("misfires_before"), nums.get("misfires_after")
    if before is not None and after is not None:
        return f"misfires {_fmt_num(before)} -> {_fmt_num(after)}"
    return None


def model_version(last: dict[str, Any]) -> str | None:
    v = _first(last, "model_version", "version", "promoted_version", "candidate_version")
    if v is None:
        model = last.get("model") or last.get("candidate")
        if isinstance(model, dict):
            v = _first(model, "version", "name", "id")
        elif isinstance(model, str):
            v = model
    if v is None:
        return None
    s = str(v)
    return s if not s[:1].isdigit() else f"v{s}"


def run_date(last: dict[str, Any]) -> str | None:
    v = _first(last, "date", "day", "when", "finished", "started", "timestamp")
    return str(v)[:10] if v is not None else None


def format_status(last: dict[str, Any] | None, marked_today: int = 0, running: bool = False) -> str:
    """One menu line: "Learning: 2026-10-09 promoted v3, misfires 12 -> 8, 2 marked today"."""
    marked = f"{marked_today} marked today" if marked_today else None
    if running:
        parts = ["running…"]
    elif last is None:
        parts = ["no run yet"]
    else:
        head = " ".join(p for p in (run_date(last), str(_first(last, "decision", "status") or "ran"), model_version(last)) if p)
        parts = [head]
        if (m := misfire_summary(last)) is not None:
            parts.append(m)
        reason = _first(last, "reason", "message", "error")
        if reason and str(_first(last, "decision", "status") or "").lower() not in ("promoted",):
            parts.append(str(reason)[:60])
    if marked:
        parts.append(marked)
    return "Learning: " + ", ".join(parts)


def marks_today(count: int, day: date, today: date) -> tuple[int, date]:
    """The "marked today" counter rolls over at midnight: (count, day) -> the pair for `today`."""
    return (count, day) if day == today else (0, today)

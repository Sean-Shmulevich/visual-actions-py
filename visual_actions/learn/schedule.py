"""The launchd agent that runs the learning job every night (macOS only; nothing else in
learn/ touches the platform).

~/Library/LaunchAgents/com.visual-actions.learn.plist runs
`<venv python> -m visual_actions.learn run` from the repo at `hour`:00 local time, with
stdout/stderr under <data dir>/learn/logs/. launchctl is called through `runner` so tests
can mock it; the plist text itself is pure.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from ..paths import REPO_ROOT, learn_dir

LABEL = "com.visual-actions.learn"
Runner = Callable[..., Any]


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def python_executable() -> str:
    """The interpreter running this code (the repo's .venv when run through uv)."""
    return sys.executable


def plist_xml(hour: int, python: str | None = None, cwd: Path | None = None, log_dir: Path | None = None, label: str = LABEL, minute: int = 0) -> str:
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError(f"hour {hour}:{minute:02d} is not a time of day")
    python = python or python_executable()
    cwd = cwd or REPO_ROOT
    log_dir = log_dir or (learn_dir() / "logs")
    args = "".join(f"\n        <string>{escape(a)}</string>" for a in (python, "-m", "visual_actions.learn", "run"))
    path_env = escape(os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{escape(label)}</string>
    <key>ProgramArguments</key>
    <array>{args}
    </array>
    <key>WorkingDirectory</key>
    <string>{escape(str(cwd))}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{hour}</integer>
        <key>Minute</key>
        <integer>{minute}</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>{escape(str(log_dir / "learn.log"))}</string>
    <key>StandardErrorPath</key>
    <string>{escape(str(log_dir / "learn.err.log"))}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{path_env}</string>
    </dict>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
"""


def _domain() -> str:
    return f"gui/{os.getuid()}"


def install(hour: int, runner: Runner = subprocess.run, path: Path | None = None, python: str | None = None, log_dir: Path | None = None) -> Path:
    """Write the plist and (re)load it. Returns the plist path."""
    path = path or plist_path()
    log_dir = log_dir or (learn_dir() / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plist_xml(hour, python=python, log_dir=log_dir), encoding="utf-8")
    runner(["launchctl", "bootout", _domain(), str(path)], capture_output=True, check=False)  # not loaded yet is fine
    runner(["launchctl", "bootstrap", _domain(), str(path)], capture_output=True, check=True)
    return path


def uninstall(runner: Runner = subprocess.run, path: Path | None = None) -> bool:
    """Unload and delete the plist. False when it was not installed."""
    path = path or plist_path()
    if not path.exists():
        return False
    runner(["launchctl", "bootout", _domain(), str(path)], capture_output=True, check=False)
    path.unlink()
    return True


def status(runner: Runner = subprocess.run, path: Path | None = None) -> dict[str, Any]:
    path = path or plist_path()
    out: dict[str, Any] = {"plist": str(path), "installed": path.exists(), "loaded": False}
    if not out["installed"]:
        return out
    r = runner(["launchctl", "print", f"{_domain()}/{LABEL}"], capture_output=True, text=True, check=False)
    out["loaded"] = getattr(r, "returncode", 1) == 0
    text = getattr(r, "stdout", "") or ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("state =", "last exit code =")):
            k, _, v = line.partition("=")
            out[k.strip().replace(" ", "_")] = v.strip()
    return out

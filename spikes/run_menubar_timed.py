"""Run the menu bar app for N seconds with a log file, for unattended testing.

    uv run python spikes/run_menubar_timed.py 30 /tmp/va.log
"""

import sys
import time

import rumps

from visual_actions.core.config import default_config
from visual_actions.core.events import ActionFired, ModeChanged, TokenEmitted
from visual_actions.paths import models_dir
from visual_actions.ui.menubar import VisualActionsApp

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
log = open(sys.argv[2], "w", buffering=1) if len(sys.argv) > 2 else sys.stdout  # noqa: SIM115

cfg = default_config()
cfg.recognizer.model = str(models_dir() / "gestures.joblib")
app = VisualActionsApp(cfg, dry_run=False, use_gate=True)
app.bus.subscribe(ModeChanged, lambda e: print(f"mode {e.old} -> {e.new}", file=log))
app.bus.subscribe(TokenEmitted, lambda e: print(f"token {e.token.name} {e.token.confidence:.2f}", file=log))
app.bus.subscribe(ActionFired, lambda e: print(f"ACTION {e.action.name} ok={e.ok} {e.message}", file=log))


deadline = time.monotonic() + seconds


def end(_t=None):
    if time.monotonic() < deadline:
        return  # a rumps.Timer started before the run loop fires once immediately
    c = app.capture
    if c:
        print(f"frames={c.frames} tracked={c.tracked} ({100 * c.tracked / max(c.frames, 1):.0f}% past gate) error={c.error}", file=log)
    log.flush()
    app.quit(None)


# rumps.Timer must be created and started on the main thread, before app.run().
rumps.Timer(end, 1.0).start()
app.run()

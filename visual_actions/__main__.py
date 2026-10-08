"""python -m visual_actions [--replay session.jsonl] [--dry] [--no-gate] [--config path]"""

from __future__ import annotations

import argparse
from pathlib import Path

from .core.config import load_config
from .paths import config_path


def main() -> int:
    ap = argparse.ArgumentParser(prog="visual_actions")
    ap.add_argument("--replay", type=Path, help="replay a JSONL session headless and exit")
    ap.add_argument("--dry", action="store_true", help="use the mock automation driver (prints instead of acting)")
    ap.add_argument("--no-gate", action="store_true", help="run the tracker on every frame")
    ap.add_argument("--config", type=Path, default=None, help=f"config TOML (default {config_path()})")
    args = ap.parse_args()

    cfg = load_config(args.config or config_path())
    if cfg.recognizer.model is None:
        from .paths import models_dir

        trained = models_dir() / "gestures.joblib"
        if trained.exists():
            cfg.recognizer.model = str(trained)
    if args.replay:
        from .tools.replay import replay

        fired = replay(args.replay, cfg, verbose=True)
        print(f"{len(fired)} action(s) fired")
        return 0
    from .app import run_live

    return run_live(cfg, dry_run=args.dry, use_gate=not args.no_gate)


if __name__ == "__main__":
    raise SystemExit(main())

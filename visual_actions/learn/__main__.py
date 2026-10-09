"""`python -m visual_actions.learn {run,status,rollback,install,uninstall}`: the nightly learning job.

    run        [--day YYYY-MM-DD] [--dry-run] [--no-judges] [--force]   process new sessions, train, gate, promote
    status     the last run (learn/last.json), the live model and the launchd agent
    rollback   repoint models/user/current.joblib at the previously promoted version
    install    [--hour H]   write and load the launchd agent (macOS)
    uninstall  unload and delete it
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from ..paths import live_model_path, models_dir, user_models_dir


def cmd_run(args: argparse.Namespace) -> int:
    from .job import run

    day = date.fromisoformat(args.day) if args.day else None
    report = run(day=day, dry_run=args.dry_run, judges=not args.no_judges, force=args.force)
    print(f"{report.status}: {report.message}")
    return 0 if report.status != "error" else 1


def cmd_status(args: argparse.Namespace) -> int:
    from . import versions
    from .job import last_report

    live = live_model_path()
    print(f"live model: {live if live else 'rules only'}")
    cur = versions.current_name(user_models_dir())
    print(f"user model pointer: {cur or 'none (shipped model)'}; versions on disk: {[p.name for p in versions.list_versions(user_models_dir())]}")
    last = last_report()
    if last is None:
        print("last run: none")
    else:
        print("last run:")
        print(json.dumps({k: last[k] for k in ("day", "status", "message", "new_frames", "champion", "candidate", "holdout", "finished_at") if k in last}, indent=2))
        for r in last.get("gate", {}).get("rules", []):
            print(f"  {'PASS' if r['passed'] else 'FAIL'} {r['name']}: {r['detail']}")
    if sys.platform == "darwin":
        from . import schedule

        print(f"launchd: {schedule.status()}")
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    from . import versions

    before = versions.current_name(user_models_dir())
    target = versions.rollback(user_models_dir())
    print(f"was {before or 'shipped model'}; now {target.name if target else f'shipped model ({models_dir() / 'gestures.joblib'})'}")
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    from ..core.config import load_config
    from ..paths import config_path
    from . import schedule

    hour = args.hour if args.hour is not None else load_config(config_path()).learn.hour
    path = schedule.install(hour)
    print(f"installed {path}: runs daily at {hour:02d}:00")
    return 0


def cmd_uninstall(args: argparse.Namespace) -> int:
    from . import schedule

    print("uninstalled" if schedule.uninstall() else "not installed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m visual_actions.learn", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="process new sessions, train a candidate, gate it, promote or discard")
    r.add_argument("--day", help="the training day, YYYY-MM-DD (default today)")
    r.add_argument("--dry-run", action="store_true", help="write nothing under datasets/ or models/; report what would happen")
    r.add_argument("--no-judges", action="store_true", help="skip the Cosmos / JEV / tagger passes")
    r.add_argument("--force", action="store_true", help="train even below min_new_frames")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("status", help="last run, live model, launchd agent")
    s.set_defaults(fn=cmd_status)
    b = sub.add_parser("rollback", help="repoint current.joblib at the previous version")
    b.set_defaults(fn=cmd_rollback)
    i = sub.add_parser("install", help="write and load the launchd agent")
    i.add_argument("--hour", type=int, default=None, help="local hour (default: [learn] hour in config)")
    i.set_defaults(fn=cmd_install)
    u = sub.add_parser("uninstall", help="unload and delete the launchd agent")
    u.set_defaults(fn=cmd_uninstall)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

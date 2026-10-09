"""The nightly job: every session the app recorded today becomes labels, a candidate model is
trained for this user, scored against the live model on held-out sessions, and promoted
only when it is better. Idempotent: processed sessions are remembered in learn/state.json,
and a session still being written (no summary line yet) waits for the next run.

Safe beside the running app: the live model file is never written in place. A candidate is a
new file under models/user/ and promotion switches the current.joblib pointer atomically.
A dry run does everything into temporary directories and writes nothing under datasets/ or
models/ (segments.jsonl and the learn/ report are still written).

    run(day, dry_run, judges)  ->  Report, learn/reports/<date>.md, learn/last.json
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..core.config import Config, load_config
from ..intent.events import parse_events
from ..intent.export import JudgeRules
from ..intent.schema import write_jsonl
from ..intent.segments import segment_session
from ..paths import config_path, datasets_dir, learn_dir, models_dir, sessions_dir, user_models_dir
from . import evaluate, versions
from .trainer import train_candidate, write_manifest

STATE = "state.json"
LAST = "last.json"
STALE_SESSION_S = 15 * 60  # an unclosed session untouched this long is a crash, not a live run
Log = Callable[[str], None]


@dataclass
class SessionResult:
    stamp: str
    segments: int = 0
    judges: dict[str, str] = field(default_factory=dict)  # pass -> "n verdicts" | "skipped: why" | "failed: why"
    frames: dict[str, int] = field(default_factory=dict)  # class -> frames exported
    skipped: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def exported(self) -> int:
        return sum(self.frames.values())


@dataclass
class Report:
    day: str
    status: str = "no-data"  # promoted | discarded | skipped | no-data | error
    dry_run: bool = False
    started_at: str = ""
    finished_at: str = ""
    sessions: list[SessionResult] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)  # sessions still open, left for the next run
    new_frames: int = 0
    champion: str = "rules"
    candidate: str | None = None
    holdout: list[str] = field(default_factory=list)
    scores: dict[str, Any] = field(default_factory=dict)  # champion / candidate -> evaluate.scores_dict
    gate: dict[str, Any] = field(default_factory=dict)  # evaluate.Decision as a dict: promote, rules, reason
    message: str = ""


@dataclass
class Judges:
    """What the judge passes run with. None = the pass is skipped. The job builds the real
    clients from the environment; tests inject fakes."""

    cosmos: Any = None
    jev: Any = None
    tagger: Any = None

    @classmethod
    def from_env(cls, log: Log) -> Judges:
        j = cls()
        if os.environ.get("NVIDIA_API_KEY"):
            from ..intent.cosmos import CosmosReason

            j.cosmos = CosmosReason()
        else:
            log("cosmos: NVIDIA_API_KEY not set, pass skipped")
        if os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY"):
            from ..intent.jev import JevClient

            j.jev = JevClient()
        elif os.environ.get("OPENROUTER_API_KEY"):
            from ..intent.jev import OpenRouterJev

            j.jev = OpenRouterJev()
        else:
            log("jev: neither TYPESAFE_API_KEY nor OPENROUTER_API_KEY is set, pass skipped")
        from ..intent import tagger as tagmod

        factory = getattr(tagmod, "make_tagger", None)
        if factory is None:
            log("tag: intent.tagger.make_tagger is not available, pass skipped")
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # make_tagger warns when it falls back to the fake
                    t = factory()
            except Exception as e:  # noqa: BLE001 - no tagger is a skipped pass, not a failed night
                log(f"tag: no tagger ({e}), pass skipped")
            else:
                if isinstance(t, tagmod.FakeTagger):
                    log("tag: no real tagger backend (codex CLI or an API key), pass skipped: canned tags must never become labels")
                else:
                    j.tagger = t
        return j


# -- state ----------------------------------------------------------------------------------


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"processed": {}}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"processed": {}}
    d.setdefault("processed", {})
    return d


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# -- one session ------------------------------------------------------------------------------


def process_session(session: Path, datasets: Path, judges: Judges, judge_min_conf: float, log: Log, judge_rules: JudgeRules | None = None) -> SessionResult:
    """segments (if missing) -> judge passes (each best-effort) -> export (humans win)."""
    res = SessionResult(session.name)
    d = session / "intent"
    seg_path = d / "segments.jsonl"
    events = parse_events(session / "events.log")
    if seg_path.exists():
        res.segments = sum(1 for line in seg_path.read_text(encoding="utf-8").splitlines() if line.strip())
    else:
        segs = segment_session(session.name, events)
        res.segments = write_jsonl(seg_path, segs)
    log(f"{session.name}: {res.segments} segments")

    if judges.cosmos is not None:
        try:
            from ..intent.cosmos import run_cosmos

            n = len(run_cosmos(session, seg_path, d / "cosmos.jsonl", judges.cosmos, log=log))
            res.judges["cosmos"] = f"{n} new verdicts"
        except Exception as e:  # noqa: BLE001 - a judge is optional; the weak labels carry the night
            res.judges["cosmos"] = f"failed: {e}"
            log(f"{session.name}: cosmos failed: {e}")
    else:
        res.judges["cosmos"] = "skipped"
    if judges.jev is not None:
        try:
            from ..intent.jev import run_jev

            n = run_jev(seg_path, d / "cosmos.jsonl", d / "jev.jsonl", judges.jev, events=events, log=log)
            res.judges["jev"] = f"{n} new verdicts"
        except Exception as e:  # noqa: BLE001
            res.judges["jev"] = f"failed: {e}"
            log(f"{session.name}: jev failed: {e}")
    else:
        res.judges["jev"] = "skipped"
    if judges.tagger is not None:
        try:
            from ..intent.tagger import run_tag

            n = run_tag(seg_path, d / "cosmos.jsonl", d / "jev.jsonl", d / "tags.jsonl", judges.tagger, log=log)
            res.judges["tag"] = f"{n} new tags"
        except Exception as e:  # noqa: BLE001
            res.judges["tag"] = f"failed: {e}"
            log(f"{session.name}: tag failed: {e}")
    else:
        res.judges["tag"] = "skipped"

    from ..intent.export import export_labels

    summary = export_labels(session, datasets, include_judge=True, judge_min_conf=judge_min_conf, rules=judge_rules)
    res.frames = dict(summary.frames)
    res.skipped = list(summary.skipped)
    log(f"{session.name}: exported {res.exported} frames " + " ".join(f"{k}={v}" for k, v in sorted(res.frames.items())))
    return res


def _session_ready(session: Path, now_s: float) -> tuple[bool, str]:
    if not (session / "events.log").exists():
        return False, "no events.log"
    if evaluate.session_closed(session):
        return True, ""
    try:
        age = now_s - (session / "events.log").stat().st_mtime
    except OSError:
        return False, "unreadable"
    if age >= STALE_SESSION_S:
        return True, "unclosed, stale"
    return False, "still open"


# -- the night ----------------------------------------------------------------------------------


def run(
    day: date | None = None,
    dry_run: bool = False,
    judges: bool = True,
    config: Config | None = None,
    log: Log = print,
    judge_clients: Judges | None = None,
    force: bool = False,
    judge_rules: JudgeRules | None = None,
) -> Report:
    day = day or date.today()  # noqa: DTZ011 - local, like the session stamps
    cfg = config or load_config(config_path())
    lc = cfg.learn
    ldir = learn_dir()
    report = Report(day.isoformat(), dry_run=dry_run, started_at=datetime.now().isoformat(timespec="seconds"))  # noqa: DTZ005
    tmp_root: Path | None = None
    try:
        tmp_root = Path(tempfile.mkdtemp(prefix="va-learn-")) if dry_run else None
        datasets = tmp_root / "datasets" if tmp_root else datasets_dir()
        if tmp_root:
            shutil.copytree(datasets_dir(), datasets, dirs_exist_ok=True) if datasets_dir().exists() else datasets.mkdir(parents=True)
        state_path = ldir / STATE
        state = load_state(state_path)
        clients = judge_clients if judge_clients is not None else (Judges.from_env(log) if judges else Judges())
        if not judges and judge_clients is None:
            log("judges off: labelling-function weak labels and human labels only")

        # 1. sessions -> labels
        sdir = sessions_dir()
        all_sessions = sorted(p for p in sdir.iterdir() if p.is_dir()) if sdir.exists() else []
        now_s = time.time()
        for s in all_sessions:
            if s.name in state["processed"]:
                continue
            ready, why = _session_ready(s, now_s)
            if not ready:
                report.pending.append(s.name)
                log(f"{s.name}: {why}, left for the next run")
                continue
            try:
                res = process_session(s, datasets, clients, lc.judge_min_conf, log, judge_rules)
                if why:
                    res.note = why
            except Exception as e:  # noqa: BLE001 - one broken session must not stop the night
                res = SessionResult(s.name, note=f"error: {e}")
                log(f"{s.name}: error: {e}")
            report.sessions.append(res)
            if not dry_run:
                state["processed"][s.name] = {"at": datetime.now().isoformat(timespec="seconds"), "frames": res.exported, "segments": res.segments}  # noqa: DTZ005
                save_state(state_path, state)
        report.new_frames = sum(r.exported for r in report.sessions)

        # 2. enough new data?
        if report.new_frames < lc.min_new_frames and not force:
            report.status = "skipped"
            report.message = f"{report.new_frames} new labelled frames < min_new_frames {lc.min_new_frames}: no training tonight"
            log(report.message)
            return report

        # 3. train the candidate
        shipped = models_dir() / "gestures.joblib"
        user_dir = user_models_dir()
        champion = versions.champion_path(user_dir, shipped)
        report.champion = champion.name if champion else "rules"
        holdout_dirs = evaluate.select_holdout([s for s in all_sessions if evaluate.session_closed(s)], day, lc.holdout_sessions)
        report.holdout = [s.name for s in holdout_dirs]
        out_dir = tmp_root / "user" if tmp_root else user_dir
        cand = train_candidate(datasets, out_dir, lc, day, frozenset(report.holdout), mirror=cfg.camera.mirror)
        if cand is None:
            report.status = "no-data"
            report.message = "no training data under datasets/"
            log(report.message)
            return report
        report.candidate = cand.name
        log(f"candidate {cand.name}: {len(cand.train)} frames {cand.train.counts()['by_source']}, held out {report.holdout} ({len(cand.holdout)} labelled frames)")

        # 4. score both on the held-out sessions
        champ_scores = evaluate.score(champion, cand.holdout, holdout_dirs, cfg, log)
        cand_scores = evaluate.score(cand.path, cand.holdout, holdout_dirs, cfg, log)
        decision = evaluate.decide(champ_scores, cand_scores, lc)
        report.scores = {"champion": evaluate.scores_dict(champ_scores), "candidate": evaluate.scores_dict(cand_scores)}
        report.gate = asdict(decision)
        for r in decision.rules:
            log(f"gate {r.name:<12} {'pass' if r.passed else 'FAIL'}  {r.detail}")

        # 5. promote or discard
        cand.manifest["metrics"] = report.scores
        cand.manifest["decision"] = {**asdict(decision), "dry_run": dry_run}
        write_manifest(cand)
        if decision.promote:
            report.status = "promoted"
            if not dry_run:
                versions.promote(user_dir, cand.path)
                removed = versions.prune(user_dir, lc.keep_versions)
                if removed:
                    log("pruned " + ", ".join(p.name for p in removed))
            report.message = f"{cand.name} {'would be promoted (dry run)' if dry_run else 'promoted'}: {decision.reason}"
        else:
            report.status = "discarded"
            report.message = f"{cand.name} discarded: {decision.reason}"
        log(report.message)
        return report
    except Exception as e:
        report.status = "error"
        report.message = f"error: {e!r}"
        log(report.message)
        raise
    finally:
        report.finished_at = datetime.now().isoformat(timespec="seconds")  # noqa: DTZ005
        write_report(ldir, report)
        if tmp_root is not None:
            shutil.rmtree(tmp_root, ignore_errors=True)


# -- outputs --------------------------------------------------------------------------------


def write_report(ldir: Path, report: Report) -> tuple[Path, Path]:
    suffix = "-dry-run" if report.dry_run else ""
    md = ldir / "reports" / f"{report.day}{suffix}.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(render_report(report), encoding="utf-8")
    last = ldir / (f"last{suffix}.json")
    _write_json(last, last_json(report))
    return md, last


def last_json(r: Report) -> dict[str, Any]:
    """The report plus the flat keys the menu bar status line reads (ui/learning_menu.py):
    decision, model_version, misfires before -> after."""
    d = asdict(r)
    d["date"] = r.day
    d["decision"] = r.status
    d["model_version"] = r.candidate if r.status == "promoted" and not r.dry_run else None
    if "champion" in r.scores and "candidate" in r.scores:
        d["misfires"] = {"champion": r.scores["champion"]["engine"]["weak_misfire_fires"], "candidate": r.scores["candidate"]["engine"]["weak_misfire_fires"]}
    return d


def _engine_row(name: str, s: dict[str, Any]) -> str:
    e = s["engine"]
    f = s["frame"]
    acc = f"{f['accuracy']:.4f}" if f.get("accuracy") is not None else "n/a"
    fist = f"{f['fist_recall']:.4f}" if f.get("fist_recall") is not None else "n/a"
    return f"| {name} | {s['model']} | {acc} | {fist} | {e['fires']} | {e['weak_misfire_fires']} | {e['weak_intended_fires']} | {e['arms']} | {e['empty_arms']} | {e['broken_holds']} | {e['fist_cancelled_drags']}/{e['drags']} |"


def render_report(r: Report) -> str:
    lines = [f"# Learning job {r.day}" + (" (dry run)" if r.dry_run else ""), "", f"**{r.status}** - {r.message}", "", f"started {r.started_at}, finished {r.finished_at}", ""]
    lines += ["## Sessions", ""]
    if r.sessions:
        lines += ["| session | segments | exported frames | judges | note |", "| --- | ---: | ---: | --- | --- |"]
        for s in r.sessions:
            frames = " ".join(f"{k}={v}" for k, v in sorted(s.frames.items())) or "-"
            judges = ", ".join(f"{k}: {v}" for k, v in s.judges.items())
            lines.append(f"| {s.stamp} | {s.segments} | {s.exported} ({frames}) | {judges} | {s.note} |")
    else:
        lines.append("no new sessions")
    if r.pending:
        lines += ["", "still open, left for the next run: " + ", ".join(r.pending)]
    lines += ["", f"new labelled frames: {r.new_frames}", ""]
    if r.candidate:
        lines += ["## Candidate", "", f"candidate `{r.candidate}` vs champion `{r.champion}`; held-out sessions: {', '.join(r.holdout) or 'none'}", ""]
        lines += ["| | model | held-out acc | fist recall | fires | weak misfire | weak intended | arms | empty arms | broken holds | fist-cancelled drags |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name in ("champion", "candidate"):
            if name in r.scores:
                lines.append(_engine_row(name, r.scores[name]))
        per_class = r.scores.get("candidate", {}).get("frame", {}).get("per_class", {})
        champ_class = r.scores.get("champion", {}).get("frame", {}).get("per_class", {})
        if per_class:
            lines += ["", "| class | frames | weight | champion acc | candidate acc |", "| --- | ---: | ---: | ---: | ---: |"]
            for cls, sc in sorted(per_class.items()):
                ca = champ_class.get(cls, {}).get("accuracy")
                lines.append(f"| {cls} | {sc['n']} | {sc['weight']} | {ca if ca is not None else 'n/a'} | {sc['accuracy']} |")
        lines += ["", "## Gate", ""]
        for rule in r.gate.get("rules", []):
            lines.append(f"- {'PASS' if rule['passed'] else 'FAIL'} {rule['name']}: {rule['detail']}")
        lines += ["", f"decision: {r.gate.get('reason', '')}"]
    return "\n".join(lines) + "\n"


def last_report(dry_run: bool = False) -> dict[str, Any] | None:
    p = learn_dir() / (f"last{'-dry-run' if dry_run else ''}.json")
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

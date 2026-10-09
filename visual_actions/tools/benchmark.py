"""Reproducible before/after benchmark of the classifier and the engine on the recorded data.

    uv run python -m visual_actions.tools.benchmark [--label before-cosmos] [--sessions 6] [--out benchmarks/]
    uv run python -m visual_actions.tools.benchmark compare benchmarks/<before>.json benchmarks/<after>.json

Three views, every one deterministic for a given set of files, the live model and the user's config:

1. classifier: tools/train.py's loader and held-out protocol (the newest user session of every
   class is the test set; public and synthetic files only ever train) on datasets/ as it is now;
2. engine: the newest N closed sessions replayed through the whole pipeline with the live model
   (learn/evaluate.replay_events) and cut by intent.segments, next to the same counts from the
   events.log the app wrote that day (what actually happened, through the same segmentation);
3. labels: the labelling functions scored against the human labels under <session>/intent/
   (intent/export.labelfn_accuracy); "n/a" until such labels exist.

Writes benchmarks/<date>-<label>.json (every number plus provenance: git hash, model and dataset
hashes, config profile, session list) and a markdown table whose value column is the label, so
`compare` lays a later run next to it with deltas. Nothing under the data directory is written.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import Config, load_config
from ..intent.events import Event, parse_events, session_duration_s
from ..intent.export import labelfn_accuracy
from ..intent.schema import Segment, SegmentKind
from ..intent.segments import segment_session
from ..learn.evaluate import EMPTY_ARM_OUTCOMES, replay_events, session_closed
from ..learn.trainer import file_sha256, source_of
from ..paths import REPO_ROOT, config_path, datasets_dir, live_model_path, sessions_dir
from .train import load as load_frames

FIST = "fist"
MISFIRE_FNS = ("fist_after_fire", "token_flip_around_fire", "immediate_redo", "low_confidence_trigger")
DEFAULT_SESSIONS = 6
DEFAULT_OUT = REPO_ROOT / "benchmarks"
Log = Callable[[str], None]


# -- engine counts -------------------------------------------------------------------------


@dataclass
class EngineCounts:
    """Counts over one session's events (replayed or live) after intent.segments cut them."""

    sessions: int = 0
    seconds: float = 0.0
    fires: int = 0
    weak_misfire_fires: int = 0  # a fire any MISFIRE_FNS function voted misfire on
    weak_intended_fires: int = 0  # combined weak label = intended
    misfire_votes: dict[str, int] = field(default_factory=lambda: {k: 0 for k in MISFIRE_FNS})
    arms: int = 0
    empty_arms: int = 0  # armed, then timeout / fist / hand lost with no command
    broken_holds: int = 0  # holds that never armed
    drags: int = 0
    fist_cancelled_drags: int = 0
    drag_ends: int = 0
    regrabs: int = 0  # a released pinch re-grabbed within the grace (a `resume` with no `pause` before it)
    hand_losses: int = 0

    def add(self, other: EngineCounts) -> None:
        for k, v in asdict(other).items():
            if k == "misfire_votes":
                for fn, n in v.items():
                    self.misfire_votes[fn] = self.misfire_votes.get(fn, 0) + n
            else:
                setattr(self, k, getattr(self, k) + v)

    def rates(self) -> dict[str, float | None]:
        return {
            "misfire_rate": _ratio(self.weak_misfire_fires, self.fires),
            "broken_hold_ratio": _ratio(self.broken_holds, self.arms + self.broken_holds),
            "releases_per_drag": _ratio(self.drag_ends + self.regrabs, self.drags),
            "hand_losses_per_min": _ratio(self.hand_losses, self.seconds / 60.0),
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["seconds"] = round(self.seconds, 1)
        d.update(self.rates())
        return d


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def engine_counts(segments: list[Segment], events: list[Event], seconds: float | None = None) -> EngineCounts:
    c = EngineCounts(sessions=1, seconds=seconds if seconds is not None else session_duration_s(events))
    for s in segments:
        if s.kind is SegmentKind.FIRE:
            c.fires += 1
            voted = [fn for fn in MISFIRE_FNS if s.labelfn_votes.get(fn) == "misfire"]
            for fn in voted:
                c.misfire_votes[fn] += 1
            if voted:
                c.weak_misfire_fires += 1
            if s.weak_label == "intended":
                c.weak_intended_fires += 1
        elif s.kind is SegmentKind.ARM:
            c.arms += 1
            if s.engine_outcome in EMPTY_ARM_OUTCOMES:
                c.empty_arms += 1
        elif s.kind is SegmentKind.BROKEN_HOLD:
            c.broken_holds += 1
        elif s.kind is SegmentKind.DRAG:
            c.drags += 1
            if s.engine_outcome == "dropped_by_fist":
                c.fist_cancelled_drags += 1
    prev_phase: str | None = None
    for e in events:
        if e.kind == "hand" and e.hand_seen is False:
            c.hand_losses += 1
        elif e.kind == "drag":
            phase = e.drag_phase
            if phase == "end":
                c.drag_ends += 1
            elif phase == "resume" and prev_phase != "pause":
                c.regrabs += 1
            prev_phase = phase
    return c


def session_counts(session_dir: Path, config: Config) -> dict[str, dict[str, Any]]:
    """{"replay": ..., "live": ...}: the session replayed through the pipeline built from
    `config`, and its own events.log, both cut by intent.segments and counted alike."""
    session_dir = Path(session_dir)
    live_events = parse_events(session_dir / "events.log")
    seconds = session_duration_s(live_events)
    replayed = replay_events(session_dir, config)
    replay = engine_counts(segment_session(session_dir.name, replayed), replayed, seconds)
    live = engine_counts(segment_session(session_dir.name, live_events), live_events, seconds)
    return {"replay": replay.to_dict(), "live": live.to_dict()}


def select_sessions(sessions: Path, n: int) -> list[Path]:
    """The `n` newest closed sessions with both files, newest first; n <= 0 means every one."""
    if not sessions.exists():
        return []
    out: list[Path] = []
    for s in sorted((p for p in sessions.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
        if not (s / "landmarks.jsonl").exists() or not session_closed(s):
            continue
        out.append(s)
        if n > 0 and len(out) >= n:
            break
    return out


def engine_benchmark(sessions: list[Path], config: Config, log: Log | None = None) -> dict[str, Any]:
    per_session: dict[str, dict[str, Any]] = {}
    totals = {"replay": EngineCounts(), "live": EngineCounts()}
    for s in sessions:
        t = time.perf_counter()
        per_session[s.name] = session_counts(s, config)
        for view in ("replay", "live"):
            c = EngineCounts(**{k: v for k, v in per_session[s.name][view].items() if k in EngineCounts.__dataclass_fields__})
            totals[view].add(c)
        if log:
            r, lv = per_session[s.name]["replay"], per_session[s.name]["live"]
            log(f"  {s.name}: replay fires={r['fires']} misfire={r['weak_misfire_fires']} arms={r['arms']} drags={r['drags']} | live fires={lv['fires']} arms={lv['arms']} ({time.perf_counter() - t:.1f}s)")
    return {"sessions": [s.name for s in sessions], "per_session": per_session, "total": {k: v.to_dict() for k, v in totals.items()}}


# -- classifier ----------------------------------------------------------------------------


def classifier_benchmark(datasets: Path, mirror: bool = True) -> dict[str, Any]:
    """train.py's loader and protocol: the newest user session of every class with at least two
    sessions is the test set; public / synthetic files only ever train."""
    x, y, g = load_frames(datasets, mirror)
    groups = sorted(set(g.tolist()))
    files = []
    for grp in groups:
        p = datasets / grp
        files.append({"file": grp, "source": source_of(p), "frames": int((g == grp).sum()), "sha256": file_sha256(p) if p.exists() else None})
    out: dict[str, Any] = {
        "frames": len(y),
        "by_class": dict(sorted(Counter(y.tolist()).items())),
        "by_source": dict(sorted(Counter(source_of(Path(grp.split("/")[-1])) for grp in g.tolist()).items())),
        "files": files,
        "status": "ok",
    }
    if len(y) == 0:
        out["status"] = "no data"
        return out
    sessions_per_class: dict[str, list[str]] = {}
    for grp in groups:
        sessions_per_class.setdefault(grp.split("/")[0], []).append(grp)
    user_sessions = {c: [s for s in ss if "/public-" not in s and "/synth" not in s] for c, ss in sessions_per_class.items()}
    testable = {c: s for c, s in user_sessions.items() if len(s) >= 1 and len(sessions_per_class[c]) >= 2}
    if not testable:
        out["status"] = "only one session per class: no held-out score"
        return out
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    test_groups = {s[-1] for s in testable.values()}
    te = np.array([grp in test_groups for grp in g])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(x[~te], y[~te])
    pred = np.asarray(model.predict(x[te])).astype(str)
    truth = y[te]
    hit = pred == truth
    labels = sorted(set(y.tolist()))
    per_class: dict[str, dict[str, Any]] = {}
    confusion: dict[str, dict[str, int]] = {}
    for cls in labels:
        sel = truth == cls
        n = int(sel.sum())
        if n == 0:
            continue
        predicted_as = int((pred == cls).sum())
        per_class[cls] = {
            "n": n,
            "accuracy": round(float(hit[sel].mean()), 4),  # recall of the class
            "precision": round(float((truth[pred == cls] == cls).mean()), 4) if predicted_as else None,
        }
        confusion[cls] = dict(sorted(Counter(pred[sel].tolist()).items()))
    top = sorted(((t, p, n) for t, row in confusion.items() for p, n in row.items() if p != t), key=lambda r: -r[2])
    fist = truth == FIST
    out["heldout"] = {
        "protocol": "newest user session per class (public / synth files train only)",
        "groups": sorted(test_groups),
        "frames": int(te.sum()),
        "train_frames": int((~te).sum()),
        "by_class": {c: s["n"] for c, s in per_class.items()},
        "accuracy": round(float(hit.mean()), 4),
        "fist_recall": round(float(hit[fist].mean()), 4) if fist.any() else None,
        "per_class": per_class,
        "confusion": confusion,
        "top_confusions": [{"true": t, "pred": p, "n": n} for t, p, n in top[:8]],
    }
    return out


# -- labels --------------------------------------------------------------------------------


def labels_benchmark(session_dirs: Iterable[Path]) -> dict[str, Any]:
    dirs = [Path(s) for s in session_dirs]
    with_human = [s for s in dirs if (s / "intent" / "human.jsonl").exists()]
    with_tags = [s for s in dirs if (s / "intent" / "tags.jsonl").exists()]

    def count(ss: list[Path], name: str) -> int:
        return sum(1 for s in ss for line in (s / "intent" / name).read_text(encoding="utf-8").splitlines() if line.strip())

    out: dict[str, Any] = {
        "sessions_with_human": [s.name for s in with_human],
        "sessions_with_tags": [s.name for s in with_tags],
        "human_labels": count(with_human, "human.jsonl"),
        "tags": count(with_tags, "tags.jsonl"),
    }
    buf = io.StringIO()
    scores = labelfn_accuracy(with_human, out=buf) if with_human else {}
    if not scores:
        out["status"] = "n/a"
        out["reason"] = "no human labels on fire / arm segments (<session>/intent/human.jsonl); labelfn_accuracy scores human labels only"
        return out
    out["status"] = "ok"
    out["functions"] = {
        name: {"kind": s.kind, "label": s.label, "votes": s.votes, "correct": s.correct, "truth": s.truth, "labelled": s.labelled, "precision": _opt_round(s.precision), "recall": _opt_round(s.recall), "coverage": round(s.coverage, 4)}
        for name, s in sorted(scores.items())
    }
    out["table"] = buf.getvalue()
    return out


def _opt_round(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


# -- provenance and the run ----------------------------------------------------------------


def git_short_hash(repo: Path = REPO_ROOT) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None


def _file_provenance(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "sha256": None}
    p = Path(path)
    return {"path": str(p), "resolved": str(p.resolve()) if p.exists() else None, "exists": p.exists(), "sha256": file_sha256(p) if p.exists() else None}


def run_benchmark(
    label: str,
    n_sessions: int = DEFAULT_SESSIONS,
    sessions: Path | None = None,
    datasets: Path | None = None,
    model: Path | None = None,
    config: Path | None = None,
    log: Log | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Every number of one run as a JSON-ready dict. `model` None = rules only; the live model
    is substituted by the CLI. The config's `recognizer.model` is overridden by `model`, as
    the app does with paths.live_model_path()."""
    started = time.perf_counter()
    sessions = sessions if sessions is not None else sessions_dir()
    datasets = datasets if datasets is not None else datasets_dir()
    cfg_path = config if config is not None else config_path()
    cfg = load_config(cfg_path)
    cfg.recognizer.model = str(model) if model else None
    log = log or (lambda s: None)

    log(f"classifier: datasets {datasets}")
    classifier = classifier_benchmark(datasets, cfg.camera.mirror)
    chosen = select_sessions(sessions, n_sessions)
    log(f"engine: {len(chosen)} session(s), model {model or 'rules'}, profile {cfg.leader.profile}")
    engine = engine_benchmark(chosen, cfg, log)
    all_sessions = sorted((p for p in sessions.iterdir() if p.is_dir()), key=lambda p: p.name) if sessions.exists() else []
    labels = labels_benchmark(all_sessions)
    log(f"labels: {labels['status']}")
    return {
        "label": label,
        "date": (today or date.today()).isoformat(),  # noqa: DTZ011 - local, like the session stamps
        "created_at": datetime.now().isoformat(timespec="seconds"),  # noqa: DTZ005
        "git": git_short_hash(),
        "model": _file_provenance(model),
        "config": {**_file_provenance(cfg_path), "profile": cfg.leader.profile},
        "paths": {"sessions": str(sessions), "datasets": str(datasets)},
        "sessions_requested": n_sessions,
        "sessions": [s.name for s in chosen],
        "classifier": classifier,
        "engine": engine,
        "labels": labels,
        "run_time_s": round(time.perf_counter() - started, 1),
    }


# -- rendering -----------------------------------------------------------------------------

ENGINE_ROWS: list[tuple[str, str]] = [
    ("fires", "fires"),
    ("weak_misfire_fires", "weak-misfire fires"),
    ("misfire_rate", "misfire rate"),
    ("weak_intended_fires", "weak-intended fires"),
    ("arms", "arms"),
    ("empty_arms", "empty arms"),
    ("broken_holds", "broken holds"),
    ("broken_hold_ratio", "broken-hold ratio"),
    ("drags", "drags"),
    ("fist_cancelled_drags", "drags cancelled by fist"),
    ("regrabs", "re-grab flaps"),
    ("releases_per_drag", "releases per drag"),
    ("hand_losses", "hand losses"),
    ("hand_losses_per_min", "hand losses / min"),
    ("seconds", "seconds"),
]

Row = tuple[str, str, Any]  # (section, metric, value)


def metric_rows(r: dict[str, Any]) -> list[Row]:
    """Every scalar the tables show, flat, in display order. The same rows feed the single-run
    markdown and `compare`, so a later run lines up metric by metric."""
    rows: list[Row] = []
    c = r.get("classifier", {})
    h = c.get("heldout")
    rows.append(("classifier", "held-out accuracy", h["accuracy"] if h else None))
    rows.append(("classifier", "fist recall", h["fist_recall"] if h else None))
    for cls, sc in sorted((h or {}).get("per_class", {}).items()):
        rows.append(("classifier", f"accuracy {cls}", sc["accuracy"]))
    rows.append(("classifier", "held-out frames", h["frames"] if h else None))
    rows.append(("classifier", "frames", c.get("frames")))
    for src, n in sorted(c.get("by_source", {}).items()):
        rows.append(("classifier", f"frames {src}", n))
    for cls, n in sorted(c.get("by_class", {}).items()):
        rows.append(("classifier", f"frames {cls}", n))
    e = r.get("engine", {})
    rows.append(("engine", "sessions", len(e.get("sessions", []))))
    for view in ("replay", "live"):
        tot = e.get("total", {}).get(view, {})
        for key, name in ENGINE_ROWS:
            rows.append((f"engine {view}", name, tot.get(key)))
        for fn in MISFIRE_FNS:
            rows.append((f"engine {view}", f"votes {fn}", tot.get("misfire_votes", {}).get(fn)))
    lab = r.get("labels", {})
    rows.append(("labels", "status", lab.get("status")))
    rows.append(("labels", "human labels", lab.get("human_labels")))
    rows.append(("labels", "tags", lab.get("tags")))
    for name, s in sorted(lab.get("functions", {}).items()):
        rows.append(("labels", f"precision {name}", s["precision"]))
        rows.append(("labels", f"recall {name}", s["recall"]))
    return rows


def fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if abs(v) >= 100:
            return f"{v:.1f}"
        s = f"{v:.4f}".rstrip("0")
        return s + "0" if s.endswith(".") else s
    return str(v)


def fmt_delta(a: Any, b: Any) -> str:
    if a is None or b is None:
        return ""  # measured on one side only
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return "" if a == b else "changed"
    d = b - a
    if d == 0:
        return "0"
    if isinstance(a, int) and isinstance(b, int):
        return f"{d:+d}"
    s = f"{d:+.4f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def _provenance_line(r: dict[str, Any]) -> str:
    m, c = r.get("model", {}), r.get("config", {})
    return (
        f"git `{r.get('git') or '?'}` · model `{m.get('path') or 'rules'}`"
        + (f" (sha {m['sha256']})" if m.get("sha256") else "")
        + f" · config profile `{c.get('profile', '?')}`"
        + (f" (sha {c['sha256']})" if c.get("sha256") else "")
        + f" · {len(r.get('sessions', []))} session(s) · run {r.get('run_time_s', '?')} s"
    )


def _table(header: list[str], align: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"] + ["| " + " | ".join(row) + " |" for row in rows]


def render_markdown(r: dict[str, Any]) -> str:
    label = r["label"]
    lines = [f"# Benchmark {label} ({r['date']})", "", _provenance_line(r), ""]
    rows = metric_rows(r)
    c = r.get("classifier", {})
    h = c.get("heldout")
    lines += ["## Classifier", "", f"status: {c.get('status', '?')}; held-out = {h['protocol'] if h else 'n/a'}", ""]
    lines += _table(["metric", label], ["---", "---:"], [[m, fmt(v)] for s, m, v in rows if s == "classifier"])
    if h and h.get("top_confusions"):
        lines += ["", "confusions (held-out, true -> predicted): " + ", ".join(f"{x['true']} -> {x['pred']} {x['n']}" for x in h["top_confusions"])]
    lines += ["", f"## Engine ({len(r.get('sessions', []))} sessions: {', '.join(r.get('sessions', [])) or 'none'})", ""]
    lines += ["replay = landmarks through the pipeline with the model above; live = the events.log the app wrote.", ""]
    engine_rows = []
    for key, name in ENGINE_ROWS:
        rep = next((v for s, m, v in rows if s == "engine replay" and m == name), None)
        liv = next((v for s, m, v in rows if s == "engine live" and m == name), None)
        engine_rows.append([name, fmt(rep), fmt(liv)])
    for fn in MISFIRE_FNS:
        rep = next((v for s, m, v in rows if s == "engine replay" and m == f"votes {fn}"), None)
        liv = next((v for s, m, v in rows if s == "engine live" and m == f"votes {fn}"), None)
        engine_rows.append([f"votes {fn}", fmt(rep), fmt(liv)])
    lines += _table(["metric", f"{label} replay", f"{label} live"], ["---", "---:", "---:"], engine_rows)
    per = r.get("engine", {}).get("per_session", {})
    if per:
        lines += ["", "### Per session (replay / live)", ""]
        body = []
        for stamp, v in per.items():
            body.append([stamp] + [f"{fmt(v['replay'].get(k))} / {fmt(v['live'].get(k))}" for k in ("fires", "weak_misfire_fires", "arms", "empty_arms", "broken_holds", "drags", "hand_losses_per_min")])
        lines += _table(["session", "fires", "weak-misfire", "arms", "empty arms", "broken holds", "drags", "losses/min"], ["---"] + ["---:"] * 7, body)
    lab = r.get("labels", {})
    lines += ["", "## Labelled-moment agreement", ""]
    if lab.get("status") == "ok":
        lines += _table(["metric", label], ["---", "---:"], [[m, fmt(v)] for s, m, v in rows if s == "labels"])
    else:
        lines.append(f"n/a: {lab.get('reason', 'no labels')} (human labels: {lab.get('human_labels', 0)}, tags: {lab.get('tags', 0)})")
    return "\n".join(lines) + "\n"


def compare(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Side by side, metric by metric, with the delta after - before."""
    a_label, b_label = before["label"], after["label"]
    lines = [f"# {a_label} ({before['date']}) vs {b_label} ({after['date']})", "", f"before: {_provenance_line(before)}", f"after: {_provenance_line(after)}", ""]
    ra = {(s, m): v for s, m, v in metric_rows(before)}
    rb = {(s, m): v for s, m, v in metric_rows(after)}
    order: list[tuple[str, str]] = []
    for key in list(ra) + list(rb):
        if key not in order:
            order.append(key)
    section = None
    body: list[list[str]] = []
    for s, m in order:
        if s != section:
            section = s
            body.append([f"**{s}**", "", "", ""])
        a, b = ra.get((s, m)), rb.get((s, m))
        body.append([m, fmt(a), fmt(b), fmt_delta(a, b)])
    lines += _table(["metric", a_label, b_label, "delta"], ["---", "---:", "---:", "---:"], body)
    return "\n".join(lines) + "\n"


def write_outputs(result: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{result['date']}-{result['label']}"
    js, md = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    js.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md.write_text(render_markdown(result), encoding="utf-8")
    return js, md


# -- CLI -----------------------------------------------------------------------------------


def _log(s: str) -> None:
    print(s, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        ap = argparse.ArgumentParser(prog="benchmark compare")
        ap.add_argument("before", type=Path)
        ap.add_argument("after", type=Path)
        a = ap.parse_args(argv[1:])
        before = json.loads(a.before.read_text(encoding="utf-8"))
        after = json.loads(a.after.read_text(encoding="utf-8"))
        print(compare(before, after), end="")
        return 0
    ap = argparse.ArgumentParser(prog="benchmark", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS, help="newest closed sessions to replay (0 = all)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--sessions-dir", type=Path, default=None)
    ap.add_argument("--datasets", type=Path, default=None)
    ap.add_argument("--model", default=None, help="joblib to replay with (default: the live model; 'none' = rules only)")
    ap.add_argument("--config", type=Path, default=None, help="config.toml (default: the user's)")
    a = ap.parse_args(argv)
    model: Path | None
    if a.model is None:
        model = live_model_path()
    elif a.model == "none":
        model = None
    else:
        model = Path(a.model)
    result = run_benchmark(a.label, a.sessions, a.sessions_dir, a.datasets, model, a.config, _log)
    js, md = write_outputs(result, a.out)
    print(md.read_text(encoding="utf-8"), end="")
    _log(f"wrote {js} and {md} in {result['run_time_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

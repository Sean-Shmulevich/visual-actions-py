"""`python -m visual_actions.intent <command> <session> [...]`: run one pass of the intent pipeline.

A session is a folder path or a stamp under the platform sessions dir. Each pass writes under
<session>/intent/. Commands dispatch through COMMANDS so new passes add one entry each.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .schema import Segment, SegmentKind, read_jsonl, write_jsonl

if TYPE_CHECKING:
    from . import jev


def resolve_session(arg: str) -> Path:
    p = Path(arg).expanduser()
    if p.is_dir():
        return p
    from ..paths import sessions_dir

    q = sessions_dir() / arg
    if q.is_dir():
        return q
    raise SystemExit(f"no session at {p} or {q}")


def _kinds(arg: str | None) -> list[str] | None:
    if not arg:
        return None
    kinds = [k.strip() for k in arg.split(",") if k.strip()]
    bad = [k for k in kinds if k not in SegmentKind.__members__.values()]
    if bad:
        raise SystemExit(f"unknown kinds {bad}; choose from {[k.value for k in SegmentKind]}")
    return kinds


def _segments(session: Path) -> list[Segment]:
    path = session / "intent" / "segments.jsonl"
    segs = list(read_jsonl(path, Segment))
    if not segs:
        raise SystemExit(f"{path} is missing or empty: run `segments` first")
    return segs


# -- commands ---------------------------------------------------------------------------


def cmd_segments(args: argparse.Namespace) -> int:
    from .events import parse_events
    from .segments import segment_session

    session = resolve_session(args.session)
    events = parse_events(session / "events.log")
    segs = segment_session(session.name, events)
    out = session / "intent" / "segments.jsonl"
    n = write_jsonl(out, segs)
    counts: dict[str, int] = {}
    for s in segs:
        counts[s.kind.value] = counts.get(s.kind.value, 0) + 1
    print(f"{out}: {n} segments " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


def cmd_clips(args: argparse.Namespace) -> int:
    from . import clips

    session = resolve_session(args.session)
    kinds = _kinds(args.kinds)
    segs = [s for s in _segments(session) if kinds is None or s.kind.value in kinds]
    if args.limit is not None:
        segs = segs[: args.limit]
    index = clips.VideoIndex.load(session)
    out_dir = session / "intent" / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"index: {'video.idx' if index.exact_index else 'approximate'}; {len(segs)} segments -> {out_dir}")
    for s in segs:
        name = s.segment_id.replace("/", "_")
        frames = clips.frames_for(session, s.t0, s.t1, fps=args.fps, index=index)
        for k, f in enumerate(frames):
            (out_dir / f"{name}.f{k:02d}.jpg").write_bytes(f)
        (out_dir / f"{name}.strip.png").write_bytes(clips.skeleton_strip(session, s.t0, s.t1))
        if args.mp4:
            clips.clip_mp4(session, s.t0, s.t1, out_dir / f"{name}.mp4", index=index)
        print(f"{s.segment_id}: {s.t0:.2f}-{s.t1:.2f}s {len(frames)} frames")
    return 0


def cmd_cosmos(args: argparse.Namespace) -> int:
    from . import cosmos

    session = resolve_session(args.session)
    kinds = _kinds(args.kinds)
    segments_path = session / "intent" / "segments.jsonl"
    if not segments_path.exists():
        raise SystemExit(f"{segments_path} is missing: run `segments` first")
    judge: cosmos.Judge
    if args.dry_run:
        out = session / "intent" / "cosmos-dry.jsonl"
        if out.exists():
            out.unlink()
        judge = cosmos.DryRunJudge(session / "intent" / "cosmos-dry")
    else:
        out = session / "intent" / "cosmos.jsonl"
        judge = cosmos.CosmosReason(max_calls=args.max_calls, media=args.media)
        if not judge.api_key:
            raise SystemExit("NVIDIA_API_KEY is not set")
    verdicts = cosmos.run_cosmos(session, segments_path, out, judge, limit=args.limit, kinds=kinds, with_strip=not args.no_strip, log=print)
    print(f"{out}: {len(verdicts)} new verdicts" + (f" (dry run, prompts under {session / 'intent' / 'cosmos-dry'})" if args.dry_run else ""))
    return 0


def make_jev() -> jev.Jev:
    """TYPESAFE_API_KEY / JEV_API_KEY -> JevClient; else OPENROUTER_API_KEY -> OpenRouterJev."""
    import os

    from . import jev as jevmod

    if os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY"):
        return jevmod.JevClient()
    if os.environ.get("OPENROUTER_API_KEY"):
        return jevmod.OpenRouterJev()
    raise SystemExit("no judge key: set TYPESAFE_API_KEY (JEV over TypeSafe) or OPENROUTER_API_KEY (a chat model over OpenRouter); --fake runs without the network")


def cmd_jev(args: argparse.Namespace) -> int:
    from . import jev

    session = resolve_session(args.session)
    d = session / "intent"
    if not (d / "segments.jsonl").exists():
        raise SystemExit(f"{d / 'segments.jsonl'} is missing: run `segments` first")
    client: jev.Jev = jev.FakeJev() if args.fake else make_jev()
    print(f"client: {type(client).__name__} model={getattr(client, 'model', None)}")
    n = jev.run_jev(d / "segments.jsonl", d / "cosmos.jsonl", d / "jev.jsonl", client, limit=args.limit, log=print)
    print(f"{d / 'jev.jsonl'}: {n} new verdicts")
    return 0


def cmd_tag(args: argparse.Namespace) -> int:
    from . import tagger

    session = resolve_session(args.session)
    d = session / "intent"
    if not (d / "segments.jsonl").exists():
        raise SystemExit(f"{d / 'segments.jsonl'} is missing: run `segments` first")
    judge: tagger.Tagger = tagger.FakeTagger() if args.dry_run else tagger.make_tagger(args.backend, max_calls=args.max_calls)
    if isinstance(judge, tagger.ClaudeTagger) and not judge.api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set; --backend codex uses the Codex CLI, --dry-run prints the prompts without calling")
    if not args.dry_run:
        print(f"backend: {type(judge).__name__} model={getattr(judge, 'model', None) or 'default'}")
    n = tagger.run_tag(d / "segments.jsonl", d / "cosmos.jsonl", d / "jev.jsonl", d / "tags.jsonl", judge, limit=args.limit, dry_run=args.dry_run, log=print)
    print(f"{d / 'tags.jsonl'}: {n} {'prompts shown (dry run)' if args.dry_run else 'new tags'}")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    from .review import serve

    return serve(resolve_session(args.session), port=args.port)


def cmd_export(args: argparse.Namespace) -> int:
    from ..paths import datasets_dir
    from .export import export_labels, print_summary

    print_summary(export_labels(resolve_session(args.session), args.datasets or datasets_dir(), out_prefix=args.prefix, include_judge=args.include_judge))
    return 0


def cmd_lf_accuracy(args: argparse.Namespace) -> int:
    from ..paths import sessions_dir
    from .export import labelfn_accuracy

    dirs = [resolve_session(s) for s in args.sessions] or sorted(p.parent.parent for p in sessions_dir().glob("*/intent/human.jsonl"))
    labelfn_accuracy(dirs)
    return 0


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "segments": cmd_segments,
    "clips": cmd_clips,
    "cosmos": cmd_cosmos,
    "jev": cmd_jev,
    "tag": cmd_tag,
    "review": cmd_review,
    "export": cmd_export,
    "lf-accuracy": cmd_lf_accuracy,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m visual_actions.intent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("segments", help="cut events.log into segments -> intent/segments.jsonl")
    s.add_argument("session")

    c = sub.add_parser("clips", help="write sampled frames and skeleton strips per segment -> intent/clips/")
    c.add_argument("session")
    c.add_argument("--kinds", help="comma list: fire,arm,drag,repeat,adjust,broken_hold,hand,dead")
    c.add_argument("--limit", type=int)
    c.add_argument("--fps", type=float, default=4.0)
    c.add_argument("--mp4", action="store_true", help="also cut an mp4 per segment (ffmpeg)")

    k = sub.add_parser("cosmos", help="pass 1: Cosmos Reason judges each segment -> intent/cosmos.jsonl")
    k.add_argument("session")
    k.add_argument("--kinds")
    k.add_argument("--limit", type=int)
    k.add_argument("--dry-run", action="store_true", help="write prompts and frame counts, call nothing")
    k.add_argument("--max-calls", type=int, default=50, help="spend cap for this run")
    k.add_argument("--media", choices=["auto", "video", "frames"], default="auto")
    k.add_argument("--no-strip", action="store_true", help="send frames only, no skeleton strip")

    j = sub.add_parser("jev", help="pass 2a: JEV answers atomic questions over each segment's text state -> intent/jev.jsonl")
    j.add_argument("session")
    j.add_argument("--limit", type=int)
    j.add_argument("--fake", action="store_true", help="canned answers, no network (default client: JevClient with TYPESAFE_API_KEY, else OpenRouterJev with OPENROUTER_API_KEY)")

    t = sub.add_parser("tag", help="pass 2b: a reasoning model tags each segment and flags the ones a human must see -> intent/tags.jsonl")
    t.add_argument("session")
    t.add_argument("--limit", type=int)
    t.add_argument("--dry-run", action="store_true", help="print the prompts, call nothing, write nothing")
    t.add_argument("--max-calls", type=int, default=400, help="spend cap for this run")
    t.add_argument("--backend", choices=["codex", "claude", "fake"], help="default: TAGGER_BACKEND, else codex when the CLI is on PATH, else claude with an API key")

    r = sub.add_parser("review", help="the human queue: one page, one key per answer -> intent/human.jsonl")
    r.add_argument("session")
    r.add_argument("--port", type=int, default=8766)

    e = sub.add_parser("export", help="labelled frames -> datasets/<class>/review-*.jsonl, segments -> datasets/intent/<session>.jsonl")
    e.add_argument("session")
    e.add_argument("--datasets", type=Path, default=None, help="default: the platform datasets dir")
    e.add_argument("--include-judge", action="store_true", help="also export confident tags that needed no human, weighted by confidence (never fist)")
    e.add_argument("--prefix", default="review")

    a = sub.add_parser("lf-accuracy", help="precision / recall / coverage of each labelling function against human labels")
    a.add_argument("sessions", nargs="*", help="default: every session with a human.jsonl")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

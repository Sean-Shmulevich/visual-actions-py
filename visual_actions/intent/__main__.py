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

from .schema import Segment, SegmentKind, by_id, read_jsonl, write_jsonl

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


def _shard(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    i, n = spec.split("/")
    return int(i), int(n)


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
        out = session / "intent" / (f"cosmos.shard-{args.shard.split('/')[0]}.jsonl" if args.shard else "cosmos.jsonl")
        judge = cosmos.CosmosReason(max_calls=args.max_calls, media=args.media)
        if not judge.api_key:
            raise SystemExit("NVIDIA_API_KEY is not set")
    verdicts = cosmos.run_cosmos(session, segments_path, out, judge, limit=args.limit, kinds=kinds, shard=_shard(args.shard), with_strip=not args.no_strip, log=print)
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
    if isinstance(judge, tagger.OpenRouterTagger) and not judge.api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set; --backend codex uses the Codex CLI, --dry-run prints the prompts without calling")
    if isinstance(judge, tagger.ClaudeTagger) and not judge.api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set; --backend codex uses the Codex CLI, --dry-run prints the prompts without calling")
    if not args.dry_run:
        print(f"backend: {type(judge).__name__} model={getattr(judge, 'model', None) or 'default'}")
    n = tagger.run_tag(d / "segments.jsonl", d / "cosmos.jsonl", d / "jev.jsonl", d / "tags.jsonl", judge, limit=args.limit, dry_run=args.dry_run, log=print, kinds=set(args.kinds.split(',')) if args.kinds else None)
    print(f"{d / 'tags.jsonl'}: {n} {'prompts shown (dry run)' if args.dry_run else 'new tags'}")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    from .review import serve

    return serve(resolve_session(args.session), port=args.port, decisions=args.decisions)


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


def _embed_sessions(args_sessions: list[str]) -> list[Path]:
    from ..paths import sessions_dir
    from .embed import embeddings_path

    dirs = [resolve_session(s) for s in args_sessions] or sorted(p.parent.parent for p in sessions_dir().glob("*/intent/embeddings.jsonl"))
    if not dirs:
        raise SystemExit("no session has an intent/embeddings.jsonl: run `embed` first")
    missing = [d for d in dirs if not embeddings_path(d).exists()]
    if missing:
        raise SystemExit(f"no embeddings.jsonl in {[str(d) for d in missing]}: run `embed` first")
    return dirs


def cmd_embed(args: argparse.Namespace) -> int:
    from . import embed

    session = resolve_session(args.session)
    kinds = _kinds(args.kinds)
    segments_path = session / "intent" / "segments.jsonl"
    if not segments_path.exists():
        raise SystemExit(f"{segments_path} is missing: run `segments` first")
    client = embed.Embed1Client(max_calls=args.max_calls)
    if not client.api_key:
        raise SystemExit("NVIDIA_API_KEY is not set")
    out = embed.embeddings_path(session)
    print(f"client: {client.url} model={client.model}")
    vecs = embed.run_embed(session, segments_path, out, client, kinds=kinds, limit=args.limit, log=print)
    print(f"{out}: {len(vecs)} new embeddings ({client.calls} calls)")
    return 0


def _print_hits(hits: list[tuple[str, float]], dirs: list[Path]) -> None:
    from .schema import CosmosVerdict

    seen: dict[str, CosmosVerdict] = {}
    for d in dirs:
        seen.update(by_id(read_jsonl(d / "intent" / "cosmos.jsonl", CosmosVerdict)))
    for sid, cos in hits:
        v = seen.get(sid)
        desc = f" {v.intent.value} {v.motion.value}: {v.hand_description[:90]}" if v else ""
        print(f"{cos:.3f} {sid}{desc}")


def cmd_similar(args: argparse.Namespace) -> int:
    from . import embed

    if bool(args.query) == bool(args.like):
        raise SystemExit("give exactly one of --query or --like")
    dirs = _embed_sessions(args.sessions)
    client = None
    if args.query:
        client = embed.Embed1Client(max_calls=1)
        if not client.api_key:
            raise SystemExit("NVIDIA_API_KEY is not set")
    hits = embed.similar(dirs, query_text=args.query, like_segment_id=args.like, k=args.k, client=client)
    print(f"{len(hits)} nearest of {sum(1 for d in dirs for _ in read_jsonl(embed.embeddings_path(d), embed.Embedding))} embedded segments in {len(dirs)} sessions")
    _print_hits(hits, dirs)
    return 0


def cmd_propagate(args: argparse.Namespace) -> int:
    from . import embed

    dirs = _embed_sessions(args.sessions)
    tags = embed.propagate(dirs, min_cos=args.min_cos, max_per_seed=args.max_per_seed, log=print)
    counts: dict[str, int] = {}
    for t in tags:
        counts[t.verdict.value] = counts.get(t.verdict.value, 0) + 1
    print(f"{len(tags)} propagated tags across {len(dirs)} sessions (min cosine {args.min_cos}) " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
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
    "embed": cmd_embed,
    "similar": cmd_similar,
    "propagate": cmd_propagate,
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
    k.add_argument("--shard", help="i/n: every n-th segment from i, into intent/cosmos.shard-i.jsonl (cat the shards into cosmos.jsonl afterwards)")

    j = sub.add_parser("jev", help="pass 2a: JEV answers atomic questions over each segment's text state -> intent/jev.jsonl")
    j.add_argument("session")
    j.add_argument("--limit", type=int)
    j.add_argument("--fake", action="store_true", help="canned answers, no network (default client: JevClient with TYPESAFE_API_KEY, else OpenRouterJev with OPENROUTER_API_KEY)")

    t = sub.add_parser("tag", help="pass 2b: a reasoning model tags each segment and flags the ones a human must see -> intent/tags.jsonl")
    t.add_argument("session")
    t.add_argument("--limit", type=int)
    t.add_argument("--kinds", help="comma-separated segment kinds to tag (default all)")
    t.add_argument("--dry-run", action="store_true", help="print the prompts, call nothing, write nothing")
    t.add_argument("--max-calls", type=int, default=400, help="spend cap for this run")
    t.add_argument("--backend", choices=["openrouter", "codex", "claude", "fake"], help="default: TAGGER_BACKEND, else openrouter with OPENROUTER_API_KEY, else codex when the CLI is on PATH, else claude with an API key")

    r = sub.add_parser("review", help="the human queue: one page, one key per answer -> intent/human.jsonl")
    r.add_argument("session")
    r.add_argument("--port", type=int, default=8766)
    r.add_argument("--decisions", action="store_true", help="read-only dashboard of every tag the AI made (verdict, reason, escalations), in time order")

    e = sub.add_parser("export", help="labelled frames -> datasets/<class>/review-*.jsonl, segments -> datasets/intent/<session>.jsonl")
    e.add_argument("session")
    e.add_argument("--datasets", type=Path, default=None, help="default: the platform datasets dir")
    e.add_argument("--include-judge", action="store_true", help="also export confident tags that needed no human, weighted by confidence (never fist)")
    e.add_argument("--prefix", default="review")

    a = sub.add_parser("lf-accuracy", help="precision / recall / coverage of each labelling function against human labels")
    a.add_argument("sessions", nargs="*", help="default: every session with a human.jsonl")

    m = sub.add_parser("embed", help="pass 1b: Cosmos Embed1 embeds each segment's clip -> intent/embeddings.jsonl")
    m.add_argument("session")
    m.add_argument("--kinds")
    m.add_argument("--limit", type=int)
    m.add_argument("--max-calls", type=int, default=500, help="spend cap for this run")

    q = sub.add_parser("similar", help="the nearest embedded segments to a text query or to one segment, across sessions")
    q.add_argument("--query", help="free text, embedded with Cosmos Embed1")
    q.add_argument("--like", help="a segment id whose stored vector is the query")
    q.add_argument("--k", type=int, default=10)
    q.add_argument("sessions", nargs="*", help="default: every session with an embeddings.jsonl")

    g = sub.add_parser("propagate", help="copy confident tags to untagged look-alikes above a cosine threshold -> intent/tags.jsonl (append)")
    g.add_argument("--min-cos", type=float, default=0.9)
    g.add_argument("--max-per-seed", type=int, default=20)
    g.add_argument("sessions", nargs="*", help="default: every session with an embeddings.jsonl")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

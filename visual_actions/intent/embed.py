"""Pass 1b: NVIDIA Cosmos Embed1 turns each clip into a 256-dim vector; text queries share the space.

The vectors make the corpus searchable ("a hand resting on the face" -> the clips that look like
that) and let a confident tag spread to its look-alikes without another model call: `propagate`
copies a seed's verdict to untagged neighbours above a cosine threshold, as Tag records with the
model "embed1-propagated", so the human queue and the export treat them like any other tag.

Endpoint: an OpenAI-style embeddings server (the hackathon NIM on port 8003)
    POST $COSMOS_EMBED_URL/embeddings          (COSMOS_EMBED_URL = http://host:8003/v1)
    Authorization: Bearer $NVIDIA_API_KEY
    {"input": "a person walking" | "data:video/mp4;base64,...", "model": "nvidia/cosmos-embed1",
     "request_type": "query", "encoding_format": "float"}
    -> {"data": [{"embedding": [256 floats]}], "usage": {"num_videos": 0|1, ...}}
`request_type` is Cosmos' own field (query | bulk_text | bulk_video); there is no `dimensions`.
Video goes in as the same 4 fps mp4 the Cosmos Reason pass sends (clips.frames_for + frames_to_mp4
over the segment's span), so the two passes see the same footage. The NIM samples MIN_FRAMES (8)
frames from the clip and fails preprocessing on a shorter one (a 1.7 s drag at 4 fps is 7 frames),
so a short clip is padded by holding its last frame.

Only the standard library talks HTTP (urllib); the opener and the clock are injectable for tests.
numpy does the similarity maths.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from . import clips
from .schema import Segment, SegmentKind, Tag, Verdict, append_jsonl, by_id, read_jsonl

DEFAULT_URL = "http://127.0.0.1:8003/v1"
DEFAULT_MODEL = "nvidia/cosmos-embed1"
DIM = 256
FPS = 4.0
MIN_FRAMES = 8  # Embed1 samples this many frames; fewer and the NIM rejects the clip
PROPAGATED_MODEL = "embed1-propagated"
SEED_VERDICTS = frozenset({Verdict.INTENDED, Verdict.MISFIRE, Verdict.MISSED, Verdict.NO_EVENT})
SEED_MIN_CONF = 0.8


@dataclass
class Embedding:
    segment_id: str
    vector: list[float]
    model: str = ""


# -- clients ---------------------------------------------------------------------------


class Embedder(Protocol):
    def embed_video(self, mp4_bytes: bytes) -> list[float]: ...

    def embed_text(self, text: str) -> list[float]: ...


class SpendCapReached(RuntimeError):
    pass


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


@dataclass
class FakeEmbed:
    """Deterministic unit vectors for tests: `vectors` by input (text or mp4 bytes), else a hash of it."""

    vectors: dict[str | bytes, list[float]] = field(default_factory=dict)
    dim: int = DIM
    calls: list[tuple[str, int]] = field(default_factory=list)  # (kind, input length)

    def _vector(self, key: str | bytes) -> list[float]:
        if key in self.vectors:
            return list(self.vectors[key])
        raw = key.encode() if isinstance(key, str) else key
        rng = np.random.default_rng(int.from_bytes(sha256(raw).digest()[:8], "little"))
        return [float(x) for x in _unit(rng.standard_normal(self.dim))]

    def embed_video(self, mp4_bytes: bytes) -> list[float]:
        self.calls.append(("video", len(mp4_bytes)))
        return self._vector(mp4_bytes)

    def embed_text(self, text: str) -> list[float]:
        self.calls.append(("text", len(text)))
        return self._vector(text)


Opener = Callable[[urllib.request.Request, float], Any]


def _urlopen(req: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.urlopen(req, timeout=timeout)


@dataclass
class Embed1Client:
    """Cosmos Embed1 over an OpenAI-style /embeddings endpoint, with retries and a spend cap."""

    api_key: str | None = None
    model: str | None = None
    url: str = DEFAULT_URL
    max_calls: int = 500  # per-run spend cap (attempts that reached the network count)
    retries: int = 4
    backoff_s: float = 1.0
    timeout_s: float = 180.0
    opener: Opener = _urlopen
    sleep: Callable[[float], None] = time.sleep
    calls: int = 0
    last_request: dict[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        env_url = os.environ.get("COSMOS_EMBED_URL")
        if env_url and self.url == DEFAULT_URL:
            self.url = env_url
        self.url = self.url.rstrip("/")
        if not self.url.endswith("/embeddings"):
            self.url += "/embeddings"
        self.api_key = self.api_key or os.environ.get("NVIDIA_API_KEY")
        self.model = self.model or os.environ.get("COSMOS_EMBED_MODEL", DEFAULT_MODEL)

    def body(self, text: str) -> dict[str, Any]:
        return {"input": text, "model": self.model, "request_type": "query", "encoding_format": "float"}

    def embed_video(self, mp4_bytes: bytes) -> list[float]:
        if not mp4_bytes:
            raise ValueError("no video bytes to embed")
        return self._post(self.body("data:video/mp4;base64," + base64.b64encode(mp4_bytes).decode()))

    def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("no text to embed")
        return self._post(self.body(text))

    def _post(self, body: dict[str, Any]) -> list[float]:
        self.last_request = {k: (v if k != "input" else v[:40]) for k, v in body.items()}
        data = json.dumps(body).encode()
        attempt = 0
        while True:
            if self.calls >= self.max_calls:
                raise SpendCapReached(f"spend cap of {self.max_calls} calls reached")
            self.calls += 1
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            req = urllib.request.Request(self.url, data=data, method="POST", headers=headers)
            try:
                with self.opener(req, self.timeout_s) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                retry = e.code == 429 or 500 <= e.code < 600
                if not retry or attempt >= self.retries:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt >= self.retries:
                    raise
            self.sleep(self.backoff_s * (2**attempt))
            attempt += 1
        return parse_embedding(payload)


def parse_embedding(payload: dict[str, Any]) -> list[float]:
    items = payload.get("data") or []
    if not items or not isinstance(items[0], dict) or not items[0].get("embedding"):
        raise ValueError(f"no embedding in response: {json.dumps(payload)[:300]}")
    vec = [float(x) for x in items[0]["embedding"]]
    if len(vec) != DIM:
        raise ValueError(f"expected a {DIM}-dim embedding, got {len(vec)}")
    return vec


# -- the pass --------------------------------------------------------------------------


def pad_frames(frames: list[bytes], n: int = MIN_FRAMES) -> list[bytes]:
    """At least `n` frames: a short clip holds its last frame."""
    return frames + [frames[-1]] * (n - len(frames)) if frames and len(frames) < n else frames


def run_embed(
    session_dir: Path,
    segments_path: Path,
    out_path: Path,
    client: Embedder,
    kinds: Iterable[SegmentKind | str] | None = None,
    limit: int | None = None,
    fps: float = FPS,
    log: Callable[[str], None] | None = None,
) -> list[Embedding]:
    """Embed every segment of the chosen kinds not yet in `out_path`; append as JSONL.

    Resumable: ids already present in `out_path` are skipped, so a crash or a spend cap
    halfway leaves a file the next run continues from.
    """
    session_dir, segments_path, out_path = Path(session_dir), Path(segments_path), Path(out_path)
    wanted = {SegmentKind(k) for k in kinds} if kinds else None
    done = set(by_id(read_jsonl(out_path, Embedding)))
    todo = [s for s in read_jsonl(segments_path, Segment) if (wanted is None or s.kind in wanted) and s.segment_id not in done]
    if limit is not None:
        todo = todo[:limit]
    index = clips.VideoIndex.load(session_dir) if todo else None
    model = str(getattr(client, "model", None) or type(client).__name__.lower())
    out: list[Embedding] = []
    for seg in todo:
        frames = clips.frames_for(session_dir, seg.t0, seg.t1, fps=fps, index=index)
        if not frames:
            if log:
                log(f"{seg.segment_id}: no frames, skipped")
            continue
        try:
            vec = client.embed_video(clips.frames_to_mp4(pad_frames(frames), fps))
        except SpendCapReached as e:
            if log:
                log(str(e))
            break
        e = Embedding(segment_id=seg.segment_id, vector=vec, model=model)
        append_jsonl(out_path, e)
        out.append(e)
        if log:
            log(f"{seg.segment_id}: {len(frames)} frames{f' (held to {MIN_FRAMES})' if len(frames) < MIN_FRAMES else ''} -> {len(vec)}-dim")
    return out


# -- search ------------------------------------------------------------------------------


def embeddings_path(session_dir: Path) -> Path:
    return Path(session_dir) / "intent" / "embeddings.jsonl"


@dataclass
class Corpus:
    """Every embedded segment across sessions, as one unit-normalised matrix."""

    ids: list[str]
    sessions: list[Path]  # per row: the session dir it came from
    matrix: np.ndarray  # (n, DIM), rows unit length

    @classmethod
    def load(cls, session_dirs: Iterable[Path]) -> Corpus:
        ids: list[str] = []
        sessions: list[Path] = []
        rows: list[np.ndarray] = []
        for d in session_dirs:
            d = Path(d)
            for e in by_id(read_jsonl(embeddings_path(d), Embedding)).values():
                ids.append(e.segment_id)
                sessions.append(d)
                rows.append(_unit(np.asarray(e.vector, dtype=np.float64)))
        matrix = np.vstack(rows) if rows else np.zeros((0, DIM))
        return cls(ids, sessions, matrix)

    def row(self, segment_id: str) -> np.ndarray | None:
        try:
            return self.matrix[self.ids.index(segment_id)]
        except ValueError:
            return None

    def nearest(self, query: np.ndarray, k: int, exclude: frozenset[str] | set[str] = frozenset()) -> list[tuple[str, float]]:
        if len(self.ids) == 0:
            return []
        cos = self.matrix @ _unit(np.asarray(query, dtype=np.float64))
        order = np.argsort(-cos, kind="stable")
        out: list[tuple[str, float]] = []
        for i in order:
            if self.ids[i] in exclude:
                continue
            out.append((self.ids[i], float(cos[i])))
            if len(out) >= k:
                break
        return out


def similar(
    session_dirs: Iterable[Path],
    query_text: str | None = None,
    like_segment_id: str | None = None,
    k: int = 10,
    client: Embedder | None = None,
) -> list[tuple[str, float]]:
    """Top-k (segment_id, cosine) across sessions, by a text query (needs a client) or by a stored segment's vector."""
    corpus = Corpus.load(session_dirs)
    if like_segment_id is not None:
        q = corpus.row(like_segment_id)
        if q is None:
            raise ValueError(f"{like_segment_id} has no embedding in the given sessions")
        return corpus.nearest(q, k, exclude={like_segment_id})
    if query_text is not None:
        if client is None:
            raise ValueError("a text query needs an embedding client")
        return corpus.nearest(np.asarray(client.embed_text(query_text)), k)
    raise ValueError("give a query_text or a like_segment_id")


# -- propagation -------------------------------------------------------------------------


def is_seed(t: Tag, min_conf: float = SEED_MIN_CONF) -> bool:
    return t.verdict in SEED_VERDICTS and t.confidence >= min_conf


def propagate(
    session_dirs: Iterable[Path],
    min_cos: float = 0.9,
    max_per_seed: int = 20,
    seed_min_conf: float = SEED_MIN_CONF,
    log: Callable[[str], None] | None = None,
) -> list[Tag]:
    """Copy each confident tag's verdict to its untagged look-alikes; append them to each session's tags.jsonl.

    A seed is a tag with a settled verdict (intended / misfire / missed / no_event) at or above
    `seed_min_conf`. A neighbour is an embedded segment with no tag at all, above `min_cos`
    from the seed; a neighbour several seeds reach takes the closest seed. Existing tags are
    never touched, and a propagated tag is never a seed for another.
    """
    dirs = [Path(d) for d in session_dirs]
    corpus = Corpus.load(dirs)
    tags_by_session: dict[Path, dict[str, Tag]] = {d: by_id(read_jsonl(d / "intent" / "tags.jsonl", Tag)) for d in dirs}
    tagged: set[str] = {sid for tags in tags_by_session.values() for sid in tags}
    seeds = [t for tags in tags_by_session.values() for t in tags.values() if is_seed(t, seed_min_conf) and t.model != PROPAGATED_MODEL and corpus.row(t.segment_id) is not None]
    best: dict[str, tuple[float, Tag]] = {}  # target id -> (cosine, seed)
    for seed in seeds:
        q = corpus.row(seed.segment_id)
        assert q is not None
        taken = 0
        for sid, cos in corpus.nearest(q, len(corpus.ids), exclude={seed.segment_id}):
            if cos < min_cos or taken >= max_per_seed:
                break
            if sid in tagged:
                continue
            taken += 1
            if sid not in best or cos > best[sid][0]:
                best[sid] = (cos, seed)
    session_of = dict(zip(corpus.ids, corpus.sessions, strict=True))
    out: list[Tag] = []
    for sid, (cos, seed) in sorted(best.items()):
        tag = Tag(
            segment_id=sid,
            verdict=seed.verdict,
            intent=seed.intent,
            motion=seed.motion,
            true_gesture=seed.true_gesture,
            tags=list(seed.tags),
            confidence=round(seed.confidence * cos, 4),
            needs_human=False,
            reason=f"looks like {seed.segment_id} (cosine {cos:.3f}), which was tagged {seed.verdict.value} at {seed.confidence:.2f}",
            model=PROPAGATED_MODEL,
        )
        append_jsonl(session_of[sid] / "intent" / "tags.jsonl", tag)
        out.append(tag)
        if log:
            log(f"{sid}: {tag.verdict.value} ({tag.confidence:.2f}) <- {seed.segment_id} @ {cos:.3f}")
    return out

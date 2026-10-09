"""Pass 1b: the Cosmos Embed1 request shape, retries and cap, run_embed's resume, similar, propagate."""

import io
import json
import urllib.error
from pathlib import Path
from typing import Any, Self

import numpy as np
import pytest

from visual_actions.intent.embed import (
    DIM,
    PROPAGATED_MODEL,
    Embed1Client,
    Embedding,
    FakeEmbed,
    SpendCapReached,
    embeddings_path,
    parse_embedding,
    propagate,
    run_embed,
    similar,
)
from visual_actions.intent.schema import (
    Intent,
    Motion,
    Segment,
    SegmentKind,
    Tag,
    Verdict,
    append_jsonl,
    read_jsonl,
    write_jsonl,
)

from .test_intent_clips import make_session


class Resp:
    def __init__(self, vec: list[float]) -> None:
        self._body = json.dumps({"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": vec}], "model": "nvidia/cosmos-embed1", "usage": {"num_videos": 0}}).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *a: object) -> None:
        pass


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(b""))  # type: ignore[arg-type]


VEC = [0.01 * (i % 7) for i in range(DIM)]


def make_client(script: list[Any], **kw: Any) -> tuple[Embed1Client, list[dict[str, Any]], list[float], list[str]]:
    requests: list[dict[str, Any]] = []
    sleeps: list[float] = []
    urls: list[str] = []

    def opener(req: Any, timeout: float) -> Any:
        requests.append(json.loads(req.data.decode()))
        urls.append(req.full_url)
        assert req.get_header("Authorization") == "Bearer key"
        assert req.get_header("Content-type") == "application/json"
        r = script.pop(0)
        if isinstance(r, Exception):
            raise r
        return Resp(r)

    c = Embed1Client(api_key="key", model="nvidia/cosmos-embed1", url="http://gpu:8003/v1", opener=opener, sleep=sleeps.append, **kw)
    return c, requests, sleeps, urls


def test_text_request_shape_and_parse():
    c, requests, _, urls = make_client([VEC])
    v = c.embed_text("a hand resting on the face")
    assert v == VEC and len(v) == DIM
    assert urls == ["http://gpu:8003/v1/embeddings"]
    assert requests == [{"input": "a hand resting on the face", "model": "nvidia/cosmos-embed1", "request_type": "query", "encoding_format": "float"}]
    assert "dimensions" not in requests[0]


def test_video_request_shape_is_a_data_url():
    c, requests, _, _ = make_client([VEC])
    c.embed_video(b"\x00\x00\x00\x1cftypisom-mp4")
    assert requests[0]["input"] == "data:video/mp4;base64,AAAAHGZ0eXBpc29tLW1wNA=="
    assert requests[0]["request_type"] == "query" and requests[0]["model"] == "nvidia/cosmos-embed1"
    with pytest.raises(ValueError):
        c.embed_video(b"")


def test_url_env_and_suffix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COSMOS_EMBED_URL", "http://166.19.38.112:8003/v1")
    monkeypatch.setenv("COSMOS_EMBED_MODEL", "nvidia/cosmos-embed1")
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    c = Embed1Client()
    assert c.url == "http://166.19.38.112:8003/v1/embeddings" and c.model == "nvidia/cosmos-embed1" and c.api_key == "k"
    assert Embed1Client(url="http://h/v1/embeddings/").url == "http://h/v1/embeddings"


def test_parse_rejects_wrong_shape():
    with pytest.raises(ValueError):
        parse_embedding({"data": []})
    with pytest.raises(ValueError):
        parse_embedding({"data": [{"embedding": [0.1, 0.2]}]})


def test_retries_on_429_and_5xx_then_gives_up_and_never_retries_4xx():
    c, requests, sleeps, _ = make_client([http_error(429), http_error(503), VEC], backoff_s=0.5)
    assert c.embed_text("x") == VEC
    assert len(requests) == 3 and sleeps == [0.5, 1.0] and c.calls == 3
    c, _, _, _ = make_client([http_error(500)] * 3, retries=2)
    with pytest.raises(urllib.error.HTTPError):
        c.embed_text("x")
    assert c.calls == 3
    c, requests, sleeps, _ = make_client([http_error(400)])
    with pytest.raises(urllib.error.HTTPError):
        c.embed_text("x")
    assert len(requests) == 1 and sleeps == []


def test_spend_cap_counts_every_attempt():
    c, _, _, _ = make_client([http_error(429), http_error(429), VEC], max_calls=2)
    with pytest.raises(SpendCapReached):
        c.embed_text("x")
    assert c.calls == 2


# -- run_embed ---------------------------------------------------------------------------


def segments_for(s: Path) -> Path:
    segs = [
        Segment(f"{s.name}/fire/0000", s.name, SegmentKind.FIRE, 1.0, 2.0),
        Segment(f"{s.name}/hand/0000", s.name, SegmentKind.HAND, 2.0, 3.0),
        Segment(f"{s.name}/dead/0000", s.name, SegmentKind.DEAD, 3.0, 3.5),
    ]
    p = s / "intent" / "segments.jsonl"
    write_jsonl(p, segs)
    return p


@pytest.mark.skipif(__import__("visual_actions.intent.clips", fromlist=["ffmpeg_bin"]).ffmpeg_bin() is None, reason="ffmpeg not found")
def test_run_embed_packs_a_4fps_mp4_and_resumes(tmp_path: Path):
    s = make_session(tmp_path, with_idx=True)
    segs = segments_for(s)
    out = embeddings_path(s)
    client = FakeEmbed()
    first = run_embed(s, segs, out, client, kinds=["fire", "hand"], log=lambda _m: None)
    assert [e.segment_id for e in first] == [f"{s.name}/fire/0000", f"{s.name}/hand/0000"]
    assert all(len(e.vector) == DIM and e.model == "fakeembed" for e in first)
    assert [k for k, _ in client.calls] == ["video", "video"] and all(n > 0 for _, n in client.calls)
    again = run_embed(s, segs, out, client, kinds=["fire", "hand", "dead"])
    assert [e.segment_id for e in again] == [f"{s.name}/dead/0000"]
    assert len(client.calls) == 3
    assert [e.segment_id for e in read_jsonl(out, Embedding)] == [f"{s.name}/fire/0000", f"{s.name}/hand/0000", f"{s.name}/dead/0000"]
    assert run_embed(s, segs, out, client, limit=5) == []


# -- similar / propagate on synthetic vectors ---------------------------------------------


def unit(*xs: float) -> list[float]:
    v = np.zeros(DIM)
    v[: len(xs)] = xs
    v /= np.linalg.norm(v)
    return [float(x) for x in v]


def corpus(tmp_path: Path) -> tuple[Path, Path]:
    """Two sessions; a's fire/0 is the hub, b/hand/1 is its twin, b/dead/2 is far away."""
    a, b = tmp_path / "a", tmp_path / "b"
    write_jsonl(embeddings_path(a), [
        Embedding("a/fire/0", unit(1, 0, 0)),
        Embedding("a/fire/1", unit(1, 0.3, 0)),  # cos 0.958 to the hub
        Embedding("a/hand/2", unit(0, 1, 0)),
    ])
    write_jsonl(embeddings_path(b), [
        Embedding("b/hand/1", unit(1, 0.05, 0)),  # cos 0.999 to the hub
        Embedding("b/dead/2", unit(0, 0, 1)),
        Embedding("b/fire/3", unit(1, 0.5, 0)),  # cos 0.894 to the hub: under 0.9
    ])
    return a, b


def test_similar_orders_by_cosine_across_sessions(tmp_path: Path):
    a, b = corpus(tmp_path)
    hits = similar([a, b], like_segment_id="a/fire/0", k=3)
    assert [h for h, _ in hits] == ["b/hand/1", "a/fire/1", "b/fire/3"]
    assert hits[0][1] == pytest.approx(0.99875, abs=1e-4) and hits[1][1] == pytest.approx(0.9578, abs=1e-3)
    assert all(h != "a/fire/0" for h, _ in hits)
    fake = FakeEmbed(vectors={"something upright": unit(0, 0, 1)})
    hits = similar([a, b], query_text="something upright", k=2, client=fake)
    assert hits[0] == ("b/dead/2", pytest.approx(1.0)) and fake.calls == [("text", 17)]
    with pytest.raises(ValueError):
        similar([a, b], like_segment_id="nope")
    with pytest.raises(ValueError):
        similar([a, b], query_text="x")


def tag(sid: str, verdict: Verdict, conf: float, model: str = "judge") -> Tag:
    return Tag(sid, verdict, Intent.COMMAND, Motion.STILL, "open_palm", ["t"], conf, False, "seed", model)


def test_propagate_writes_only_untagged_neighbours_and_never_overwrites(tmp_path: Path):
    a, b = corpus(tmp_path)
    write_jsonl(a / "intent" / "tags.jsonl", [
        tag("a/fire/0", Verdict.INTENDED, 0.9),
        tag("a/fire/1", Verdict.MISFIRE, 0.95),  # already tagged: a seed too, and never overwritten
        tag("a/hand/2", Verdict.NO_EVENT, 0.5),  # under the seed confidence: not a seed
    ])
    before_a = list(read_jsonl(a / "intent" / "tags.jsonl", Tag))
    written = propagate([a, b], min_cos=0.9)
    assert [t.segment_id for t in written] == ["b/fire/3", "b/hand/1"]
    by = {t.segment_id: t for t in written}
    t = by["b/hand/1"]
    # the closest seed wins: b/hand/1 is 0.999 from a/fire/0 and 0.96 from a/fire/1
    assert t.verdict == Verdict.INTENDED and t.model == PROPAGATED_MODEL and t.needs_human is False
    assert t.confidence == pytest.approx(0.9 * 0.99875, abs=1e-3)
    assert "a/fire/0" in t.reason and "0.999" in t.reason
    assert t.intent == Intent.COMMAND and t.true_gesture == "open_palm" and t.tags == ["t"]
    # b/fire/3 is 0.894 from a/fire/0 (under the bar) but 0.985 from a/fire/1, so it takes that seed's verdict
    u = by["b/fire/3"]
    assert u.verdict == Verdict.MISFIRE and u.confidence == pytest.approx(0.95 * 0.985, abs=2e-3) and "a/fire/1" in u.reason
    tags_a = list(read_jsonl(a / "intent" / "tags.jsonl", Tag))
    assert tags_a == before_a  # a's file untouched: its neighbours were all tagged already
    tags_b = list(read_jsonl(b / "intent" / "tags.jsonl", Tag))
    assert [x.segment_id for x in tags_b] == ["b/fire/3", "b/hand/1"]
    # a second run adds nothing: the neighbours are tagged now and propagated tags never seed
    assert propagate([a, b], min_cos=0.5) == []
    assert [x.segment_id for x in read_jsonl(b / "intent" / "tags.jsonl", Tag)] == ["b/fire/3", "b/hand/1"]
    assert list(read_jsonl(a / "intent" / "tags.jsonl", Tag)) == before_a


def test_propagate_respects_max_per_seed_and_ambiguous_is_no_seed(tmp_path: Path):
    a, b = corpus(tmp_path)
    append_jsonl(a / "intent" / "tags.jsonl", tag("a/fire/0", Verdict.AMBIGUOUS, 0.99))
    assert propagate([a, b], min_cos=0.5) == []
    append_jsonl(a / "intent" / "tags.jsonl", tag("a/fire/0", Verdict.MISSED, 0.85))  # last record per id wins
    written = propagate([a, b], min_cos=0.5, max_per_seed=1)
    assert [t.segment_id for t in written] == ["b/hand/1"] and written[0].verdict == Verdict.MISSED

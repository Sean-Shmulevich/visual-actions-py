"""Pass 2a: JEV answers atomic typed questions over a text state of the moment.

JEV (TypeSafe AI, https://docs.typesafe.ai) is a hosted text-only typed-decision model.
One request carries a `state` plus a map of typed questions evaluated in parallel:
`noul` (is this statement true -> probability 0..1), `choice` (pick one option -> choice,
per-option probabilities, confidence), `score` (ordered rubric). It is strong when a task
is decomposed into small atomic questions whose probabilities are combined in code, weak
at one broad question and at arithmetic, so nothing here asks it to count.

The state deliberately carries no engine vocabulary: no gesture names (shapes are S1, S2,
...), no engine outcome ("cancelled_by_fist"), no weak label. The engine's outcome is the
thing under judgement, and the judge must not read the answer off the question.

Verified REST shape (docs.typesafe.ai/api, /introduction/quickstart, /primitives/*):

    POST https://api.typesafe.ai/v1/systemone
    Authorization: Bearer $TYPESAFE_API_KEY
    Content-Type: application/json
    {"state": "...", "model": "jev-latest",
     "questions": {"<id>": {"type": "noul", "instructions": "...",
                            "criteria": {"true": "...", "false": "..."}},
                   "<id>": {"type": "choice", "instructions": "...",
                            "criteria": {"<option>": "<description>", ...}}}}
    -> {"model": "jev-1.13.0",
        "answers": {"<id>": {"type": "noul", "noul": 0.93},
                    "<id>": {"type": "choice", "choice": "<option>",
                             "probabilities": {"<option>": p, ...}, "confidence": 0.8}},
        "usage": {"input_tokens": n, "output_tokens": n}}
    Errors: 401 bad key, 422 malformed question, 429 rate limit, 529 overloaded.

OpenRouterJev answers the same questions with a chat model over OpenRouter (POST
https://openrouter.ai/api/v1/chat/completions, OpenAI shape, Bearer OPENROUTER_API_KEY) and
returns TypeSafe's response shape, so parse_response and everything downstream are shared.
The chat model is asked for JSON only: one probability per noul id and, for the choice
question, the picked option plus a probability per option; probabilities are normalised in
code and the choice confidence is derived as (p_max - 1/n) / (1 - 1/n).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from .events import Event, parse_events
from .schema import CosmosVerdict, JevVerdict, Segment, SegmentKind, append_jsonl, by_id, read_jsonl

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_JEV_MODEL = "openai/gpt-6.1-sol"
MAX_STATE_CHARS = 6000  # well under JEV's context; it prefers a small clean state
NONE_OPTION = "none"

# -- questions -----------------------------------------------------------------------------
# Ids are stable: downstream code and the tagger read answers by these keys.

Q_DELIBERATE = "deliberate_sign"
Q_UNRELATED = "unrelated_activity"
Q_HELD_STILL = "held_still"
Q_STROKE = "sideways_stroke"
Q_REACTS = "reacts_unwanted"
Q_REPEATED = "repeated_sign"
Q_NOBODY = "nobody_at_desk"
Q_SHAPE = "hand_shape"

NOUL_QUESTIONS: dict[str, dict[str, Any]] = {
    Q_DELIBERATE: {
        "type": "noul",
        "instructions": "The person is deliberately showing a hand sign to the camera.",
        "criteria": {
            "true": "The hand is raised and held toward the camera as a signal, the person's attention is on the screen.",
            "false": "The hand is busy with something else, or passes through the frame, or nobody is signalling.",
        },
    },
    Q_UNRELATED: {
        "type": "noul",
        "instructions": "The hand is doing something unrelated to the camera: typing, touching the face, holding a drink or a phone, gesturing while talking.",
    },
    Q_HELD_STILL: {
        "type": "noul",
        "instructions": "One hand shape is held still for at least half a second.",
        "criteria": {
            "true": "A single shape id persists over a span of 0.5 s or more with little movement.",
            "false": "Shapes flicker between ids, or the hand keeps moving, or the shape lasts well under 0.5 s.",
        },
    },
    Q_STROKE: {
        "type": "noul",
        "instructions": "The hand moves sideways across the frame as a stroke, and the stroke itself is the gesture.",
    },
    Q_REACTS: {
        "type": "noul",
        "instructions": "Right after the system acted, the person reacts as if something unwanted happened: a cancelling shape, a sharp withdrawal, a repeat attempt, a change of posture.",
        "criteria": {
            "true": "Within about three seconds of the system acting there is a reaction that reads as 'no, not that'.",
            "false": "The person carries on, or the system did not act, or the hand simply leaves after the sign.",
        },
    },
    Q_REPEATED: {
        "type": "noul",
        "instructions": "The same hand sign is shown again shortly after the first time.",
    },
    Q_NOBODY: {
        "type": "noul",
        "instructions": "Nobody is at the desk.",
        "criteria": {
            "true": "No person and no hand is present for the whole span.",
            "false": "A person or a hand is present for at least part of the span.",
        },
    },
}

# Choice options are physical descriptions. The keys are what the model picks and must not
# be engine names; OPTION_TO_ENGINE maps them back in code. "none" is mandatory.
SHAPE_OPTIONS: dict[str, str] = {
    NONE_OPTION: "no clear hand sign, or no hand in view",
    "spread_palm": "open hand, palm toward the camera, fingers spread",
    "closed_hand": "all fingers curled into the palm, a closed hand",
    "index_left": "index finger extended pointing sideways to the viewer's left, other fingers curled",
    "index_right": "index finger extended pointing sideways to the viewer's right, other fingers curled",
    "index_up": "index finger alone extended straight up",
    "two_fingers_up": "index and middle fingers extended up together, a V, others curled",
    "middle_finger_up": "middle finger alone extended up, others curled",
    "thumb_up": "thumb extended up, all other fingers curled",
    "thumb_down": "thumb extended down, all other fingers curled",
    "pinched_tips": "thumb tip and index tip touching, a pinch",
}

OPTION_TO_ENGINE: dict[str, str | None] = {
    NONE_OPTION: None,
    "spread_palm": "open_palm",
    "closed_hand": "fist",
    "index_left": "h_left",
    "index_right": "h_right",
    "index_up": "point_up",
    "two_fingers_up": "two_up",
    "middle_finger_up": "middle_up",
    "thumb_up": "thumbs_up",
    "thumb_down": "thumbs_down",
    "pinched_tips": "pinch",
}

SHAPE_QUESTION: dict[str, Any] = {
    "type": "choice",
    "instructions": "Which hand sign, if any, is the person showing to the camera? Pick 'none' unless one sign is clearly shown.",
    "criteria": dict(SHAPE_OPTIONS),
}

QUESTION_IDS: tuple[str, ...] = (*NOUL_QUESTIONS.keys(), Q_SHAPE)


def questions() -> dict[str, dict[str, Any]]:
    """The full question map for one request (fresh copy)."""
    q: dict[str, dict[str, Any]] = {k: json.loads(json.dumps(v)) for k, v in NOUL_QUESTIONS.items()}
    q[Q_SHAPE] = json.loads(json.dumps(SHAPE_QUESTION))
    return q


# -- state ---------------------------------------------------------------------------------

_SYSTEM_ACTED_KINDS = (SegmentKind.FIRE, SegmentKind.DRAG, SegmentKind.REPEAT, SegmentKind.ADJUST)


def build_trace(segment: Segment, events: Sequence[Event]) -> str:
    """A de-named token trace for the segment span: shape ids (S1, S2, ... by first
    appearance; a frame with no shape is '-'), confidences, timings relative to t0,
    plus hand seen/lost and the moments the system acted. Never a gesture name."""
    ids: dict[str, str] = {}
    parts: list[str] = []
    last: tuple[str, bool] | None = None
    for e in events:
        if e.t < segment.t0 or e.t > segment.t1:
            continue
        rel = f"+{e.t - segment.t0:.2f}s"
        if e.kind == "token":
            tok = e.token or "none"
            if tok == "none":
                sid = "-"
            else:
                sid = ids.setdefault(tok, f"S{len(ids) + 1}")
            still = bool(e.still)
            if last == (sid, still):
                continue  # collapse runs, the span of a run is what matters
            last = (sid, still)
            conf = f" c={e.confidence:.2f}" if e.confidence is not None else ""
            parts.append(f"{rel} {sid}{conf}{' still' if still else ' moving'}")
        elif e.kind == "hand":
            parts.append(f"{rel} hand {'seen' if e.hand_seen else 'lost'}")
            last = None
        elif e.kind == "action":
            parts.append(f"{rel} system acted")
        elif e.kind == "drag" and e.drag_phase in ("start", "end"):
            parts.append(f"{rel} system {'began' if e.drag_phase == 'start' else 'ended'} moving the pointer")
        elif e.kind == "veto":
            parts.append(f"{rel} hand near the face")
    return "; ".join(parts)


def acted_at(segment: Segment, events: Sequence[Event] | None = None) -> float | None:
    """Seconds into the clip at which the system acted, if it did."""
    if events:
        for e in events:
            if segment.t0 <= e.t <= segment.t1 and (e.kind == "action" or (e.kind == "drag" and e.drag_phase == "start")):
                return round(e.t - segment.t0, 2)
    if segment.kind in _SYSTEM_ACTED_KINDS:
        from .segments import CONTEXT_BEFORE_S

        return CONTEXT_BEFORE_S if segment.kind is SegmentKind.FIRE else 1.0
    return None


def render_state(segment: Segment, cosmos: CosmosVerdict | None, trace: str, acted_s: float | None = None) -> str:
    """A compact fixed-format text state. No engine gesture names, no engine verdicts.

    `trace` comes from build_trace (already de-named). `acted_s` defaults to the segment's
    kind-based estimate when the events are not at hand."""
    if acted_s is None:
        acted_s = acted_at(segment)
    dur = segment.t1 - segment.t0
    lines = [
        "A short clip of a person at a desk, filmed by the webcam above the screen.",
        f"clip: {dur:.1f} s long",
        f"hand in view: {_pct(segment.hand_present_fraction)} of the clip",
    ]
    if cosmos is not None:
        lines += [
            f"person present: {_yn(cosmos.person_present)}",
            f"attention on the screen: {cosmos.attention_to_screen}",
            f"arm raised toward the camera: {_yn(cosmos.arm_raised_toward_camera)}",
            f"hand touching the face: {_yn(cosmos.face_touched)}",
            f"hand motion: {cosmos.motion.value}",
            f"hand, as seen: {cosmos.hand_description.strip() or 'not described'}",
        ]
    else:
        lines.append("video description: not available")
    lines.append(f"shape trace (shape ids by order of appearance, c = detector confidence): {trace or 'no shapes detected'}")
    if acted_s is not None:
        lines.append(f"the system acted at +{acted_s:.1f} s")
    else:
        lines.append("the system did not act during the clip")
    state = "\n".join(lines)
    if len(state) > MAX_STATE_CHARS:
        state = state[: MAX_STATE_CHARS - 3] + "..."
    return state


def _pct(x: float | None) -> str:
    return "unknown" if x is None else f"{round(x * 100)}%"


def _yn(b: bool) -> str:
    return "yes" if b else "no"


# -- combining answers in code ---------------------------------------------------------------


def deliberate_sign_score(answers: dict[str, float]) -> float:
    """One number for 'this was a sign meant for the camera' from the atomic answers.

    The direct answer is discounted by the competing explanations (unrelated activity,
    nobody there) and nudged by physical evidence of a sign (a held shape or a stroke)."""
    p = answers.get(Q_DELIBERATE, 0.5)
    p *= 1.0 - answers.get(Q_UNRELATED, 0.0)
    p *= 1.0 - answers.get(Q_NOBODY, 0.0)
    evidence = max(answers.get(Q_HELD_STILL, 0.0), answers.get(Q_STROKE, 0.0))
    p *= 0.5 + 0.5 * evidence
    return round(min(1.0, max(0.0, p)), 4)


def noul_confidence(p: float) -> float:
    """|2p - 1|: the docs' way to put a Noul on the same scale as Choice confidence."""
    return abs(2.0 * p - 1.0)


def overall_confidence(answers: dict[str, float], choice_confidence: float | None) -> float:
    cs = [noul_confidence(p) for p in answers.values()]
    if choice_confidence is not None:
        cs.append(choice_confidence)
    return round(sum(cs) / len(cs), 4) if cs else 0.0


def parse_response(segment_id: str, resp: dict[str, Any]) -> JevVerdict:
    """Turn one raw response into a JevVerdict. Missing or malformed answers are skipped,
    so a partial response still yields a record (its confidence reflects what came back)."""
    raw = resp.get("answers") or {}
    answers: dict[str, float] = {}
    for qid in NOUL_QUESTIONS:
        a = raw.get(qid)
        if isinstance(a, dict) and isinstance(a.get("noul"), (int, float)):
            answers[qid] = round(float(a["noul"]), 4)
    choice: str | None = None
    probs: dict[str, float] = {}
    choice_conf: float | None = None
    a = raw.get(Q_SHAPE)
    if isinstance(a, dict):
        if isinstance(a.get("choice"), str):
            choice = a["choice"]
        pr = a.get("probabilities")
        if isinstance(pr, dict):
            probs = {str(k): round(float(v), 4) for k, v in pr.items() if isinstance(v, (int, float))}
        if isinstance(a.get("confidence"), (int, float)):
            choice_conf = float(a["confidence"])
        elif probs:
            n = len(probs)
            choice_conf = (max(probs.values()) - 1 / n) / (1 - 1 / n) if n > 1 else 1.0
    return JevVerdict(segment_id, answers, choice, probs, overall_confidence(answers, choice_conf), str(resp.get("model") or ""))


def choice_to_engine(choice: str | None) -> str | None:
    return OPTION_TO_ENGINE.get(choice or NONE_OPTION)


# -- clients -------------------------------------------------------------------------------


class JevError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"JEV HTTP {status}: {body[:300]}")
        self.status = status


class Jev(Protocol):
    def request(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]: ...


class JevClient:
    """The hosted model over urllib. Retries 429 / 5xx (529 = overloaded) with backoff."""

    RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})

    def __init__(
        self,
        api_key: str | None = None,
        model: str = JEV_MODEL,
        url: str = JEV_URL,
        timeout_s: float = 60.0,
        max_retries: int = 4,
        backoff_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        urlopen: Callable[..., Any] = urllib.request.urlopen,
    ):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY") or ""
        self.model = model
        self.url = url
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._sleep = sleep
        self._urlopen = urlopen
        self.calls = 0

    def request(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if not self.api_key:
            raise JevError(401, "no API key: set TYPESAFE_API_KEY")
        body = json.dumps({"state": state, "model": self.model, "questions": questions}).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last: JevError | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
            self.calls += 1
            try:
                with self._urlopen(req, timeout=self.timeout_s) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", "replace") if hasattr(e, "read") else ""
                last = JevError(e.code, text)
                if e.code not in self.RETRY_STATUSES:
                    raise last from None
            except urllib.error.URLError as e:
                last = JevError(0, str(e.reason))
            if attempt < self.max_retries:
                self._sleep(self.backoff_s * (2**attempt))
        assert last is not None
        raise last


class OpenRouterJev:
    """The same questions put to a chat model over OpenRouter; answers come back in
    TypeSafe's response shape. Model from `model`, else JEV_MODEL env, else
    openai/gpt-6.1-sol. Retries and backoff as JevClient."""

    RETRY_STATUSES = JevClient.RETRY_STATUSES
    SYSTEM = (
        "You are a calibrated decision model. You get a STATE (a text description of a moment) and numbered QUESTIONS. "
        "Answer each by id with a probability between 0 and 1 that the statement is true, read against its criteria when given. "
        "For a choice question give the picked option and a probability per option (they should sum to 1). "
        "Answer with ONE JSON object and nothing else."
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        url: str = OPENROUTER_URL,
        timeout_s: float = 120.0,
        max_retries: int = 4,
        backoff_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        urlopen: Callable[..., Any] = urllib.request.urlopen,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or ""
        self.model = model or os.environ.get("JEV_MODEL") or OPENROUTER_JEV_MODEL
        self.url = url
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._sleep = sleep
        self._urlopen = urlopen
        self.calls = 0
        self.last_raw = ""

    @staticmethod
    def build_prompt(state: str, questions: dict[str, dict[str, Any]]) -> str:
        lines = ["STATE:", state, "", "QUESTIONS:"]
        shape: dict[str, Any] = {}
        for qid, q in questions.items():
            if q.get("type") == "choice":
                lines.append(f"- {qid} (choice): {q.get('instructions', '')}")
                for opt, desc in (q.get("criteria") or {}).items():
                    lines.append(f"    option {opt}: {desc}")
                shape[qid] = {"choice": "<option>", "probabilities": {opt: "<0..1>" for opt in (q.get("criteria") or {})}}
            else:
                lines.append(f"- {qid} (probability true): {q.get('instructions', '')}")
                for k, desc in (q.get("criteria") or {}).items():
                    lines.append(f"    {k}: {desc}")
                shape[qid] = "<0..1>"
        lines += ["", "Answer with exactly this JSON shape and nothing else:", json.dumps(shape)]
        return "\n".join(lines)

    def request(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if not self.api_key:
            raise JevError(401, "no API key: set OPENROUTER_API_KEY")
        text = self.complete(self.build_prompt(state, questions))
        return self.to_response(text, questions, self.model)

    def complete(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "messages": [{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": prompt}], "temperature": 0}).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last: JevError | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
            self.calls += 1
            try:
                with self._urlopen(req, timeout=self.timeout_s) as r:
                    resp = json.loads(r.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", "replace") if hasattr(e, "read") else ""
                last = JevError(e.code, text)
                if e.code not in self.RETRY_STATUSES:
                    raise last from None
            except urllib.error.URLError as e:
                last = JevError(0, str(e.reason))
            if attempt < self.max_retries:
                self._sleep(self.backoff_s * (2**attempt))
        else:
            assert last is not None
            raise last
        if isinstance(resp, dict) and resp.get("error") and not resp.get("choices"):
            raise JevError(0, json.dumps(resp.get("error")))
        self.last_raw = json.dumps(resp)
        choices = resp.get("choices") or [] if isinstance(resp, dict) else []
        msg = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
        content = msg.get("content", "")
        if isinstance(content, list):  # some providers return content parts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return str(content or "")

    @staticmethod
    def to_response(text: str, questions: dict[str, dict[str, Any]], model: str) -> dict[str, Any]:
        """The chat model's JSON (first {...} block, leniently) -> TypeSafe's response shape.
        Missing answers are left out so parse_response's partial handling applies."""
        m = _JSON_BLOCK.search(text or "")
        d: Any = {}
        if m:
            try:
                d = json.loads(m.group(0))
            except json.JSONDecodeError:
                d = {}
        if not isinstance(d, dict):
            d = {}
        answers: dict[str, Any] = {}
        for qid, q in questions.items():
            a = d.get(qid)
            if q.get("type") == "choice":
                opts = list(q.get("criteria") or {})
                probs = _normalise(a.get("probabilities") if isinstance(a, dict) else None, opts)
                choice = a.get("choice") if isinstance(a, dict) else (a if isinstance(a, str) else None)
                if choice not in opts:
                    choice = max(probs, key=lambda o: probs[o]) if probs else None
                if choice is None:
                    continue
                n = len(opts)
                conf = (probs.get(choice, 0.0) - 1 / n) / (1 - 1 / n) if n > 1 else 1.0
                answers[qid] = {"type": "choice", "choice": choice, "probabilities": probs, "confidence": round(min(1.0, max(0.0, conf)), 4)}
            else:
                p = a.get("probability", a.get("noul")) if isinstance(a, dict) else a
                if isinstance(p, bool) or not isinstance(p, (int, float)):
                    continue
                answers[qid] = {"type": "noul", "noul": round(min(1.0, max(0.0, float(p))), 4)}
        return {"model": model, "answers": answers}


def _normalise(raw: Any, opts: list[str]) -> dict[str, float]:
    probs = {o: max(0.0, float(raw[o])) for o in opts if isinstance(raw, dict) and isinstance(raw.get(o), (int, float)) and not isinstance(raw.get(o), bool)}
    total = sum(probs.values())
    if total > 0:
        return {o: round(probs.get(o, 0.0) / total, 4) for o in opts}
    return {}


class FakeJev:
    """Canned answers for tests and dry runs. `answer(state, questions)` may be swapped
    for a function; by default every Noul is 0.5 and the choice is 'none'."""

    def __init__(self, answer: Callable[[str, dict[str, dict[str, Any]]], dict[str, Any]] | None = None, model: str = "jev-fake"):
        self.model = model
        self._answer = answer
        self.requests: list[tuple[str, dict[str, dict[str, Any]]]] = []

    def request(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        self.requests.append((state, questions))
        if self._answer is not None:
            return self._answer(state, questions)
        answers: dict[str, Any] = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.5}
            elif q["type"] == "choice":
                opts = list(q["criteria"])
                answers[qid] = {"type": "choice", "choice": NONE_OPTION, "confidence": 0.0, "probabilities": {o: 1 / len(opts) for o in opts}}
        return {"model": self.model, "answers": answers, "usage": {"input_tokens": 0, "output_tokens": 0}}


# -- the pass --------------------------------------------------------------------------------


def state_digest(state: str) -> str:
    return hashlib.sha1(state.encode("utf-8")).hexdigest()[:12]


def run_jev(
    segments_path: Path,
    cosmos_path: Path,
    out_path: Path,
    client: Jev,
    limit: int | None = None,
    events: Sequence[Event] | None = None,
    log: Callable[[str], None] = print,
) -> int:
    """Ask JEV about every segment not yet in out_path; append one JevVerdict per segment.
    Resumable: segments already answered are skipped. Returns the number answered."""
    segments = list(read_jsonl(segments_path, Segment))
    cosmos = by_id(read_jsonl(cosmos_path, CosmosVerdict))
    done = by_id(read_jsonl(out_path, JevVerdict))
    if events is None:
        log_path = segments_path.parent.parent / "events.log"
        events = parse_events(log_path) if log_path.exists() else []
    n = 0
    for seg in segments:
        if seg.segment_id in done:
            continue
        if limit is not None and n >= limit:
            break
        state = render_state(seg, cosmos.get(seg.segment_id), build_trace(seg, events), acted_at(seg, events))
        resp = client.request(state, questions())
        verdict = parse_response(seg.segment_id, resp)
        append_jsonl(out_path, verdict)
        n += 1
        log(f"{seg.segment_id}: sign={deliberate_sign_score(verdict.answers):.2f} shape={verdict.choice} conf={verdict.confidence:.2f}")
    return n

"""Pass 2b: a reasoning model tags the moment and decides whether a human must look.

Unlike the Cosmos and JEV passes, this one sees the engine's side (what token fired, what
action ran, the weak label from the log) and reconciles it against the person's intent
as the earlier passes saw it. It answers in JSON only; the needs_human rule is code, not
the model's opinion, so it can be audited and tightened without re-tagging.

Two backends share the prompt: CodexTagger runs the OpenAI Codex CLI non-interactively
(`codex exec`, prompt on stdin, the answer shape enforced by --output-schema, the last
message read from a file) and is the default when the binary is on PATH; ClaudeTagger
calls the Anthropic Messages API directly over urllib (no SDK in this project): POST
https://api.anthropic.com/v1/messages with x-api-key and anthropic-version headers.
OpenRouterTagger reuses ClaudeTagger's flow over the OpenAI chat shape at OpenRouter.
`make_tagger` picks one from TAGGER_BACKEND, the PATH and the environment.

Never-ask-the-user policy: whenever the first answer is ambiguous or under the confidence
floor the same prompt goes to a stronger reasoning model (Astra over OpenRouter) and that
answer wins; `needs_human` is still computed and kept in the reason (it feeds the decisions
dashboard and the audit), but a tag only asks for a human when TAGGER_ASK_HUMAN=1.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from . import jev as jevmod
from .schema import (
    CosmosVerdict,
    Intent,
    JevVerdict,
    Motion,
    Segment,
    Tag,
    Verdict,
    append_jsonl,
    by_id,
    read_jsonl,
)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OAUTH_BETA = "oauth-2025-04-20"  # an OAuth access token (sk-ant-oat...) goes on Authorization: Bearer with this beta
DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_ESCALATE = "claude-sonnet-5-5"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-6.1-sol"
OPENROUTER_ESCALATE = "openai/gpt-6-astra"  # a stronger reasoning model for the ambiguous ones
ASK_HUMAN_ENV = "TAGGER_ASK_HUMAN"
CODEX_BINARY = "codex"
_REAL_API_KEY = re.compile(r"^sk-ant-api\d{2}-[A-Za-z0-9_-]{30,}$")
CONFIDENCE_FLOOR = 0.75
STRONG_WEAK_LABEL = 0.5
AUDIT_PERCENT = 10
MAX_TOKENS = 1024

ENGINE_GESTURES = ("none", "open_palm", "fist", "h_left", "h_right", "point_up", "two_up", "middle_up", "thumbs_up", "thumbs_down", "pinch")
KNOWN_TAGS = ("face_touch", "typing", "drink", "phone", "talking", "fast_exit", "transition", "held_from_previous_command", "late", "wrong_gesture", "low_light", "off_screen")

SYSTEM_PROMPT = """You label moments from a webcam session of a person controlling their computer with hand gestures.
Three sources describe each moment: what the gesture ENGINE did (its own vocabulary: tokens, actions, outcomes, a weak label from log heuristics), what a VIDEO model saw (physical description only), and calibrated answers from a DECISION model to atomic questions over a de-named shape trace.
Reconcile them. The engine is what is under judgement: decide whether the person meant what the engine did.

Engine gesture vocabulary: none, open_palm, fist, h_left (index pointing to the viewer's left), h_right (index pointing right), point_up, two_up (V), middle_up, thumbs_up, thumbs_down, pinch (thumb and index tips touching).

Verdicts:
- intended: the engine did what the person wanted
- misfire: the engine acted, the person did not mean it
- missed: the person tried a sign, the engine did not act (or acted late or on the wrong sign)
- no_event: nothing to judge, the hand was incidental or nobody was there and the engine stayed quiet
- ambiguous: the sources genuinely conflict or say too little

Answer with ONE JSON object and nothing else:
{"verdict": "intended|misfire|missed|no_event|ambiguous",
 "intent": "command|incidental|dead|unsure",
 "motion": "still|moving|swipe|slide|pinch_drag",
 "true_gesture": "<engine vocabulary or null>",
 "tags": ["face_touch","typing","drink","phone","talking","fast_exit","transition","held_from_previous_command","late","wrong_gesture", ...],
 "confidence": 0.0-1.0,
 "reason": "one or two sentences"}
true_gesture is the sign the person actually showed, in engine vocabulary, or null when there was none or you cannot tell. Use "transition" when a shape was read while the hand was moving between shapes, "held_from_previous_command" when the sign is a leftover of the previous command, "fast_exit" when the hand flicked out of frame."""

HARDER_INSTRUCTION = "The first pass at this moment came back ambiguous or unsure. Think harder: weigh each source again against the others, pick the single best-supported verdict, and give a calibrated confidence. JSON only."

# The answer shape, as a JSON schema for backends that enforce one (codex --output-schema).
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": [v.value for v in Verdict]},
        "intent": {"type": "string", "enum": [v.value for v in Intent]},
        "motion": {"type": "string", "enum": [v.value for v in Motion]},
        "true_gesture": {"anyOf": [{"type": "string", "enum": list(ENGINE_GESTURES)}, {"type": "null"}]},
        "tags": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "intent", "motion", "true_gesture", "tags", "confidence", "reason"],
    "additionalProperties": False,
}


def build_prompt(segment: Segment, cosmos: CosmosVerdict | None, jev: JevVerdict | None, votes: dict[str, str] | None = None) -> str:
    """The user turn: the three inputs plus the weak votes, in a fixed order."""
    votes = segment.labelfn_votes if votes is None else votes
    dur = segment.t1 - segment.t0
    lines = [
        "## Engine",
        f"segment: {segment.segment_id} kind={segment.kind.value} duration={dur:.1f}s hand_present={segment.hand_present_fraction}",
        f"token={segment.engine_gesture} action={segment.engine_action} outcome={segment.engine_outcome} namespace={segment.engine_namespace} token_confidence={segment.mean_confidence}",
        f"log heuristics: votes={json.dumps(votes, sort_keys=True)} weak_label={segment.weak_label} weight={segment.weak_weight:.2f}",
    ]
    if segment.notes:
        lines.append(f"notes: {segment.notes}")
    lines.append("")
    lines.append("## Video model")
    if cosmos is None:
        lines.append("not available")
    else:
        lines += [
            f"person_present={cosmos.person_present} hand_present={cosmos.hand_present} attention_to_screen={cosmos.attention_to_screen} arm_raised={cosmos.arm_raised_toward_camera} face_touched={cosmos.face_touched}",
            f"intent={cosmos.intent.value} motion={cosmos.motion.value} confidence={cosmos.confidence:.2f}",
            f"hand: {cosmos.hand_description}",
            f"reasoning: {cosmos.reasoning}",
        ]
        if cosmos.sub_spans:
            lines.append(f"sub_spans: {json.dumps(cosmos.sub_spans)}")
    lines.append("")
    lines.append("## Decision model (probability each statement is true)")
    if jev is None:
        lines.append("not available")
    else:
        for qid in jevmod.NOUL_QUESTIONS:
            if qid in jev.answers:
                lines.append(f"{qid}: {jev.answers[qid]:.2f}")
        lines.append(f"combined deliberate-sign score: {jevmod.deliberate_sign_score(jev.answers):.2f}")
        shape = jevmod.choice_to_engine(jev.choice)
        lines.append(f"hand shape pick: {jev.choice} (engine: {shape}) probabilities={json.dumps(jev.choice_probs)}")
    lines.append("")
    lines.append("Tag this moment. JSON only.")
    return "\n".join(lines)


def single_turn_prompt(user: str, harder: bool = False) -> str:
    """System and user turns as one text, for a backend with no system slot (codex exec).
    `harder` appends the escalation instruction for a second try on the same model."""
    text = f"{SYSTEM_PROMPT}\n\n{user}"
    return f"{text}\n\n{HARDER_INSTRUCTION}" if harder else text


# -- parsing ---------------------------------------------------------------------------------


@dataclass
class Answer:
    verdict: Verdict
    intent: Intent
    motion: Motion
    true_gesture: str | None
    tags: list[str]
    confidence: float
    reason: str


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_answer(text: str) -> Answer:
    """Lenient: finds the first {...} block, tolerates code fences, unknown values fall back
    to ambiguous / unsure so a bad answer routes to a human instead of crashing."""
    m = _JSON_BLOCK.search(text or "")
    d: dict[str, Any] = {}
    if m:
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            d = {}
    if not isinstance(d, dict):
        d = {}
    verdict = _enum(Verdict, d.get("verdict"), Verdict.AMBIGUOUS)
    intent = _enum(Intent, d.get("intent"), Intent.UNSURE)
    motion = _enum(Motion, d.get("motion"), Motion.STILL)
    tg = d.get("true_gesture")
    true_gesture = tg if isinstance(tg, str) and tg in ENGINE_GESTURES and tg != "none" else None
    tags = [str(t) for t in d.get("tags", []) if isinstance(t, (str, int))] if isinstance(d.get("tags"), list) else []
    conf = d.get("confidence")
    confidence = min(1.0, max(0.0, float(conf))) if isinstance(conf, (int, float)) else 0.0
    if not d:
        verdict, confidence = Verdict.AMBIGUOUS, 0.0
    return Answer(verdict, intent, motion, true_gesture, tags, confidence, str(d.get("reason", "") or ("unparseable answer" if not d else "")))


def _enum(cls: Any, v: Any, default: Any) -> Any:
    try:
        return cls(v)
    except (ValueError, TypeError):
        return default


# -- the needs_human rule ----------------------------------------------------------------


def audit_sample(segment_id: str, percent: int = AUDIT_PERCENT) -> bool:
    """Deterministic: the same ids are audited on every run."""
    h = int(hashlib.sha1(segment_id.encode("utf-8")).hexdigest(), 16)
    return h % 100 < percent


def needs_human(answer: Answer, segment: Segment, cosmos: CosmosVerdict | None, jev: JevVerdict | None, floor: float = CONFIDENCE_FLOOR) -> tuple[bool, list[str]]:
    """True with the reasons when: the verdict is ambiguous; confidence < floor; the tag
    disagrees with a weak label of weight >= 0.5; Cosmos's intent and JEV's deliberate-sign
    score disagree; or the id falls in the 10 % audit sample of confident tags."""
    why: list[str] = []
    if answer.verdict is Verdict.AMBIGUOUS:
        why.append("ambiguous")
    if answer.confidence < floor:
        why.append(f"confidence {answer.confidence:.2f} < {floor}")
    if segment.weak_label in ("intended", "misfire", "missed") and segment.weak_weight >= STRONG_WEAK_LABEL and answer.verdict.value != segment.weak_label:
        why.append(f"disagrees with weak label {segment.weak_label} (w={segment.weak_weight:.2f})")
    if cosmos is not None and jev is not None and jev.answers:
        sign = jevmod.deliberate_sign_score(jev.answers)
        cosmos_says_command = cosmos.intent is Intent.COMMAND
        jev_says_command = sign >= 0.5
        if cosmos.intent is not Intent.UNSURE and cosmos_says_command != jev_says_command:
            why.append(f"video intent {cosmos.intent.value} vs decision sign score {sign:.2f}")
    if not why and audit_sample(segment.segment_id):
        why.append("audit sample")
    return bool(why), why


# -- taggers ---------------------------------------------------------------------------------


class Tagger(Protocol):
    def tag(self, segment: Segment, cosmos: CosmosVerdict | None, jev: JevVerdict | None, votes: dict[str, str] | None = None) -> Tag: ...


class TaggerError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"Anthropic HTTP {status}: {body[:300]}")
        self.status = status


class CodexError(TaggerError):
    def __init__(self, body: str, status: int = 0):
        RuntimeError.__init__(self, f"codex exec: {body[:400]}")
        self.status = status


class CallCapExceeded(RuntimeError):
    pass


def ask_human_enabled() -> bool:
    """The never-ask-the-user policy: tags route to a human only when TAGGER_ASK_HUMAN=1."""
    return os.environ.get(ASK_HUMAN_ENV, "").strip().lower() in ("1", "true", "yes")


def escalation_note(model: str, why: str) -> str:
    return f"[escalated to {model}: {why}]"


def escalation_reason(answer: Answer, floor: float) -> str | None:
    """Why a first answer is sent to the stronger model, or None when it stands."""
    if answer.verdict is Verdict.AMBIGUOUS:
        return "ambiguous"
    if answer.confidence < floor:
        return f"confidence {answer.confidence:.2f} < {floor}"
    return None


def make_tag(segment: Segment, answer: Answer, cosmos: CosmosVerdict | None, jev: JevVerdict | None, model: str, escalated: str | None = None) -> Tag:
    """`escalated` is the escalation note when a stronger model answered; it goes on the
    reason so the decisions dashboard shows which decisions the AI took on its own."""
    human, why = needs_human(answer, segment, cosmos, jev)
    reason = answer.reason if not why else f"{answer.reason} [human: {'; '.join(why)}]"
    if escalated:
        reason = f"{reason} {escalated}".strip()
    if not ask_human_enabled():
        human = False
    return Tag(segment.segment_id, answer.verdict, answer.intent, answer.motion, answer.true_gesture, list(answer.tags), round(answer.confidence, 3), human, reason, model)


class ClaudeTagger:
    """Messages API over urllib. First answer from `model`; if it is ambiguous or under the
    confidence floor the same prompt goes to `escalate_model` and that answer wins.
    Retries 429 / 5xx / overloaded with backoff; `max_calls` caps one run."""

    RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        escalate_model: str | None = None,
        floor: float = CONFIDENCE_FLOOR,
        max_calls: int = 400,
        max_retries: int = 4,
        backoff_s: float = 1.0,
        timeout_s: float = 120.0,
        url: str = ANTHROPIC_URL,
        sleep: Callable[[float], None] = time.sleep,
        urlopen: Callable[..., Any] = urllib.request.urlopen,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        self.model = model or os.environ.get("TAGGER_MODEL", DEFAULT_MODEL)
        self.escalate_model = escalate_model or os.environ.get("TAGGER_ESCALATE", DEFAULT_ESCALATE)
        self.floor = floor
        self.max_calls = max_calls
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.timeout_s = timeout_s
        self.url = url
        self._sleep = sleep
        self._urlopen = urlopen
        self.calls = 0
        self.last_raw = ""

    def tag(self, segment: Segment, cosmos: CosmosVerdict | None, jev: JevVerdict | None, votes: dict[str, str] | None = None) -> Tag:
        prompt = build_prompt(segment, cosmos, jev, votes)
        text = self.complete(self.model, prompt)
        answer = parse_answer(text)
        model = self.model
        why = escalation_reason(answer, self.floor)
        note = None
        if why and self.escalate_model and self.escalate_model != self.model:
            text = self.complete(self.escalate_model, prompt)
            answer = parse_answer(text)
            model = self.escalate_model
            note = escalation_note(model, why)
        return make_tag(segment, answer, cosmos, jev, model, note)

    def auth_headers(self) -> dict[str, str]:
        """API keys go on x-api-key; an OAuth access token on Authorization: Bearer + the oauth beta."""
        if self.api_key.startswith("sk-ant-oat"):
            return {"Authorization": f"Bearer {self.api_key}", "anthropic-beta": OAUTH_BETA}
        return {"x-api-key": self.api_key}

    def complete(self, model: str, prompt: str) -> str:
        """One Messages call; returns the concatenated text blocks. A refusal returns ''."""
        if self.calls >= self.max_calls:
            raise CallCapExceeded(f"tagger call cap {self.max_calls} reached")
        if not self.api_key:
            raise TaggerError(401, "no API key: set ANTHROPIC_API_KEY")
        body = json.dumps({"model": model, "max_tokens": MAX_TOKENS, "system": SYSTEM_PROMPT, "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
        headers = {**self.auth_headers(), "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}
        last: TaggerError | None = None
        for attempt in range(self.max_retries + 1):
            self.calls += 1
            req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
            try:
                with self._urlopen(req, timeout=self.timeout_s) as r:
                    resp = json.loads(r.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", "replace") if hasattr(e, "read") else ""
                last = TaggerError(e.code, text)
                if e.code not in self.RETRY_STATUSES:
                    raise last from None
            except urllib.error.URLError as e:
                last = TaggerError(0, str(e.reason))
            if attempt < self.max_retries:
                self._sleep(self.backoff_s * (2**attempt))
        else:
            assert last is not None
            raise last
        if resp.get("type") == "error":  # overloaded_error arrives as HTTP 529 normally, but be lenient
            raise TaggerError(0, json.dumps(resp.get("error")))
        self.last_raw = json.dumps(resp)
        if resp.get("stop_reason") == "refusal":
            return ""
        return "".join(b.get("text", "") for b in resp.get("content", []) if isinstance(b, dict) and b.get("type") == "text")


class OpenRouterTagger(ClaudeTagger):
    """ClaudeTagger's flow over the OpenAI chat shape at OpenRouter: the system prompt is the
    system message, Bearer OPENROUTER_API_KEY. Model from TAGGER_MODEL (default
    openai/gpt-6.1-sol); an ambiguous or under-floor answer goes to TAGGER_ESCALATE (default
    openai/gpt-6-astra, a stronger reasoning model) and Tag.model names the one that answered."""

    def __init__(self, api_key: str | None = None, model: str | None = None, escalate_model: str | None = None, url: str = OPENROUTER_URL, **kw: Any):
        super().__init__(api_key=api_key or os.environ.get("OPENROUTER_API_KEY", ""), model=model or os.environ.get("TAGGER_MODEL", OPENROUTER_MODEL), escalate_model=escalate_model or os.environ.get("TAGGER_ESCALATE", OPENROUTER_ESCALATE), url=url, **kw)

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def complete(self, model: str, prompt: str) -> str:
        """One chat completion; returns the assistant text ('' when there is none)."""
        if self.calls >= self.max_calls:
            raise CallCapExceeded(f"tagger call cap {self.max_calls} reached")
        if not self.api_key:
            raise TaggerError(401, "no API key: set OPENROUTER_API_KEY")
        body = json.dumps({"model": model, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], "temperature": 0}).encode("utf-8")
        headers = {**self.auth_headers(), "content-type": "application/json"}
        last: TaggerError | None = None
        for attempt in range(self.max_retries + 1):
            self.calls += 1
            req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
            try:
                with self._urlopen(req, timeout=self.timeout_s) as r:
                    resp = json.loads(r.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", "replace") if hasattr(e, "read") else ""
                last = TaggerError(e.code, text)
                if e.code not in self.RETRY_STATUSES:
                    raise last from None
            except urllib.error.URLError as e:
                last = TaggerError(0, str(e.reason))
            if attempt < self.max_retries:
                self._sleep(self.backoff_s * (2**attempt))
        else:
            assert last is not None
            raise last
        if not isinstance(resp, dict):
            raise TaggerError(0, "malformed response")
        if resp.get("error") and not resp.get("choices"):
            raise TaggerError(0, json.dumps(resp.get("error")))
        self.last_raw = json.dumps(resp)
        choices = resp.get("choices") or []
        msg = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return str(content or "")


class CodexTagger:
    """The OpenAI Codex CLI, non-interactive: `codex exec --ephemeral -s read-only
    --skip-git-repo-check -C <tmp> --output-schema <schema> -o <out> -` with the prompt on
    stdin; the last message (the JSON answer) is read from <out>. The model is `model`,
    else TAGGER_MODEL, else the CLI's config default (`-m` omitted). An ambiguous or
    under-floor answer is escalated once: through an OpenRouterTagger to the Astra model
    when OPENROUTER_API_KEY is set (`escalate_via`), else to TAGGER_ESCALATE (or
    `escalate_model`) when set, else a second call on the same model with the think-harder
    instruction. `max_calls` caps one run. Never a shell string: argv only."""

    def __init__(
        self,
        model: str | None = None,
        escalate_model: str | None = None,
        floor: float = CONFIDENCE_FLOOR,
        max_calls: int = 400,
        timeout_s: float = 120.0,
        binary: str = CODEX_BINARY,
        run: Callable[..., Any] = subprocess.run,
        escalate_via: OpenRouterTagger | None = None,
    ):
        self.model = model or os.environ.get("TAGGER_MODEL") or None
        self.escalate_model = escalate_model or os.environ.get("TAGGER_ESCALATE") or None
        if escalate_via is None and os.environ.get("OPENROUTER_API_KEY"):
            escalate_via = OpenRouterTagger(model=OPENROUTER_ESCALATE, escalate_model=OPENROUTER_ESCALATE, max_calls=max_calls)
        self.escalate_via = escalate_via
        self.floor = floor
        self.max_calls = max_calls
        self.timeout_s = timeout_s
        self.binary = binary
        self._run = run
        self.calls = 0
        self.last_raw = ""
        self.last_argv: list[str] = []
        self.last_seconds = 0.0

    @staticmethod
    def model_name(model: str | None) -> str:
        return f"codex/{model}" if model else "codex/default"

    def tag(self, segment: Segment, cosmos: CosmosVerdict | None, jev: JevVerdict | None, votes: dict[str, str] | None = None) -> Tag:
        user = build_prompt(segment, cosmos, jev, votes)
        text = self.complete(self.model, single_turn_prompt(user))
        answer = parse_answer(text)
        model = self.model_name(self.model)
        why = escalation_reason(answer, self.floor)
        note = None
        if why:
            if self.escalate_via is not None:
                text = self.escalate_via.complete(self.escalate_via.model, user)
                model = self.escalate_via.model
            elif self.escalate_model and self.escalate_model != self.model:
                text = self.complete(self.escalate_model, single_turn_prompt(user))
                model = self.model_name(self.escalate_model)
            else:
                text = self.complete(self.model, single_turn_prompt(user, harder=True))
            answer = parse_answer(text)
            note = escalation_note(model, why)
        return make_tag(segment, answer, cosmos, jev, model, note)

    def complete(self, model: str | None, prompt: str) -> str:
        """One `codex exec` run; returns the last message text ('' when none was written)."""
        if self.calls >= self.max_calls:
            raise CallCapExceeded(f"tagger call cap {self.max_calls} reached")
        exe = shutil.which(self.binary)
        if exe is None:
            raise CodexError(f"'{self.binary}' is not on PATH: install the OpenAI Codex CLI (brew install codex) or set TAGGER_BACKEND=claude with ANTHROPIC_API_KEY")
        with tempfile.TemporaryDirectory(prefix="va-tagger-") as d:
            schema = Path(d) / "answer.schema.json"
            schema.write_text(json.dumps(ANSWER_SCHEMA, indent=1), encoding="utf-8")
            out = Path(d) / "answer.json"
            argv = [exe, "exec", *(["-m", model] if model else []), "--ephemeral", "-s", "read-only", "--skip-git-repo-check", "-C", d, "--output-schema", str(schema), "-o", str(out), "-"]
            self.calls += 1
            self.last_argv = argv
            t0 = time.monotonic()
            try:
                r = self._run(argv, input=prompt, capture_output=True, text=True, timeout=self.timeout_s, cwd=d)
            except FileNotFoundError as e:
                raise CodexError(f"cannot run {exe}: {e}") from None
            except subprocess.TimeoutExpired:
                raise CodexError(f"timed out after {self.timeout_s:.0f} s") from None
            finally:
                self.last_seconds = time.monotonic() - t0
            if r.returncode != 0:
                tail = (r.stderr or r.stdout or "").strip()[-400:]
                raise CodexError(f"exit {r.returncode}: {tail}", status=r.returncode)
            text = out.read_text(encoding="utf-8") if out.exists() else ""
        self.last_raw = text
        return text


class FakeTagger:
    """Answers from a function of the segment (default: a confident intended tag), with
    every prompt recorded for tests."""

    def __init__(self, answer: Callable[[Segment], Answer] | None = None, model: str = "fake-tagger"):
        self._answer = answer
        self.model = model
        self.prompts: list[str] = []

    def tag(self, segment: Segment, cosmos: CosmosVerdict | None, jev: JevVerdict | None, votes: dict[str, str] | None = None) -> Tag:
        self.prompts.append(build_prompt(segment, cosmos, jev, votes))
        answer = self._answer(segment) if self._answer else Answer(Verdict.INTENDED, Intent.COMMAND, Motion.STILL, segment.engine_gesture, [], 0.9, "fake")
        return make_tag(segment, answer, cosmos, jev, self.model)


def make_tagger(backend: str | None = None, max_calls: int = 400) -> Tagger:
    """`backend`, else TAGGER_BACKEND, else: openrouter when OPENROUTER_API_KEY is set, codex
    when the codex binary is on PATH, claude when ANTHROPIC_API_KEY looks like a real
    sk-ant-api key, else a FakeTagger with a warning."""
    name = (backend or os.environ.get("TAGGER_BACKEND") or "").strip().lower()
    if not name:
        if os.environ.get("OPENROUTER_API_KEY"):
            name = "openrouter"
        elif shutil.which(CODEX_BINARY):
            name = "codex"
        elif _REAL_API_KEY.match(os.environ.get("ANTHROPIC_API_KEY", "")):
            name = "claude"
        else:
            warnings.warn("no tagger backend: OPENROUTER_API_KEY is not set, codex is not on PATH and ANTHROPIC_API_KEY is not an sk-ant-api key; using FakeTagger (canned answers)", RuntimeWarning, stacklevel=2)
            return FakeTagger()
    if name == "openrouter":
        return OpenRouterTagger(max_calls=max_calls)
    if name == "codex":
        return CodexTagger(max_calls=max_calls)
    if name == "claude":
        return ClaudeTagger(max_calls=max_calls)
    if name == "fake":
        return FakeTagger()
    raise ValueError(f"unknown tagger backend {name!r}: choose openrouter, codex, claude or fake")


# -- the pass --------------------------------------------------------------------------------


def run_tag(
    segments_path: Path,
    cosmos_path: Path,
    jev_path: Path,
    out_path: Path,
    tagger: Tagger,
    limit: int | None = None,
    kinds: set[str] | None = None,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> int:
    """Tag every segment not yet in out_path; append one Tag each. Resumable. With
    dry_run the prompts are logged and nothing is called or written. Returns the count;
    a call cap stops the run cleanly with what was tagged so far on disk."""
    segments = list(read_jsonl(segments_path, Segment))
    cosmos = by_id(read_jsonl(cosmos_path, CosmosVerdict))
    jev = by_id(read_jsonl(jev_path, JevVerdict))
    done = by_id(read_jsonl(out_path, Tag))
    n = 0
    for seg in segments:
        if seg.segment_id in done or (kinds is not None and seg.kind.value not in kinds):
            continue
        if limit is not None and n >= limit:
            break
        c, j = cosmos.get(seg.segment_id), jev.get(seg.segment_id)
        if dry_run:
            log(f"--- {seg.segment_id}\n{build_prompt(seg, c, j)}")
            n += 1
            continue
        try:
            tag = tagger.tag(seg, c, j)
        except CallCapExceeded as e:
            log(f"stopped: {e}")
            break
        append_jsonl(out_path, tag)
        n += 1
        log(f"{seg.segment_id}: {tag.verdict.value} {tag.intent.value} gesture={tag.true_gesture} conf={tag.confidence:.2f} human={tag.needs_human}")
    return n

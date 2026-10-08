# Visual Actions — Technical Design

Status: draft v0.1 · 2026-10-07 · companion to the PRD (Claude Doc "Visual Actions PRD")

This document turns the PRD's architecture into concrete modules, types and
contracts. It is the file a coding agent should read before touching the code.
When the code and this document disagree, fix one of them in the same commit.

## 1. Decisions already made

| Decision | Choice | Consequence in code |
| --- | --- | --- |
| Shell | Python 3.12 end to end, PyObjC for macOS | One process, `uv`-managed, no Swift in v0.1 |
| Tracker | MediaPipe Hand Landmarker | 21 landmarks, handedness, world coords; datasets recorded now stay valid cross-platform |
| Coordinate frame | User's point of view, mirrored preview | Normalizer flips x once; every rule and label is written as the user experiences it |
| Concurrency | Two threads | `capture` thread: camera + gate + tracker. Main thread: bus, modes, dispatcher, UI, Quartz |
| Stage wiring | Typed synchronous pub/sub | Frozen dataclass events, one in-process `Bus` |
| Gesture 2 | Right-hand H pointing right → Cmd+Shift+Tab | Same classifier label family, direction feature decides |
| Escape | Fist 1.0 s, or no hand for 1.5 s | Two timers in the mode engine, both tested |
| Tier 2 | scikit-learn on normalized landmarks | 63-float feature vector, logistic regression or MLP, `joblib` artifact |
| Feedback | PyObjC overlay `NSWindow` + `NSSound` | Main-thread only; the overlay subscribes to the bus |
| Data dir | Platform-native via `platformdirs` | `~/Library/Application Support/visual-actions/` on macOS |
| Docs | This file in the repo; PRD stays the product source of truth | |

## 2. Package layout

```
visual_actions/
  __init__.py
  __main__.py            # `python -m visual_actions` → app.run()
  app.py                 # wires factory, bus, threads, UI; nothing else lives here
  core/                  # imports NOTHING from platform/ or ui/ (enforced by a test)
    events.py            # event dataclasses + Bus
    types.py             # Landmark, HandFrame, Token, Action, Namespace, Binding
    camera.py            # CameraSource protocol
    gate.py              # tier 0: HandGate protocol + MotionSkinGate
    tracker.py           # tier 1: HandTracker protocol + MediaPipeTracker
    normalize.py         # landmarks → feature vector, the ONE place x is flipped
    recognizer.py        # tier 2: Recognizer protocol, RuleRecognizer, SklearnRecognizer, Smoother
    arbiter.py           # tier 3: Arbiter protocol + NullArbiter (JEV later)
    modes.py             # ModeEngine state machine + Idle/Hold/Armed states
    bindings.py          # namespace → gesture → action tables, loaded from config
    dispatcher.py        # Action → DesktopAutomation calls
    automation.py        # DesktopAutomation protocol
    config.py            # Config dataclass + TOML load/save + defaults
    recorder.py          # writes LandmarkEvents to JSONL datasets
  platform/
    factory.py           # PlatformFactory.create() → (CameraSource, DesktopAutomation, Permissions)
    base.py              # Permissions protocol
    macos/
      camera.py          # OpenCVCamera (AVFoundation backend)
      permissions.py     # AVCaptureDevice + AXIsProcessTrusted checks
      automation.py      # MacAutomation: Quartz, AX, osascript
      sound.py           # NSSound wrapper
    windows/             # stubs: log and return success
    linux/               # mocks: synthetic frames, log automation
    mock/                # used by tests on every OS
  ui/
    menubar.py           # rumps app, toggles, Actions panel entry point
    overlay.py           # NSWindow indicator: mode label, countdown ring, last token
  plugins/
    loader.py            # discovers ~/…/actions/*/action.toml, validates, builds PluginActions
    runner.py            # picks script by platform, runs via automation.run_native
    schema.py            # manifest schema
  tools/
    record.py            # CLI: record a labelled dataset
    train.py             # CLI: train tier 2 from datasets → models/gestures.joblib
    replay.py            # CLI: replay a JSONL dataset through the pipeline headless
tests/
  fixtures/*.jsonl       # recorded landmark sessions
  test_core_isolation.py # core/ imports no platform/ or ui/
  test_pipeline_replay.py
  test_modes.py
  test_normalize.py
  test_plugins.py
skills/
  visual-actions-plugin/SKILL.md
```

## 3. Core types (`core/types.py`)

```python
from dataclasses import dataclass
from enum import Enum

class Hand(Enum):
    LEFT = "left"
    RIGHT = "right"

@dataclass(frozen=True)
class Landmark:
    x: float   # 0..1, user's frame: +x is the user's right
    y: float   # 0..1, +y is down
    z: float   # relative depth, MediaPipe convention
    visibility: float | None = None

@dataclass(frozen=True)
class HandFrame:
    t_ns: int                      # monotonic capture time
    hand: Hand
    landmarks: tuple[Landmark, ...]  # exactly 21, MediaPipe index order
    confidence: float              # handedness/presence score

@dataclass(frozen=True)
class Token:
    """A recognized gesture, already smoothed. The mode engine consumes tokens, never frames."""
    t_ns: int
    name: str          # "open_palm", "fist", "h_left", "h_right", "none"
    confidence: float
    hand: Hand
    still: bool        # centroid drift under threshold over the smoothing window

class ActionKind(Enum):
    KEY = "key"
    SCROLL = "scroll"
    WINDOW = "window"
    MEDIA = "media"
    PLUGIN = "plugin"

@dataclass(frozen=True)
class Action:
    kind: ActionKind
    name: str                  # human label for the popup: "Cmd+Tab"
    args: tuple[tuple[str, str], ...] = ()   # ordered, hashable kwargs

@dataclass(frozen=True)
class Binding:
    namespace: str   # "window"
    gesture: str     # token name
    action: Action
```

Rules:
- Everything crossing a thread or the bus is frozen and hashable.
- `Action` is data, never a closure. The dispatcher interprets it. This is what makes
  replay, logging and the plugin kind possible.
- Landmark order is MediaPipe's (0 wrist, 4 thumb tip, 8 index tip, 12 middle tip,
  16 ring tip, 20 pinky tip). Apple Vision, if added, maps into this order in its tracker.

## 4. Events and the bus (`core/events.py`)

```python
@dataclass(frozen=True) class FrameCaptured:   t_ns: int; frame: "np.ndarray"; gate_open: bool
@dataclass(frozen=True) class HandSeen:        hand_frame: HandFrame
@dataclass(frozen=True) class HandLost:        t_ns: int
@dataclass(frozen=True) class TokenEmitted:    token: Token
@dataclass(frozen=True) class ModeChanged:     t_ns: int; old: str; new: str; namespace: str | None; deadline_ns: int | None
@dataclass(frozen=True) class ActionFired:     t_ns: int; action: Action; ok: bool; message: str
@dataclass(frozen=True) class Tick:            t_ns: int   # 20 Hz, drives timeouts
```

`Bus.subscribe(EventType, handler)` and `Bus.publish(event)`. Synchronous, in order,
on the publishing thread. The capture thread publishes only `FrameCaptured`,
`HandSeen`, `HandLost`; those handlers do nothing but put the event on a
`queue.Queue`. The main thread drains the queue each `Tick` and re-publishes on the
main-thread bus. Net effect: every subscriber except the queue bridge runs on the
main thread, so UI and Quartz calls are always safe.

`FrameCaptured.frame` is published only when a debug preview is on; otherwise the
event carries `frame=None` to avoid copying 30 images a second.

## 5. Threads and timing (`app.py`)

```
capture thread                      main thread (rumps run loop)
──────────────                      ───────────────────────────
loop:                               every 50 ms (NSTimer → Tick):
  frame = camera.read()               drain queue → bus.publish(ev)
  if gate.open(frame):                bus.publish(Tick)
    hands = tracker.track(frame)      (ModeEngine checks deadlines on Tick)
    for h in hands: q.put(HandSeen)
  else if gate.was_open: q.put(HandLost)
```

Latency budget, gesture complete → action fired, target 150 ms:
- capture 33 ms worst case (one frame at 30 fps)
- tracker 15 ms
- smoothing window 250 ms is *not* part of the budget; it is part of "gesture complete"
- queue hop + tick alignment ≤ 50 ms
- recognizer + modes + dispatch < 5 ms
- Quartz post ~5 ms

Sum ≈ 110 ms. If the tick is too coarse, drop to 10 ms; it is a config value.

## 6. Tier 0 gate (`core/gate.py`)

`HandGate.open(frame_bgr, t_ns) -> bool` plus `notify(t_ns, hand_seen)`. A skin
mask alone opened on 99 % of frames (face, desk, walls), so `MotionGate` is
motion-driven with tracker feedback:
1. Downscale to 160 px wide, grey, blur.
2. Open when ≥ 0.4 % of pixels changed by more than 24 levels since the last frame.
3. Stay open for 1.5 s after the last motion or the last frame where the tracker
   reported a hand (`notify`), so a still hand keeps the tracker running.

Measured: 0.14 ms per frame; empty still desk ≈ 0 motion; with no hand present the
tracker runs on 14–39 % of frames (camera auto-exposure and the user moving).

## 7. Tier 1 tracker (`core/tracker.py`)

`HandTracker.track(frame_bgr, t_ns) -> list[HandFrame]`. `MediaPipeTracker` wraps
the Tasks `HandLandmarker` in VIDEO mode, `num_hands=1` in v0.1, min confidences
0.6. It produces landmarks in the raw camera frame; `normalize.py` is the only place
the user-frame flip happens, so the tracker stays convention-free.

Spike result (2026-10-07, M-series MacBook, 640x480, AVFoundation via OpenCV):
capture 29.9 fps, tracker p50 13.7 ms, p95 15.2 ms, right hand reported as
`Right` on the raw (unmirrored) frame. MediaPipe's handedness already assumes a
selfie-style input, so on the raw frame the label matches the user's real hand;
do not re-flip handedness in the normalizer.

## 8. Normalization (`core/normalize.py`)

`to_user_frame(hf: HandFrame, mirror: bool) -> HandFrame` flips x when `mirror` is
true (default on laptops, off for a rear-facing camera; config).

`features(hf: HandFrame) -> np.ndarray[63]`:
1. Translate so the wrist is the origin.
2. Scale by the wrist → middle-MCP distance.
3. Rotate so the wrist → middle-MCP vector points up (removes hand roll; the
   *direction* of the H sign is then recovered separately as a feature).
4. Do NOT mirror by handedness. MediaPipe labelled 854 of 854 frames of a right-hand
   sideways H as "left" (2026-10-08); mirroring on that label inverted every
   direction. v0.1 is right-hand data only; left-handed support means recording
   left-hand sessions.
5. Append the user-frame direction of wrist → index-tip as two features (cos, sin),
   taken from the unmirrored landmarks, so `h_left` vs `h_right` is learnable and
   immune to the handedness label.

The same function feeds the recorder, the trainer and the live recognizer. A test
asserts that recording and live paths produce identical vectors for the same frame.

## 9. Tier 2 recognizer (`core/recognizer.py`)

`Recognizer.classify(features) -> tuple[str, float]`.

- `RuleRecognizer`: hand-written rules for `open_palm` (all four fingers extended,
  spread, palm facing camera) and `fist` (all tips within 0.35 of their MCPs). These
  two drive the leader and escape and must not depend on training.
- `SklearnRecognizer`: loads `models/gestures.joblib`; classes include `"none"`.
- `Smoother`: majority vote with confidence mean over a 250 ms window, plus a
  stillness check on the wrist centroid (< 12 px drift at 640 px width). Emits one
  `Token` per window; emits `none` tokens too, so the mode engine can see "a hand is
  here but doing nothing".

Order: rules first; if a rule fires with confidence ≥ 0.9 it wins, else the model's
answer is used. Ambiguity band for tier 3 is 0.45–0.75 for 300 ms (config).

Calibration note (2026-10-08, three 8 s webcam samples in `tests/fixtures/real/`):
- Straight fingers score 0.8–0.95 on the path-ratio extension metric, not 1.0, because
  of foreshortening. Thresholds are now extended ≥ 0.7, curled ≤ 0.45.
- Thumb "out" vs "tucked" is best read as thumb-tip to index-MCP distance: ~0.55 on an
  open palm, ~0.16 in the H sign. Threshold 0.35.
- With those, the rules hit the H sign on 212/212 frames and the open palm on 120/227.
  The open-palm misses are foreshortened frames where the middle/ring/pinky ratio dips
  under 0.7. The leader therefore needs either a looser rule with the smoother's
  majority vote doing the work, or (better) the trained tier 2 model covering
  `open_palm` as well, with the rule kept only as a fallback.
- A fist seen knuckles-first is indistinguishable from extended fingers in the 2D
  ratio; the `fist` recording classified as 113 none / 36 h_left / 6 open_palm. The
  fist escape must come from the trained model, not the rule. Until then the
  hand-out-of-frame escape is the reliable one.

## 10. Tier 3 arbiter (`core/arbiter.py`)

`Arbiter.decide(features, candidate: str) -> float | None` returns a probability or
`None` if unavailable. `NullArbiter` returns `None`. `JevArbiter` (v0.2) renders
features as a fixed text state and asks atomic Noul questions. It is called from the
main thread with a 300 ms deadline on a worker; if it misses, the token is treated
as `none`. Nothing visual ever leaves the process.

## 11. Mode engine (`core/modes.py`)

States and transitions (the PRD's leader flow):

```
IDLE      --open_palm & still--> HOLDING(t0)
HOLDING   --token != open_palm or !still--> IDLE
HOLDING   --Tick, now-t0 >= leader_hold_s--> ARMED(namespace, deadline)
ARMED     --token in bindings[namespace]--> FIRE(action) --> IDLE
ARMED     --Tick, now >= deadline--> TIMEOUT --> IDLE
ANY       --fist held 1.0 s--> IDLE (escape)
ANY       --HandLost for 1.5 s--> IDLE (escape)
```

Implemented as one engine with `on_token`, `on_hand_lost`, `on_tick`; it publishes
`ModeChanged` on every transition, `HoldProgress` while holding, and the dispatcher
publishes `ActionFired`. No state reads the clock itself; time comes in on tokens
and `Tick`, which is what makes the engine replayable and testable with fake time.

**Confidence-weighted timing (2026-10-08).** Both phases accumulate evidence
rather than wall-clock time, so the system speeds up when it is sure and stalls
when it is not:

- Hold: each palm token adds `dt * rate(c)` where `rate(c) = clamp((c - 0.5) / 0.5,
  -1, 1) * confidence_gain`. At gain 1.5 a confident palm arms in ~1.3 s, a 0.75
  palm at half speed, 0.5 stalls, below 0.5 drains. A moving palm pauses (rate 0).
  The overlay ring draws this evidence fraction, not elapsed time.
- Fire: tokens for a bound gesture add their confidence to a running mass; the
  action fires when mass ≥ `fire_evidence` (0.9). One sure token fires at once, two
  0.5 tokens fire on the second, tokens under `min_token_confidence` (0.3) never
  count. Switching gesture restarts the mass.

## 11b. Pinch-and-drag (`core/pinch.py`, `core/pointer.py`, `core/drag.py`)

Added 2026-10-08. Moves the window under the hand while the user pinches.

- **Pinch** is geometric and per frame: 3-D thumb-tip to index-tip distance in
  canonical units (wrist→middle-MCP = 1), with hysteresis (on < 0.30, off > 0.50,
  calibrated from data by `tools/calibrate.py`) and a 2-frame debounce. While a
  change is pending the detector emits nothing, so the opening hand's jumping
  midpoint never yanks the window on release. The pinch point is the thumb/index
  midpoint in the user frame; hand scale (wrist→middle-MCP in frame units) rides
  along as a depth proxy.
- **Pointer** maps the user-frame point to screen points through a *reach box*
  (the part of the frame a seated user can cover, default 0.15..0.85, calibrated
  from the 5th..95th percentile of recorded wrist positions). Optional depth
  normalization scales the box with hand size (`depth_gain`, off by default). A
  One Euro filter per axis smooths jitter without adding lag at speed.
- **Drag controller**: on pinch START it maps the point, asks the driver for the
  window under it, records the grab pointer and window origin; each MOVE sets the
  window to origin + (pointer − grab pointer) × gain; END (or hand lost) releases in
  place. A pinch over nothing is a MISS and the armed window stays armed.
- **Mode engine**: ARMED + pinch START over a window → DRAGGING; tokens are ignored
  while dragging (the pinched hand classifies as anything); END or hand-lost →
  IDLE. The 5 s command deadline does not apply while dragging.
- **Driver**: `window_at` via CGWindowList (~2 ms), `move_window` via the AX
  position attribute with a per-window element cache (first move ~84 ms, then
  ~0.5 ms). Mock driver holds fake windows for the headless suite.
- **Tests**: `test_pinch.py`, `test_pointer.py`, `test_drag.py`,
  `test_pipeline_drag.py` (synthetic palm→pinch→drag sessions through the whole
  pipeline on fake time, asserting the mock window's final position), plus the
  driver's `test_windows_mock.py`.

## 12. Bindings and config (`core/bindings.py`, `core/config.py`)

```toml
# ~/Library/Application Support/visual-actions/config.toml
[timing]
leader_hold_s = 1.1
command_timeout_s = 5.0
escape_fist_s = 1.0
escape_lost_s = 1.5
popup_ms = 900
tick_ms = 50

[feedback]
audio = true
popup = true

[camera]
index = 0
mirror = true

[recognizer]
model = "models/gestures.joblib"
ambiguous_low = 0.45
ambiguous_high = 0.75

[namespaces.window]
leader = "open_palm"

[[namespaces.window.bindings]]
gesture = "h_left"
action = { kind = "key", name = "Cmd+Tab", chord = "cmd+tab" }

[[namespaces.window.bindings]]
gesture = "h_right"
action = { kind = "key", name = "Cmd+Shift+Tab", chord = "cmd+shift+tab" }

[[namespaces.window.bindings]]
gesture = "thumbs_up"
action = { kind = "plugin", name = "toggle-dark-mode" }
```

Config is loaded once at start and on a menu bar "Reload". Bindings are validated
against known gesture names and discovered plugins; an unknown reference is a
warning in the menu, not a crash.

## 13. Dispatcher and automation (`core/dispatcher.py`, `core/automation.py`)

```python
class DesktopAutomation(Protocol):
    def press(self, chord: str) -> None: ...
    def scroll(self, dx: int, dy: int) -> None: ...
    def set_window_frame(self, target: WindowRef, frame: Rect) -> None: ...
    def media(self, verb: MediaVerb) -> None: ...
    def run_native(self, script_path: Path, timeout_s: float) -> CompletedProcess: ...
```

`MacAutomation.press` posts `CGEvent` key down/up with modifier flags; chord parsing
is shared across platforms in `core/chords.py`.

Spike result (2026-10-07): the app switcher ignores a Tab event that merely carries
the Command flag. The working sequence on `kCGHIDEventTap` is: Command key down
(keycode 0x37, flag set) → Tab down (flag set) → Tab up (flag set) → ~150 ms →
Command key up. So `press` always posts real modifier key events around the main
key. Also: `NSWorkspace.frontmostApplication()` is stale in a process with no run
loop; verify the front app through System Events or a live run loop, never from a
plain script. `run_native` chooses the runner by
extension: `.applescript` → `osascript`, `.py` → the app's interpreter, `.sh` → `sh`.
Windows stub and Linux mock log and return success, so the full pipeline runs in CI.

Dispatcher: `PLUGIN` actions go to `plugins.runner`; everything else maps 1:1 onto
the protocol. It always publishes `ActionFired`, with `ok=False` and the error
message on failure, which the overlay shows.

## 14. Permissions (`platform/macos/permissions.py`)

On launch, in order: camera (`AVCaptureDevice.authorizationStatusForMediaType_`,
request if undetermined), accessibility (`AXIsProcessTrustedWithOptions` with the
prompt option). Each result is a menu bar item with a deep link to the right System
Settings pane. The app runs with everything denied; it just cannot see or act.

## 15. Plugins (`plugins/`)

Manifest schema as in the PRD. Loader runs at start and on "Reload": validates,
builds a `PluginAction(name, title, script_for_platform, confirm)`, and exposes the
list to the bindings validator and the Actions panel. Runner enforces a 10 s timeout
and captures stdout for the popup. The skill at `skills/visual-actions-plugin/
SKILL.md` documents the format for agents.

## 16. Recording, training, replay (`tools/`)

- `record --label h_left --seconds 30` streams `HandSeen` events to
  `datasets/<label>/<timestamp>.jsonl` (one `HandFrame` per line, raw camera frame,
  so normalization changes do not invalidate data).
- `train` loads every dataset, applies `features()`, holds out 20 % by *session*
  (never by frame, frames within a session are correlated), fits, prints a confusion
  matrix, writes `models/gestures.joblib`.
- `replay path.jsonl` feeds a session through gate-skipped tracker output → full
  pipeline with mock automation and fake time, printing tokens, mode changes and
  actions. This is the headless integration test.

Minimum data for v0.1: 2,000 frames each of `h_left`, `h_right`, `open_palm`,
`fist`, `none`, from at least two sessions on different days.

Gesture set as of 2026-10-08 evening: `open_palm` (leader), `fist` (escape),
`h_left`, `h_right`, `point_up`, `two_up`, `none`. Seven classes, 9,786 frames, 12
sessions; the model classifies 99 % of frames in every recording correctly.
`point_up` and `two_up` have one session each; their bindings (Mission Control,
App Exposé) are placeholders.

Status 2026-10-08 (earlier): one guided session recorded (~890 frames per gesture, 207 none).
Logistic regression on the 65-float vector: 0.999 train accuracy, every recording
classified correctly, and the first live arm-and-fire succeeded with the model
(`--dry`). No held-out score yet; the trainer prints one automatically once a second
session per class exists. `none` needs far more variety than one 30 s take.

## 17. Testing

- `test_core_isolation.py`: walks `core/` AST, fails on any import of `platform`,
  `ui`, `plugins`, `cv2`, `mediapipe`, `objc`. `tracker.py` and `gate.py` import
  their libraries lazily inside the concrete classes; the protocols stay pure.
- `test_normalize.py`: invariance to translation, scale, roll; mirror symmetry.
- `test_modes.py`: fake clock, scripted tokens, asserts every transition in §11
  including both escapes and a timeout.
- `test_pipeline_replay.py`: fixture session with a known H sign → exactly one
  `ActionFired(Cmd+Tab)`.
- `test_plugins.py`: good manifest, bad manifest, missing platform.

## 18. Milestone 1 definition of done

`uv run pytest` green on a machine with no camera, `uv run python -m visual_actions
--replay tests/fixtures/h_left.jsonl` prints one Cmd+Tab action, and this document
matches the code.

Status 2026-10-08: done. 33 tests, replay fires exactly one Cmd+Tab, ruff and
pyright clean. Bonus beyond scope: the live loop (`python -m visual_actions --dry`)
already runs camera → gate → tracker → rules → smoother → modes → mock automation
with terminal output, and `tools/record.py` writes datasets. Not yet shown: a full
live arm-and-fire with a real hand, blocked on leader detection reliability (§9
calibration note). That is milestone 3/4 work.

## 19. Open questions carried from the PRD

- Tick at 50 ms vs 10 ms: measure once the loop exists.
- Whether `FrameCaptured` should ever carry pixels on the bus, or the preview should
  read from the capture thread directly.
- JEV: latency from home wifi and the off-device policy before it leaves v0.2.

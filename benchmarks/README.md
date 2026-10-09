# Benchmarks

Reproducible before/after numbers for the classifier and the engine, computed from the recorded
data (`~/Library/Application Support/visual-actions/{datasets,sessions}`), the live model
(`paths.live_model_path()`) and the user's config (`paths.config_path()`). Nothing under the data
directory is written.

```
uv run python -m visual_actions.tools.benchmark --label before-cosmos            # -> benchmarks/<date>-before-cosmos.{json,md}
uv run python -m visual_actions.tools.benchmark --label after-cosmos --sessions 6
uv run python -m visual_actions.tools.benchmark compare benchmarks/2026-10-09-before-cosmos.json benchmarks/<date>-after-cosmos.json
```

`--sessions N` replays the newest N closed sessions (default 6; 0 = every closed session).
`--model path|none` and `--config path` override the live model and the user's config; `--out`
the output directory. Every run records its provenance in the JSON: git short hash, model path
and sha256, config path, profile and sha256, the session list, and every dataset file with its
sha256 and frame count. `compare` prints one table, metric by metric, with `after - before`
deltas; the markdown of a single run uses the same rows with the label as its value column, so an
"after" run reads as a second column next to it.

## Metrics

### Classifier (held-out frame accuracy)

`tools/train.py`'s loader and protocol on `datasets/` as it is now: the newest user session of
every class that has at least two sessions is the test set; public (HaGRID) and synthetic files
only ever train; the model is train.py's StandardScaler + logistic regression (deterministic).

| metric | meaning |
| --- | --- |
| held-out accuracy | fraction of held-out frames predicted as their class (weighted by frames) |
| fist recall | held-out fist frames predicted fist: the cancel must not regress |
| accuracy `<class>` | per-class recall on the held-out frames |
| held-out frames | size of the test set |
| frames, frames `<source>`, frames `<class>` | every frame the loader read, by source (user / public / synth / review) and by class |
| confusions | the largest off-diagonal cells of the held-out confusion matrix, true -> predicted |

### Engine (replay vs live)

The newest N closed sessions, each seen two ways and cut by `intent.segments` either way, so the
counts are the same functions over two event streams:

- replay: `landmarks.jsonl` through the whole pipeline on fake time with the live model and the
  user's config (`learn/evaluate.replay_events`, mock automation), i.e. what today's code and
  model would do on that footage;
- live: the session's own `events.log`, i.e. what the app actually did that day with whatever
  code, model and config it ran.

| metric | meaning |
| --- | --- |
| fires | commands fired (`action` lines) |
| weak-misfire fires | fires on which any of `fist_after_fire`, `token_flip_around_fire`, `immediate_redo`, `low_confidence_trigger` voted misfire (`intent/labelfns.py`) |
| misfire rate | weak-misfire fires / fires |
| weak-intended fires | fires whose combined weak label is `intended` |
| arms | leader holds that reached ARMED |
| empty arms | arms that ended with no command: timeout, fist, or hand lost |
| broken holds | holds that never armed |
| broken-hold ratio | broken holds / (arms + broken holds) |
| drags | pinch-drags started |
| drags cancelled by fist | drags whose release was followed by a fist drop |
| re-grab flaps | a released pinch re-grabbed within the release grace (a drag `resume` with no `pause` before it) |
| releases per drag | (drag ends + re-grab flaps) / drags; 1.0 = every drag released once, cleanly |
| hand losses, hand losses / min | `hand lost` lines; the replay feeds the recorded frames and lost markers as they were, so this is the recording's number in both columns |
| seconds | session length from the live log, shared by both columns |
| votes `<fn>` | fires each misfire labelling function voted on |

Totals are over the N sessions; the per-session table shows `replay / live` side by side.

### Labelled-moment agreement

When human labels exist under `<session>/intent/human.jsonl`, `intent/export.labelfn_accuracy`
scores every labelling function against them: precision (votes the human agreed with / votes)
and recall (human labels of that verdict the function caught / human labels of that verdict) on
fire and arm segments. Judge tags (`tags.jsonl`) are counted but not scored. `n/a` until labels
exist.

## Runs

| file | what |
| --- | --- |
| `2026-10-09-before-cosmos.{json,md}` | baseline before the Cosmos judge: live model `models/gestures.joblib`, strict profile, newest 6 closed sessions |

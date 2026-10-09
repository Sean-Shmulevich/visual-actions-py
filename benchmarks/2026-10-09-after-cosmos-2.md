# Benchmark after-cosmos-2 (2026-10-09)

git `dac24cf` · model `/Users/shmul/Library/Application Support/visual-actions/models/gestures.joblib` (sha d33ede93c2738e80) · config profile `strict` (sha 61cff0b2d0905543) · 6 session(s) · run 7.9 s

## Classifier

status: ok; held-out = newest user session per class (public / synth files train only)

| metric | after-cosmos-2 |
| --- | ---: |
| held-out accuracy | 0.6633 |
| fist recall | 0.5295 |
| accuracy fist | 0.5295 |
| accuracy h_left | 1.0 |
| accuracy h_right | 0.9175 |
| accuracy none | 0.4623 |
| accuracy open_palm | 1.0 |
| accuracy pinch | 0.9809 |
| accuracy point_up | 1.0 |
| accuracy two_up | 1.0 |
| held-out frames | 2813 |
| frames | 27385 |
| frames public | 16160 |
| frames review | 1439 |
| frames user | 9786 |
| frames fist | 3409 |
| frames h_left | 1779 |
| frames h_right | 1702 |
| frames none | 3466 |
| frames open_palm | 4382 |
| frames pinch | 2380 |
| frames point_up | 3087 |
| frames thumbs_down | 2000 |
| frames thumbs_up | 2000 |
| frames two_up | 3180 |

confusions (held-out, true -> predicted): none -> h_left 440, none -> point_up 111, fist -> thumbs_down 110, fist -> thumbs_up 86, h_right -> none 62, none -> thumbs_up 49, fist -> h_left 23, none -> two_up 19

## Engine (6 sessions: 20261009-154651, 20261009-154245, 20261009-152005, 20261009-151601, 20261009-150908, 20261009-145832)

replay = landmarks through the pipeline with the model above; live = the events.log the app wrote.

| metric | after-cosmos-2 replay | after-cosmos-2 live |
| --- | ---: | ---: |
| fires | 456 | 277 |
| weak-misfire fires | 228 | 141 |
| misfire rate | 0.5 | 0.509 |
| weak-intended fires | 96 | 65 |
| arms | 69 | 67 |
| empty arms | 34 | 24 |
| broken holds | 25 | 18 |
| broken-hold ratio | 0.266 | 0.2118 |
| drags | 27 | 52 |
| drags cancelled by fist | 6 | 7 |
| re-grab flaps | 0 | 1 |
| releases per drag | 1.0 | 1.0192 |
| hand losses | 323 | 323 |
| hand losses / min | 5.2662 | 5.2662 |
| seconds | 3680.1 | 3680.1 |
| votes fist_after_fire | 14 | 13 |
| votes token_flip_around_fire | 76 | 48 |
| votes immediate_redo | 14 | 15 |
| votes low_confidence_trigger | 171 | 94 |

### Per session (replay / live)

| session | fires | weak-misfire | arms | empty arms | broken holds | drags | losses/min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261009-154651 | 81 / 78 | 52 / 50 | 20 / 21 | 5 / 5 | 4 / 2 | 8 / 12 | 4.9106 / 4.9106 |
| 20261009-154245 | 2 / 1 | 0 / 0 | 1 / 1 | 0 / 0 | 0 / 0 | 0 / 0 | 9.0864 / 9.0864 |
| 20261009-152005 | 339 / 176 | 160 / 80 | 23 / 20 | 13 / 5 | 13 / 10 | 11 / 22 | 6.3258 / 6.3258 |
| 20261009-151601 | 0 / 5 | 0 / 3 | 7 / 7 | 3 / 2 | 8 / 6 | 4 / 12 | 5.964 / 5.964 |
| 20261009-150908 | 34 / 15 | 16 / 6 | 13 / 12 | 8 / 7 | 0 / 0 | 4 / 6 | 4.8363 / 4.8363 |
| 20261009-145832 | 0 / 2 | 0 / 2 | 5 / 6 | 5 / 5 | 0 / 0 | 0 / 0 | 3.2353 / 3.2353 |

## Labelled-moment agreement

n/a: no human labels on fire / arm segments (<session>/intent/human.jsonl); labelfn_accuracy scores human labels only (human labels: 0, tags: 1206)

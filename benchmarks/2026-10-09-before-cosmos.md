# Benchmark before-cosmos (2026-10-09)

git `632f2e3` · model `/Users/shmul/Library/Application Support/visual-actions/models/gestures.joblib` (sha d33ede93c2738e80) · config profile `strict` (sha 61cff0b2d0905543) · 6 session(s) · run 13.6 s

## Classifier

status: ok; held-out = newest user session per class (public / synth files train only)

| metric | before-cosmos |
| --- | ---: |
| held-out accuracy | 0.8721 |
| fist recall | 0.5601 |
| accuracy fist | 0.5601 |
| accuracy h_left | 0.9126 |
| accuracy h_right | 0.9257 |
| accuracy none | 0.7136 |
| accuracy open_palm | 0.9679 |
| accuracy point_up | 0.961 |
| accuracy two_up | 0.9855 |
| held-out frames | 6037 |
| frames | 25946 |
| frames public | 16160 |
| frames user | 9786 |
| frames fist | 3409 |
| frames h_left | 1768 |
| frames h_right | 1702 |
| frames none | 3466 |
| frames open_palm | 3746 |
| frames pinch | 2014 |
| frames point_up | 2910 |
| frames thumbs_down | 2000 |
| frames thumbs_up | 2000 |
| frames two_up | 2931 |

confusions (held-out, true -> predicted): none -> h_left 227, fist -> thumbs_down 110, fist -> thumbs_up 98, h_left -> none 61, h_right -> none 55, none -> thumbs_up 48, none -> point_up 40, point_up -> none 35

## Engine (6 sessions: 20261009-142733, 20261009-132737, 20261009-124833, 20261009-120535, 20261009-104823, 20261009-101425)

replay = landmarks through the pipeline with the model above; live = the events.log the app wrote.

| metric | before-cosmos replay | before-cosmos live |
| --- | ---: | ---: |
| fires | 203 | 299 |
| weak-misfire fires | 83 | 115 |
| misfire rate | 0.4089 | 0.3846 |
| weak-intended fires | 73 | 104 |
| arms | 61 | 71 |
| empty arms | 18 | 10 |
| broken holds | 78 | 66 |
| broken-hold ratio | 0.5612 | 0.4818 |
| drags | 39 | 135 |
| drags cancelled by fist | 4 | 3 |
| re-grab flaps | 2 | 1 |
| releases per drag | 1.0513 | 1.0074 |
| hand losses | 843 | 843 |
| hand losses / min | 5.9112 | 5.9112 |
| seconds | 8556.7 | 8556.7 |
| votes fist_after_fire | 17 | 27 |
| votes token_flip_around_fire | 43 | 46 |
| votes immediate_redo | 3 | 27 |
| votes low_confidence_trigger | 46 | 45 |

### Per session (replay / live)

| session | fires | weak-misfire | arms | empty arms | broken holds | drags | losses/min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261009-142733 | 26 / 57 | 5 / 29 | 11 / 12 | 5 / 6 | 10 / 16 | 4 / 10 | 5.2105 / 5.2105 |
| 20261009-132737 | 60 / 63 | 39 / 41 | 23 / 24 | 3 / 3 | 34 / 36 | 13 / 19 | 5.7992 / 5.7992 |
| 20261009-124833 | 0 / 0 | 0 / 0 | 4 / 4 | 0 / 0 | 4 / 0 | 10 / 42 | 8.8602 / 8.8602 |
| 20261009-120535 | 117 / 115 | 39 / 34 | 21 / 21 | 8 / 1 | 16 / 14 | 12 / 47 | 5.9851 / 5.9851 |
| 20261009-104823 | 0 / 15 | 0 / 3 | 0 / 3 | 0 / 0 | 3 / 0 | 0 / 2 | 10.1848 / 10.1848 |
| 20261009-101425 | 0 / 49 | 0 / 8 | 2 / 7 | 2 / 0 | 11 / 0 | 0 / 15 | 6.9372 / 6.9372 |

## Labelled-moment agreement

n/a: no human labels on fire / arm segments (<session>/intent/human.jsonl); labelfn_accuracy scores human labels only (human labels: 0, tags: 2)

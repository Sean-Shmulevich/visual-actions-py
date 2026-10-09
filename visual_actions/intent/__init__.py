"""Intent labelling: turn recorded sessions into labelled moments of intent.

Offline tooling, platform-free, nothing here runs in the live loop.

    session (video + events.log + landmarks.jsonl)
      -> events.py      typed events from events.log
      -> segments.py    candidate moments: command attempts, incidental hand, dead footage
      -> labelfns.py    weak labels from the log alone (fist after a fire, empty arms, ...)
      -> clips.py       frames / clips / skeleton strips per segment
      -> cosmos.py      pass 1: NVIDIA Cosmos Reason watches the clip (person, face, arm, hand, motion)
      -> jev.py         pass 2a: JEV answers atomic typed questions over the text state
      -> tagger.py      pass 2b: a reasoning LLM tags the moment and decides if a human is needed
      -> review.py      the human queue (one key per answer)
      -> export.py      labelled frames into datasets/ and the intent dataset for training

All passes read and write the JSONL records in schema.py, so each can be re-run, mocked
in tests, or skipped.
"""

"""The end-of-day learning job: sessions -> labels -> a per-user candidate model -> gate -> promote.

Platform-free except schedule.py (launchd). Nothing here runs in the live loop; the app only
reads the pointer the job switches (paths.live_model_path).

    job.py        run(): new sessions -> segments -> optional judge passes -> export -> train
                  -> evaluate -> promote or discard; writes learn/reports/<date>.md, last.json
    trainer.py    per-user weighted training from datasets/, versioned joblib + manifest
    evaluate.py   candidate vs champion: held-out frame accuracy, replay engine metrics, the gate
    versions.py   the models/user/current.joblib pointer: promote, rollback, prune
    schedule.py   the launchd agent (macOS)
    __main__.py   python -m visual_actions.learn {run,status,rollback,install,uninstall}
"""

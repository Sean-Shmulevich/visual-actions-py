"""Where things live on disk, platform-native via platformdirs."""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_config_dir, user_data_dir

APP = "visual-actions"
REPO_ROOT = Path(__file__).resolve().parents[1]


def config_path() -> Path:
    return Path(user_config_dir(APP)) / "config.toml"


def data_dir() -> Path:
    return Path(user_data_dir(APP))


def datasets_dir() -> Path:
    return data_dir() / "datasets"


def models_dir() -> Path:
    return data_dir() / "models"


def user_models_dir() -> Path:
    """Per-user models trained by the nightly learning job (learn/)."""
    return models_dir() / "user"


def learn_dir() -> Path:
    """State, reports and logs of the nightly learning job."""
    return data_dir() / "learn"


def live_model_path() -> Path | None:
    """The classifier the app loads: the promoted per-user model (models/user/current.joblib,
    a pointer the learning job switches atomically) when it exists, else the shipped
    models/gestures.joblib, else None (rules only)."""
    current = user_models_dir() / "current.joblib"
    if current.exists():  # follows the symlink; a dangling pointer falls through to the shipped model
        return current
    shipped = models_dir() / "gestures.joblib"
    return shipped if shipped.exists() else None


def sessions_dir() -> Path:
    return data_dir() / "sessions"


def actions_dir() -> Path:
    return data_dir() / "actions"


def face_model_path() -> Path | None:
    for candidate in (models_dir() / "blaze_face_short_range.tflite", REPO_ROOT / "models" / "blaze_face_short_range.tflite"):
        if candidate.exists():
            return candidate
    return None


def model_path() -> Path:
    """The MediaPipe landmarker bundle: user data dir first, then the repo's models/."""
    for candidate in (models_dir() / "hand_landmarker.task", REPO_ROOT / "models" / "hand_landmarker.task"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("hand_landmarker.task not found; see README")

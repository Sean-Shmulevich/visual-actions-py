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


def actions_dir() -> Path:
    return data_dir() / "actions"


def model_path() -> Path:
    """The MediaPipe landmarker bundle: user data dir first, then the repo's models/."""
    for candidate in (models_dir() / "hand_landmarker.task", REPO_ROOT / "models" / "hand_landmarker.task"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("hand_landmarker.task not found; see README")

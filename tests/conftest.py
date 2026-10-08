"""Most tests were written against the FAST leader profile and feed fixtures with ~1.5 s
palms; the app's default is STRICT. Tests that want STRICT call apply_profile(cfg, STRICT)
through their own imported name, which this fixture does not touch."""

import pytest

import visual_actions.core.config as config_mod


@pytest.fixture(autouse=True)
def _fast_profile_by_default(monkeypatch):
    real = config_mod.apply_profile

    def fast_unless_explicit(cfg, profile):
        return real(cfg, config_mod.FAST if profile == config_mod.STRICT else profile)

    monkeypatch.setattr(config_mod, "apply_profile", fast_unless_explicit)

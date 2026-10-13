"""The managed-storage paths a node reports when none were passed in."""

from __future__ import annotations

import pytest

from tensorstead.agent.resources import default_managed_paths

pytestmark = pytest.mark.unit


def test_the_installed_store_variable_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """The install writes TENSORSTEAD_MODEL_STORE_PATH; that is the store."""
    monkeypatch.delenv("TENSORSTEAD_MODEL_STORE", raising=False)
    monkeypatch.setenv("TENSORSTEAD_MODEL_STORE_PATH", "/srv/models")
    assert default_managed_paths()[0] == "/srv/models"


def test_the_shorter_name_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TENSORSTEAD_MODEL_STORE_PATH", raising=False)
    monkeypatch.setenv("TENSORSTEAD_MODEL_STORE", "/legacy/models")
    assert default_managed_paths()[0] == "/legacy/models"


def test_unset_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TENSORSTEAD_MODEL_STORE_PATH", raising=False)
    monkeypatch.delenv("TENSORSTEAD_MODEL_STORE", raising=False)
    assert default_managed_paths()[0] == "/var/lib/tensorstead/models"

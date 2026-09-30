"""Contract tests for the production agent application factory."""

from __future__ import annotations

from pathlib import Path

import pytest

from tensorstead.agent.app import build_agent_app_from_env

pytestmark = pytest.mark.contract


def test_environment_factory_requires_both_role_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """A systemd agent must not silently expose an unauthenticated API."""
    monkeypatch.delenv("TENSORSTEAD_MGMT_TOKEN", raising=False)
    monkeypatch.delenv("TENSORSTEAD_REPLICATION_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="TENSORSTEAD_MGMT_TOKEN"):
        build_agent_app_from_env()


def test_environment_factory_rejects_reused_role_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replication credentials must not inherit the management role."""
    monkeypatch.setenv("TENSORSTEAD_MGMT_TOKEN", "management")
    monkeypatch.setenv("TENSORSTEAD_REPLICATION_TOKEN", "management")

    with pytest.raises(RuntimeError, match="must differ"):
        build_agent_app_from_env()


def test_environment_factory_reads_managed_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The system service can set explicit node-local paths without argv secrets."""
    model_store = tmp_path / "models"
    state_dir = tmp_path / "state"
    image_store = tmp_path / "images"
    monkeypatch.setenv("TENSORSTEAD_MGMT_TOKEN", "management")
    monkeypatch.setenv("TENSORSTEAD_REPLICATION_TOKEN", "replication")
    monkeypatch.setenv("TENSORSTEAD_MODEL_STORE_PATH", str(model_store))
    monkeypatch.setenv("TENSORSTEAD_STATE_DIR", str(state_dir))
    monkeypatch.setenv("TENSORSTEAD_IMAGE_STORE_PATH", str(image_store))

    app = build_agent_app_from_env()

    assert app.state.store_dir == model_store
    assert app.state.marker_dir == state_dir
    assert app.state.model_store_path == str(model_store)
    assert app.state.image_store_path == str(image_store)

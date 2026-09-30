"""Contract coverage for the coordinator's explicit persistent-store path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import tensorstead.coordinator.app as coordinator_app_module
from tensorstead.adapters.credentials.local_file import LocalFileCredentialProvider
from tensorstead.cli import coordinator_cmds
from tensorstead.coordinator.app import build_coordinator_app
from tests.fakes.node_agent import FakeNodeAgent
from tests.helpers import FakeNodeClient, make_node

pytestmark = pytest.mark.contract


def _build_with_store(store: Path, credential_root: Path) -> Any:
    return build_coordinator_app(
        management_token="test",
        store=store,
        node_client=FakeNodeClient(FakeNodeAgent()),
        credential_provider=LocalFileCredentialProvider(root=credential_root),
    )


def test_explicit_store_survives_coordinator_rebuild(tmp_path: Path) -> None:
    """The store selected for serving remains authoritative after restart."""
    store = tmp_path / "tensorstead.db"
    node = make_node()

    first_app = _build_with_store(store, tmp_path / "credentials-first")
    first_app.state.repository.save_node(node)

    second_app = _build_with_store(store, tmp_path / "credentials-second")
    restored = second_app.state.repository.get_node(node.id)

    assert restored is not None
    assert restored.name == node.name


def test_coordinator_serve_passes_store_to_app_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI's ``--store`` option must reach coordinator app construction."""
    selected_store = tmp_path / "selected.db"
    selected_store.touch()
    sentinel_app = object()
    captured: dict[str, object] = {}

    def fake_build_coordinator_app(*, store: Path) -> object:
        captured["store"] = store
        return sentinel_app

    def fake_uvicorn_run(app: object, *, host: str, port: int) -> None:
        captured.update(app=app, host=host, port=port)

    monkeypatch.setattr(
        coordinator_app_module,
        "build_coordinator_app",
        fake_build_coordinator_app,
    )
    monkeypatch.setattr("uvicorn.run", fake_uvicorn_run)
    # Unrelated to what this test checks, but serve() now refuses to start
    # with no TENSORSTEAD_MGMT_TOKEN and no --insecure-dev-mode regardless of
    # host.
    monkeypatch.setenv("TENSORSTEAD_MGMT_TOKEN", "test-token")

    coordinator_cmds.serve(host="127.0.0.2", port=9080, store=selected_store)

    assert captured == {
        "store": selected_store,
        "app": sentinel_app,
        "host": "127.0.0.2",
        "port": 9080,
    }

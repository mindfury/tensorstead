"""Supported first-run path: a local, open coordinator."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from tensorstead.cli import config
from tensorstead.cli.main import app

pytestmark = pytest.mark.unit

runner = CliRunner()


def test_login_can_save_an_open_loopback_coordinator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A coordinator without a configured token is valid on loopback."""
    config_file = tmp_path / "config.yml"
    monkeypatch.setenv("TENSORSTEAD_CONFIG", str(config_file))

    def fake_get(self: httpx.Client, url: str, *, headers: dict[str, str]) -> httpx.Response:
        assert url == "/v1/nodes"
        assert headers == {}
        return httpx.Response(200, json=[])

    monkeypatch.setattr(httpx.Client, "get", fake_get)

    result = runner.invoke(
        app,
        ["login", "--api", "http://127.0.0.1:8080", "--token-file", ""],
        input="\n",
    )

    assert result.exit_code == 0, result.output
    assert config.load_config_file() == {"coordinator": "http://127.0.0.1:8080"}


def test_status_checks_a_configured_open_coordinator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Status must not mistake missing credentials for a broken open coordinator."""
    config_file = tmp_path / "config.yml"
    config_file.write_text("coordinator: http://127.0.0.1:8080\n", encoding="utf-8")
    monkeypatch.setenv("TENSORSTEAD_CONFIG", str(config_file))

    def fake_get(self: httpx.Client, url: str, *, headers: dict[str, str]) -> httpx.Response:
        assert headers == {}
        if url == "/v1/version":
            return httpx.Response(200, json={"version": "0.1.8", "contract_version": "1"})
        assert url == "/v1/nodes"
        return httpx.Response(200, json=[])

    monkeypatch.setattr(httpx.Client, "get", fake_get)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    assert "reachable          yes" in result.output

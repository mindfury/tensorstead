"""CLI ``deployment modify --config`` vs ``--replace-config``.

``--config`` merges and sends ``replace_config: false``; ``--replace-config``
replaces the whole map and sends ``replace_config: true``. The two are mutually
exclusive. The recorded config is printed back.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from tensorstead.cli import commands as cli_commands
from tensorstead.cli.main import app as cli_app

pytestmark = pytest.mark.unit

_DEPLOYMENTS = [
    {"declared": {"id": "dep-123", "name": "qwen36-27b"}},
]


class _MockClient:
    """Records the last PATCH payload; serves the deployment list for lookup."""

    def __init__(self) -> None:
        self.patched: dict[str, Any] | None = None

    def request(
        self, method: str, path: str, *, json: Any = None, headers: Any = None
    ) -> httpx.Response:
        if method == "GET" and path == "/v1/deployments":
            return httpx.Response(200, json=_DEPLOYMENTS)
        if method == "PATCH" and path.startswith("/v1/deployments/"):
            self.patched = json
            # Echo the recorded config back so the human renderer prints it.
            recorded = dict((json or {}).get("runtime_config") or {})
            return httpx.Response(
                200,
                json={
                    "revision": 2,
                    "restart_required": False,
                    "applied": False,
                    "runtime_config": recorded,
                    "warnings": [],
                },
            )
        return httpx.Response(404, json={"code": "not_found", "message": path})


@pytest.fixture
def mock_client(monkeypatch: pytest.MonkeyPatch) -> _MockClient:
    client = _MockClient()
    monkeypatch.setattr(cli_commands, "_client", lambda **_kwargs: client)
    return client


def test_config_flag_merges_and_sends_replace_config_false(mock_client: _MockClient) -> None:
    result = CliRunner().invoke(
        cli_app,
        ["deployment", "modify", "qwen36-27b", "--config", "tool_call_parser=qwen3_xml"],
    )
    assert result.exit_code == 0, result.output
    assert mock_client.patched is not None
    assert mock_client.patched["runtime_config"] == {"tool_call_parser": "qwen3_xml"}
    assert mock_client.patched["replace_config"] is False


def test_replace_config_flag_sends_replace_config_true(mock_client: _MockClient) -> None:
    result = CliRunner().invoke(
        cli_app,
        ["deployment", "modify", "qwen36-27b", "--replace-config", "tool_call_parser=qwen3_json"],
    )
    assert result.exit_code == 0, result.output
    assert mock_client.patched is not None
    assert mock_client.patched["runtime_config"] == {"tool_call_parser": "qwen3_json"}
    assert mock_client.patched["replace_config"] is True


def test_config_and_replace_config_are_mutually_exclusive(mock_client: _MockClient) -> None:
    result = CliRunner().invoke(
        cli_app,
        [
            "deployment",
            "modify",
            "qwen36-27b",
            "--config",
            "tool_call_parser=qwen3_xml",
            "--replace-config",
            "tool_call_parser=qwen3_json",
        ],
    )
    # Refused before any coordinator call: the destructive reading must not be
    # reachable by accident, and a conflicting pair is not a valid instruction.
    assert result.exit_code == 2, result.output
    assert "--config merges and --replace-config replaces" in result.output
    assert mock_client.patched is None


def test_modify_prints_the_recorded_config(mock_client: _MockClient) -> None:
    """A drop is visible at the moment it happens, not only at the next show."""
    result = CliRunner().invoke(
        cli_app,
        ["deployment", "modify", "qwen36-27b", "--config", "gpu_memory_utilization=0.5"],
    )
    assert result.exit_code == 0, result.output
    assert "runtime config" in result.output
    assert "gpu_memory_utilization = 0.5" in result.output

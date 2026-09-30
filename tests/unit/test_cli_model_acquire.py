"""CLI ``model acquire`` polls its operation to terminal.

``model_acquire`` is the first truly async CLI command. The coordinator returns
202 immediately and runs the download on a background thread, so the CLI must poll
``GET /v1/operations/{id}`` and emit the terminal record — the operation on
success, the structured failure reason on failure — instead of blocking on the
POST response (which used to exceed the MCP client's 60s read timeout and
surface as ``coordinator_unreachable``).
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from tensorstead.cli import commands as cli_commands
from tensorstead.cli.main import app as cli_app

pytestmark = pytest.mark.unit

_NODES = [{"id": "node-a", "name": "spark-01"}]


class _AcquireClient:
    """Serves node lookup, the 202 acquire, and a poll sequence to terminal."""

    def __init__(self, *, terminal: dict[str, Any], running_first: bool = True) -> None:
        self.terminal = terminal
        self._running_first = running_first
        self.posted: dict[str, Any] | None = None
        self.polls = 0

    def request(
        self, method: str, path: str, *, json: Any = None, headers: Any = None
    ) -> httpx.Response:
        # Node-name resolution done by the CLI before the POST.
        if method == "GET" and path == "/v1/nodes":
            return httpx.Response(200, json=_NODES)
        return httpx.Response(404, json={"code": "not_found", "message": path})

    def post(self, path: str, *, json: Any = None, headers: Any = None) -> httpx.Response:
        self.posted = json
        return httpx.Response(202, json={"operation_id": "op-123"})

    def get(self, path: str, *, headers: Any = None) -> httpx.Response:
        if path == "/v1/operations/op-123":
            self.polls += 1
            # The GIL schedules the test thread before the background worker, so the first
            # poll is not guaranteed terminal — exercise the loop.
            if self._running_first and self.polls == 1:
                return httpx.Response(200, json={"id": "op-123", "state": "running"})
            return httpx.Response(200, json=self.terminal)
        return httpx.Response(404, json={"code": "not_found", "message": path})


def _invoke(client: _AcquireClient, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(cli_commands, "_client", lambda **_kwargs: client)
    return CliRunner().invoke(
        cli_app,
        [
            "model",
            "acquire",
            "--source",
            "huggingface",
            "--id",
            "org/model",
            "--node",
            "spark-01",
            "--revision",
            "e1f2a3b",
        ],
    )


def test_model_acquire_polls_and_emits_the_terminal_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI accepts the 202, polls past ``running`` to ``succeeded``, prints it."""
    terminal = {
        "id": "op-123",
        "kind": "model_acquire",
        "state": "succeeded",
        "per_node_outcomes": {"node-a": {"state": "succeeded"}},
    }
    client = _AcquireClient(terminal=terminal, running_first=True)
    result = _invoke(client, monkeypatch)

    assert result.exit_code == 0, result.output
    # The accept line appears immediately, before the poll result.
    assert "Operation accepted: op-123" in result.output
    # The CLI polled at least once past the non-terminal "running" state.
    assert client.polls >= 2
    # The resolved node id (not the name) was sent on the acquire POST.
    assert client.posted == {
        "source_id": "huggingface",
        "source_model_id": "org/model",
        "revision": "e1f2a3b",
        "nodes": ["node-a"],
    }


def test_model_acquire_renders_a_failed_operation_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal ``failed`` operation is rendered as its failure reason."""
    terminal = {
        "id": "op-123",
        "kind": "model_acquire",
        "state": "failed",
        "failure_reason": {
            "code": "authorization_refused",
            "message": "huggingface: access to org/model is gated",
            "detail": {"source_id": "huggingface"},
        },
    }
    client = _AcquireClient(terminal=terminal, running_first=False)
    result = _invoke(client, monkeypatch)

    assert result.exit_code != 0, result.output
    assert "authorization_refused" in result.output
    assert "access to org/model is gated" in result.output

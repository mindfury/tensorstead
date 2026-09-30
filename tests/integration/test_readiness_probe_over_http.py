"""The readiness probe against a real HTTP server.

`tests/unit/test_readiness_probe.py` covers the branch table with httpx
monkeypatched. That proves the decisions are right; it does not prove the probe
can talk to anything. These tests run it against a real socket serving real
responses, because the defect being fixed was precisely a check that never
performed the I/O its name claimed.

No Docker and no GPU: the server is `http.server` in a thread, which is enough
to stand in for a runtime's OpenAI-compatible surface.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tensorstead.agent.readiness import ReadinessProbe, probe_endpoint

pytestmark = pytest.mark.integration

_PROBE = ReadinessProbe(path="/v1/models", expect_status=200)


class _RuntimeHandler(BaseHTTPRequestHandler):
    """Stands in for a runtime's model-listing endpoint."""

    # Set per-server below.
    status_for_models = 200
    require_auth = False
    seen_authorization: str | None = None

    def do_GET(self) -> None:  # BaseHTTPRequestHandler's spelling, not ours
        type(self).seen_authorization = self.headers.get("Authorization")

        if self.path != "/v1/models":
            self.send_response(404)
            self.end_headers()
            return

        if self.require_auth and not self.headers.get("Authorization"):
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b"{}")
            return

        self.send_response(self.status_for_models)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"object":"list","data":[{"id":"qwen36-27b"}]}')

    def log_message(self, *args: object) -> None:
        """Silence the default stderr access log."""


def _serve(**attrs: object) -> Iterator[tuple[int, type[_RuntimeHandler]]]:
    """Run a one-off HTTP server on an ephemeral port.

    Yields the port and the handler class itself — the class is where the
    request is recorded, and each server gets its own subclass so concurrent
    fixtures cannot overwrite each other's observations.
    """
    handler = type("_Handler", (_RuntimeHandler,), dict(attrs))
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1]), handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def serving_port() -> Iterator[int]:
    for port, _handler in _serve(status_for_models=200):
        yield port


@pytest.fixture
def loading_port() -> Iterator[int]:
    """A runtime that is up but has not finished loading its model."""
    for port, _handler in _serve(status_for_models=503):
        yield port


@pytest.fixture
def secured() -> Iterator[tuple[int, type[_RuntimeHandler]]]:
    yield from _serve(status_for_models=200, require_auth=True)


def test_a_serving_runtime_is_reported_ready(serving_port: int) -> None:
    """The happy path, over a real connection rather than a stubbed one."""
    result = probe_endpoint(f"0.0.0.0:{serving_port}", probe=_PROBE)

    assert result.reachable is True
    assert result.inference_ready is True
    assert result.detail is None


def test_a_loading_runtime_is_reachable_but_not_ready(loading_port: int) -> None:
    """The window the whole feature exists for: listening, not yet serving."""
    result = probe_endpoint(f"0.0.0.0:{loading_port}", probe=_PROBE)

    assert result.reachable is True
    assert result.inference_ready is False
    assert "503" in (result.detail or "")


def test_a_secured_runtime_without_a_credential_is_unknown(
    secured: tuple[int, type[_RuntimeHandler]],
) -> None:
    """401 means we cannot see, not that the runtime is broken."""
    port, _handler = secured

    result = probe_endpoint(f"0.0.0.0:{port}", probe=_PROBE)

    assert result.reachable is True
    assert result.inference_ready is None
    assert "credential" in (result.detail or "")


def test_a_secured_runtime_with_the_credential_is_ready(
    secured: tuple[int, type[_RuntimeHandler]],
) -> None:
    """The credential actually travels, and the runtime actually accepts it."""
    port, handler = secured

    result = probe_endpoint(f"0.0.0.0:{port}", probe=_PROBE, api_key="sk-test-value")

    assert result.inference_ready is True
    assert handler.seen_authorization == "Bearer sk-test-value"


def test_a_closed_port_is_unreachable() -> None:
    """Nothing listening: measured, not assumed."""
    for port, _handler in _serve(status_for_models=200):
        closed_port = port  # the server shuts down when this loop exits

    result = probe_endpoint(f"0.0.0.0:{closed_port}", probe=_PROBE)

    assert result.reachable is False
    assert result.inference_ready is False

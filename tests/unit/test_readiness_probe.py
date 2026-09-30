"""Endpoint readiness probing.

The branch table here is the whole point of the feature, so it is tested
directly rather than only through the agent route. What separates this from the
implementation it replaced is that every outcome is one of *three* values: a
probe that cannot establish a fact must say so rather than picking a side.

Reachability is exercised against a real loopback socket. Nothing is mocked at
the transport layer, because "does a connect succeed" was precisely the question
the old implementation never asked.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest

from tensorstead.agent import readiness
from tensorstead.agent.readiness import EndpointReadiness, ReadinessProbe, probe_endpoint

pytestmark = pytest.mark.unit

_PROBE = ReadinessProbe(path="/v1/models", expect_status=200)


@pytest.fixture
def listening_port() -> Iterator[int]:
    """A real bound-and-listening loopback port, released after the test."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        yield int(sock.getsockname()[1])


@pytest.fixture
def closed_port() -> int:
    """A port nothing is listening on: bound to learn the number, then released."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    return port


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def _respond(monkeypatch: pytest.MonkeyPatch, status_code: int) -> dict[str, Any]:
    """Make the runtime answer with ``status_code``; capture the request."""
    import httpx

    captured: dict[str, Any] = {}

    def fake_get(url: str, **kwargs: Any) -> _Response:
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        captured["timeout"] = kwargs.get("timeout")
        return _Response(status_code)

    monkeypatch.setattr(httpx, "get", fake_get)
    return captured


# ------------------------------------------------------------------ transport


def test_nothing_listening_is_not_ready(closed_port: int) -> None:
    """A closed port is unambiguous: unreachable, and therefore not serving."""
    result = probe_endpoint(f"0.0.0.0:{closed_port}", probe=_PROBE)

    assert result.reachable is False
    assert result.inference_ready is False
    assert str(closed_port) in (result.detail or "")


def test_missing_endpoint_reports_unknown_not_broken() -> None:
    """A deployment with no recorded endpoint has not been shown to be broken."""
    result = probe_endpoint(None, probe=_PROBE)

    assert result == EndpointReadiness(
        reachable=None,
        inference_ready=None,
        detail="no endpoint recorded for this deployment",
    )


def test_malformed_endpoint_reports_unknown() -> None:
    result = probe_endpoint("not-a-host-port", probe=_PROBE)

    assert result.reachable is None
    assert result.inference_ready is None
    assert "host:port" in (result.detail or "")


def test_bind_address_endpoint_is_probed_on_loopback(
    listening_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``0.0.0.0`` is a bind address, not a destination — probe 127.0.0.1.

    Probing the declared host verbatim would attempt to connect to 0.0.0.0,
    which is not a thing you can dial.
    """
    captured = _respond(monkeypatch, 200)

    result = probe_endpoint(f"0.0.0.0:{listening_port}", probe=_PROBE)

    assert result.inference_ready is True
    assert captured["url"] == f"http://127.0.0.1:{listening_port}/v1/models"


# ------------------------------------------------------------------ inference


def test_serving_runtime_is_ready(listening_port: int, monkeypatch: pytest.MonkeyPatch) -> None:
    _respond(monkeypatch, 200)

    result = probe_endpoint(f"127.0.0.1:{listening_port}", probe=_PROBE)

    assert result.reachable is True
    assert result.inference_ready is True
    assert result.detail is None


def test_listening_but_not_serving_is_reachable_and_not_ready(
    listening_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The distinction the incident needed: the socket is up, the model is not."""
    _respond(monkeypatch, 503)

    result = probe_endpoint(f"127.0.0.1:{listening_port}", probe=_PROBE)

    assert result.reachable is True
    assert result.inference_ready is False
    assert "503" in (result.detail or "")


def test_connection_reset_after_accept_is_not_ready(
    listening_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exactly what the incident showed: the connect succeeds, the request dies."""
    import httpx

    def fake_get(url: str, **kwargs: Any) -> _Response:
        raise httpx.ReadError("connection reset by peer")

    monkeypatch.setattr(httpx, "get", fake_get)

    result = probe_endpoint(f"127.0.0.1:{listening_port}", probe=_PROBE)

    assert result.reachable is True
    assert result.inference_ready is False
    assert "ReadError" in (result.detail or "")


@pytest.mark.parametrize("status_code", [401, 403])
def test_unauthorized_is_unknown_not_unhealthy(
    status_code: int, listening_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """We cannot see past the runtime's own auth, so we do not claim to.

    Reporting this as ``false`` would mark every correctly-secured deployment
    broken whenever the agent lacks the credential — a false alarm that would
    teach operators to ignore the field.
    """
    _respond(monkeypatch, status_code)

    result = probe_endpoint(f"127.0.0.1:{listening_port}", probe=_PROBE)

    assert result.reachable is True
    assert result.inference_ready is None
    assert "credential" in (result.detail or "")


def test_credential_is_sent_as_a_bearer_token(
    listening_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _respond(monkeypatch, 200)

    probe_endpoint(f"127.0.0.1:{listening_port}", probe=_PROBE, api_key="sk-secret")

    assert captured["headers"] == {"Authorization": "Bearer sk-secret"}


def test_no_declared_probe_reports_unknown_readiness(listening_port: int) -> None:
    """A runtime we cannot ask is not thereby broken."""
    result = probe_endpoint(f"127.0.0.1:{listening_port}", probe=None)

    assert result.reachable is True
    assert result.inference_ready is None
    assert "no readiness probe" in (result.detail or "")


def test_probe_is_bounded_in_time(listening_port: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """A hung runtime must not hang the coordinator's observation request."""
    captured = _respond(monkeypatch, 200)

    probe_endpoint(f"127.0.0.1:{listening_port}", probe=_PROBE)

    assert captured["timeout"] == readiness._HTTP_TIMEOUT_SECONDS
    assert readiness._HTTP_TIMEOUT_SECONDS <= 5.0

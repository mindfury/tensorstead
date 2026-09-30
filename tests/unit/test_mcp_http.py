from __future__ import annotations

import asyncio

import pytest

from tensorstead.mcp.server import _BearerAuth, _http_settings


def _request(token: str | None) -> list[dict[object, object]]:
    sent: list[dict[object, object]] = []

    async def app(_scope: object, _receive: object, send: object) -> None:
        await send({"type": "http.response.start", "status": 204})  # type: ignore[operator]

    scope = {
        "type": "http",
        "headers": [] if token is None else [(b"authorization", token.encode())],
    }

    async def send(message: dict[object, object]) -> None:
        sent.append(message)

    asyncio.run(_BearerAuth(app, "secret")(scope, None, send))
    return sent


def test_streamable_http_requires_bearer_token() -> None:
    assert _request(None)[0]["status"] == 401
    assert _request("Bearer wrong")[0]["status"] == 401
    assert _request("Bearer secret")[0]["status"] == 204


def test_streamable_http_requires_tls_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TENSORSTEAD_MCP_TLS_CERT", raising=False)
    monkeypatch.delenv("TENSORSTEAD_MCP_TLS_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TLS_CERT"):
        _http_settings()


def test_streamable_http_settings_are_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TENSORSTEAD_MCP_PORT", "9443")
    monkeypatch.setenv("TENSORSTEAD_MCP_PATH", "/tensorstead/mcp")
    monkeypatch.setenv("TENSORSTEAD_MCP_TLS_CERT", "/etc/tensorstead/mcp.crt")
    monkeypatch.setenv("TENSORSTEAD_MCP_TLS_KEY", "/etc/tensorstead/mcp.key")
    assert _http_settings() == (
        "127.0.0.1",
        9443,
        "/tensorstead/mcp",
        "/etc/tensorstead/mcp.crt",
        "/etc/tensorstead/mcp.key",
    )

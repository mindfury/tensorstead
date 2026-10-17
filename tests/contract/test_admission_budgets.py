"""Request bodies, nesting and background work are bounded and refused explicitly.

CodeGuard CG-15: no body limit, nesting limit or bound on queued background work
was configured anywhere, so oversized requests or a burst of expensive
operations could consume the management plane.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tensorstead.contracts.admission import BodyLimitMiddleware, json_depth_exceeds
from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer test", "Content-Type": "application/json"}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("TENSORSTEAD_MAX_REQUEST_BYTES", "4096")
    monkeypatch.setenv("TENSORSTEAD_MAX_BACKGROUND_OPERATIONS", "2")
    app, _ = build_test_coordinator()
    return TestClient(app, raise_server_exceptions=False)


def test_a_declared_oversized_body_is_refused(client: TestClient) -> None:
    resp = client.put("/v1/buildspecs/s", content=b"{" + b" " * 5000 + b"}", headers=_AUTH)
    assert resp.status_code == 413
    assert resp.json()["code"] == "request_too_large"


def test_a_chunked_body_is_refused_once_it_passes_the_limit(client: TestClient) -> None:
    def chunks() -> Iterator[bytes]:
        yield b'{"base_image": "'
        for _ in range(100):
            yield b"x" * 1024
        yield b'"}'

    resp = client.put("/v1/buildspecs/s", content=chunks(), headers=_AUTH)
    assert resp.status_code == 413


def test_deep_nesting_is_a_400_not_a_500(client: TestClient) -> None:
    body = b"[" * 2000 + b"]" * 2000
    resp = client.put("/v1/buildspecs/s", content=body, headers=_AUTH)
    assert resp.status_code == 400


def test_depth_scan_ignores_brackets_inside_strings() -> None:
    assert not json_depth_exceeds(b'{"a": "' + b"[" * 500 + b'"}', limit=4)
    assert json_depth_exceeds(b"[[[[[[]]]]]]", limit=4)


def _node(client: TestClient, name: str, ip: str) -> str:
    resp = client.post(
        "/v1/nodes", json={"name": name, "agent_endpoint": f"https://{ip}:8443"}, headers=_AUTH
    )
    return str(resp.json()["id"])


def test_background_work_beyond_its_slots_is_refused_with_429(client: TestClient) -> None:
    node = _node(client, "spark-01", "10.0.0.11")
    admission = client.app.state.work_admission  # type: ignore[attr-defined]
    admission.admit("held 1")
    admission.admit("held 2")

    resp = client.post(
        "/v1/images:pull", json={"reference": "repo/x:1", "nodes": [node]}, headers=_AUTH
    )
    assert resp.status_code == 429
    assert resp.json()["code"] == "busy"
    # Control and health stay available while background work is saturated.
    assert client.get("/health").status_code == 200
    assert client.get("/v1/nodes", headers=_AUTH).status_code == 200

    admission.release()
    admission.release()


def test_a_finished_operation_gives_its_slot_back(client: TestClient) -> None:
    node = _node(client, "spark-01", "10.0.0.11")
    for i in range(5):  # more than the 2 slots, sequentially
        resp = client.post(
            "/v1/images:pull", json={"reference": f"repo/x:{i}", "nodes": [node]}, headers=_AUTH
        )
        assert resp.status_code == 202, resp.text
        poll_operation(client, resp.json()["operation_id"])


def test_the_mcp_wrapper_refuses_auth_before_reading_any_body() -> None:
    from tensorstead.mcp.server import _BearerAuth

    read: list[bool] = []

    async def receive() -> dict[str, Any]:
        read.append(True)
        return {"type": "http.request", "body": b"x" * 10, "more_body": False}

    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def inner(_s: Any, _r: Any, _w: Any) -> None:
        pass

    app = _BearerAuth(BodyLimitMiddleware(inner, max_bytes=4), "tok")
    asyncio.run(app({"type": "http", "headers": []}, receive, send))
    assert sent[0]["status"] == 401
    assert read == [], "the body was read before authentication"

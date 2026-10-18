"""Malformed Authorization headers are a plain 401 on every server, never an exception.

CodeGuard CG-19: ``hmac.compare_digest`` on ``str`` raises ``TypeError`` for
non-ASCII input, so ``Authorization: Bearer \\xff`` raised instead of denying.
Duplicate headers were resolved by whichever copy the parser kept.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tensorstead.agent.app import build_agent_app
from tensorstead.domain.tokens import tokens_match
from tensorstead.mcp.server import _BearerAuth
from tests.fakes.container_engine import FakeContainerEngine
from tests.fakes.service_manager import FakeServiceManager
from tests.helpers import build_test_coordinator

pytestmark = pytest.mark.contract

_BAD: list[list[tuple[bytes, bytes]]] = [
    [],  # absent
    [(b"authorization", b"")],
    [(b"authorization", b"Bearer")],
    [(b"authorization", b"Basic dGVzdA==")],
    [(b"authorization", b"Bearer \xff")],  # the reviewer's probe
    [(b"authorization", b"Bearer t\xe9st")],
]


@pytest.fixture
def coordinator() -> TestClient:
    app, _ = build_test_coordinator()
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def agent(tmp_path: Path) -> TestClient:
    app = build_agent_app(
        management_token="mgmt",
        container_engine=FakeContainerEngine(),
        service_manager=FakeServiceManager(),
        store_dir=tmp_path / "store",
        marker_dir=tmp_path / "markers",
    )
    return TestClient(app, raise_server_exceptions=True)


@pytest.mark.parametrize("headers", _BAD)
def test_the_coordinator_denies_without_raising(
    coordinator: TestClient, headers: list[tuple[bytes, bytes]]
) -> None:
    assert coordinator.get("/v1/nodes", headers=headers).status_code == 401


@pytest.mark.parametrize("headers", _BAD)
def test_the_agent_denies_without_raising(
    agent: TestClient, headers: list[tuple[bytes, bytes]]
) -> None:
    assert agent.get("/agent/v1/images", headers=headers).status_code == 401


def test_duplicate_headers_are_refused_even_if_one_is_right(
    coordinator: TestClient, agent: TestClient
) -> None:
    dup = [(b"authorization", b"Bearer test"), (b"authorization", b"Bearer wrong")]
    assert coordinator.get("/v1/nodes", headers=dup).status_code == 401
    dup_agent = [(b"authorization", b"Bearer mgmt"), (b"authorization", b"Bearer mgmt")]
    assert agent.get("/agent/v1/images", headers=dup_agent).status_code == 401


def test_the_right_token_still_authenticates(coordinator: TestClient, agent: TestClient) -> None:
    assert coordinator.get("/v1/nodes", headers={"Authorization": "Bearer test"}).status_code == 200
    assert (
        agent.get("/agent/v1/images", headers={"Authorization": "Bearer mgmt"}).status_code == 200
    )


def _mcp_status(headers: list[tuple[bytes, bytes]]) -> int:
    sent: list[dict[str, Any]] = []
    reached: list[bool] = []

    async def inner(_scope: Any, _receive: Any, _send: Any) -> None:
        reached.append(True)

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def receive() -> dict[str, Any]:
        return {"type": "http.request"}

    auth = _BearerAuth(inner, "mcp-token")
    asyncio.run(auth({"type": "http", "headers": headers}, receive, send))
    return 200 if reached else int(sent[0]["status"])


@pytest.mark.parametrize(
    "headers",
    [
        [],
        [(b"authorization", b"Bearer \xff")],
        [(b"authorization", b"Bearer mcp-token"), (b"authorization", b"Bearer mcp-token")],
    ],
)
def test_mcp_denies_without_raising(headers: list[tuple[bytes, bytes]]) -> None:
    assert _mcp_status(headers) == 401


def test_mcp_right_token_passes() -> None:
    assert _mcp_status([(b"authorization", b"Bearer mcp-token")]) == 200


@pytest.mark.parametrize("supplied", [None, "", "\xff", "tést", "中", b"\xff\xfe"])
def test_tokens_match_is_total(supplied: Any) -> None:
    assert tokens_match(supplied, "secret") is False

"""No API documentation or schema is served without the management token.

CodeGuard CG-18: FastAPI's default /docs, /redoc and /openapi.json sat outside
every route dependency, so an unauthenticated request learned the release
(``1.6.0``) and the whole API surface -- though ``/v1/version`` itself is
deliberately behind auth.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tensorstead.agent.app import build_agent_app
from tensorstead.version import VERSION
from tests.fakes.container_engine import FakeContainerEngine
from tests.fakes.service_manager import FakeServiceManager
from tests.helpers import build_test_coordinator

pytestmark = pytest.mark.contract

_PUBLIC = ["/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"]


def test_the_coordinator_serves_no_public_docs() -> None:
    app, _ = build_test_coordinator()
    client = TestClient(app)
    for path in _PUBLIC:
        resp = client.get(path)
        assert resp.status_code in (401, 404), path
        assert VERSION not in resp.text, path


def test_the_schema_is_available_with_the_token() -> None:
    app, _ = build_test_coordinator()
    client = TestClient(app)
    assert client.get("/v1/openapi.json").status_code == 401
    resp = client.get("/v1/openapi.json", headers={"Authorization": "Bearer test"})
    assert resp.status_code == 200
    assert "/v1/deployments" in resp.json()["paths"]


def test_the_agent_serves_no_docs(tmp_path: Path) -> None:
    app = build_agent_app(
        management_token="mgmt",
        container_engine=FakeContainerEngine(),
        service_manager=FakeServiceManager(),
        store_dir=tmp_path / "store",
        marker_dir=tmp_path / "markers",
    )
    client = TestClient(app)
    for path in _PUBLIC:
        assert client.get(path).status_code == 404, path


def test_health_stays_minimal_and_public() -> None:
    app, _ = build_test_coordinator()
    resp = TestClient(app).get("/health")
    assert resp.status_code == 200
    assert VERSION not in resp.text

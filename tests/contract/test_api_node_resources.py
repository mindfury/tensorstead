"""Contract test node resources.

``GET /v1/nodes/{id}/resources`` returns accelerator, memory, and
managed-storage figures on demand. ``memory_is_unified`` is reported rather
than a discrete-VRAM model being assumed. ``status`` may be
``unknown`` or ``unreachable``.

Reported for operator judgment only — these values never gate an operation.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from tests.fakes.node_agent import FakeNodeAgent
from tests.helpers import build_test_coordinator

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer test"}


def _register_node(client: TestClient, name: str = "spark-01") -> str:
    resp = client.post(
        "/v1/nodes",
        json={"name": name, "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    assert resp.status_code == 201
    return str(resp.json()["id"])


def test_resources_returns_accelerator_and_memory() -> None:
    """Accelerator and memory figures are returned."""
    agent = FakeNodeAgent()
    agent.set_resources(
        {
            "status": "ok",
            "observed_at": datetime.now().astimezone().isoformat(),
            "accelerator_utilization_pct": 63.0,
            "accelerator_memory_used": 41231686042,
            "accelerator_memory_total": 137438953472,
            "memory_is_unified": True,
            "storage": [],
        }
    )
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    node_id = _register_node(client)

    resp = client.get(f"/v1/nodes/{node_id}/resources", headers=_AUTH)
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "ok"
    assert "observed_at" in body
    assert body["accelerator_utilization_pct"] == 63.0
    assert body["accelerator_memory_used"] == 41231686042
    assert body["accelerator_memory_total"] == 137438953472


def test_memory_is_unified_reported_not_assumed() -> None:
    """memory_is_unified is reported, never a discrete-VRAM assumption."""
    agent = FakeNodeAgent()
    agent.set_resources(
        {
            "status": "ok",
            "observed_at": datetime.now().astimezone().isoformat(),
            "accelerator_utilization_pct": 10.0,
            "accelerator_memory_used": 1000,
            "accelerator_memory_total": 2000,
            "memory_is_unified": True,
            "storage": [],
        }
    )
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    node_id = _register_node(client)

    body = client.get(f"/v1/nodes/{node_id}/resources", headers=_AUTH).json()
    assert "memory_is_unified" in body
    assert body["memory_is_unified"] is True


def test_storage_figures_returned() -> None:
    """Managed-storage capacity and available space are returned."""
    agent = FakeNodeAgent()
    agent.set_resources(
        {
            "status": "ok",
            "observed_at": datetime.now().astimezone().isoformat(),
            "accelerator_utilization_pct": 0.0,
            "accelerator_memory_used": 0,
            "accelerator_memory_total": 0,
            "memory_is_unified": True,
            "storage": [
                {
                    "purpose": "models",
                    "path": "/var/lib/tensorstead/models",
                    "capacity_bytes": 3995598389248,
                    "available_bytes": 1204738879488,
                },
                {
                    "purpose": "images",
                    "path": "/var/lib/docker",
                    "capacity_bytes": 3995598389248,
                    "available_bytes": 1204738879488,
                },
            ],
        }
    )
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    node_id = _register_node(client)

    body = client.get(f"/v1/nodes/{node_id}/resources", headers=_AUTH).json()
    assert len(body["storage"]) == 2
    for entry in body["storage"]:
        assert "purpose" in entry
        assert "path" in entry
        assert "capacity_bytes" in entry
        assert "available_bytes" in entry
    purposes = {e["purpose"] for e in body["storage"]}
    assert "models" in purposes
    assert "images" in purposes


def test_unreachable_node_resources_reports_unreachable() -> None:
    """An unreachable node reports unreachable, not an error."""
    agent = FakeNodeAgent()
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    node_id = _register_node(client)

    agent.unreachable = True
    resp = client.get(f"/v1/nodes/{node_id}/resources", headers=_AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "unreachable"
    assert "observed_at" in body


def test_resources_requires_auth() -> None:
    """Management token gates the resources route."""
    app, _ = build_test_coordinator()
    client = TestClient(app)
    node_id = _register_node(client)
    resp = client.get(f"/v1/nodes/{node_id}/resources")
    assert resp.status_code == 401

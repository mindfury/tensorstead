"""Integration test node concurrency.

Two deployments on the same node, on different endpoints, are both created and
both started, and neither is refused on resource grounds.
The product never refuses an operation because of resource capacity — only the
runtime's own bind failure is authoritative.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.integration

_AUTH = {"Authorization": "Bearer test"}


@pytest.fixture
def client() -> TestClient:
    app, _ = build_test_coordinator()
    return TestClient(app)


def _register_node(client: TestClient, name: str = "spark-01") -> str:
    resp = client.post(
        "/v1/nodes",
        json={"name": name, "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    assert resp.status_code == 201
    return str(resp.json()["id"])


def _acquire_model(client: TestClient, node_id: str, repo: str = "org/model") -> str:
    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": repo,
            "revision": "main",
            "nodes": [node_id],
        },
        headers=_AUTH,
    )
    # Acquisition runs on a background thread; poll to terminal before reading the model list.
    poll_operation(client, resp.json()["operation_id"])
    models = client.get("/v1/models", headers=_AUTH).json()
    return str(next(m["id"] for m in models if m["source_model_id"] == repo))


def test_two_deployments_per_node_on_different_endpoints(client: TestClient) -> None:
    """Two deployments on one node, different endpoints, both start."""
    node_id = _register_node(client)
    model_a = _acquire_model(client, node_id, "org/model-a")
    model_b = _acquire_model(client, node_id, "org/model-b")

    # Create both deployments on distinct endpoints.
    payloads = [
        {
            "name": "model-a",
            "model_id": model_a,
            "runtime_type": "vllm",
            "runtime_version": "0.6.0",
            "image_reference": "repo/vllm:a",
            "runtime_config": {"tensor_parallel_size": 1},
            "participating_nodes": [node_id],
            "endpoint": "10.0.0.11:8000",
        },
        {
            "name": "model-b",
            "model_id": model_b,
            "runtime_type": "vllm",
            "runtime_version": "0.6.0",
            "image_reference": "repo/vllm:b",
            "runtime_config": {"tensor_parallel_size": 1},
            "participating_nodes": [node_id],
            "endpoint": "10.0.0.11:8001",
        },
    ]
    for payload in payloads:
        resp = client.post("/v1/deployments", json=payload, headers=_AUTH)
        assert resp.status_code == 202, f"create {payload['name']} should not be refused"

    # Start both.
    deployments = {
        d["declared"]["name"]: d for d in client.get("/v1/deployments", headers=_AUTH).json()
    }
    for name in ("model-a", "model-b"):
        start = client.post(
            f"/v1/deployments/{deployments[name]['declared']['id']}:start",
            headers=_AUTH,
        )
        assert start.status_code == 202, f"start {name} should not be refused"

    # Neither is refused on resource grounds — both are running.
    refreshed = {
        d["declared"]["name"]: d for d in client.get("/v1/deployments", headers=_AUTH).json()
    }
    assert refreshed["model-a"]["declared"]["desired_state"] == "running"
    assert refreshed["model-b"]["declared"]["desired_state"] == "running"

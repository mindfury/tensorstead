"""Integration test the whole deployment path against fakes.

register → acquire → create → start → deployment reports serving at the
recorded endpoint. This exercises the coordinator service layer driving the
fake agent end to end, with no external anything (tier 1).
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


def _acquire_model(client: TestClient, node_id: str) -> str:
    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "org/model",
            "revision": "e1f2a3b",
            "nodes": [node_id],
        },
        headers=_AUTH,
    )
    assert resp.status_code == 202
    # Acquisition runs on a background thread; poll to terminal before reading the model list.
    poll_operation(client, resp.json()["operation_id"])
    models = client.get("/v1/models", headers=_AUTH).json()
    return str(next(m["id"] for m in models if m["source_model_id"] == "org/model"))


def test_us1_full_path(client: TestClient) -> None:
    """Register → acquire → create → start → reports serving."""
    node_id = _register_node(client)
    model_id = _acquire_model(client, node_id)

    # Create the deployment (host untouched).
    create = client.post(
        "/v1/deployments",
        json={
            "name": "llama-70b",
            "model_id": model_id,
            "runtime_type": "vllm",
            "runtime_version": "0.6.0",
            "image_reference": "repo/vllm:tag",
            "runtime_config": {"tensor_parallel_size": 1},
            "participating_nodes": [node_id],
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )
    assert create.status_code == 202

    # Start the deployment (desired state → running).
    deployments = client.get("/v1/deployments", headers=_AUTH).json()
    deployment = next(d for d in deployments if d["declared"]["name"] == "llama-70b")
    start = client.post(f"/v1/deployments/{deployment['declared']['id']}:start", headers=_AUTH)
    assert start.status_code == 202

    # The deployment reports serving at the recorded endpoint.
    refreshed = client.get(f"/v1/deployments/{deployment['declared']['id']}", headers=_AUTH).json()
    assert refreshed["declared"]["desired_state"] == "running"
    assert refreshed["declared"]["running_revision"] == 1
    assert refreshed["declared"]["revision"]["endpoint"] == "10.0.0.11:8000"

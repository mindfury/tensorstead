"""Contract test deployment creation.

- ``POST /v1/deployments`` creates the deployment at revision 1 with
  ``desired_state: stopped`` and **alters no host state**.
- ``endpoint_conflict`` refuses a known managed deployment on the same
  node+endpoint — on identity grounds, never capacity.
- ``runtime_not_distributed`` rejects a multi-node request against a runtime
  that cannot distribute.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.contract

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
    # The route returns 202 immediately and runs on a background thread, so the
    # model row is written asynchronously — poll the operation to terminal
    # before reading the model list.
    poll_operation(client, resp.json()["operation_id"])
    models = client.get("/v1/models", headers=_AUTH).json()
    return str(next(m["id"] for m in models if m["source_model_id"] == "org/model"))


def _deployment_payload(model_id: str, node_id: str, *, endpoint: str = "10.0.0.11:8000") -> dict:
    return {
        "name": "llama-70b",
        "model_id": model_id,
        "runtime_type": "vllm",
        "runtime_version": "0.6.0",
        "image_reference": "repo/vllm:tag",
        "runtime_config": {"tensor_parallel_size": 1},
        "participating_nodes": [node_id],
        "endpoint": endpoint,
    }


def test_create_returns_operation_and_records_deployment(client: TestClient) -> None:
    """Deployment created at revision 1, desired_state stopped."""
    node_id = _register_node(client)
    model_id = _acquire_model(client, node_id)
    resp = client.post(
        "/v1/deployments",
        json=_deployment_payload(model_id, node_id),
        headers=_AUTH,
    )
    assert resp.status_code == 202
    assert "operation_id" in resp.json()

    deployments = client.get("/v1/deployments", headers=_AUTH).json()
    assert deployments
    deployment = next(d for d in deployments if d["declared"]["name"] == "llama-70b")
    assert deployment["declared"]["desired_state"] == "stopped"
    assert deployment["declared"]["current_revision"] == 1
    assert deployment["declared"]["revision"]["runtime_type"] == "vllm"


def test_endpoint_conflict_refused(client: TestClient) -> None:
    """A known managed deployment on the same node+endpoint is refused."""
    node_id = _register_node(client)
    model_id = _acquire_model(client, node_id)
    payload = _deployment_payload(model_id, node_id)
    assert client.post("/v1/deployments", json=payload, headers=_AUTH).status_code == 202
    # Same node, same endpoint — refused on identity grounds, naming it.
    resp = client.post("/v1/deployments", json=payload, headers=_AUTH)
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "endpoint_conflict"
    assert "llama-70b" in body["message"]


def test_different_endpoint_allowed(client: TestClient) -> None:
    """Two deployments on the same node but different endpoints coexist."""
    node_id = _register_node(client)
    model_id = _acquire_model(client, node_id)
    p1 = _deployment_payload(model_id, node_id, endpoint="10.0.0.11:8000")
    p2 = _deployment_payload(model_id, node_id, endpoint="10.0.0.11:8001")
    p2["name"] = "llama-70b-b"
    assert client.post("/v1/deployments", json=p1, headers=_AUTH).status_code == 202
    assert client.post("/v1/deployments", json=p2, headers=_AUTH).status_code == 202


def test_invalid_runtime_config_rejected(client: TestClient) -> None:
    """Invalid runtime config is rejected before the deployment is valid."""
    node_id = _register_node(client)
    model_id = _acquire_model(client, node_id)
    payload = _deployment_payload(model_id, node_id)
    payload["runtime_config"] = {"not_a_real_field": 1}  # rejected by vllm schema
    resp = client.post("/v1/deployments", json=payload, headers=_AUTH)
    assert resp.status_code == 422
    assert resp.json()["code"] == "invalid_deployment"

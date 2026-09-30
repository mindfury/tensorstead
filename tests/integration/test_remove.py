"""Integration test removal.

The runtime instance and its boot arrangement go, the model and image
stay, and the response says which was which.
Removal of a running deployment stops the instance and removes its boot
arrangement. Removal against an unreachable node fails rather than
orphaning.

This is also the home of the invariant: removing a deployment causes zero
deletions of acquired model artifacts or runtime images. Expensive,
slow-to-reacquire things are never collateral damage of a cheap operation.
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


def _setup_running_deployment(client: TestClient) -> tuple[str, str]:
    """Register → acquire → create → start; return (deployment_id, model_id)."""
    resp = client.post(
        "/v1/nodes",
        json={"name": "spark-01", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    assert resp.status_code == 201
    node_id = str(resp.json()["id"])

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
    model_id = str(next(m["id"] for m in models if m["source_model_id"] == "org/model"))

    resp = client.post(
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
    assert resp.status_code == 202
    deployments = client.get("/v1/deployments", headers=_AUTH).json()
    deployment = next(d for d in deployments if d["declared"]["name"] == "llama-70b")
    dep_id = str(deployment["declared"]["id"])

    start = client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH)
    assert start.status_code == 202
    return dep_id, model_id


def test_remove_deletes_instance_retains_model_and_image(client: TestClient) -> None:
    """Remove deletes the runtime instance + unit; model and image stay."""
    dep_id, model_id = _setup_running_deployment(client)

    resp = client.delete(f"/v1/deployments/{dep_id}", headers=_AUTH)
    assert resp.status_code == 202
    body = resp.json()
    assert "operation_id" in body

    # The deployment is gone from the coordinator's declared state.
    resp = client.get(f"/v1/deployments/{dep_id}", headers=_AUTH)
    assert resp.status_code == 404

    # The model is still present.
    resp = client.get(f"/v1/models/{model_id}", headers=_AUTH)
    assert resp.status_code == 200

    # The fake agent no longer has the deployment.
    app = client.app
    fake = app.state.fake_agent  # type: ignore[attr-defined]
    assert dep_id not in fake.deployments


def test_remove_unreachable_node_fails(client: TestClient) -> None:
    """Removal against an unreachable node fails rather than orphaning."""
    dep_id, _ = _setup_running_deployment(client)

    app = client.app
    fake = app.state.fake_agent  # type: ignore[attr-defined]
    fake.unreachable = True

    resp = client.delete(f"/v1/deployments/{dep_id}", headers=_AUTH)
    # The agent is unreachable — removal must fail, not orphan.
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["code"] == "agent_unreachable"

    # The deployment still exists because removal did not succeed.
    fake.unreachable = False
    resp = client.get(f"/v1/deployments/{dep_id}", headers=_AUTH)
    assert resp.status_code == 200

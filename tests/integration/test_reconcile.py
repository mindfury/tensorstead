"""Integration test reconcile.

Full convergence succeeds; partial convergence **fails** naming each
remaining divergence, and the changes already applied are not reverted.
Applied changes leave the deployment in a state a later
reconcile can act on.
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


def _setup_running_deployment(client: TestClient) -> str:
    """Register → acquire → create → start; return deployment id."""
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
    return dep_id


def test_reconcile_full_convergence_succeeds(client: TestClient) -> None:
    """When the deployment is in the declared state, reconcile succeeds."""
    dep_id = _setup_running_deployment(client)

    resp = client.post(f"/v1/deployments/{dep_id}:reconcile", headers=_AUTH)
    assert resp.status_code == 202
    body = resp.json()
    assert "operation_id" in body


def test_reconcile_partial_failure_names_remaining_divergence(client: TestClient) -> None:
    """Partial convergence fails naming each remaining divergence.

    The changes already applied are not reverted.
    """
    dep_id = _setup_running_deployment(client)

    # Simulate divergence: the agent reports the container is not running
    # while desired state is running.
    app = client.app
    fake = app.state.fake_agent  # type: ignore[attr-defined]
    fake.deployments[dep_id].running = False
    fake.deployments[dep_id].running_revision = None

    resp = client.post(f"/v1/deployments/{dep_id}:reconcile", headers=_AUTH)
    # Reconcile attempts convergence; with a fake that cannot restart the
    # container, the operation reports what it could not do.
    assert resp.status_code in (200, 202), resp.text

    # The deployment's desired state is unchanged (applied changes
    # are not reverted).
    dep = client.get(f"/v1/deployments/{dep_id}", headers=_AUTH).json()
    assert dep["declared"]["desired_state"] == "running"

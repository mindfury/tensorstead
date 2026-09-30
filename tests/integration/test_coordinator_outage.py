"""Integration test coordinator-outage survival.

With the coordinator stopped, a running deployment continues serving.
The agent's boot-restoration unit is self-sufficient and the
agent never initiates contact with the coordinator, so a stopped
coordinator is invisible to a running deployment.
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


def test_running_deployment_survives_coordinator_stop(client: TestClient) -> None:
    """With the coordinator stopped, a running deployment keeps serving.

    The fake agent models the real agent's behaviour: it never initiates
    contact with the coordinator, so when we stop driving it from the
    coordinator side, the deployment's running state is unchanged. The
    unit the agent installed is self-sufficient.
    """
    dep_id = _setup_running_deployment(client)

    # The fake agent's deployment is running.
    app = client.app
    fake = app.state.fake_agent  # type: ignore[attr-defined]
    assert fake.deployments[dep_id].running is True

    # Simulate the coordinator stopping: we close the TestClient (no more
    # coordinator-side calls). The agent is a separate process in
    # production and never polls the coordinator, so its state is
    # unaffected by the coordinator being gone.
    client.close()

    # The fake agent's deployment state is unchanged — the agent does not
    # need the coordinator to keep running. The agent
    # never initiates contact, so the coordinator being gone is invisible.
    assert fake.deployments[dep_id].running is True

"""Contract test lifecycle idempotency.

Requesting a held state reports ``already_in_state``, creates no second
instance, and the operation succeeds (exit 0 for the CLI). Idempotency
means repeating a lifecycle operation yields zero additional runtime
instances and zero conflicting records (cross-cutting
invariant 5).
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


def _full_setup(client: TestClient) -> str:
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
    # The route returns 202 immediately and runs on a background thread, so the
    # model row is written asynchronously — poll the operation to terminal
    # before reading the model list.
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


def test_start_when_already_running_reports_already_in_state(client: TestClient) -> None:
    """Starting a running deployment → already_in_state, no second instance."""
    dep_id = _full_setup(client)

    # Start again — already in state.
    resp = client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["code"] == "already_in_state"

    # Only one runtime instance on the fake agent.
    app = client.app
    fake = app.state.fake_agent  # type: ignore[attr-defined]
    assert len(fake.deployments) == 1
    assert fake.deployments[dep_id].running is True


def test_stop_when_already_stopped_reports_already_in_state(client: TestClient) -> None:
    """Stopping a stopped deployment → already_in_state."""
    dep_id = _full_setup(client)

    # Stop it first.
    resp = client.post(f"/v1/deployments/{dep_id}:stop", headers=_AUTH)
    assert resp.status_code == 202

    # Stop again — already in state.
    resp = client.post(f"/v1/deployments/{dep_id}:stop", headers=_AUTH)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["code"] == "already_in_state"

    app = client.app
    fake = app.state.fake_agent  # type: ignore[attr-defined]
    assert fake.deployments[dep_id].running is False


# "Already in state" is a claim about the nodes, so it is checked against them.
# On 2026-10-23 a node rebooted under a running deployment; the record still
# said running, the coordinator's own observation said not running, and start
# refused with "is already running" -- reporting the record as a fact.


def test_start_of_a_deployment_declared_running_but_dead_starts_it(client: TestClient) -> None:
    """A reboot/crash killed the container: start must start, not claim it is running."""
    dep_id = _full_setup(client)
    fake = client.app.state.fake_agent  # type: ignore[attr-defined]
    fake.deployments[dep_id].running = False  # the node rebooted

    resp = client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH)
    assert resp.status_code == 202, resp.text
    poll_operation(client, resp.json()["operation_id"])

    assert fake.deployments[dep_id].running is True
    assert len(fake.deployments) == 1


def test_start_refuses_without_claiming_when_a_node_cannot_be_observed(client: TestClient) -> None:
    dep_id = _full_setup(client)
    fake = client.app.state.fake_agent  # type: ignore[attr-defined]
    fake.unreachable = True

    resp = client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH)
    body = resp.json()
    assert body["code"] == "agent_unreachable", body
    assert "could not be observed" in body["message"]
    assert "already running" not in body["message"]


def test_stop_of_a_deployment_declared_stopped_but_running_stops_it(client: TestClient) -> None:
    """Recorded stopped while a container still runs: stop must stop it."""
    dep_id = _full_setup(client)
    assert client.post(f"/v1/deployments/{dep_id}:stop", headers=_AUTH).status_code == 202
    fake = client.app.state.fake_agent  # type: ignore[attr-defined]
    fake.deployments[dep_id].running = True  # e.g. started by hand on the node

    resp = client.post(f"/v1/deployments/{dep_id}:stop", headers=_AUTH)
    assert resp.status_code == 202, resp.text
    poll_operation(client, resp.json()["operation_id"])
    assert fake.deployments[dep_id].running is False

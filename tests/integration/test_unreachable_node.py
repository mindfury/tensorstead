"""Integration test unreachability.

With the node unreachable, observed reports ``unreachable`` and declared is
returned in full. No previously observed value is presented as current
Declared state is still returned in full alongside an unobtainable
observation.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.fakes.node_agent import FakeNodeAgent
from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.integration

_AUTH = {"Authorization": "Bearer test"}


@pytest.fixture
def client_and_agent() -> tuple[TestClient, FakeNodeAgent]:
    agent = FakeNodeAgent()
    app, _ = build_test_coordinator(agent=agent)
    return TestClient(app), agent


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


def _create_and_start(client: TestClient, node_id: str, model_id: str) -> str:
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
    deployments = client.get("/v1/deployments", headers=_AUTH).json()
    deployment = next(d for d in deployments if d["declared"]["name"] == "llama-70b")
    start = client.post(f"/v1/deployments/{deployment['declared']['id']}:start", headers=_AUTH)
    assert start.status_code == 202
    return str(deployment["declared"]["id"])


def test_unreachable_node_observed_reports_unreachable(
    client_and_agent: tuple[TestClient, FakeNodeAgent],
) -> None:
    """Observed reports unreachable; declared returned in full."""
    client, agent = client_and_agent
    node_id = _register_node(client)
    model_id = _acquire_model(client, node_id)
    deployment_id = _create_and_start(client, node_id, model_id)

    # Confirm observed is running before the node goes unreachable.
    before = client.get(f"/v1/deployments/{deployment_id}", headers=_AUTH).json()
    assert before["observed"]["status"] == "running"

    # Make the agent unreachable — observed degrades to unreachable.
    agent.unreachable = True
    resp = client.get(f"/v1/deployments/{deployment_id}", headers=_AUTH)
    assert resp.status_code == 200
    body = resp.json()

    # Observed reports unreachable, never a previously observed value.
    assert body["observed"]["status"] == "unreachable"
    assert body["observed"]["observed_at"]  # still carries when-it-was-read

    # Declared is returned in full alongside the unobtainable observation.
    declared = body["declared"]
    assert declared["id"] == deployment_id
    assert declared["name"] == "llama-70b"
    assert declared["desired_state"] == "running"
    assert declared["current_revision"] == 1
    assert declared["running_revision"] == 1


def test_no_previously_observed_value_presented_as_current(
    client_and_agent: tuple[TestClient, FakeNodeAgent],
) -> None:
    """A stale value is never presented as current."""
    client, agent = client_and_agent
    node_id = _register_node(client)
    model_id = _acquire_model(client, node_id)
    deployment_id = _create_and_start(client, node_id, model_id)

    # While reachable, observed reports running.
    running = client.get(f"/v1/deployments/{deployment_id}", headers=_AUTH).json()
    assert running["observed"]["status"] == "running"

    # Node goes unreachable — the status must not stay "running".
    agent.unreachable = True
    unreachable = client.get(f"/v1/deployments/{deployment_id}", headers=_AUTH).json()
    assert unreachable["observed"]["status"] == "unreachable"
    assert unreachable["observed"]["status"] != "running"


def test_declared_returned_in_full_when_unreachable(
    client_and_agent: tuple[TestClient, FakeNodeAgent],
) -> None:
    """Declared state is fully returned even when observation fails."""
    client, agent = client_and_agent
    node_id = _register_node(client)
    model_id = _acquire_model(client, node_id)
    deployment_id = _create_and_start(client, node_id, model_id)

    agent.unreachable = True
    resp = client.get(f"/v1/deployments/{deployment_id}", headers=_AUTH)
    body = resp.json()
    declared = body["declared"]
    # Every declared field is present — not just a subset.
    for field in (
        "id",
        "name",
        "desired_state",
        "current_revision",
        "running_revision",
        "revision",
    ):
        assert field in declared, f"declared missing {field!r} when unreachable"
    assert declared["revision"]["runtime_type"] == "vllm"
    assert declared["revision"]["endpoint"] == "10.0.0.11:8000"

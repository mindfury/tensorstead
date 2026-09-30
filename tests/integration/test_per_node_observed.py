"""Integration test per-node observation.

One unreachable node reports ``unreachable`` **for itself** and is never
folded into a healthy verdict for the whole deployment.

The failure mode this guards against is the tempting summary: two nodes, one
answered "running", so call the deployment running. That reads as reassurance
and is exactly wrong — the truthful answer is that half of it is unknown. So
the per-node block keeps each node's own answer, and the overall status is
degraded by the node that could not be reached rather than carried by the one
that could.

Declared state is still returned in full throughout: losing
sight of a host does not mean losing the record of what was asked for.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.fakes.node_agent import FakeNodeAgent
from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.integration

_AUTH = {"Authorization": "Bearer test"}


@pytest.fixture
def two_nodes() -> tuple[TestClient, str, str, str, FakeNodeAgent]:
    """A running two-node deployment; node B has its own agent."""
    app, _ = build_test_coordinator()
    client = TestClient(app)
    node_client = app.state.node_client

    node_a = _register(client, "spark-01", "https://10.0.0.11:8443")
    node_b = _register(client, "spark-02", "https://10.0.0.12:8443")
    agent_b = FakeNodeAgent()
    node_client.agents[node_b] = agent_b

    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "org/model",
            "revision": "e1f2a3b",
            "nodes": [node_a, node_b],
        },
        headers=_AUTH,
    )
    assert resp.status_code == 202
    # Acquisition runs on a background thread; poll to terminal before reading the model list.
    poll_operation(client, resp.json()["operation_id"])
    models = client.get("/v1/models", headers=_AUTH).json()
    model_id = str(next(m["id"] for m in models if m["source_model_id"] == "org/model"))

    created = client.post(
        "/v1/deployments",
        json={
            "name": "llama-70b",
            "model_id": model_id,
            "runtime_type": "vllm",
            "runtime_version": "0.6.0",
            "image_reference": "repo/vllm:tag",
            "runtime_config": {"tensor_parallel_size": 2},
            "participating_nodes": [node_a, node_b],
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )
    assert created.status_code == 202
    deployments = client.get("/v1/deployments", headers=_AUTH).json()
    dep_id = str(deployments[0]["declared"]["id"])
    assert client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH).status_code == 202
    return client, dep_id, node_a, node_b, agent_b


def _register(client: TestClient, name: str, endpoint: str) -> str:
    resp = client.post("/v1/nodes", json={"name": name, "agent_endpoint": endpoint}, headers=_AUTH)
    assert resp.status_code == 201
    return str(resp.json()["id"])


def test_all_nodes_report_themselves_when_healthy(
    two_nodes: tuple[TestClient, str, str, str, FakeNodeAgent],
) -> None:
    """Every participating node appears in the per-node block."""
    client, dep_id, node_a, node_b, _agent_b = two_nodes

    observed = client.get(f"/v1/deployments/{dep_id}/status", headers=_AUTH).json()["observed"]

    assert set(observed["per_node"]) == {node_a, node_b}
    assert observed["per_node"][node_a]["status"] == "running"
    assert observed["per_node"][node_b]["status"] == "running"
    assert observed["status"] == "running"


def test_one_unreachable_node_reports_unreachable_for_itself(
    two_nodes: tuple[TestClient, str, str, str, FakeNodeAgent],
) -> None:
    """The unreachable node says so; the reachable one still answers."""
    client, dep_id, node_a, node_b, agent_b = two_nodes
    agent_b.unreachable = True

    observed = client.get(f"/v1/deployments/{dep_id}/status", headers=_AUTH).json()["observed"]

    assert observed["per_node"][node_b]["status"] == "unreachable"
    assert observed["per_node"][node_a]["status"] == "running", (
        "a reachable node still reports its own real state"
    )


def test_the_whole_is_never_reported_healthy_on_a_partial_answer(
    two_nodes: tuple[TestClient, str, str, str, FakeNodeAgent],
) -> None:
    """One node answering "running" does not make the deployment running.

    This is the assertion the whole test file exists for.
    """
    client, dep_id, _node_a, _node_b, agent_b = two_nodes
    agent_b.unreachable = True

    observed = client.get(f"/v1/deployments/{dep_id}/status", headers=_AUTH).json()["observed"]

    assert observed["status"] != "running"
    assert observed["status"] == "unreachable"


def test_declared_is_returned_in_full_while_a_node_is_unreachable(
    two_nodes: tuple[TestClient, str, str, str, FakeNodeAgent],
) -> None:
    """Losing a host loses no part of the record."""
    client, dep_id, node_a, node_b, agent_b = two_nodes
    agent_b.unreachable = True

    body = client.get(f"/v1/deployments/{dep_id}", headers=_AUTH).json()
    declared = body["declared"]

    assert declared["desired_state"] == "running"
    assert declared["revision"]["participating_nodes"] == [node_a, node_b]
    assert declared["revision"]["runtime_version"] == "0.6.0"
    assert declared["revision"]["image_reference"] == "repo/vllm:tag"


def test_observation_mutates_nothing_on_either_node(
    two_nodes: tuple[TestClient, str, str, str, FakeNodeAgent],
) -> None:
    """Reading state changes none of it, unreachable node included."""
    client, dep_id, _node_a, _node_b, agent_b = two_nodes
    agent_b.unreachable = True
    before = dict(agent_b.deployments)

    client.get(f"/v1/deployments/{dep_id}/status", headers=_AUTH)
    client.get(f"/v1/deployments/{dep_id}", headers=_AUTH)

    assert agent_b.deployments == before
    declared = client.get(f"/v1/deployments/{dep_id}", headers=_AUTH).json()["declared"]
    assert declared["desired_state"] == "running", "observation did not rewrite desired state"


def test_recovery_is_visible_without_any_write(
    two_nodes: tuple[TestClient, str, str, str, FakeNodeAgent],
) -> None:
    """When the node comes back, the next read simply tells the truth again."""
    client, dep_id, _node_a, node_b, agent_b = two_nodes
    agent_b.unreachable = True
    assert (
        client.get(f"/v1/deployments/{dep_id}/status", headers=_AUTH).json()["observed"][
            "per_node"
        ][node_b]["status"]
        == "unreachable"
    )

    agent_b.unreachable = False

    observed = client.get(f"/v1/deployments/{dep_id}/status", headers=_AUTH).json()["observed"]
    assert observed["per_node"][node_b]["status"] == "running"
    assert observed["status"] == "running"

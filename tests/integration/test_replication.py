"""Integration test acquire-once-then-replicate.

A model wanted on two nodes is fetched from upstream **once** and then copied
node-to-node. Three claims, each asserted separately because each could fail on
its own:

- upstream is contacted exactly once, no matter how many nodes want the model;
- the second node's copy arrives from the *first node*, named explicitly, over
  the agents' own connectivity;
- **local replication is an optimization, not a precondition** — when the peer
  hop is unavailable the second node falls back to acquiring from upstream and
  the request still succeeds. A performance feature that can fail the whole
  operation is not an optimization.

Underlying all of it: no model byte passes through the coordinator.
The coordinator names a source and reads back an outcome; the transfer is
between the two agents.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tests.fakes.node_agent import FakeNodeAgent
from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.integration

_AUTH = {"Authorization": "Bearer test"}


@pytest.fixture
def two_nodes() -> tuple[TestClient, str, str, FakeNodeAgent, FakeNodeAgent]:
    """Two registered nodes, each with its own agent."""
    app, _ = build_test_coordinator()
    client = TestClient(app)
    node_client = app.state.node_client

    node_a = _register(client, "spark-01", "https://10.0.0.11:8443")
    node_b = _register(client, "spark-02", "https://10.0.0.12:8443")
    agent_a = node_client.agent
    agent_b = FakeNodeAgent()
    node_client.agents[node_b] = agent_b
    return client, node_a, node_b, agent_a, agent_b


def _register(client: TestClient, name: str, endpoint: str) -> str:
    resp = client.post("/v1/nodes", json={"name": name, "agent_endpoint": endpoint}, headers=_AUTH)
    assert resp.status_code == 201
    return str(resp.json()["id"])


def _acquire(client: TestClient, nodes: list[str], **extra: object) -> int:
    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "org/model",
            "revision": "e1f2a3b",
            "nodes": nodes,
            **extra,
        },
        headers=_AUTH,
    )
    # Acquisition runs on a background thread; poll to terminal before reading the model list.
    poll_operation(client, resp.json()["operation_id"])
    return int(resp.status_code)


def test_upstream_is_contacted_once_for_two_nodes(
    two_nodes: tuple[TestClient, str, str, FakeNodeAgent, FakeNodeAgent],
) -> None:
    """Acquire once from upstream, then replicate."""
    client, node_a, node_b, agent_a, agent_b = two_nodes

    assert _acquire(client, [node_a, node_b]) == 202

    assert agent_a.acquisition_calls == 1, "the first node fetched from upstream"
    assert agent_b.acquisition_calls == 0, "the second node did not re-fetch upstream"


def test_the_second_node_pulls_from_the_first(
    two_nodes: tuple[TestClient, str, str, FakeNodeAgent, FakeNodeAgent],
) -> None:
    """The transfer is agent-to-agent and names its source."""
    client, node_a, node_b, _agent_a, agent_b = two_nodes

    _acquire(client, [node_a, node_b])

    assert len(agent_b.replication_pulls) == 1
    pull = agent_b.replication_pulls[0]
    assert pull["from_node"] == node_a
    assert pull["from_endpoint"] == "https://10.0.0.11:8443"
    assert pull["content_digest"] == "sha256:fake-digest", (
        "the destination was given something to verify the transfer against"
    )


def test_both_nodes_end_up_with_an_available_replica(
    two_nodes: tuple[TestClient, str, str, FakeNodeAgent, FakeNodeAgent],
) -> None:
    client, node_a, node_b, _agent_a, _agent_b = two_nodes

    _acquire(client, [node_a, node_b])

    models = client.get("/v1/models", headers=_AUTH).json()
    model = next(m for m in models if m["source_model_id"] == "org/model")
    states = {r["node_id"]: r["state"] for r in model["replicas"]}
    assert states == {node_a: "available", node_b: "available"}


def test_replication_failure_is_refused_not_refetched(
    two_nodes: tuple[TestClient, str, str, FakeNodeAgent, FakeNodeAgent],
) -> None:
    """No usable peer hop → the request fails naming the holder; upstream is not hit twice.

    This used to fall back to upstream silently, and with the real Hugging Face
    adapter replication never worked at all, so every second node quietly
    downloaded the whole model again (165 GiB for GLM-5.3, 2026-10-16).
    """
    client, node_a, node_b, agent_a, agent_b = two_nodes
    agent_b.replication_unavailable = True

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
    operation = poll_operation(client, resp.json()["operation_id"])

    assert agent_a.acquisition_calls == 1
    assert agent_b.acquisition_calls == 0, "the second node must not download it again"
    assert operation["state"] == "failed"
    assert "replication_failed" in json.dumps(operation)

    models = client.get("/v1/models", headers=_AUTH).json()
    model = next(m for m in models if m["source_model_id"] == "org/model")
    states = {r["node_id"]: r["state"] for r in model["replicas"]}
    assert states.get(node_a) == "available", "the replica already acquired is kept"
    assert states.get(node_b) != "available"


def test_upstream_fallback_happens_only_when_asked(
    two_nodes: tuple[TestClient, str, str, FakeNodeAgent, FakeNodeAgent],
) -> None:
    """``allow_upstream_fallback`` restores the old behaviour, explicitly."""
    client, node_a, node_b, agent_a, agent_b = two_nodes
    agent_b.replication_unavailable = True

    assert _acquire(client, [node_a, node_b], allow_upstream_fallback=True) == 202

    assert agent_a.acquisition_calls == 1
    assert agent_b.acquisition_calls == 1
    models = client.get("/v1/models", headers=_AUTH).json()
    model = next(m for m in models if m["source_model_id"] == "org/model")
    states = {r["node_id"]: r["state"] for r in model["replicas"]}
    assert states == {node_a: "available", node_b: "available"}


def test_a_node_that_already_holds_the_revision_transfers_nothing(
    two_nodes: tuple[TestClient, str, str, FakeNodeAgent, FakeNodeAgent],
) -> None:
    """Re-acquiring is idempotent: no upstream fetch, no peer pull."""
    client, node_a, node_b, agent_a, agent_b = two_nodes
    _acquire(client, [node_a, node_b])
    pulls_before = len(agent_b.replication_pulls)

    _acquire(client, [node_a, node_b])

    assert agent_a.acquisition_calls == 1
    assert len(agent_b.replication_pulls) == pulls_before


def test_adding_a_third_node_replicates_rather_than_refetching(
    two_nodes: tuple[TestClient, str, str, FakeNodeAgent, FakeNodeAgent],
) -> None:
    """A later node joins by peer copy, not another upstream download."""
    client, node_a, node_b, agent_a, _agent_b = two_nodes
    _acquire(client, [node_a, node_b])

    app_client = client.app.state.node_client  # type: ignore[attr-defined]
    node_c = _register(client, "spark-03", "https://10.0.0.13:8443")
    agent_c = FakeNodeAgent()
    app_client.agents[node_c] = agent_c

    _acquire(client, [node_a, node_b, node_c])

    assert agent_a.acquisition_calls == 1, "still only one upstream fetch overall"
    assert agent_c.acquisition_calls == 0
    assert len(agent_c.replication_pulls) == 1
    assert agent_c.replication_pulls[0]["from_node"] == node_a


def test_asking_for_only_the_new_node_replicates_from_an_existing_holder(
    two_nodes: tuple[TestClient, str, str, FakeNodeAgent, FakeNodeAgent],
) -> None:
    """``nodes=[new]`` copies from a holder the request does not name.

    Measured on the estate: a replica of a model held on spark-471b, requested
    for spark-6d42 alone, downloaded the whole checkpoint from upstream again,
    because only the nodes named were considered as sources.
    """
    client, node_a, node_b, agent_a, agent_b = two_nodes
    _acquire(client, [node_a])
    assert agent_a.acquisition_calls == 1

    _acquire(client, [node_b])

    assert agent_b.acquisition_calls == 0, "the new node fetched from upstream again"
    assert len(agent_b.replication_pulls) == 1
    assert agent_b.replication_pulls[0]["from_node"] == node_a
    models = client.get("/v1/models", headers=_AUTH).json()
    model = next(m for m in models if m["source_model_id"] == "org/model")
    assert {r["node_id"]: r["state"] for r in model["replicas"]} == {
        node_a: "available",
        node_b: "available",
    }


def test_a_holder_elsewhere_that_cannot_copy_refuses_rather_than_refetching(
    two_nodes: tuple[TestClient, str, str, FakeNodeAgent, FakeNodeAgent],
) -> None:
    """The fetch-once rule applies to a holder found outside the request too."""
    client, node_a, node_b, _agent_a, agent_b = two_nodes
    _acquire(client, [node_a])
    agent_b.replication_unavailable = True

    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "org/model",
            "revision": "e1f2a3b",
            "nodes": [node_b],
        },
        headers=_AUTH,
    )
    operation = poll_operation(client, resp.json()["operation_id"])

    assert agent_b.acquisition_calls == 0, "a failed copy must not become a second download"
    assert operation["state"] == "failed"
    assert "replication_failed" in json.dumps(operation)


def test_no_model_byte_transits_the_coordinator(
    two_nodes: tuple[TestClient, str, str, FakeNodeAgent, FakeNodeAgent],
) -> None:
    """The coordinator names a peer and reads an outcome — nothing more.

    The replication response is metadata only: an id, a state, a digest, a
    size. There is no field that could carry artifact content, which is what
    keeps the management plane out of the data path structurally rather than
    by convention.
    """
    client, node_a, node_b, _agent_a, agent_b = two_nodes
    _acquire(client, [node_a, node_b])

    outcome = agent_b.replicate_model(
        source_id="huggingface",
        source_model_id="org/model",
        resolved_revision="e1f2a3b",
        content_digest="sha256:fake-digest",
        source_node_id=node_a,
        source_agent_endpoint="https://10.0.0.11:8443",
    )
    assert set(outcome) == {"model_id", "state", "content_digest", "size_bytes"}


def test_a_second_acquire_while_one_is_running_is_refused(
    two_nodes: tuple[TestClient, str, str, FakeNodeAgent, FakeNodeAgent],
) -> None:
    """One download per repository: the second request is told, at once, to wait and replicate."""
    client, node_a, node_b, _agent_a, agent_b = two_nodes
    service = client.app.state.models_service  # type: ignore[attr-defined]
    service.reserve_acquisition(
        source_id="huggingface",
        source_model_id="org/model",
        file_selector=(),
        operation_id="01RUNNINGOPERATION000000000",
        nodes=[node_a],
    )

    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "org/model",
            "revision": "main",  # a different revision spelling is the same download
            "nodes": [node_b],
        },
        headers=_AUTH,
    )

    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "acquisition_in_progress"
    assert "01RUNNINGOPERATION000000000" in body["message"]
    assert "spark-01" in body["message"]
    assert agent_b.acquisition_calls == 0


def test_the_reservation_is_released_when_the_acquire_ends(
    two_nodes: tuple[TestClient, str, str, FakeNodeAgent, FakeNodeAgent],
) -> None:
    client, node_a, node_b, agent_a, _agent_b = two_nodes
    assert _acquire(client, [node_a]) == 202
    # A follow-up for the other node is admitted, and replicates rather than downloading.
    assert _acquire(client, [node_a, node_b]) == 202
    assert agent_a.acquisition_calls == 1

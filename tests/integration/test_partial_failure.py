"""Integration test partial failure across nodes.

Succeeding on one node and failing on another is an **overall failure**, not a
qualified success. Three things follow from that and each is asserted here:

1. the failure names the node that failed, so an operator knows where to look;
2. declared desired state is **not** recorded as changed — a half-applied
   operation did not achieve what was asked, so recording it as achieved would
   make the coordinator's record a lie;
3. the node that *did* change surfaces as **divergence** on the next
   observation, rather than being rolled back. Rolling it back would be a
   second unrequested mutation, and the product's whole posture is to report
   what it finds and change nothing without being asked.
"""

from __future__ import annotations

import threading
import time
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from tensorstead.ports.runtime_adapter import ContainerRequirements
from tests.fakes.node_agent import FakeNodeAgent
from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.integration

_AUTH = {"Authorization": "Bearer test"}


@pytest.fixture
def two_node_deployment() -> tuple[TestClient, str, str, str, FakeNodeAgent]:
    """A vLLM deployment across two nodes, node B backed by its own agent.

    Returns ``(client, deployment_id, node_a, node_b, agent_b)`` so a test can
    make exactly one of the two nodes misbehave.
    """
    app, _ = build_test_coordinator()
    client = TestClient(app)
    node_client = app.state.node_client

    node_a = _register(client, "spark-01", "https://10.0.0.11:8443")
    node_b = _register(client, "spark-02", "https://10.0.0.12:8443")

    agent_b = FakeNodeAgent()
    node_client.agents[node_b] = agent_b

    model_id = _acquire(client, [node_a, node_b])
    dep_id = _create(client, model_id, [node_a, node_b])
    return client, dep_id, node_a, node_b, agent_b


def _register(client: TestClient, name: str, endpoint: str) -> str:
    resp = client.post("/v1/nodes", json={"name": name, "agent_endpoint": endpoint}, headers=_AUTH)
    assert resp.status_code == 201
    return str(resp.json()["id"])


def _acquire(client: TestClient, nodes: list[str]) -> str:
    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "org/model",
            "revision": "e1f2a3b",
            "nodes": nodes,
        },
        headers=_AUTH,
    )
    assert resp.status_code == 202
    # Acquisition runs on a background thread; poll to terminal before reading the model list.
    poll_operation(client, resp.json()["operation_id"])
    models = client.get("/v1/models", headers=_AUTH).json()
    return str(next(m["id"] for m in models if m["source_model_id"] == "org/model"))


def _create(client: TestClient, model_id: str, nodes: list[str]) -> str:
    resp = client.post(
        "/v1/deployments",
        json={
            "name": "llama-70b",
            "model_id": model_id,
            "runtime_type": "vllm",
            "runtime_version": "0.6.0",
            "image_reference": "repo/vllm:tag",
            "runtime_config": {"tensor_parallel_size": 2},
            "participating_nodes": nodes,
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )
    assert resp.status_code == 202
    deployments = client.get("/v1/deployments", headers=_AUTH).json()
    return str(deployments[0]["declared"]["id"])


def test_partial_start_is_an_overall_failure_naming_the_node(
    two_node_deployment: tuple[TestClient, str, str, str, FakeNodeAgent],
) -> None:
    """One node up, one down → failure that names the failed node."""
    client, dep_id, _node_a, node_b, agent_b = two_node_deployment
    _without_staging(client)
    agent_b.unreachable = True

    resp = client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH)

    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "partial_failure"
    assert "spark-02" in body["message"], "the failed node is named"
    per_node = body["detail"]["per_node"]
    assert per_node[node_b]["state"] == "failed"
    assert body["detail"]["desired_state_changed"] is False


def test_partial_start_leaves_desired_state_unchanged(
    two_node_deployment: tuple[TestClient, str, str, str, FakeNodeAgent],
) -> None:
    """The coordinator does not record a half-applied change."""
    client, dep_id, _node_a, _node_b, agent_b = two_node_deployment
    agent_b.unreachable = True

    client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH)

    declared = client.get(f"/v1/deployments/{dep_id}", headers=_AUTH).json()["declared"]
    assert declared["desired_state"] == "stopped"
    assert declared["running_revision"] is None


def test_the_changed_node_surfaces_as_divergence(
    two_node_deployment: tuple[TestClient, str, str, str, FakeNodeAgent],
) -> None:
    """The node that did start is reported, not silently reverted."""
    client, dep_id, node_a, _node_b, agent_b = two_node_deployment
    _without_staging(client)
    agent_b.unreachable = True

    client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH)

    # Node A really did start — the fake agent records the deployment.
    status = client.get(f"/v1/deployments/{dep_id}/status", headers=_AUTH).json()
    per_node = status["observed"]["per_node"]
    assert per_node[node_a]["status"] == "running"

    # And that mismatch against a 'stopped' declared state is a divergence the
    # operator is shown rather than something the product quietly undoes.
    assert status["divergences"], "the started node is reported as divergence"


def test_the_operation_record_carries_per_node_outcomes(
    two_node_deployment: tuple[TestClient, str, str, str, FakeNodeAgent],
) -> None:
    """The failure is durable and attributable afterwards."""
    client, dep_id, node_a, node_b, agent_b = two_node_deployment
    _without_staging(client)
    agent_b.unreachable = True

    client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH)

    operations = client.get(f"/v1/operations?deployment_id={dep_id}", headers=_AUTH).json()
    start_ops = [op for op in operations if op["kind"] == "start"]
    assert start_ops, "the attempt was recorded"
    failure = start_ops[-1]["failure_reason"]
    assert start_ops[-1]["state"] == "failed"
    assert failure["code"] == "partial_failure"
    assert failure["detail"]["per_node"][node_a]["state"] == "succeeded"
    assert failure["detail"]["per_node"][node_b]["state"] == "failed"


def _without_staging(client: TestClient) -> None:
    """Make this deployment's runtime one whose ranks may start concurrently.

    The generic contract -- every nominated node attempted, dispatched
    without serializing behind a slow one -- is what these two tests are about.
    It is unchanged.

    What changed is that vLLM now *declares* it cannot start its ranks
    concurrently, and the fixture happens to be a
    two-node vLLM deployment. Asserting the generic rule through a runtime that
    declares an exception to it would be testing the exception. So these use a
    stand-in that declares no rendezvous port, which is what every
    non-distributed and every ordinary multi-node runtime declares.

    Staged behaviour has its own file: test_staged_distributed_start.py.
    """

    class _ConcurrentRuntime:
        @staticmethod
        def container_requirements(_config: dict, _position: object) -> object:
            return ContainerRequirements()

    client.app.state.lifecycle_service._adapters = {  # type: ignore[attr-defined]
        "vllm": _ConcurrentRuntime()
    }


def test_every_node_is_attempted_before_the_verdict(
    two_node_deployment: tuple[TestClient, str, str, str, FakeNodeAgent],
) -> None:
    """A failure on the first node does not skip the second.

    The reported outcome has to describe the cluster as it actually is, which
    means finding out about every node rather than stopping at the first bad
    news.
    """
    client, dep_id, node_a, node_b, agent_b = two_node_deployment
    _without_staging(client)
    app_state_client = client.app.state.node_client  # type: ignore[attr-defined]
    # Make the *first* node fail instead, and confirm B was still contacted.
    app_state_client.agent.unreachable = True

    resp = client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH)

    assert resp.status_code == 409
    per_node = resp.json()["detail"]["per_node"]
    assert set(per_node) == {node_a, node_b}
    assert per_node[node_a]["state"] == "failed"
    assert per_node[node_b]["state"] == "succeeded"
    assert agent_b.deployments, "the second node was attempted despite the first failing"


def test_a_slow_first_node_does_not_prevent_starting_the_second(
    two_node_deployment: tuple[TestClient, str, str, str, FakeNodeAgent],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every nominated node is contacted even while another is still starting.

    A runtime image pull or cold GPU initialization can take minutes.  The
    coordinator must dispatch to the other nodes during that interval rather
    than serializing the cluster behind the first slow agent.
    """
    client, dep_id, _node_a, _node_b, agent_b = two_node_deployment
    _without_staging(client)
    first_agent = client.app.state.node_client.agent  # type: ignore[attr-defined]
    entered = threading.Event()
    release = threading.Event()
    original = first_agent.create_deployment

    def slow_create(deployment_id: str, *, endpoint: str) -> dict[str, object]:
        entered.set()
        assert release.wait(timeout=2.0), "test did not release the first agent"
        return cast(dict[str, object], original(deployment_id, endpoint=endpoint))

    monkeypatch.setattr(first_agent, "create_deployment", slow_create)
    response: list[Any] = []

    def start() -> None:
        response.append(client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH))

    worker = threading.Thread(target=start)
    worker.start()
    assert entered.wait(timeout=1.0), "first agent was not contacted"

    deadline = time.monotonic() + 1.0
    while not agent_b.deployments and time.monotonic() < deadline:
        time.sleep(0.01)
    assert agent_b.deployments, "second agent was not contacted while first was slow"

    release.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert response[0].status_code == 202


def test_all_nodes_failing_keeps_the_single_node_shape(
    two_node_deployment: tuple[TestClient, str, str, str, FakeNodeAgent],
) -> None:
    """Nothing succeeded → ``agent_unreachable``, not ``partial_failure``.

    "Partial" has to mean partial. A total failure reports the same way a
    one-node deployment always has.
    """
    client, dep_id, _node_a, _node_b, agent_b = two_node_deployment
    client.app.state.node_client.agent.unreachable = True  # type: ignore[attr-defined]
    agent_b.unreachable = True

    resp = client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH)

    assert resp.status_code == 503
    assert resp.json()["code"] == "agent_unreachable"

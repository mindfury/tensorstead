"""Stop and remove reach every node the deployment was materialized on.

CodeGuard CG-09: stop drove only the current revision's nodes. Start on A,
modify placement to B, stop: only B was asked, A kept serving, and the
coordinator recorded ``stopped``. Remove had the same flaw and would orphan A's
container and unit when it deleted the record.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.fakes.node_agent import FakeNodeAgent
from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer test"}


@pytest.fixture
def moved() -> tuple[TestClient, Any, str, str, str, FakeNodeAgent]:
    """Running on A, then re-placed onto B while still running."""
    app, _ = build_test_coordinator()
    client = TestClient(app)
    node_client = app.state.node_client
    a = client.post(
        "/v1/nodes",
        json={"name": "spark-01", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    ).json()["id"]
    b = client.post(
        "/v1/nodes",
        json={"name": "spark-02", "agent_endpoint": "https://10.0.0.12:8443"},
        headers=_AUTH,
    ).json()["id"]
    agent_a = FakeNodeAgent()
    node_client.agents[a] = agent_a
    node_client.agents[b] = FakeNodeAgent()
    op = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "org/model",
            "revision": "e1f2a3b",
            "nodes": [a, b],
        },
        headers=_AUTH,
    ).json()["operation_id"]
    poll_operation(client, op)
    model_id = client.get("/v1/models", headers=_AUTH).json()[0]["id"]
    client.post(
        "/v1/deployments",
        json={
            "name": "d",
            "model_id": model_id,
            "runtime_type": "vllm",
            "runtime_version": "latest",
            "image_reference": "repo/vllm:tag",
            "runtime_config": {},
            "participating_nodes": [a],
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )
    dep = client.get("/v1/deployments", headers=_AUTH).json()[0]["declared"]["id"]
    assert client.post(f"/v1/deployments/{dep}:start", headers=_AUTH).status_code == 202
    resp = client.patch(
        f"/v1/deployments/{dep}",
        json={"participating_nodes": [b], "endpoint": "10.0.0.12:8000"},
        headers=_AUTH,
    )
    assert resp.status_code == 200, resp.text
    node_client.stops.clear()
    return client, node_client, dep, a, b, agent_a


def test_stop_reaches_the_node_it_actually_runs_on(moved: Any) -> None:
    client, node_client, dep, a, _b, _ = moved
    assert client.post(f"/v1/deployments/{dep}:stop", headers=_AUTH).status_code == 202
    assert a in node_client.stops, "the node still running it was never asked to stop"


def test_an_unreachable_old_node_is_not_reported_stopped(moved: Any) -> None:
    client, _node_client, dep, _a, _b, agent_a = moved
    agent_a.unreachable = True
    resp = client.post(f"/v1/deployments/{dep}:stop", headers=_AUTH)
    assert resp.status_code >= 400
    assert client.app.state.repository.get_deployment(dep).desired_state == "running"


def test_restart_after_a_move_stops_the_old_node(moved: Any) -> None:
    client, node_client, dep, a, b, _ = moved
    assert client.post(f"/v1/deployments/{dep}:restart", headers=_AUTH).status_code in (200, 202)
    assert a in node_client.stops
    created_on = [node for node, _payload in node_client.create_payloads][-1:]
    assert created_on == [b]


def test_remove_reaches_every_placement(moved: Any) -> None:
    client, node_client, dep, a, b, _ = moved
    assert client.delete(f"/v1/deployments/{dep}", headers=_AUTH).status_code in (200, 202, 204)
    assert set(node_client.removes) == {a, b}

"""An unexpected managed container is detected, and a legitimate one is not.

The requirement promises that a container running in the product's namespace that the
product did not record is **reported, never killed**. Before it was implemented,
that promise was empty in two separate ways:

1. The agent asked its container engine for a ``containers`` dict. Only the
   test fake has one, so on real hardware the check evaluated
   ``isinstance(None, dict)`` and returned false every single time. The
   guarantee reported nothing in production, and the tests passed because they
   ran against the one object that answered.

2. Where it did run, it was wrong. Any *other* ``tensorstead-`` container counted
   as unexpected, so a node legitimately hosting two deployments accused itself
   the moment either one's container went away.

The fix is a split rather than a better heuristic: the agent enumerates what is
on its node, and the coordinator — the only side holding the deployment records
— decides what is unaccounted for.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.fakes.node_agent import FakeNodeAgent
from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.integration

_AUTH = {"Authorization": "Bearer test"}


def _kinds(body: dict) -> list[str]:
    return [d["kind"] for d in body.get("divergences", [])]


def _node(client: TestClient) -> str:
    return str(
        client.post(
            "/v1/nodes",
            json={"name": "spark-alpha", "agent_endpoint": "https://10.0.0.11:8443"},
            headers=_AUTH,
        ).json()["id"]
    )


def _deploy(client: TestClient, node_id: str, name: str, port: int) -> str:
    """Create and start one deployment on ``node_id``. Returns its id."""
    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": f"nvidia/{name}",
            "nodes": [node_id],
        },
        headers=_AUTH,
    )
    # Acquisition runs on a background thread; poll to terminal before reading the model list.
    poll_operation(client, resp.json()["operation_id"])
    models = client.get("/v1/models", headers=_AUTH).json()
    model_id = next(m["id"] for m in models if m["source_model_id"] == f"nvidia/{name}")

    create = client.post(
        "/v1/deployments",
        json={
            "name": name,
            "model_id": model_id,
            "runtime_type": "vllm",
            "runtime_version": "0.6.0",
            "image_reference": "local/vllm:26.07-xgrammar-0.2.1",
            "runtime_config": {"tensor_parallel_size": 1},
            "participating_nodes": [node_id],
            "endpoint": f"10.0.0.11:{port}",
        },
        headers=_AUTH,
    )
    assert create.status_code == 202, create.text
    deployments = client.get("/v1/deployments", headers=_AUTH).json()
    dep_id = str(next(d["declared"]["id"] for d in deployments if d["declared"]["name"] == name))
    assert client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH).status_code == 202
    return dep_id


def test_a_hand_started_container_is_reported() -> None:
    """The case of something in the namespace the coordinator never recorded."""
    agent = FakeNodeAgent()
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    deployment_id = _deploy(client, _node(client), "qwen36-27b", 8000)

    agent.add_unmanaged_container("tensorstead-01HANDSTARTED")

    body = client.get(f"/v1/deployments/{deployment_id}/status", headers=_AUTH).json()

    assert "unexpected_instance" in _kinds(body)
    unexpected = next(d for d in body["divergences"] if d["kind"] == "unexpected_instance")
    assert "tensorstead-01HANDSTARTED" in str(unexpected["observed"]), (
        "the divergence must name the container an operator has to go and look at"
    )


def test_a_second_legitimate_deployment_is_not_an_unexpected_instance() -> None:
    """The false alarm the old check generated on any multi-deployment node.

    Two recorded deployments on one node is an ordinary arrangement. The
    previous implementation reported each as evidence of the other being
    unexpected, which would have taught an operator to ignore the field
    entirely — the failure mode that makes a real alarm useless.
    """
    agent = FakeNodeAgent()
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    node_id = _node(client)
    first = _deploy(client, node_id, "qwen36-27b", 8000)
    _deploy(client, node_id, "qwen36-35b-a3b", 8001)

    body = client.get(f"/v1/deployments/{first}/status", headers=_AUTH).json()

    assert "unexpected_instance" not in _kinds(body), (
        "a second recorded deployment on the same node was reported as unexpected"
    )


def test_a_stopped_deployments_sibling_is_still_not_unexpected() -> None:
    """The exact shape of the old false positive.

    The stub only looked when the observed deployment's own container was
    absent — so stopping one of two deployments made the surviving, perfectly
    legitimate one look like an intrusion.
    """
    agent = FakeNodeAgent()
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    node_id = _node(client)
    first = _deploy(client, node_id, "qwen36-27b", 8000)
    _deploy(client, node_id, "qwen36-35b-a3b", 8001)

    client.post(f"/v1/deployments/{first}:stop", headers=_AUTH)

    body = client.get(f"/v1/deployments/{first}/status", headers=_AUTH).json()

    assert "unexpected_instance" not in _kinds(body)


def test_an_agent_that_cannot_enumerate_makes_no_claim() -> None:
    """Silence from an old agent is not a clean bill of health (contract < 1.3).

    An agent predating this field omits it. Reading that as an empty namespace
    would report every node as verified-clean on the strength of a question it
    was never asked — the 009 failure mode in a new place.
    """
    agent = FakeNodeAgent()
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    deployment_id = _deploy(client, _node(client), "qwen36-27b", 8000)

    agent.enumerates_containers = False

    body = client.get(f"/v1/deployments/{deployment_id}/status", headers=_AUTH).json()

    assert "unexpected_instance" not in _kinds(body)

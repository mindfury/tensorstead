"""A deployment that is up but not serving is a divergence.

The 2026-08-10 incident's control-plane report, verbatim:

    desired_state: running
    observed.status: running
    observed.endpoint_reachable: true
    divergences: []

— issued while every client request returned HTTP 500 and every direct endpoint
connection was reset. The empty divergence list was not bad luck. Every check
that could have fired was guarded on a field the agent never populated
truthfully, so ``[]`` was the only reachable answer.

These tests drive the coordinator through the fake agent and assert that the
state the incident was actually in now produces a named divergence.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.fakes.node_agent import FakeNodeAgent
from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.integration

_AUTH = {"Authorization": "Bearer test"}


def _running_deployment(client: TestClient) -> tuple[str, str]:
    """Create, then start, a single-node deployment. Returns (deployment_id, node_id)."""
    node_id = client.post(
        "/v1/nodes",
        json={"name": "spark-alpha", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    ).json()["id"]

    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "nvidia/Qwen3.6-27B-NVFP4",
            "revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
            "nodes": [node_id],
        },
        headers=_AUTH,
    )
    # Acquisition runs on a background thread; poll to terminal before reading the model list.
    poll_operation(client, resp.json()["operation_id"])
    models = client.get("/v1/models", headers=_AUTH).json()
    model_id = next(m["id"] for m in models)

    create = client.post(
        "/v1/deployments",
        json={
            "name": "qwen36-27b",
            "model_id": model_id,
            "runtime_type": "vllm",
            "runtime_version": "0.6.0",
            "image_reference": "local/vllm:26.07-xgrammar-0.2.1",
            "runtime_config": {"tensor_parallel_size": 1},
            "participating_nodes": [node_id],
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )
    assert create.status_code == 202, create.text
    deployments = client.get("/v1/deployments", headers=_AUTH).json()
    dep_id = next(d["declared"]["id"] for d in deployments if d["declared"]["name"] == "qwen36-27b")

    assert client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH).status_code == 202
    return dep_id, node_id


def test_healthy_deployment_reports_no_divergence() -> None:
    """The control: a genuinely serving deployment still reports clean."""
    agent = FakeNodeAgent()
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    dep_id, _ = _running_deployment(client)

    body = client.get(f"/v1/deployments/{dep_id}", headers=_AUTH).json()

    assert body["observed"]["status"] == "running"
    assert body["divergences"] == []


def test_listener_up_but_runtime_not_serving_is_a_divergence() -> None:
    """The incident state produces ``inference_not_ready``, not silence.

    Reachable transport, dead runtime — which is what "connection reset by
    peer" on an endpoint whose container is still up looks like from the
    coordinator.
    """
    agent = FakeNodeAgent()
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    dep_id, _node_id = _running_deployment(client)

    deployment = agent.deployments[dep_id]
    deployment.endpoint_reachable = True
    deployment.inference_ready = False

    body = client.get(f"/v1/deployments/{dep_id}", headers=_AUTH).json()

    # Declared state is still returned in full alongside the divergence.
    assert body["declared"]["name"] == "qwen36-27b"
    assert body["observed"]["inference_ready"] is False

    kinds = [d["kind"] for d in body["divergences"]]
    assert "inference_not_ready" in kinds, (
        f"a deployment that is up but not serving reported divergences={kinds!r}; "
        "this is the state the 2026-08-10 incident reported as healthy"
    )


def test_unknown_readiness_is_not_reported_as_a_divergence() -> None:
    """``None`` readiness raises nothing — we did not observe a problem.

    Trading the old false negative for a false positive would be no better. A
    runtime whose adapter declares no probe, or one the agent cannot
    authenticate to, is not thereby broken.
    """
    agent = FakeNodeAgent()
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    dep_id, _ = _running_deployment(client)

    agent.deployments[dep_id].inference_ready = None

    body = client.get(f"/v1/deployments/{dep_id}", headers=_AUTH).json()

    kinds = [d["kind"] for d in body["divergences"]]
    assert "inference_not_ready" not in kinds, (
        "unknown readiness was reported as a divergence; None means we could "
        "not establish the fact, not that the runtime is broken"
    )

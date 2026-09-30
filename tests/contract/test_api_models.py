"""Contract test model acquisition.

- ``POST /v1/models:acquire`` returns a 202 operation id and records
  the resolved revision.
- A staging or failed replica is never presented as available.
- ``GET /v1/models`` reports identity, source, and resolved revision.
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


def _register_node(client: TestClient, name: str = "spark-01") -> str:
    resp = client.post(
        "/v1/nodes",
        json={"name": name, "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    assert resp.status_code == 201
    return str(resp.json()["id"])


def test_acquire_returns_operation_and_records_revision(client: TestClient) -> None:
    """Acquire returns a 202 operation id and records the resolved revision."""
    node_id = _register_node(client)
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
    assert "operation_id" in resp.json()
    # The route returns 202 immediately and runs on a background thread, so the
    # model row is written asynchronously — poll the operation to terminal
    # before reading the model list.
    poll_operation(client, resp.json()["operation_id"])

    # The model is listed with the resolved revision recorded.
    models = client.get("/v1/models", headers=_AUTH).json()
    assert models
    model = next(m for m in models if m["source_model_id"] == "org/model")
    assert model["resolved_revision"] == "e1f2a3b"
    assert model["revision_pinned"] is True


def test_available_replica_is_recorded(client: TestClient) -> None:
    """After a successful acquire the node holds an available replica."""
    node_id = _register_node(client)
    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "org/model",
            "nodes": [node_id],
        },
        headers=_AUTH,
    )
    poll_operation(client, resp.json()["operation_id"])
    models = client.get("/v1/models", headers=_AUTH).json()
    model = next(m for m in models if m["source_model_id"] == "org/model")
    replicas = model["replicas"]
    assert replicas, "a replica should be recorded"
    assert all(r["state"] == "available" for r in replicas), (
        "no staging/failed replica may be presented as available"
    )
    assert any(r["node_id"] == node_id for r in replicas)


def test_default_branch_is_recorded_as_the_agent_resolved_revision(client: TestClient) -> None:
    """A moving upstream label must never be stored as a pinned model identity."""
    node_id = _register_node(client)
    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "org/model",
            "nodes": [node_id],
        },
        headers=_AUTH,
    )
    poll_operation(client, resp.json()["operation_id"])

    model = next(
        item
        for item in client.get("/v1/models", headers=_AUTH).json()
        if item["source_model_id"] == "org/model"
    )
    assert model["resolved_revision"] != "main"
    assert model["revision_pinned"] is True


def test_acquire_reuses_verified_replica(client: TestClient) -> None:
    """Re-acquiring at the same revision transfers nothing.

    The fake reports the replica is already available, so the coordinator does
    not re-acquire from upstream.
    """
    node_id = _register_node(client)
    for _ in range(2):
        resp = client.post(
            "/v1/models:acquire",
            json={
                "source_id": "huggingface",
                "source_model_id": "org/model",
                "revision": "main",
                "nodes": [node_id],
            },
            headers=_AUTH,
        )
        poll_operation(client, resp.json()["operation_id"])
    models = client.get("/v1/models", headers=_AUTH).json()
    model = next(m for m in models if m["source_model_id"] == "org/model")
    # Exactly one available replica, not a duplicate.
    assert len(model["replicas"]) == 1


def test_acquire_requires_auth(client: TestClient) -> None:
    node_id = _register_node(client)
    resp = client.post(
        "/v1/models:acquire",
        json={"source_id": "huggingface", "source_model_id": "org/model", "nodes": [node_id]},
    )
    assert resp.status_code == 401


def test_acquire_operation_polls_to_terminal(client: TestClient) -> None:
    """The 202 returns immediately; the operation reaches ``succeeded``.

    This is the fix for the incident's upstream cause: ``model_acquire`` used to
    block until the download finished, so any pull longer than the MCP client's
    60s read timeout returned ``coordinator_unreachable`` — indistinguishable
    from a dead coordinator, which is what made a reasonable retry produce the
    duplicate acquire. The route now returns 202 at once and runs on a background
    thread; the operation's terminal state is what a client polls for.
    """
    node_id = _register_node(client)
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
    operation_id = resp.json()["operation_id"]

    op = poll_operation(client, operation_id)
    assert op["state"] == "succeeded"
    # The per-node outcome records which node the work completed on.
    assert op["per_node_outcomes"][node_id]["state"] == "succeeded"


def test_acquire_failure_recorded_on_agent_error(client: TestClient) -> None:
    """A failed acquire records the reason on the operation, not as a 5xx.

    The route has already returned 202, so a failure that surfaces on the background
    thread cannot reach the client as an HTTP error — it must be recorded on the
    operation or it hangs in ``running`` forever. A gated model
    produces an ``authorization_refused`` whose structured detail survives into
    ``failure_reason``, so an operator polling the operation sees the same
    actionable refusal the synchronous 403 used to carry.
    """
    from tests.fakes.node_agent import FakeNodeAgent

    # Gate the model so the upstream refuses with no credential.
    agent: FakeNodeAgent = client.app.state.fake_agent  # type: ignore[attr-defined]
    agent.gate_model("org/gated-model")

    node_id = _register_node(client)
    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "org/gated-model",
            "revision": "e1f2a3b",
            "nodes": [node_id],
        },
        headers=_AUTH,
    )
    assert resp.status_code == 202
    op = poll_operation(client, resp.json()["operation_id"])
    assert op["state"] == "failed"
    assert op["failure_reason"]["code"] == "authorization_refused"
    assert "huggingface" in op["failure_reason"]["message"]

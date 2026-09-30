"""Integration test the gated-model path.

The independent test, end to end through the coordinator API: attempt a
gated model with no credential and expect a refusal that **names the source and
the nature of the refusal**; supply a credential once; retry and
succeed.

Two things this test is careful about:

- The refusal must be *terminal and legible*, not a generic 500. An operator
  reading it should know which upstream refused and why, without a log dive.
- Unaccepted access terms are reported as something to settle with the provider.
  The product never accepts them and never routes around them — the
  test asserts the acquisition stays refused, because "we tried harder" would be
  precisely the wrong behaviour.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.fakes.node_agent import FakeNodeAgent
from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.integration

_AUTH = {"Authorization": "Bearer test"}
_SECRET = "hf_gated_model_token_9f3c1a"


@pytest.fixture
def client() -> TestClient:
    app, _ = build_test_coordinator()
    return TestClient(app)


def _agent(client: TestClient) -> FakeNodeAgent:
    """The fake node agent behind this coordinator (tests/helpers.py)."""
    agent: FakeNodeAgent = client.app.state.fake_agent  # type: ignore[attr-defined]
    return agent


def _register_node(client: TestClient) -> str:
    resp = client.post(
        "/v1/nodes",
        json={"name": "spark-01", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    assert resp.status_code == 201
    return str(resp.json()["id"])


def _acquire(client: TestClient, node_id: str, *, credential: str | None = None) -> Any:
    payload: dict = {
        "source_id": "huggingface",
        "source_model_id": "org/gated-model",
        "revision": "e1f2a3b",
        "nodes": [node_id],
    }
    if credential is not None:
        payload["credential"] = credential
    return client.post("/v1/models:acquire", json=payload, headers=_AUTH)


def test_us5_gated_model_refused_then_acquired(client: TestClient) -> None:
    """No credential → named refusal; supply one → acquisition succeeds.

    ``model_acquire`` returns 202 immediately and runs on a background
    thread, the refusal is no longer a synchronous 403 — it is the terminal
    state of the operation the route accepted. The structured refusal (which
    source, which model, why) is preserved verbatim in the operation's
    ``failure_reason``, so an operator polling the operation sees the same
    actionable detail the 403 body used to carry.
    """
    agent = _agent(client)
    agent.gate_model("org/gated-model")
    node_id = _register_node(client)

    # 1. No credential: accepted (202), then the operation fails with a named
    #    refusal that says which source and why.
    refused = _acquire(client, node_id)
    assert refused.status_code == 202
    op = poll_operation(client, refused.json()["operation_id"])
    assert op["state"] == "failed"
    reason = op["failure_reason"]
    assert reason["code"] == "authorization_refused"
    assert "huggingface" in reason["message"]
    assert "org/gated-model" in reason["message"]
    assert reason["detail"]["source_id"] == "huggingface"

    # A refused acquisition leaves no available replica.
    models = client.get("/v1/models", headers=_AUTH).json()
    for model in models:
        assert all(r["state"] != "available" for r in model["replicas"])

    # 2. Supply the credential once.
    stored = client.put(
        "/v1/credentials/huggingface/personal",
        json={"secret": _SECRET, "default": True},
        headers=_AUTH,
    )
    assert stored.status_code == 204

    # 3. Retry: it succeeds, using the source default without being named.
    accepted = _acquire(client, node_id)
    assert accepted.status_code == 202
    poll_operation(client, accepted.json()["operation_id"])

    models = client.get("/v1/models", headers=_AUTH).json()
    gated = next(m for m in models if m["source_model_id"] == "org/gated-model")
    assert any(r["state"] == "available" for r in gated["replicas"])

    # The value reached the agent per-request — proving the
    # coordinator resolved the reference rather than passing the name through.
    assert agent.credentials_seen[-1] == _SECRET


def test_us5_named_credential_is_used_over_the_default(client: TestClient) -> None:
    """Naming a credential selects it; naming is optional and additive."""
    agent = _agent(client)
    agent.gate_model("org/gated-model")
    node_id = _register_node(client)

    client.put(
        "/v1/credentials/huggingface/personal",
        json={"secret": "hf_personal_token"},
        headers=_AUTH,
    )
    client.put(
        "/v1/credentials/huggingface/org",
        json={"secret": "hf_org_token"},
        headers=_AUTH,
    )

    accepted = _acquire(client, node_id, credential="org")
    assert accepted.status_code == 202
    poll_operation(client, accepted.json()["operation_id"])
    assert agent.credentials_seen[-1] == "hf_org_token"


def test_us5_naming_an_unknown_credential_is_reported(client: TestClient) -> None:
    """A named credential that does not exist is an error, not a silent fallback.

    The operation is accepted (202) and runs to ``failed`` with ``not_found``,
    naming the credential — the same legible refusal the synchronous 404 used
    to carry, now the terminal state of the accepted operation.
    """
    node_id = _register_node(client)
    client.put(
        "/v1/credentials/huggingface/personal",
        json={"secret": _SECRET},
        headers=_AUTH,
    )

    resp = _acquire(client, node_id, credential="nonexistent")
    assert resp.status_code == 202
    op = poll_operation(client, resp.json()["operation_id"])
    assert op["state"] == "failed"
    assert op["failure_reason"]["code"] == "not_found"
    assert "nonexistent" in op["failure_reason"]["message"]


def test_us5_unaccepted_access_terms_are_reported_never_accepted(client: TestClient) -> None:
    """Access terms are settled with the provider, never by us."""
    agent = _agent(client)
    agent.gate_model("org/gated-model", requires_access_terms=True)
    node_id = _register_node(client)

    client.put(
        "/v1/credentials/huggingface/personal",
        json={"secret": _SECRET},
        headers=_AUTH,
    )

    # A valid credential is not enough: the terms are the operator's to accept.
    # The operation is accepted (202) and fails with the named refusal.
    resp = _acquire(client, node_id)
    assert resp.status_code == 202
    op = poll_operation(client, resp.json()["operation_id"])
    assert op["state"] == "failed"
    reason = op["failure_reason"]
    assert reason["code"] == "authorization_refused"
    assert reason["detail"]["reason"] == "access_terms_not_accepted"
    assert "huggingface" in reason["message"]
    # The message directs the operator to the provider rather than offering a
    # way around it.
    assert "provider" in reason["message"].lower()

    # Nothing was accepted or bypassed on the operator's behalf.
    assert agent.access_terms_accepted == []
    models = client.get("/v1/models", headers=_AUTH).json()
    for model in models:
        assert all(r["state"] != "available" for r in model["replicas"])


def test_us5_credential_list_shows_names_and_default_only(client: TestClient) -> None:
    """``GET /v1/credentials`` returns references, never values."""
    client.put(
        "/v1/credentials/huggingface/personal",
        json={"secret": _SECRET, "default": True},
        headers=_AUTH,
    )
    client.put(
        "/v1/credentials/huggingface/org",
        json={"secret": "hf_org_token"},
        headers=_AUTH,
    )

    resp = client.get("/v1/credentials", headers=_AUTH)
    assert resp.status_code == 200
    listed = resp.json()
    assert {(c["source_id"], c["name"], c["is_default"]) for c in listed} == {
        ("huggingface", "personal", True),
        ("huggingface", "org", False),
    }
    assert _SECRET not in resp.text


def test_us5_deleting_the_default_reports_the_consequence(client: TestClient) -> None:
    """The default is never silently reassigned."""
    client.put(
        "/v1/credentials/huggingface/personal",
        json={"secret": _SECRET, "default": True},
        headers=_AUTH,
    )
    client.put(
        "/v1/credentials/huggingface/org",
        json={"secret": "hf_org_token"},
        headers=_AUTH,
    )

    resp = client.request("DELETE", "/v1/credentials/huggingface/personal", headers=_AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["was_default"] is True
    assert body["default_now"] is None

    remaining = client.get("/v1/credentials", headers=_AUTH).json()
    assert [c["is_default"] for c in remaining] == [False]


def test_us5_credentials_require_the_management_token(client: TestClient) -> None:
    """Every credential route is management-token gated."""
    assert client.get("/v1/credentials").status_code == 401
    assert (
        client.put("/v1/credentials/huggingface/personal", json={"secret": _SECRET}).status_code
        == 401
    )
    assert client.request("DELETE", "/v1/credentials/huggingface/personal").status_code == 401

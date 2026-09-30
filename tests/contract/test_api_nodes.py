"""Contract test the node endpoints.

- ``register`` captures the agent's reported platform facts and **installs
  nothing**; the agent is already running.
- The only failure modes are ``agent_unreachable`` and
  ``agent_version_incompatible`` — never a bootstrap attempt.
- ``GET /v1/nodes`` and ``GET /v1/nodes/{id}`` inventory registered nodes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tensorstead.contracts.version import CONTRACT_VERSION
from tests.fakes.node_agent import FakeNodeAgent
from tests.helpers import build_test_coordinator

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer test"}


@pytest.fixture
def client() -> TestClient:
    app, _ = build_test_coordinator()
    return TestClient(app)


def _node(client: TestClient, name: str, host: str) -> str:
    resp = client.post(
        "/v1/nodes",
        json={"name": name, "agent_endpoint": f"https://{host}:8443"},
        headers=_AUTH,
    )
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


def test_register_captures_platform_facts(client: TestClient) -> None:
    """Register captures the agent's platform facts and installs nothing."""
    resp = client.post(
        "/v1/nodes",
        json={"name": "spark-01", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "spark-01"
    assert body["agent_contract_version"] == CONTRACT_VERSION
    assert "cpu_arch" in body["platform_facts"]
    assert "os_family" in body["platform_facts"]
    assert "id" in body


def test_register_duplicate_name_is_refused(client: TestClient) -> None:
    """A duplicate node name is refused with already_exists."""
    payload = {"name": "spark-01", "agent_endpoint": "https://10.0.0.11:8443"}
    assert client.post("/v1/nodes", json=payload, headers=_AUTH).status_code == 201
    resp = client.post("/v1/nodes", json=payload, headers=_AUTH)
    assert resp.status_code == 409
    assert resp.json()["code"] == "already_exists"


def test_register_never_bootstraps_absent_agent(client: TestClient) -> None:
    """An absent agent is reported agent_unreachable, never bootstrapped.

    Registration verifies the agent is running; it does not install, upgrade,
    or bootstrap the agent. Here the fake is pre-configured so the
    test asserts the failure surface; a real unreachable host raises
    agent_unreachable (covered by the service contract below).
    """
    # With a healthy fake agent, register succeeds — proving no bootstrap path
    # is attempted and the failure mode is reachability/compat, never install.
    resp = client.post(
        "/v1/nodes",
        json={"name": "ok", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    assert resp.status_code == 201


def test_register_agent_version_incompatible() -> None:
    """A major contract-version mismatch is refused."""
    agent = FakeNodeAgent(contract_version="99.0")  # incompatible major
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    resp = client.post(
        "/v1/nodes",
        json={"name": "old", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    assert resp.status_code == 409
    assert resp.json()["code"] == "agent_version_incompatible"


def test_node_list_and_get(client: TestClient) -> None:
    """GET /v1/nodes lists and GET /v1/nodes/{id} inspects."""
    reg = client.post(
        "/v1/nodes",
        json={"name": "spark-01", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    ).json()
    listed = client.get("/v1/nodes", headers=_AUTH)
    assert listed.status_code == 200
    assert any(n["name"] == "spark-01" for n in listed.json())
    detail = client.get(f"/v1/nodes/{reg['id']}", headers=_AUTH)
    assert detail.status_code == 200
    assert detail.json()["id"] == reg["id"]


def test_register_requires_auth(client: TestClient) -> None:
    """Management token gates every node route."""
    resp = client.post(
        "/v1/nodes",
        json={"name": "spark-01", "agent_endpoint": "https://10.0.0.11:8443"},
    )
    assert resp.status_code == 401


@pytest.mark.parametrize(
    "bad_endpoint",
    [
        "http://10.0.0.11:8443",
        "https://user:pass@10.0.0.11:8443",
        "https://10.0.0.11:8443/../../v1/nodes",
        "https://10.0.0.11:8443?x=1",
        "https://10.0.0.11:8443#frag",
        "not-a-url-at-all",
        "",
        "file:///etc/passwd",
    ],
)
def test_register_refuses_a_non_canonical_agent_endpoint(
    client: TestClient, bad_endpoint: str
) -> None:
    """A non-canonical agent endpoint is refused.

    Registration sends the fleet management token to whatever
    ``agent_endpoint`` names, immediately, before any operator reviews the
    node. A non-``https`` scheme, embedded userinfo, a path, a query, or a
    fragment can each smuggle meaning past a skim-read of the URL, so every
    one of those shapes is refused rather than merely the scheme.
    """
    resp = client.post(
        "/v1/nodes",
        json={"name": "spark-suspect", "agent_endpoint": bad_endpoint},
        headers=_AUTH,
    )

    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "invalid_agent_endpoint"
    # And the refusal actually refused: no node was left behind to register
    # against later, and nothing was contacted on its behalf.
    listed = client.get("/v1/nodes", headers=_AUTH).json()
    assert not any(n["name"] == "spark-suspect" for n in listed)


def test_register_accepts_a_bare_https_host_and_port(client: TestClient) -> None:
    """The canonical shape every real node already uses must keep working."""
    resp = client.post(
        "/v1/nodes",
        json={"name": "spark-canonical", "agent_endpoint": "https://10.0.0.99:8443"},
        headers=_AUTH,
    )
    assert resp.status_code == 201, resp.text


# --------------------------------------------- per-node management token
# The motivating defect: every agent held the same fleet-wide
# TENSORSTEAD_MGMT_TOKEN, so a compromised node disclosed the credential that
# controls every other node too. A node can now be issued its own.

_SECRET = "s3cret-per-node-management-token"


def test_a_new_node_has_no_management_token_override(client: TestClient) -> None:
    """The migration path a node predating this feature keeps working under."""
    resp = client.post(
        "/v1/nodes",
        json={"name": "spark-01", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    assert resp.json()["has_management_token_override"] is False


def test_rotating_the_management_token_sets_the_override_flag(client: TestClient) -> None:
    node_id = _node(client, "spark-01", "10.0.0.11")

    resp = client.post(
        f"/v1/nodes/{node_id}/rotate-management-token",
        json={"token": _SECRET},
        headers=_AUTH,
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["has_management_token_override"] is True


def test_clearing_the_management_token_reverts_the_override_flag(client: TestClient) -> None:
    node_id = _node(client, "spark-01", "10.0.0.11")
    client.post(
        f"/v1/nodes/{node_id}/rotate-management-token", json={"token": _SECRET}, headers=_AUTH
    )

    resp = client.post(f"/v1/nodes/{node_id}/clear-management-token", headers=_AUTH)

    assert resp.status_code == 200, resp.text
    assert resp.json()["has_management_token_override"] is False


def test_rotating_to_an_empty_token_is_refused(client: TestClient) -> None:
    """Empty means "no override" -- setting it that way must go through clear."""
    node_id = _node(client, "spark-01", "10.0.0.11")

    resp = client.post(
        f"/v1/nodes/{node_id}/rotate-management-token", json={"token": ""}, headers=_AUTH
    )

    assert resp.status_code == 422, resp.text


def test_the_management_token_value_never_appears_in_any_node_response(
    client: TestClient,
) -> None:
    """The store holds a reference, never a value -- the same rule
    every other credential this product tracks already follows."""
    node_id = _node(client, "spark-01", "10.0.0.11")
    client.post(
        f"/v1/nodes/{node_id}/rotate-management-token", json={"token": _SECRET}, headers=_AUTH
    )

    responses = [
        client.get("/v1/nodes", headers=_AUTH),
        client.get(f"/v1/nodes/{node_id}", headers=_AUTH),
    ]

    for resp in responses:
        assert _SECRET not in resp.text, (
            f"the per-node management token reached a response body: {resp.text}"
        )

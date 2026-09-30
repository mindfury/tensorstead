"""Contract tests for coordinator-to-agent TLS verification and auth."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

import tensorstead.coordinator.app as coordinator_app_module
from tensorstead.adapters.credentials.local_file import LocalFileCredentialProvider
from tensorstead.coordinator.node_http import NodeHTTPClient
from tensorstead.domain.identity import new_ulid
from tensorstead.domain.models import Node

pytestmark = pytest.mark.contract


def _node(*, agent_management_token_ref: str = "") -> Node:
    return Node(
        id=new_ulid(),
        name="spark-01",
        agent_endpoint="https://10.0.0.11:8443",
        agent_contract_version="1.0",
        agent_cert_fingerprint="",
        platform_facts={},
        registered_at=datetime.now().astimezone(),
        agent_management_token_ref=agent_management_token_ref,
    )


def test_node_client_verifies_tls_by_default() -> None:
    assert NodeHTTPClient(management_token="test")._verify is True


def test_coordinator_uses_ca_bundle_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The configured private CA is passed to the node transport unchanged."""
    captured: dict[str, Any] = {}

    class StubNodeClient:
        def __init__(self, *, management_token: str, verify: bool | str, **_kwargs: Any) -> None:
            captured["management_token"] = management_token
            captured["verify"] = verify

    monkeypatch.setenv("TENSORSTEAD_MGMT_TOKEN", "test-token")
    monkeypatch.setenv("TENSORSTEAD_AGENT_CA_BUNDLE", "/etc/tensorstead/agents-ca.pem")
    monkeypatch.setattr("tensorstead.coordinator.node_http.NodeHTTPClient", StubNodeClient)

    app = coordinator_app_module.build_coordinator_app()

    assert captured == {
        "management_token": "test-token",
        "verify": "/etc/tensorstead/agents-ca.pem",
    }
    assert isinstance(app.state.node_client, StubNodeClient)


# ------------------------------------------------- per-node management token
# Every agent held the same fleet-wide
# TENSORSTEAD_MGMT_TOKEN, so a compromised node disclosed the credential that
# controls every other node too. These test the actual mechanism that
# closes it: which bearer value NodeHTTPClient presents to a given node.


def test_a_node_with_no_override_gets_the_fleet_wide_token() -> None:
    """The migration path: unset means every existing node is unaffected."""
    client = NodeHTTPClient(management_token="fleet-wide-secret")

    headers = client._headers(_node())

    assert headers["Authorization"] == "Bearer fleet-wide-secret"


def test_a_node_with_an_override_gets_its_own_token_instead(tmp_path: Path) -> None:
    """The actual fix: a node issued its own credential gets presented with it."""
    provider = LocalFileCredentialProvider(root=tmp_path / "credentials")
    ref = provider.store("node-management-token", "spark-01-node-id", "this-nodes-own-secret")
    client = NodeHTTPClient(management_token="fleet-wide-secret", credential_provider=provider)

    headers = client._headers(_node(agent_management_token_ref=ref))

    assert headers["Authorization"] == "Bearer this-nodes-own-secret"


def test_other_nodes_are_unaffected_by_one_nodes_override(tmp_path: Path) -> None:
    """The property that actually matters: nodes do not share a token anymore."""
    provider = LocalFileCredentialProvider(root=tmp_path / "credentials")
    ref = provider.store("node-management-token", "spark-01-node-id", "spark-01-only-secret")
    client = NodeHTTPClient(management_token="fleet-wide-secret", credential_provider=provider)

    overridden = client._headers(_node(agent_management_token_ref=ref))
    plain = client._headers(_node())

    assert overridden["Authorization"] == "Bearer spark-01-only-secret"
    assert plain["Authorization"] == "Bearer fleet-wide-secret"


def test_an_override_ref_with_no_provider_configured_falls_back_safely() -> None:
    """A node record with a ref but a client built without a provider must
    not crash or silently send an unresolved reference as a bearer token."""
    client = NodeHTTPClient(management_token="fleet-wide-secret")

    headers = client._headers(_node(agent_management_token_ref="some-ref-the-client-cannot-read"))

    assert headers["Authorization"] == "Bearer fleet-wide-secret"

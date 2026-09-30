"""The inference-credential surface (Phase 4).

The service, storage, and delivery were built and deployed before any surface
existed, so the feature was unreachable by an operator. These exercise the API
the way a client does, and pin the properties that must survive it becoming
reachable: no read path, and no value in the store.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.helpers import build_test_coordinator

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer test"}
_SECRET = "sk-inference-do-not-leak"


@pytest.fixture
def client() -> TestClient:
    app, _ = build_test_coordinator()
    return TestClient(app)


def _store(client: TestClient, name: str, tmp_path: Path) -> None:
    secret_file = tmp_path / f"{name}.secret"
    secret_file.write_text(_SECRET, encoding="utf-8")
    resp = client.put(
        f"/v1/inference-credentials/{name}",
        json={"from_file": str(secret_file)},
        headers=_AUTH,
    )
    assert resp.status_code == 200, resp.text


def test_a_stored_credential_is_listed_by_name_only(client: TestClient, tmp_path: Path) -> None:
    """No interface returns a stored value, for either credential kind."""
    _store(client, "prod", tmp_path)

    listed = client.get("/v1/inference-credentials", headers=_AUTH)

    assert listed.status_code == 200
    assert [c["name"] for c in listed.json()] == ["prod"]
    assert _SECRET not in listed.text, "listing must never carry the value"
    assert "secret_ref" not in listed.text, "not even the reference"


def test_the_value_never_appears_in_any_response(client: TestClient, tmp_path: Path) -> None:
    """The success criterion, checked across the surface rather than only at the store."""
    _store(client, "prod", tmp_path)

    for method, path in [
        ("get", "/v1/inference-credentials"),
        ("get", "/v1/nodes"),
        ("get", "/v1/deployments"),
        ("get", "/v1/operations"),
    ]:
        response = getattr(client, method)(path, headers=_AUTH)
        assert _SECRET not in response.text, f"{path} leaked the credential"


def test_binding_requires_a_stored_credential(client: TestClient) -> None:
    """A dangling binding would be a reference to something that does not exist."""
    resp = client.put(
        "/v1/deployments/does-not-matter/inference-credential",
        json={"name": "absent"},
        headers=_AUTH,
    )

    assert resp.status_code == 404
    assert "absent" in resp.text


def test_clearing_a_binding_is_permitted(client: TestClient) -> None:
    """Clearing returns the deployment to the node's provisioning."""
    resp = client.put("/v1/deployments/d1/inference-credential", json={"name": None}, headers=_AUTH)

    assert resp.status_code == 200
    assert resp.json()["inference_credential"] is None


def test_deleting_an_absent_credential_is_a_clean_refusal(client: TestClient) -> None:
    resp = client.delete("/v1/inference-credentials/absent", headers=_AUTH)

    assert resp.status_code == 404


def test_delete_removes_a_credential_nothing_binds(client: TestClient, tmp_path: Path) -> None:
    _store(client, "prod", tmp_path)

    assert client.delete("/v1/inference-credentials/prod", headers=_AUTH).status_code == 200
    assert client.get("/v1/inference-credentials", headers=_AUTH).json() == []


def test_every_route_requires_authentication(client: TestClient) -> None:
    for method, path in [
        ("get", "/v1/inference-credentials"),
        ("put", "/v1/inference-credentials/x"),
        ("delete", "/v1/inference-credentials/x"),
        ("put", "/v1/deployments/d1/inference-credential"),
    ]:
        call = getattr(client, method)
        response = call(path) if method in ("get", "delete") else call(path, json={})
        assert response.status_code in (401, 403), f"{method} {path} was not gated"

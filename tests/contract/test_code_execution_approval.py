"""Authorizing ``trust_remote_code`` for one exact tuple, and nothing else.

NVIDIA's published deployment guidance for ``Qwen3.8-Flash-Next-NVFP4`` includes
``--trust-remote-code``. The authorization policy refused it outright, so the
product could not deploy a model its own estate had already acquired and built a
runtime for -- and the operator's remaining route was a hand-started container,
which is the drift the management plane exists to prevent. A safety feature that
manufactures the failure it guards against is not one.

The refusal's own wording named the way out: *what executes on an appliance is
not something a deployment record may decide **on its own***. So a second
authority decides with it -- a reviewed, immutable approval naming one exact
tuple, created through its own route.

What these tests hold to:

- an ordinary deployment is refused exactly as before;
- an approved exact tuple is accepted, and the argv proves the flag was passed;
- ``extra_args`` is not a second door;
- changing the model revision, the image digest, or the runtime loses the
  authorization -- not by revocation, but because the key stops fitting;
- the agent re-derives the tuple from its own disk rather than trusting the
  grant it was handed.

The scenario is the real one throughout: the checkpoint, revision, and imported
runtime image id from the qualification that prompted this.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tensorstead.domain.models import ImageOrigin, ImageRecord
from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer test"}

_MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"
_REVISION = "fab0aecb760cec45227f6656abcaafa11abca87a"
_IMAGE = "local/vllm-qwen38-flashnext:d4d703c"
_DIGEST = "sha256:02366b8f87b8490c137b49f91beb7046904b16c70b7af2408753854d94e69170"
_OTHER_DIGEST = "sha256:" + "b" * 64
_CONFIG = {"tensor_parallel_size": 2, "trust_remote_code": True}


def _estate(*, image_digest: str = _DIGEST) -> tuple[TestClient, str, str]:
    """A coordinator with the model acquired and the runtime image recorded.

    The image record matters: an approval binds to the digest of the runtime
    that will execute the code, so a deployment declaring a code-loading option
    is refused until the image is actually present on its nodes.
    """
    app, repo = build_test_coordinator()
    client = TestClient(app)
    node_id = client.post(
        "/v1/nodes",
        json={"name": "spark-alpha", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    ).json()["id"]

    acquire = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": _MODEL,
            "revision": _REVISION,
            "nodes": [node_id],
        },
        headers=_AUTH,
    )
    poll_operation(client, acquire.json()["operation_id"])
    model_id = next(m["id"] for m in client.get("/v1/models", headers=_AUTH).json())

    repo.save_image(
        ImageRecord(
            node_id=node_id,
            reference=_IMAGE,
            digest=image_digest,
            pulled_at=datetime.now().astimezone(),
            origin=ImageOrigin.IMPORTED,
        )
    )
    return client, node_id, model_id


def _approve(client: TestClient, **overrides: Any) -> Any:
    body = {
        "option": "trust_remote_code",
        "runtime_type": "vllm",
        "model_source_id": "huggingface",
        "source_model_id": _MODEL,
        "model_revision": _REVISION,
        "image_digest": _DIGEST,
        "reason": "NVIDIA deployment guidance for this checkpoint requires it",
        "approved_by": "operator",
    }
    body.update(overrides)
    return client.post("/v1/code-approvals", json=body, headers=_AUTH)


def _create(
    client: TestClient, node_id: str, model_id: str, *, config: dict[str, Any] | None = None
) -> Any:
    return client.post(
        "/v1/deployments",
        json={
            "name": "qwen38-flashnext",
            "model_id": model_id,
            "runtime_type": "vllm",
            "runtime_version": "0.28.0",
            "image_reference": _IMAGE,
            "runtime_config": _CONFIG if config is None else config,
            "participating_nodes": [node_id],
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )


# --------------------------------------------------------------- the refusal


def test_an_unapproved_deployment_is_still_refused() -> None:
    """The guard is unchanged for everyone who has not been through review."""
    client, node_id, model_id = _estate()

    resp = _create(client, node_id, model_id)

    assert resp.status_code >= 400, resp.text
    assert "trust_remote_code" in resp.text
    # The refusal must name the remedy. Without it an operator holding NVIDIA's
    # own command line learns only that the product disagrees with it, which is
    # how a safety feature becomes a hand-started container.
    assert "code_approval_create" in resp.text


def test_the_refusal_names_the_exact_tuple_that_would_authorize_it() -> None:
    """An operator should not have to guess what to approve."""
    client, node_id, model_id = _estate()

    body = _create(client, node_id, model_id).text

    for expected in (_MODEL, _REVISION, _DIGEST, "huggingface", "vllm"):
        assert expected in body, f"refusal did not name {expected!r}"


# -------------------------------------------------------------- the approval


def test_an_approved_tuple_may_declare_the_flag() -> None:
    client, node_id, model_id = _estate()
    assert _approve(client).status_code == 201

    resp = _create(client, node_id, model_id)

    assert resp.status_code == 202, resp.text


def test_the_revision_records_which_approval_authorized_it() -> None:
    """Provenance in the immutable history, not merely in the live table."""
    client, node_id, model_id = _estate()
    approval = _approve(client).json()
    _create(client, node_id, model_id)

    dep_id = client.get("/v1/deployments", headers=_AUTH).json()[0]["declared"]["id"]
    export = client.get(f"/v1/deployments/{dep_id}/export", headers=_AUTH).json()

    block = export["code_execution_approval"]
    assert block["fingerprint"] == approval["fingerprint"]
    assert block["approval_id"] == approval["id"]
    assert block["status"] == "active"
    assert block["approved_by"] == "operator"
    assert block["reason"]


def test_an_export_reports_an_approval_that_has_since_been_revoked() -> None:
    """History cannot change; whether it still authorizes anything can.

    An export exists to be carried to another estate. Discovering there that the
    definition will not start is discovering it too late.
    """
    client, node_id, model_id = _estate()
    approval = _approve(client).json()
    _create(client, node_id, model_id)
    client.delete(f"/v1/code-approvals/{approval['id']}", headers=_AUTH)

    dep_id = client.get("/v1/deployments", headers=_AUTH).json()[0]["declared"]["id"]
    block = client.get(f"/v1/deployments/{dep_id}/export", headers=_AUTH).json()[
        "code_execution_approval"
    ]

    assert block["status"] == "revoked"
    assert block["fingerprint"] == approval["fingerprint"], "history was rewritten"


def test_a_deployment_that_loads_no_code_gains_no_approval_key() -> None:
    """No existing export grows a line announcing a feature it does not use."""
    client, node_id, model_id = _estate()
    _create(client, node_id, model_id, config={"tensor_parallel_size": 2})

    dep_id = client.get("/v1/deployments", headers=_AUTH).json()[0]["declared"]["id"]
    export = client.get(f"/v1/deployments/{dep_id}/export", headers=_AUTH).json()

    assert "code_execution_approval" not in export


# ------------------------------------------------- the key stops fitting


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_model_id", "nvidia/Some-Other-Model"),
        ("model_revision", "a" * 40),
        ("image_digest", _OTHER_DIGEST),
        ("model_source_id", "local"),
    ],
)
def test_an_approval_for_a_different_tuple_does_not_authorize_this_one(
    field: str, value: str
) -> None:
    """Authorization is lost by the key ceasing to fit, not by anyone noticing."""
    client, node_id, model_id = _estate()
    assert _approve(client, **{field: value}).status_code == 201

    resp = _create(client, node_id, model_id)

    assert resp.status_code >= 400, f"an approval differing in {field} authorized this deployment"


def test_a_different_runtime_is_a_different_review() -> None:
    """Approving a flag on vLLM is not approving it on SGLang."""
    client, node_id, model_id = _estate()
    assert _approve(client, runtime_type="sglang").status_code == 201

    assert _create(client, node_id, model_id).status_code >= 400


def test_revoking_the_approval_refuses_the_next_modify() -> None:
    """Revocation is decided fresh, never inherited from the previous revision."""
    client, node_id, model_id = _estate()
    approval = _approve(client).json()
    _create(client, node_id, model_id)
    dep_id = client.get("/v1/deployments", headers=_AUTH).json()[0]["declared"]["id"]

    client.delete(f"/v1/code-approvals/{approval['id']}", headers=_AUTH)
    resp = client.post(
        f"/v1/deployments/{dep_id}:modify",
        json={"runtime_config": dict(_CONFIG, max_num_seqs=4)},
        headers=_AUTH,
    )

    assert resp.status_code >= 400, "a revoked approval was carried forward by modify"


def test_an_approval_cannot_name_a_moving_tag() -> None:
    """An approval is granted for code read at one commit.

    Refused at the approval, with a reason, rather than accepted and left to
    fail later as an unexplained fingerprint miss.
    """
    client, _, _ = _estate()

    resp = _approve(client, model_revision="main")

    assert resp.status_code >= 400, "an approval bound to a tag was accepted"
    assert "immutable" in resp.text
    # A domain invariant must not reach the operator as "Internal Server Error":
    # that names no field and sends them to the wrong half
    # of the system.
    assert resp.status_code != 500, resp.text


def test_an_absent_image_refuses_rather_than_binding_to_a_reference() -> None:
    """A tag is not a runtime; the approval binds to the digest that will run."""
    app, _repo = build_test_coordinator()
    client = TestClient(app)
    node_id = client.post(
        "/v1/nodes",
        json={"name": "spark-alpha", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    ).json()["id"]
    acquire = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": _MODEL,
            "revision": _REVISION,
            "nodes": [node_id],
        },
        headers=_AUTH,
    )
    poll_operation(client, acquire.json()["operation_id"])
    model_id = next(m["id"] for m in client.get("/v1/models", headers=_AUTH).json())
    _approve(client)

    resp = _create(client, node_id, model_id)

    assert resp.status_code >= 400
    assert "image" in resp.text.lower()


# ------------------------------------------------------- not a second door


def test_extra_args_is_not_a_bypass_even_with_an_approval() -> None:
    """The negative test this capability was required to carry.

    An approval authorizes a *modelled* option, whose value the product records
    and renders itself. The passthrough is unvalidated by construction, so an
    approvable option arriving through it is not an approval question at all --
    it is the bypass ``authorize_extra_arg`` already refuses by reserved name.
    Letting an approval reach it would turn one reviewed opening into two doors,
    one of which nobody reviewed.
    """
    client, node_id, model_id = _estate()
    assert _approve(client).status_code == 201

    for spelling in (
        {"trust-remote-code": True},
        {"trust_remote_code": True},
        {"trust-remote-code=true": True},
    ):
        resp = _create(
            client,
            node_id,
            model_id,
            config={"tensor_parallel_size": 2, "extra_args": spelling},
        )
        assert resp.status_code >= 400, f"extra_args {spelling} bypassed policy"


def test_only_loads_code_options_can_be_approved_at_all() -> None:
    """An approval that could never match is worse than a refusal.

    It reads to a later operator as a permission that exists.
    """
    for option in ("api_key", "model", "port", "headless"):
        resp = _approve(TestClient(build_test_coordinator()[0]), option=option)
        assert resp.status_code >= 400, f"{option} was accepted as approvable"


def test_re_approving_the_same_tuple_is_a_conflict_not_an_overwrite() -> None:
    """Deleting one of two rows authorizing the same thing revokes nothing."""
    client, _, _ = _estate()
    assert _approve(client).status_code == 201

    second = _approve(client, approved_by="someone-else", reason="different reason")

    assert second.status_code >= 400
    assert len(client.get("/v1/code-approvals", headers=_AUTH).json()) == 1

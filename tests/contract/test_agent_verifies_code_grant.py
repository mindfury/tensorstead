"""The agent checks the grant against its own disk, not against the request body.

A grant travelling in a request body is an assertion. What makes it evidence is
that the node re-derives every fact in it: the model source, id and revision
come from *this node's* replica marker, and the image digest from the image it
actually resolved. A grant that does not describe what is about to run is
refused here, however well-formed it looks.

That matters because the coordinator cannot do this part. It records
``image_digest=""`` and resolves nothing -- the digest is only a fact once the
agent has it -- so an approval bound to a runtime can only be checked where the
runtime is. Splitting the check is not redundancy; each half is the only place
its fact exists.

**The honest limit**, stated here because a test file is where a security claim
should be falsifiable: a caller holding this agent's management token can
already name any image and any model path, so it can construct a deployment that
satisfies this check. The token is the boundary and always was. What the
verification buys is that a *stale*, *replayed*, or *mismatched* approval is
refused rather than honoured -- which is the failure that actually happens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from tensorstead.agent.app import build_agent_app
from tensorstead.domain.approvals import fingerprint
from tests.fakes.container_engine import FakeContainerEngine
from tests.fakes.service_manager import FakeServiceManager

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer mgmt"}
_MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"
_REVISION = "fab0aecb760cec45227f6656abcaafa11abca87a"
_LOCAL_ID = f"huggingface:{_MODEL}"


@pytest.fixture
def agent(tmp_path: Path) -> TestClient:
    app = build_agent_app(
        management_token="mgmt",
        container_engine=FakeContainerEngine(),
        service_manager=FakeServiceManager(),
        store_dir=tmp_path / "store",
        marker_dir=tmp_path / "markers",
    )
    return TestClient(app)


@pytest.fixture
def model_path(tmp_path: Path) -> str:
    """A model tree the pre-flight accepts, with a replica marker beside it.

    The marker is the point: it is what the agent reads to learn which model and
    revision actually sit at this path.
    """
    d = tmp_path / "store" / "model"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"arch": "qwen3"}))
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}})
    )
    (d / "model-00001-of-00001.safetensors").write_bytes(b"shard")

    markers = tmp_path / "markers"
    markers.mkdir(parents=True, exist_ok=True)
    (markers / "model.json").write_text(
        json.dumps(
            {
                "model_id": _LOCAL_ID,
                "resolved_revision": _REVISION,
                "state": "available",
                "local_path": str(d),
                "verified_at": None,
                "size_bytes": 5,
                "content_digest": None,
            }
        )
    )
    return str(d)


def _digest_for(agent: TestClient, reference: str) -> str:
    """Whatever this node resolves the reference to — never a value we invent."""
    return str(
        agent.post("/agent/v1/images:pull", json={"reference": reference}, headers=_AUTH).json()[
            "digest"
        ]
    )


def _grant(digest: str, **overrides: Any) -> dict[str, Any]:
    grant = {
        "approval_id": "01J00000000000000000000001",
        "option": "trust_remote_code",
        "runtime_type": "vllm",
        "model_source_id": "huggingface",
        "source_model_id": _MODEL,
        "model_revision": _REVISION,
        "image_digest": digest,
        "policy_version": "2026-09-05.1",
    }
    grant.update(overrides)
    grant["fingerprint"] = fingerprint(
        option=str(grant["option"]),
        runtime_type=str(grant["runtime_type"]),
        model_source_id=str(grant["model_source_id"]),
        source_model_id=str(grant["source_model_id"]),
        model_revision=str(grant["model_revision"]),
        image_digest=str(grant["image_digest"]),
    )
    return grant


def _create(agent: TestClient, model_path: str, grant: dict[str, Any] | None) -> Any:
    body: dict[str, Any] = {
        "deployment_id": "01J00000000000000000000000",
        "revision": 1,
        "runtime_type": "vllm",
        "image_reference": "repo/vllm:tag",
        "runtime_config": {"tensor_parallel_size": 1, "trust_remote_code": True},
        "model_path": model_path,
        "endpoint": "0.0.0.0:8000",
    }
    if grant is not None:
        body["code_execution_grant"] = grant
    return agent.post("/agent/v1/deployments", json=body, headers=_AUTH)


def test_without_a_grant_the_agent_refuses_exactly_as_before(
    agent: TestClient, model_path: str
) -> None:
    """The node is not a place where the coordinator's guard can be skipped."""
    resp = _create(agent, model_path, None)

    assert resp.status_code == 422, resp.text
    assert "trust_remote_code" in resp.text


def test_a_matching_grant_is_accepted(agent: TestClient, model_path: str) -> None:
    digest = _digest_for(agent, "repo/vllm:tag")

    resp = _create(agent, model_path, _grant(digest))

    assert resp.status_code == 200, resp.text


def test_matching_grant_accepts_a_migrated_legacy_model_path(
    agent: TestClient, tmp_path: Path
) -> None:
    """Approval lookup uses the mount path after agent-side legacy migration.

    The coordinator's durable replica record predates Docker-safe store names
    and sends ``huggingface:nvidia/...``.  An agent that has already migrated
    those bytes records the encoded path in its local marker.  The grant must
    match that on-disk marker, not reject before the same request reaches the
    migration that will supply ``--model``.
    """
    acquisition = cast(Any, agent.app).state.acquisition
    legacy_path = str(tmp_path / "store" / _LOCAL_ID)
    effective_path = acquisition.local_path(_LOCAL_ID)
    effective_path.mkdir(parents=True)
    (effective_path / "config.json").write_text(json.dumps({"arch": "qwen3"}))
    (effective_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}})
    )
    (effective_path / "model-00001-of-00001.safetensors").write_bytes(b"shard")
    markers = tmp_path / "markers"
    markers.mkdir(parents=True, exist_ok=True)
    (markers / "migrated.json").write_text(
        json.dumps(
            {
                "model_id": _LOCAL_ID,
                "resolved_revision": _REVISION,
                "state": "available",
                "local_path": str(effective_path),
                "verified_at": None,
                "size_bytes": 5,
                "content_digest": None,
            }
        )
    )

    resp = _create(agent, legacy_path, _grant(_digest_for(agent, "repo/vllm:tag")))

    assert resp.status_code == 200, resp.text


def test_the_argv_proves_the_flag_was_actually_passed(agent: TestClient, model_path: str) -> None:
    """The acceptance criterion that cannot be satisfied by a record.

    Read back from the container the node actually created, not from the config
    the node was sent. A product that reported its own intent here would be
    reporting the thing this whole repository exists to distrust.
    """
    digest = _digest_for(agent, "repo/vllm:tag")
    assert _create(agent, model_path, _grant(digest)).status_code == 200

    report = agent.get(
        "/agent/v1/deployments/01J00000000000000000000000/runtime", headers=_AUTH
    ).json()

    assert "--trust-remote-code" in report["argv"], report["argv"]


def test_argv_omits_the_flag_when_nothing_approved_it(agent: TestClient, model_path: str) -> None:
    """The same proof in the negative, so the assertion above means something."""
    body = {
        "deployment_id": "01J00000000000000000000002",
        "revision": 1,
        "runtime_type": "vllm",
        "image_reference": "repo/vllm:tag",
        "runtime_config": {"tensor_parallel_size": 1},
        "model_path": model_path,
        "endpoint": "0.0.0.0:8000",
    }
    assert agent.post("/agent/v1/deployments", json=body, headers=_AUTH).status_code == 200

    report = agent.get(
        "/agent/v1/deployments/01J00000000000000000000002/runtime", headers=_AUTH
    ).json()

    assert "--trust-remote-code" not in report["argv"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_revision", "a" * 40),
        ("source_model_id", "nvidia/Some-Other-Model"),
        ("model_source_id", "local"),
    ],
)
def test_a_grant_disagreeing_with_this_node_is_refused(
    agent: TestClient, model_path: str, field: str, value: str
) -> None:
    """Replaying an approval for one model against another fails on this disk."""
    digest = _digest_for(agent, "repo/vllm:tag")

    resp = _create(agent, model_path, _grant(digest, **{field: value}))

    assert resp.status_code == 422, f"a grant with a wrong {field} was honoured"
    assert "code_execution_not_authorized" in resp.text


def test_a_grant_for_another_image_is_refused(agent: TestClient, model_path: str) -> None:
    """The approved code would otherwise run in a runtime nobody approved."""
    resp = _create(agent, model_path, _grant("sha256:" + "c" * 64))

    assert resp.status_code == 422, resp.text
    assert "code_execution_not_authorized" in resp.text


def test_a_grant_whose_fingerprint_does_not_match_its_contents_is_refused(
    agent: TestClient, model_path: str
) -> None:
    """A key that is not the hash of its own tuple authorizes nothing."""
    digest = _digest_for(agent, "repo/vllm:tag")
    grant = _grant(digest)
    grant["fingerprint"] = "sha256:" + "0" * 64

    resp = _create(agent, model_path, grant)

    assert resp.status_code == 422
    assert "fingerprint" in resp.text


def test_a_grant_for_another_runtime_is_refused(agent: TestClient, model_path: str) -> None:
    """Approving a flag on SGLang is not approving it on vLLM."""
    digest = _digest_for(agent, "repo/vllm:tag")

    resp = _create(agent, model_path, _grant(digest, runtime_type="sglang"))

    assert resp.status_code == 422
    assert "code_execution_not_authorized" in resp.text


def test_a_grant_for_an_unapprovable_option_is_refused(agent: TestClient, model_path: str) -> None:
    """The narrow opening stays narrow at the node too, not only at the coordinator."""
    digest = _digest_for(agent, "repo/vllm:tag")

    resp = _create(agent, model_path, _grant(digest, option="api_key"))

    assert resp.status_code == 422
    assert "code_execution_not_authorized" in resp.text

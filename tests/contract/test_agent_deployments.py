"""Contract test the agent's deployment and image endpoints.

- ``POST /agent/v1/images:pull`` returns the **platform-specific** digest, not
  the multi-arch manifest-list digest.
- ``POST /agent/v1/deployments`` materializes a deployment and returns the
  platform-specific image digest.

The agent is built with the fake container engine, which models the
platform-specific digest (tests/fakes/container_engine.py). Management-token
gated.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorstead.agent.app import build_agent_app
from tests.fakes.container_engine import FakeContainerEngine
from tests.fakes.service_manager import FakeServiceManager

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer mgmt"}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
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
    """A real on-disk vLLM model tree the pre-flight check will accept.

    Previously these tests pointed ``model_path`` at ``/var/lib/...`` -- a path
    that does not exist on the test host -- which is exactly what let the
    missing-config.json incident hide: the agent launched a container against
    a dir it never inspected. The pre-flight now refuses that; the fixture
    builds the tree vLLM's adapter will accept (config + sharded index + shards).
    """
    import json

    d = tmp_path / "store" / "model"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"arch": "qwen3"}))
    (d / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (d / "model-00001-of-00002.safetensors").write_bytes(b"shard1")
    (d / "model-00002-of-00002.safetensors").write_bytes(b"shard2")
    return str(d)


def test_images_pull_returns_platform_digest(client: TestClient) -> None:
    """``/agent/v1/images:pull`` returns the platform-specific digest.

    The fake engine returns a deterministic ``sha256:platform-...`` digest for
    the reference — distinct from a multi-arch manifest-list digest. The
    assertion is that the returned value is the platform-specific one and not a
    manifest-list reference.
    """
    resp = client.post("/agent/v1/images:pull", json={"reference": "repo/vllm:tag"}, headers=_AUTH)
    assert resp.status_code == 200
    digest = resp.json()["digest"]
    assert digest.startswith("sha256:")
    assert "platform" in digest  # the fake marks the platform-specific digest


def test_deployment_create_returns_platform_digest(client: TestClient, model_path: str) -> None:
    """Deployment create returns the platform-specific image digest."""
    resp = client.post(
        "/agent/v1/deployments",
        json={
            "deployment_id": "01J00000000000000000000000",
            "revision": 1,
            "runtime_type": "vllm",
            "image_reference": "repo/vllm:tag",
            "runtime_config": {"tensor_parallel_size": 1},
            "model_path": model_path,
            "endpoint": "0.0.0.0:8000",
        },
        headers=_AUTH,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "created"
    assert body["image_digest"].startswith("sha256:")
    assert "platform" in body["image_digest"]


def test_deployment_create_starts_container_and_enables_unit(
    client: TestClient, model_path: str
) -> None:
    """Deployment create starts the container and enables a boot-restoration unit."""
    resp = client.post(
        "/agent/v1/deployments",
        json={
            "deployment_id": "01J00000000000000000000000",
            "revision": 1,
            "runtime_type": "vllm",
            "image_reference": "repo/vllm:tag",
            "runtime_config": {"tensor_parallel_size": 1},
            "model_path": model_path,
            "endpoint": "0.0.0.0:8000",
        },
        headers=_AUTH,
    )
    assert resp.status_code == 200


def test_vllm_inference_key_comes_from_the_agent_environment(
    client: TestClient, model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inference authentication is not accepted from deployment configuration."""
    monkeypatch.setenv("TENSORSTEAD_INFERENCE_API_KEY", "inference-secret")
    resp = client.post(
        "/agent/v1/deployments",
        json={
            "deployment_id": "01J00000000000000000000001",
            "revision": 1,
            "runtime_type": "vllm",
            "image_reference": "repo/vllm:tag",
            "runtime_config": {"served_model_name": "qwen36-27b"},
            "model_path": model_path,
            "endpoint": "0.0.0.0:8000",
        },
        headers=_AUTH,
    )

    assert resp.status_code == 200
    app = cast(FastAPI, client.app)
    engine = app.state.container_engine
    args = engine.containers["tensorstead-01J00000000000000000000001"].command_args
    assert args is not None
    assert args[-2:] == ["--served-model-name", "qwen36-27b"]
    environment = engine.containers["tensorstead-01J00000000000000000000001"].environment
    assert environment is not None
    # The credential is delivered as environment, not argv. Asserted as a
    # membership check rather than dict equality because the adapter also
    # contributes cache-root variables -- but the property this test
    # guards is unchanged and is checked on both sides: the secret is in the
    # environment, and nowhere in the command line.
    assert environment["VLLM_API_KEY"] == "inference-secret"
    assert not any("inference-secret" in arg for arg in args), (
        "the inference credential reached the host-visible process command line"
    )


def test_stop_removes_the_container_name_for_a_later_restart(
    client: TestClient, model_path: str
) -> None:
    """A stopped deployment must not leave Docker's name reservation behind."""
    deployment_id = "01J00000000000000000000002"
    created = client.post(
        "/agent/v1/deployments",
        json={
            "deployment_id": deployment_id,
            "revision": 1,
            "runtime_type": "vllm",
            "image_reference": "repo/vllm:tag",
            "runtime_config": {},
            "model_path": model_path,
            "endpoint": "0.0.0.0:8000",
        },
        headers=_AUTH,
    )
    assert created.status_code == 200

    stopped = client.post(f"/agent/v1/deployments/{deployment_id}:stop", headers=_AUTH)

    assert stopped.status_code == 200
    app = cast(FastAPI, client.app)
    assert f"tensorstead-{deployment_id}" not in app.state.container_engine.containers


def test_agent_endpoints_require_management_token(client: TestClient) -> None:
    """Agent management endpoints are gated by the management token."""
    assert client.post("/agent/v1/images:pull", json={"reference": "x"}).status_code == 401
    assert client.post("/agent/v1/deployments", json={}).status_code == 401


def test_create_deployment_fails_fast_on_invalid_model_dir(
    client: TestClient, tmp_path: Path
) -> None:
    """A structurally invalid model dir fails the start at 422, not "succeeded".

    This is the incident's other half. The corrupted tree was missing
    ``config.json``; the deployment route bind-mounted it and started the
    container anyway, and the failure surfaced only minutes later as an exited
    container with no classified cause. The pre-flight now inspects the dir
    before container creation, so a missing ``config.json`` fails fast with
    ``model_directory_invalid`` — a named reason the coordinator records as a
    node failure rather than a silent "created".

    The error shape mirrors ``invalid_runtime`` (code + message + detail) so
    ``node_http._error_from_response`` unwraps it into an ``AgentCallError``.

    Deliberately placed *inside* the store (``tmp_path / "store" / ...``, the
    same root the ``client`` fixture configures): a path outside the store is
    a different refusal now (``model_path_not_contained``, covered separately
    below), and this test's own job is the
    shape check that runs after containment passes.
    """
    bad_dir = tmp_path / "store" / "broken-model"
    bad_dir.mkdir(parents=True)
    # config.json deliberately absent — the incident shape.
    resp = client.post(
        "/agent/v1/deployments",
        json={
            "deployment_id": "01J00000000000000000000003",
            "revision": 1,
            "runtime_type": "vllm",
            "image_reference": "repo/vllm:tag",
            "runtime_config": {"tensor_parallel_size": 1},
            "model_path": str(bad_dir),
            "endpoint": "0.0.0.0:8000",
        },
        headers=_AUTH,
    )

    assert resp.status_code == 422, resp.text
    body = resp.json()
    detail = body["detail"]
    assert detail["code"] == "model_directory_invalid"
    assert "config.json" in detail["message"]
    assert detail["detail"]["model_path"] == str(bad_dir)
    assert detail["detail"]["runtime_type"] == "vllm"

    # Nothing was started — the failure is pre-flight, before container creation.
    app = cast(FastAPI, client.app)
    assert "tensorstead-01J00000000000000000000003" not in app.state.container_engine.containers


def test_create_refuses_a_model_path_outside_the_managed_store(
    client: TestClient, tmp_path: Path
) -> None:
    """A model path outside the managed store is refused.

    The primary model path used to reach Docker as a read-only host bind
    mount with no containment check at all -- a management/MCP caller
    naming any outside directory with a minimal valid model shape (exactly
    what this test builds) reached the host filesystem directly. Same shape
    as the config.json test above, at a path a real replica could never be:
    outside ``tmp_path / "store"``, the root the ``client`` fixture
    configures.
    """
    import json

    outside = tmp_path / "not-the-store" / "looks-like-a-model"
    outside.mkdir(parents=True)
    (outside / "config.json").write_text(json.dumps({"arch": "qwen3"}))

    resp = client.post(
        "/agent/v1/deployments",
        json={
            "deployment_id": "01J00000000000000000000004",
            "revision": 1,
            "runtime_type": "vllm",
            "image_reference": "repo/vllm:tag",
            "runtime_config": {"tensor_parallel_size": 1},
            "model_path": str(outside),
            "endpoint": "0.0.0.0:8000",
        },
        headers=_AUTH,
    )

    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "model_path_not_contained"
    assert detail["detail"]["runtime_type"] == "vllm"

    app = cast(FastAPI, client.app)
    assert "tensorstead-01J00000000000000000000004" not in app.state.container_engine.containers


@pytest.mark.parametrize(
    "bad_id",
    [
        "not-a-ulid",
        "../../../../etc/cron.d/evil",
        "/etc/ssh/sshd_config",
        "",
    ],
)
def test_create_refuses_a_non_ulid_deployment_id(
    client: TestClient, model_path: str, bad_id: str
) -> None:
    """The cache-root-escape boundary.

    ``deployment_id`` is joined onto the cache root and then ``chmod``/
    ``chown``ed; an absolute or traversal value used to reach outside it
    silently, because ``Path`` join discards the left side for an absolute
    right side. Every entry point now refuses anything that is not shaped
    like a ULID before it is used for anything.
    """
    resp = client.post(
        "/agent/v1/deployments",
        json={
            "deployment_id": bad_id,
            "revision": 1,
            "runtime_type": "vllm",
            "image_reference": "repo/vllm:tag",
            "runtime_config": {"tensor_parallel_size": 1},
            "model_path": model_path,
            "endpoint": "0.0.0.0:8000",
        },
        headers=_AUTH,
    )

    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"]["code"] == "invalid_deployment_id"
    app = cast(FastAPI, client.app)
    assert app.state.container_engine.containers == {}, "nothing should have started"


_STOP_REMOVE_RECONCILE = ((":stop", "post"), ("", "delete"), (":reconcile", "post"))


def test_stop_remove_reconcile_refuse_a_non_ulid_deployment_id(client: TestClient) -> None:
    """The same boundary on the three path-parameter routes, not only create.

    A multi-segment value (``/etc/passwd``, ``../../x``) never reaches the
    handler at all -- Starlette's default path converter does not match ``/``
    within a single ``{deployment_id}`` segment, so it 404s at routing before
    Tensorstead ever sees it.
    """
    for suffix, method in _STOP_REMOVE_RECONCILE:
        resp = getattr(client, method)(f"/agent/v1/deployments/not-a-ulid{suffix}", headers=_AUTH)
        assert resp.status_code == 422, f"{method} {suffix}: {resp.text}"
        assert resp.json()["detail"]["code"] == "invalid_deployment_id"


def test_stop_and_reconcile_refuse_a_single_segment_traversal_id(client: TestClient) -> None:
    """``..`` reaches the handler when a suffix follows it in the same segment.

    Excludes bare ``DELETE .../deployments/..``: the HTTP layer itself
    collapses that dot-segment during URL normalization before routing sees
    it, landing on a different, nonexistent route (404) rather than this
    handler -- already a safe outcome, just not one this handler produces.
    """
    for suffix, method in ((":stop", "post"), (":reconcile", "post")):
        resp = getattr(client, method)(f"/agent/v1/deployments/..{suffix}", headers=_AUTH)
        assert resp.status_code == 422, f"{method} {suffix}: {resp.text}"
        assert resp.json()["detail"]["code"] == "invalid_deployment_id"


def test_create_classifies_a_refused_runtime_config_instead_of_500(
    client: TestClient, tmp_path: Path
) -> None:
    """A config the adapter refuses is the operator's to fix, so it must say so.

    ``validate_config`` sat two lines above ``validate_model_path`` and was the
    one pre-flight that never had its ``ValueError`` wrapped. Every other
    refusal on this route -- a bad id, an uncontained path, a structurally
    invalid tree -- returns a classified 422 the coordinator unwraps into an
    ``AgentCallError``; this one escaped as an unhandled exception, so FastAPI
    returned a bare ``Internal Server Error`` and the coordinator turned that
    into a 503 naming nothing.

    Found the expensive way. Seven deployment records carry
    ``trust_remote_code: true``, refused from the start. They
    kept running because nothing restarted them, and when one finally was
    stopped, the restore answered "Internal Server Error" -- the operator could
    not tell an unstartable *record* from a broken *agent*, and the one fact
    that would have distinguished them was in the node's journal.
    """
    import json

    good_dir = tmp_path / "store" / "fine-model"
    good_dir.mkdir(parents=True)
    (good_dir / "config.json").write_text(json.dumps({"arch": "qwen3"}))

    resp = client.post(
        "/agent/v1/deployments",
        json={
            "deployment_id": "01J00000000000000000000009",
            "revision": 1,
            "runtime_type": "vllm",
            "image_reference": "repo/vllm:tag",
            # Refused from the start: it loads code into the runtime.
            "runtime_config": {"tensor_parallel_size": 1, "trust_remote_code": True},
            "model_path": str(good_dir),
            "endpoint": "0.0.0.0:8000",
        },
        headers=_AUTH,
    )

    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "invalid_runtime_config"
    # The adapter's own reason, verbatim: the operator has to know *which*
    # option is refused to be able to act on it.
    assert "trust_remote_code" in detail["message"]
    assert detail["detail"]["runtime_type"] == "vllm"

    # Pre-flight, so nothing was created.
    app = cast(FastAPI, client.app)
    assert "tensorstead-01J00000000000000000000009" not in app.state.container_engine.containers


def test_a_node_key_does_not_stop_a_runtime_without_a_mechanism(
    client: TestClient, model_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The node's inference key is optional for runtimes that cannot use it.

    SGLang declares no credential mechanism. With a node-wide key present this
    start used to be refused, which made one installer-provisioned key a
    requirement for every runtime on the node.
    """
    monkeypatch.setenv("TENSORSTEAD_INFERENCE_API_KEY", "inference-secret")
    resp = client.post(
        "/agent/v1/deployments",
        json={
            "deployment_id": "01J00000000000000000000002",
            "revision": 1,
            "runtime_type": "sglang",
            "image_reference": "repo/sglang:tag",
            "runtime_config": {},
            "model_path": model_path,
            "endpoint": "0.0.0.0:30000",
        },
        headers=_AUTH,
    )

    assert resp.status_code == 200, resp.text
    app = cast(FastAPI, client.app)
    container = app.state.container_engine.containers["tensorstead-01J00000000000000000000002"]
    assert "inference-secret" not in (container.environment or {}).values()


def test_a_bound_key_still_refuses_a_runtime_without_a_mechanism(
    client: TestClient, model_path: str
) -> None:
    """A key bound to the deployment is an explicit request, so it is not skipped."""
    resp = client.post(
        "/agent/v1/deployments",
        json={
            "deployment_id": "01J00000000000000000000003",
            "revision": 1,
            "runtime_type": "sglang",
            "image_reference": "repo/sglang:tag",
            "runtime_config": {},
            "model_path": model_path,
            "endpoint": "0.0.0.0:30000",
            "inference_credential": "bound-secret",
        },
        headers=_AUTH,
    )

    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "credential_not_enforceable"

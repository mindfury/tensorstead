"""Observed state must describe the runtime, not the paperwork.

The 2026-08-10 incident: a deployment whose vLLM process was dead reported
``status: running``, ``endpoint_reachable: true``, and ``divergences: []`` for
the entire window in which every inference request returned HTTP 500 and every
direct endpoint connection was reset. Nothing was wrong with the reporting
pipeline — each field was faithfully reporting the wrong thing:

- ``status`` came from "does a container object exist", never from its state;
- ``endpoint_reachable`` came from ``systemctl is-enabled``, a boot-time
  configuration flag that is true for a unit that has never run;
- ``running_revision`` was hardcoded ``None``, which made the coordinator's
  ``revision_mismatch`` check unreachable by construction.

These tests pin the three lies. Each one fails against the older agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorstead.agent.app import build_agent_app
from tests.fakes.container_engine import FakeContainerEngine
from tests.fakes.service_manager import FakeServiceManager

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer mgmt"}
_DEPLOYMENT_ID = "01J00000000000000000000010"
_CONTAINER = f"tensorstead-{_DEPLOYMENT_ID}"


@pytest.fixture
def engine() -> FakeContainerEngine:
    return FakeContainerEngine()


@pytest.fixture
def services() -> FakeServiceManager:
    return FakeServiceManager()


@pytest.fixture
def client(tmp_path: Path, engine: FakeContainerEngine, services: FakeServiceManager) -> TestClient:
    app = build_agent_app(
        management_token="mgmt",
        container_engine=engine,
        service_manager=services,
        store_dir=tmp_path / "store",
        marker_dir=tmp_path / "markers",
    )
    return TestClient(app)


def _deploy(
    client: TestClient,
    *,
    revision: int = 7,
    endpoint: str = "0.0.0.0:8000",
    restore_on_boot: bool = False,
) -> None:
    """Materialize a deployment through the real agent route."""
    # Build a real model tree the pre-flight will accept. Previously the
    # path was a /var/lib/... path that does not exist on the test host -- the
    # agent launched against a directory it never inspected, which is what hid
    # the missing-config.json incident. The tree lives in the agent's own store.
    import json

    model_dir = cast(FastAPI, client.app).state.acquisition.local_path("qwen")
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps({"arch": "qwen3"}))
    response = client.post(
        "/agent/v1/deployments",
        json={
            "deployment_id": _DEPLOYMENT_ID,
            "revision": revision,
            "runtime_type": "vllm",
            "image_reference": "local/vllm:26.07-xgrammar-0.2.1",
            "runtime_config": {"tensor_parallel_size": 1},
            "model_path": str(model_dir),
            "endpoint": endpoint,
            "restore_on_boot": restore_on_boot,
        },
        headers=_AUTH,
    )
    assert response.status_code == 200, response.text


def _observe(client: TestClient) -> dict:
    response = client.get(
        f"/agent/v1/deployments/{_DEPLOYMENT_ID}/observed",
        headers=_AUTH,
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


def test_dead_container_is_not_reported_running(
    client: TestClient, engine: FakeContainerEngine
) -> None:
    """A container that exists but is not running reports ``not_running``.

    This is the incident itself. The container object survives the death of the
    process inside it, so "the engine knows this name" is not evidence that
    anything is serving.
    """
    _deploy(client)
    assert _observe(client)["status"] == "running"

    # The runtime dies; the container remains, as Docker leaves it.
    engine.stop_container(_CONTAINER)

    observed = _observe(client)
    assert observed["status"] == "not_running", (
        "a stopped container was reported as running — the observed status is "
        "reading container existence, not container state"
    )


def test_endpoint_reachable_is_not_the_boot_enabled_flag(
    tmp_path: Path,
    engine: FakeContainerEngine,
    services: FakeServiceManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``endpoint_reachable`` must probe the endpoint, not read unit config.

    This is the incident's shape at the route level: the container is up, the
    systemd unit is enabled, and nothing is serving. ``systemctl is-enabled``
    answers "will this start at boot", which stays true across a crash — so the
    old implementation reported this exact state as reachable.
    """
    # Boot restoration is opt-in, and this test needs the
    # unit *enabled* -- an enabled unit is the misleading signal it guards
    # against, so the scenario has to ask for one. Both halves of the grant are
    # required now: the node permits it and the deployment asks.
    monkeypatch.setenv("TENSORSTEAD_ALLOW_BOOT_RESTORATION", "true")
    client = TestClient(
        build_agent_app(
            management_token="mgmt",
            container_engine=engine,
            service_manager=services,
            store_dir=tmp_path / "store",
            marker_dir=tmp_path / "markers",
        )
    )
    _deploy(client, endpoint="0.0.0.0:59999", restore_on_boot=True)

    # The unit is still enabled — that is precisely the misleading signal.
    assert services.is_enabled(_CONTAINER) is True

    observed = _observe(client)
    assert observed["status"] == "running"
    assert observed["endpoint_reachable"] is False, (
        "endpoint reported reachable with nothing listening — reachability is "
        "being read from the boot-enabled flag rather than probed"
    )
    assert observed["inference_ready"] is False


def test_a_dead_container_claims_nothing_it_did_not_measure(
    client: TestClient, engine: FakeContainerEngine
) -> None:
    """Unmeasured facts report ``null``, not ``false``.

    No probe is issued against a dead container, so reachability is unknown
    rather than denied — a port this deployment no longer holds may well be
    bound by something else. ``status`` carries the alarm, and ``detail``
    carries the engine's own account.
    """
    _deploy(client)
    engine.stop_container(_CONTAINER, exit_code=1)

    observed = _observe(client)
    assert observed["status"] == "not_running"
    assert observed["endpoint_reachable"] is None
    assert observed["inference_ready"] is None
    assert "exit code 1" in (observed["detail"] or "")


def test_reconcile_restarts_a_dead_container(
    client: TestClient, engine: FakeContainerEngine
) -> None:
    """Reconcile must act on container state, not on container existence.

    The incident's most damning line: an explicit reconcile against a
    deployment whose runtime was dead reported "no changes required". It read
    ``get_digest(name) is not None`` as "running", so the dead container looked
    healthy, the start branch never fired, and the one operation invoked to
    repair the fault was structurally incapable of repairing it.
    """
    _deploy(client)
    engine.stop_container(_CONTAINER, exit_code=1)

    response = client.post(
        f"/agent/v1/deployments/{_DEPLOYMENT_ID}:reconcile",
        headers=_AUTH,
    )
    assert response.status_code == 200, response.text
    body = response.json()

    actions = [change.get("action") for change in body.get("changed", [])]
    assert "started" in actions, (
        f"reconcile reported changed={body.get('changed')!r} for a dead container; "
        "it is reading container existence rather than container state"
    )
    assert engine.containers[_CONTAINER].running is True


def test_an_unlabelled_container_says_what_is_missing_and_how_to_fix_it(
    client: TestClient, engine: FakeContainerEngine
) -> None:
    """The message an operator lives with until deployments are re-created.

    Every deployment that predates endpoint labelling lands here. "No endpoint recorded"
    reads as a contradiction to someone looking at the declared endpoint in the
    block above, and names no way out -- the coordinator plainly has an
    endpoint; it is the container that carries none.
    """
    _deploy(client)
    container = engine.containers[_CONTAINER]
    container.labels.pop("tensorstead.endpoint")

    observed = _observe(client)

    assert observed["status"] == "running"
    assert observed["endpoint_reachable"] is None
    assert observed["inference_ready"] is None
    detail = observed["detail"] or ""
    assert "predates endpoint labelling" in detail
    assert "re-create" in detail


def test_the_wire_contract_declares_inference_readiness() -> None:
    """An additive optional field, and the compatibility signal moved with it.

    The contract version is the *only* signal that tells a coordinator whether a
    node can answer a question. Adding the field without moving
    the version would leave a 1.1 agent's silence indistinguishable from a 1.2
    agent's "cannot tell".
    """
    from tensorstead.contracts.agent import ObservedStateResponse
    from tensorstead.contracts.version import CONTRACT_VERSION

    fields = ObservedStateResponse.model_fields
    assert "inference_ready" in fields
    # Optional, so an agent below 1.2 simply omits it.
    assert fields["inference_ready"].default is None
    assert CONTRACT_VERSION == "1.15"


def test_an_agent_too_old_to_answer_is_unknown_not_broken() -> None:
    """A 1.1 agent omits the field; that is not a fault and not a clean bill.

    Reporting the omission as ready would restore the false confidence the change
    removed. Reporting it as a fault would mark every un-upgraded node broken
    during a rolling upgrade, which is its own kind of untrue.
    """
    from tensorstead.contracts.agent import ObservedStateResponse

    older_agent_payload: dict[str, Any] = {
        "status": "running",
        "observed_at": "2026-08-10T22:07:00-04:00",
        "running_image_digest": "sha256:abc",
        "running_revision": 9,
        "endpoint_reachable": True,
    }

    parsed = ObservedStateResponse(**older_agent_payload)

    assert parsed.inference_ready is None


def test_running_revision_is_reported(client: TestClient) -> None:
    """The agent reports the revision it was asked to run.

    The contract document already specifies an integer here. Returning ``None``
    made the coordinator's ``revision_mismatch`` divergence unreachable: its
    guard is ``running_revision is not None``.
    """
    _deploy(client, revision=7)

    observed = _observe(client)
    assert observed["running_revision"] == 7, (
        "the agent discarded the revision it was handed at materialization, so "
        "revision_mismatch can never fire"
    )


def test_inference_readiness_is_distinct_from_reachability(client: TestClient) -> None:
    """Transport reachability and serving inference are separate facts.

    A bound port proves a listener exists. It does not prove the model loaded or
    that the runtime answers its own API — the distinction the incident needed.
    """
    _deploy(client)

    observed = _observe(client)
    assert "inference_ready" in observed, (
        "observed state carries no inference-readiness fact, so a listening "
        "socket remains indistinguishable from a serving runtime"
    )

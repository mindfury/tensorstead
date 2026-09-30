"""Boot restoration is asked for and earned, never a side effect.

On 2026-08-15 a deployment deadlocked the NVIDIA kernel driver on its first
start. It had never completed a single successful run. Boot restoration brought
it back on every reboot, into the same deadlock,
until the node was reimaged.

Nothing about the hang was this product's fault — it was a known driver defect.
What *was* this product's fault is that a deployment acquired the right to run
on every future boot as a side effect of being started once. The operator's
escape hatch from a wedged appliance is a reboot; unconditional boot restoration
is what took the escape hatch away.

So there are now two gates, and they answer different questions:

- **`restore_on_boot`** is intent. Nobody's deployment becomes persistent
  without saying so, and it defaults to off.
- **the container actually running** is evidence. A definition nobody has seen
  start does not get to start unattended before there is a login prompt.

The honest limit, stated here because the tests cannot state it: the deployment
that prompted all this took four minutes to deadlock and *would* have passed the
evidence gate. Default-off is what would have prevented that incident. The
evidence gate is the second line, and it catches the container that never came
up at all.

A third mechanism lives in the unit file rather than here: `StartLimitBurst`,
so a deployment that fails at boot gives up instead of re-entering
the failure on every restart.
"""

from __future__ import annotations

import json
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
def permitted_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """A node provisioned to allow boot restoration at all.

    Most cases here are about the *deployment's* half of the decision, so they
    need the node's half already granted. That the node can withhold it is the
    subject of its own tests below.
    """
    monkeypatch.setenv("TENSORSTEAD_ALLOW_BOOT_RESTORATION", "true")


@pytest.fixture
def client(
    tmp_path: Path,
    engine: FakeContainerEngine,
    services: FakeServiceManager,
    permitted_node: None,
) -> TestClient:
    return TestClient(
        build_agent_app(
            management_token="mgmt",
            container_engine=engine,
            service_manager=services,
            store_dir=tmp_path / "store",
            marker_dir=tmp_path / "markers",
        )
    )


def _deploy(client: TestClient, body: dict[str, Any] | None = None) -> Any:
    model_dir = cast(FastAPI, client.app).state.acquisition.local_path("qwen")
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps({"arch": "qwen3"}))
    payload: dict[str, Any] = {
        "deployment_id": _DEPLOYMENT_ID,
        "revision": 1,
        "runtime_type": "vllm",
        "image_reference": "local/vllm:26.07",
        "runtime_config": {"tensor_parallel_size": 1},
        "model_path": str(model_dir),
        "endpoint": "0.0.0.0:8000",
    }
    payload.update(body or {})
    return client.post("/agent/v1/deployments", json=payload, headers=_AUTH)


def test_a_deployment_that_did_not_ask_is_not_restored(
    client: TestClient, services: FakeServiceManager
) -> None:
    """The default, and the one that would have prevented the incident."""
    assert _deploy(client).status_code == 200

    assert services.is_enabled(_CONTAINER) is False, (
        "a deployment acquired boot persistence without asking for it -- which is "
        "how an unproven deployment came back into its own deadlock on every reboot"
    )


def test_a_deployment_that_asked_and_started_is_restored(
    client: TestClient, services: FakeServiceManager
) -> None:
    """Opting in still works. The capability is gated, not removed."""
    assert _deploy(client, {"restore_on_boot": True}).status_code == 200

    assert services.is_enabled(_CONTAINER) is True


class _ImmediatelyExitingEngine(FakeContainerEngine):
    """A container that starts and is dead by the time anyone looks.

    Docker's real behaviour for a runtime that exits on launch: the container
    object exists and resolves by name, and only ``running`` distinguishes it
    from a healthy one. That distinction is exactly what the evidence gate checks.
    """

    def start_container(self, name: str) -> None:
        super().start_container(name)
        self.containers[name].running = False
        self.containers[name].exit_code = 1


def test_a_deployment_that_asked_but_never_started_is_not_restored(
    tmp_path: Path, services: FakeServiceManager, permitted_node: None
) -> None:
    """Intent is not evidence.

    Declaring boot restoration says what you want. It says nothing about whether
    this definition can start, and a definition that cannot start is exactly the
    one that must not be started unattended before anyone can log in.
    """
    client = TestClient(
        build_agent_app(
            management_token="mgmt",
            container_engine=_ImmediatelyExitingEngine(),
            service_manager=services,
            store_dir=tmp_path / "store",
            marker_dir=tmp_path / "markers",
        )
    )
    _deploy(client, {"restore_on_boot": True})

    assert services.is_enabled(_CONTAINER) is False, (
        "a container that never came up was granted the right to come up on every boot"
    )


def test_the_unit_is_still_written_so_reconcile_can_repair(
    client: TestClient, services: FakeServiceManager, engine: FakeContainerEngine
) -> None:
    """Installed and enabled are different facts, and this is why it matters.

    The unit's presence is how the agent records "I was told to run this", and
    reconcile reads it to decide whether a dead container should be restarted.
    Keying that on the boot flag would make reconcile a no-op for every
    deployment with `restore_on_boot: false` -- the operation an operator invokes
    to fix a dead container, quietly declining to fix it.
    """
    _deploy(client)
    assert services.is_installed(_CONTAINER) is True
    assert services.is_enabled(_CONTAINER) is False

    engine.stop_container(_CONTAINER, exit_code=1)
    response = client.post(
        f"/agent/v1/deployments/{_DEPLOYMENT_ID}:reconcile", json={}, headers=_AUTH
    )
    assert response.status_code == 200, response.text

    actions = [change.get("action") for change in response.json().get("changed", [])]
    assert "started" in actions, (
        "reconcile refused to restart a dead container because the deployment had not "
        "opted into boot restoration -- two unrelated questions answered by one flag"
    )
    assert engine.containers[_CONTAINER].running is True


def test_the_unit_gives_up_rather_than_fighting_the_machine(tmp_path: Path) -> None:
    """A failing unit must stop retrying.

    ``Restart=on-failure`` with no ceiling means a deployment that wedges its
    node does so again after every restart, which is what turns a reboot from an
    escape hatch into another lap of the same loop.

    systemd's own facility, which is what the design requires -- we configure the
    mechanism it already delegates to rather than writing a supervisor.
    """
    from tensorstead.agent.service_manager.systemd import SystemdServiceManager

    manager = SystemdServiceManager(systemd_dir=tmp_path, dry_run=True)
    manager.install("tensorstead-test")
    unit = (tmp_path / "tensorstead-tensorstead-test.service").read_text()

    assert "StartLimitBurst=" in unit, "a unit with no retry ceiling retries forever"
    assert "StartLimitIntervalSec=" in unit
    # The mechanism is still systemd's, not ours.
    assert "Restart=on-failure" in unit


def test_the_revision_carries_the_declaration_so_a_modify_re_earns_it() -> None:
    """It lives on the revision, and that placement is the point.

    ``modify`` produces revision *n+1* -- a definition nobody has run. If boot
    restoration lived on the deployment, an untested change would inherit the
    right to start unattended from the version it replaced.
    """
    from tensorstead.domain.models import DeploymentRevision

    fields = DeploymentRevision.__dataclass_fields__
    assert "restore_on_boot" in fields, (
        "boot restoration must be part of the versioned definition, not a mutable "
        "property a new revision inherits without review"
    )
    assert fields["restore_on_boot"].default is False, "the safe default is off"


# --------------------------------------------------------------------------
# The node grants the capability, and an agent cannot reach it
# --------------------------------------------------------------------------


def test_a_node_that_was_not_provisioned_for_it_refuses(
    tmp_path: Path,
    engine: FakeContainerEngine,
    services: FakeServiceManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The half of the guard that actually holds.

    ``restore_on_boot`` can be set by anything holding a management token --
    including an autonomous agent, which is the threat the opt-in flag does not
    address. This one is read from the node's own provisioned environment,
    written by the deployment playbook, which needs credentials no agent has
    been given.

    So the request asks for boot restoration in the strongest terms available to
    it and the node still declines.
    """
    monkeypatch.delenv("TENSORSTEAD_ALLOW_BOOT_RESTORATION", raising=False)
    client = TestClient(
        build_agent_app(
            management_token="mgmt",
            container_engine=engine,
            service_manager=services,
            store_dir=tmp_path / "store",
            marker_dir=tmp_path / "markers",
        )
    )

    assert _deploy(client, {"restore_on_boot": True}).status_code == 200

    assert services.is_enabled(_CONTAINER) is False, (
        "a node that was never provisioned to permit boot restoration arranged it "
        "anyway because the request asked -- the capability and its exercise must "
        "not share a credential"
    )
    # The deployment still runs. The node withheld persistence, not service.
    assert engine.containers[_CONTAINER].running is True
    assert services.is_installed(_CONTAINER) is True


@pytest.mark.parametrize("value", ["", "false", "no", "0", "off", "maybe", "TRUE-ish"])
def test_only_an_affirmative_provisioning_value_grants_it(
    tmp_path: Path,
    engine: FakeContainerEngine,
    services: FakeServiceManager,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    """A misread must deny, never permit.

    This gates a capability whose failure mode is an unrecoverable node, so
    anything that is not plainly an operator writing "yes" is a no -- including
    a typo, which is exactly the case where a permissive parser would be worst.
    """
    monkeypatch.setenv("TENSORSTEAD_ALLOW_BOOT_RESTORATION", value)
    client = TestClient(
        build_agent_app(
            management_token="mgmt",
            container_engine=engine,
            service_manager=services,
            store_dir=tmp_path / "store",
            marker_dir=tmp_path / "markers",
        )
    )
    _deploy(client, {"restore_on_boot": True})

    assert services.is_enabled(_CONTAINER) is False, f"{value!r} was read as permission"


def test_the_agent_surface_cannot_ask_for_boot_restoration() -> None:
    """The MCP tool does not carry the field at all.

    Not "refuses it" -- **cannot express it**. An agent reaching this product
    only through MCP has no way to say the word, which is the same narrowing
    ``credential_set`` already has and for the same reason: some authorities do
    not belong on the surface an autonomous caller drives.

    Asserted against the tool's real signature rather than its documentation,
    because a docstring saying the field is absent is not the field being
    absent.
    """
    import inspect

    from tensorstead.mcp.tools import register_tools

    captured: dict[str, Any] = {}

    class _Server:
        # Some tools register with an explicit name and some take the function's
        # own, so this accepts both rather than assuming one.
        def tool(self, *args: Any, **kwargs: Any) -> Any:
            def decorate(fn: Any) -> Any:
                captured[kwargs.get("name") or fn.__name__] = fn
                return fn

            return decorate

    register_tools(_Server(), client=object())  # type: ignore[arg-type]

    create = captured["deployment_create"]
    parameters = set(inspect.signature(create).parameters)
    assert "restore_on_boot" not in parameters, (
        "the MCP surface offers boot restoration; an agent can grant itself "
        "persistence across reboots"
    )
    # The human-facing surfaces still have it -- this is a narrowing, not a removal.
    from tensorstead.contracts.api import DeploymentCreateRequest

    assert "restore_on_boot" in DeploymentCreateRequest.model_fields

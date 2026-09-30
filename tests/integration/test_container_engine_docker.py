"""The docker-py backend against a real Docker Engine.

Everything else in the suite exercises the container-engine seam through
`FakeContainerEngine`. That is the right default -- but the 2026-08-10 incident
was caused by an assumption about what the *real* Engine returns, and a fake
written by the same hand that held the assumption cannot falsify it. These tests
ask the daemon.

What they pin, all of it previously unverified against real Docker:

- `containers.get(name)` resolves an exited container, so its success is not
  evidence of liveness -- the original defect, asserted against the daemon;
- `State.Running` and `State.ExitCode` are where the truth actually lives;
- a restarting container is not reported as running;
- `labels=` on create round-trips through `Config.Labels`, which is how the
  agent recovers a deployment's revision and endpoint.

No GPU and no vLLM: any tiny image exercises all of it. `DockerEngine.
create_container` itself is *not* covered here -- it issues a GPU device request
that no Mac can satisfy, so the full materialization path remains verifiable
only on the appliance. Containers are therefore created through docker-py
directly, with the same `labels=` kwarg the agent passes.

Skipped automatically when no daemon is reachable, so `make check` stays green
on a host without Docker.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

import pytest

from tensorstead.agent.container_engine.base import ContainerState
from tensorstead.agent.container_engine.docker_py import DockerEngine

pytestmark = pytest.mark.integration

_IMAGE = "alpine:3.20"
_PREFIX = "tensorstead-test-009-"


def _docker_or_skip() -> Any:
    """The Docker client, or skip the module if the daemon is unreachable."""
    docker = pytest.importorskip("docker", reason="docker SDK not installed")
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:  # daemon down, socket missing, permissions
        pytest.skip(f"no reachable Docker daemon: {type(exc).__name__}: {exc}")
    return client


@pytest.fixture(scope="module")
def client() -> Any:
    docker_client = _docker_or_skip()
    try:
        docker_client.images.get(_IMAGE)
    except Exception:
        docker_client.images.pull(_IMAGE)
    return docker_client


@pytest.fixture
def engine(client: Any) -> DockerEngine:
    return DockerEngine(client=client)


@pytest.fixture
def make_container(client: Any, request: pytest.FixtureRequest) -> Any:
    """Create a uniquely named container, removed however the test ends."""
    created: list[Any] = []

    def _make(*, command: list[str], labels: dict[str, str] | None = None, **kwargs: Any) -> Any:
        name = f"{_PREFIX}{request.node.name[:40]}-{len(created)}"
        with contextlib.suppress(Exception):
            client.containers.get(name).remove(force=True)
        container = client.containers.create(
            _IMAGE, command=command, name=name, labels=labels or {}, **kwargs
        )
        created.append(container)
        return container

    yield _make

    for container in created:
        with contextlib.suppress(Exception):
            container.remove(force=True)


def _wait_for(predicate: Any, *, timeout: float = 10.0, interval: float = 0.1) -> bool:
    """Poll until ``predicate`` holds. Real daemons are not instantaneous."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_absent_container_inspects_to_none(engine: DockerEngine) -> None:
    assert engine.inspect_container(f"{_PREFIX}definitely-not-created") is None


def test_running_container_reports_running(engine: DockerEngine, make_container: Any) -> None:
    container = make_container(command=["sleep", "60"])
    container.start()

    state = engine.inspect_container(container.name)

    assert state is not None
    assert state.running is True
    assert state.exit_code is None
    assert state.image_digest and state.image_digest.startswith("sha256:")


def test_exited_container_is_resolvable_but_not_running(
    engine: DockerEngine, client: Any, make_container: Any
) -> None:
    """The incident's root assumption, put to the daemon.

    `containers.get` succeeding proves the container was created, not that
    anything is alive inside it. Reading a digest through it and calling the
    answer "running" is what reported a dead deployment as healthy.
    """
    container = make_container(command=["sh", "-c", "exit 1"])
    container.start()
    assert _wait_for(lambda: client.containers.get(container.name).status == "exited")

    # The old code path: the name still resolves and still yields a digest.
    assert engine.get_digest(container.name) is not None

    state = engine.inspect_container(container.name)

    assert state is not None
    assert state.running is False, (
        "a real exited container was reported as running -- the assumption "
        "behind the 2026-08-10 incident holds against the daemon"
    )
    assert state.exit_code == 1
    assert state.detail == "exited"


def test_restarting_container_is_not_reported_as_running(
    engine: DockerEngine, client: Any, make_container: Any
) -> None:
    """A crash-looping runtime is not serving, whatever its restart policy says."""
    container = make_container(
        command=["sh", "-c", "exit 1"],
        restart_policy={"Name": "always"},
    )
    container.start()

    observed_restarting = _wait_for(
        lambda: client.containers.get(container.name).status == "restarting"
    )
    if not observed_restarting:
        pytest.skip("could not catch the container mid-restart on this host")

    state = engine.inspect_container(container.name)

    assert state is not None
    assert state.running is False, "a crash-looping container was reported as running"


def test_labels_round_trip_through_the_daemon(engine: DockerEngine, make_container: Any) -> None:
    """The agent recovers revision and endpoint from here."""
    labels = {
        "tensorstead.deployment_id": "01KZJ7AV2QGYD6SNH1QVYH0S80",
        "tensorstead.revision": "9",
        "tensorstead.endpoint": "0.0.0.0:8000",
        "tensorstead.runtime_type": "vllm",
    }
    container = make_container(command=["sleep", "60"], labels=labels)
    container.start()

    state = engine.inspect_container(container.name)

    assert state is not None
    for key, value in labels.items():
        assert state.labels.get(key) == value, f"label {key} did not survive the round trip"


def test_labels_survive_the_container_dying(
    engine: DockerEngine, client: Any, make_container: Any
) -> None:
    """Revision must still be reportable for a deployment that has died.

    This is why labels were chosen over agent-side state: the metadata lives on
    the object being described, so a dead deployment can still say which
    revision it was.
    """
    container = make_container(
        command=["sh", "-c", "exit 3"],
        labels={"tensorstead.revision": "9"},
    )
    container.start()
    assert _wait_for(lambda: client.containers.get(container.name).status == "exited")

    state = engine.inspect_container(container.name)

    assert state is not None
    assert state.running is False
    assert state.exit_code == 3
    assert state.labels.get("tensorstead.revision") == "9"


def test_a_container_without_labels_reports_none(engine: DockerEngine, make_container: Any) -> None:
    """A deployment created before container labels existed has none, and says so."""
    container = make_container(command=["sleep", "60"])
    container.start()

    state = engine.inspect_container(container.name)

    assert state is not None
    assert state.labels.get("tensorstead.revision") is None


def test_managed_namespace_enumeration_includes_stopped_containers(
    engine: DockerEngine, make_container: Any
) -> None:
    """The managed-namespace enumeration, asked of the real daemon.

    A container whose runtime died is the case most worth surfacing, so
    ``all=True`` is load-bearing rather than incidental — a listing that showed
    only live containers would hide exactly the wreckage an operator is looking
    for.
    """
    alive = make_container(command=["sleep", "60"], labels={"tensorstead.deployment_id": "01ALIVE"})
    alive.start()
    dead = make_container(command=["true"], labels={"tensorstead.deployment_id": "01DEAD"})
    dead.start()
    assert _wait_for(lambda: engine.inspect_container(dead.name) is not None)
    assert _wait_for(lambda: engine.inspect_container(dead.name).running is False)  # type: ignore[union-attr]

    by_name = {state.name: state for state in engine.list_managed_containers()}

    assert alive.name in by_name, "a running managed container was not enumerated"
    assert dead.name in by_name, "a dead container was filtered out of the namespace listing"
    assert by_name[alive.name].running is True
    assert by_name[dead.name].running is False
    assert by_name[alive.name].labels.get("tensorstead.deployment_id") == "01ALIVE"


def test_enumeration_is_confined_to_the_managed_prefix(
    engine: DockerEngine, client: Any, request: pytest.FixtureRequest
) -> None:
    """Docker's name filter is an unanchored substring match, so we re-check.

    ``my-tensorstead-thing`` contains the prefix without being in the namespace.
    Trusting the daemon's matching would have the product claim ownership of a
    container it never created.
    """
    name = f"not-{_PREFIX}{request.node.name[:30]}"
    with contextlib.suppress(Exception):
        client.containers.get(name).remove(force=True)
    outsider = client.containers.create(_IMAGE, command=["sleep", "60"], name=name)
    try:
        listed = {state.name for state in engine.list_managed_containers()}
        assert name not in listed, "a container outside the managed prefix was claimed"
    finally:
        with contextlib.suppress(Exception):
            outsider.remove(force=True)


# --------------------------------------------------- runtime output


def test_log_tail_carries_both_streams_demultiplexed(
    engine: DockerEngine, make_container: Any
) -> None:
    """Docker frames non-TTY output; a raw read would interleave binary headers.

    The Engine multiplexes stdout and stderr into one stream with an 8-byte
    frame header per chunk. If docker-py did not demultiplex, the log tail this
    product shows an operator would carry that framing — readable enough to look
    fine in a test that only greps for a substring, and corrupt in exactly the
    place a traceback matters. Asserted against the daemon rather than assumed.

    Both streams, because a runtime dying at startup writes to whichever it
    happens to use.
    """
    container = make_container(command=["sh", "-c", "echo to-stdout; echo to-stderr >&2"])
    container.start()
    assert _wait_for(lambda: engine.inspect_container(container.name).running is False)  # type: ignore[union-attr]

    logs = engine.container_logs(container.name)

    assert logs is not None
    assert "to-stdout" in logs
    assert "to-stderr" in logs
    assert "\x00" not in logs, "stream framing leaked into the operator-visible log tail"
    assert "\x01" not in logs


def test_log_tail_of_an_absent_container_is_none_not_empty(engine: DockerEngine) -> None:
    """`None` means we could not read; empty means the runtime said nothing."""
    assert engine.container_logs(f"{_PREFIX}never-created") is None


def test_the_effective_argv_joins_entrypoint_and_command(
    engine: DockerEngine, make_container: Any
) -> None:
    """The process did not experience them separately, so neither should a reader."""
    container = make_container(command=["-c", "sleep 30"], entrypoint=["/bin/sh"])
    container.start()

    state = engine.inspect_container(container.name)

    assert state is not None
    assert state.command == ["/bin/sh", "-c", "sleep 30"]


def test_a_crash_looping_container_reports_a_rising_relaunch_count(
    engine: DockerEngine, make_container: Any
) -> None:
    """The fact that separates a crash loop from a healthy runtime.

    Asserted against a container the daemon is genuinely restarting, because
    `RestartCount` is the Engine's own bookkeeping and nothing in this product
    could produce it.
    """
    container = make_container(
        command=["sh", "-c", "exit 1"],
        restart_policy={"Name": "on-failure", "MaximumRetryCount": 5},
    )
    container.start()

    assert _wait_for(
        lambda: (
            (
                engine.inspect_container(container.name) or ContainerState(running=False)
            ).restart_count
            not in (None, 0)
        ),
        timeout=20.0,
    ), "the daemon restarted the container but the seam reported no relaunches"


# ---------------------------------------- container-level requirements


def test_declared_requirements_reach_the_daemon(
    client: Any, request: pytest.FixtureRequest
) -> None:
    """Shared memory, IPC mode, ulimits, ports and environment, as Docker stores them.

    `ulimits` in particular is not a plain mapping at the API boundary — it
    wants `docker.types.Ulimit` objects — and getting that wrong would fail only
    at container creation, on the appliance, during the bring-up of a model
    somebody was waiting for.

    `DockerEngine.create_container` cannot be exercised here: it issues a GPU
    device request no Mac can satisfy. So the requirement-folding step is
    applied to the same create arguments and passed to the same API.
    """
    import docker as docker_module  # type: ignore[import-untyped]

    from tensorstead.agent.container_engine.docker_py import _apply_requirements
    from tensorstead.ports.runtime_adapter import ContainerRequirements

    name = f"{_PREFIX}{request.node.name[:40]}"
    with contextlib.suppress(Exception):
        client.containers.get(name).remove(force=True)

    create_args: dict[str, Any] = {
        "name": name,
        "command": ["sleep", "30"],
        "ports": {"8000/tcp": 8000},
    }
    _apply_requirements(
        create_args,
        ContainerRequirements(
            shm_size=2 * 1024 * 1024 * 1024,
            ipc_mode="host",
            extra_ports={"6379/tcp": 6379},
            ulimits={"memlock": (-1, -1)},
            environment={"NCCL_DEBUG": "INFO"},
        ),
        docker_module,
    )

    container = client.containers.create(_IMAGE, **create_args)
    try:
        container.reload()
        host_config = container.attrs["HostConfig"]
        assert host_config["ShmSize"] == 2 * 1024 * 1024 * 1024
        assert host_config["IpcMode"] == "host"
        assert host_config["Ulimits"] == [{"Name": "memlock", "Soft": -1, "Hard": -1}]
        assert "6379/tcp" in host_config["PortBindings"]
        assert any(e.startswith("NCCL_DEBUG=") for e in container.attrs["Config"]["Env"])
    finally:
        with contextlib.suppress(Exception):
            container.remove(force=True)


def test_a_requirement_cannot_move_the_declared_endpoint(
    client: Any, request: pytest.FixtureRequest
) -> None:
    """The port operators are told to connect to is not the runtime's to change.

    A requirement claiming the endpoint's own port would leave the product
    reporting an address nothing is listening on — the class of untruth this design
    removed from observed state, reintroduced through configuration.
    """
    import docker as docker_module

    from tensorstead.agent.container_engine.docker_py import _apply_requirements
    from tensorstead.ports.runtime_adapter import ContainerRequirements

    name = f"{_PREFIX}{request.node.name[:40]}"
    with contextlib.suppress(Exception):
        client.containers.get(name).remove(force=True)

    create_args: dict[str, Any] = {
        "name": name,
        "command": ["sleep", "30"],
        "ports": {"8000/tcp": 8000},
    }
    _apply_requirements(
        create_args,
        ContainerRequirements(extra_ports={"8000/tcp": 9999}),
        docker_module,
    )

    container = client.containers.create(_IMAGE, **create_args)
    try:
        container.reload()
        binding = container.attrs["HostConfig"]["PortBindings"]["8000/tcp"]
        assert binding[0]["HostPort"] == "8000", (
            "a runtime requirement overrode the deployment's declared endpoint"
        )
    finally:
        with contextlib.suppress(Exception):
            container.remove(force=True)


def test_the_engine_reports_what_is_running_not_what_was_configured(
    engine: DockerEngine, make_container: Any
) -> None:
    """Asked of the daemon rather than assumed.

    A container configured with one command line whose entrypoint runs another
    is the shape the contract forbids, and the only way to see it is to ask what is
    actually running. ``ContainerState.command`` reports the configured value
    and would look correct.
    """
    container = make_container(
        command=["--model", "/models/deepseek", "--tensor-parallel-size", "2"],
        entrypoint=["/bin/sh", "-c", "exec sleep 60"],
    )
    container.start()

    assert _wait_for(lambda: engine.container_processes(container.name) is not None)
    running = engine.container_processes(container.name)
    configured = engine.inspect_container(container.name)

    assert running is not None
    assert configured is not None
    assert "--model" in configured.command, "the configured argv should look correct"
    assert not any("/models/deepseek" in line for line in running), (
        "the entrypoint discarded the argv and the process listing did not show it"
    )

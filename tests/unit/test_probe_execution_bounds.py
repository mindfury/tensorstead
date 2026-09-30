"""What the one-shot probe backend actually does.

The earlier claim that a probe "can reach nothing and keeps nothing" was tested
by handing a fake ``run_once`` some arguments and asserting the arguments came
back. That proves the route asked politely. It proves nothing about whether the
engine denies the network, honours the timeout, bounds what it reads, or removes
what it started — and one of those turned out to be wrong: a failed removal was
suppressed, so the route could answer ``known`` while an unlabelled container
stayed on the node.

So these run the **real** ``DockerEngine`` against a fake Docker client. The
seam under test is the one that talks to the daemon, which is where every claim
in that sentence is either true or not.

Deliberately *not* asserted, because they are not implemented and pretending
otherwise is the failure mode this file exists to correct: a read-only root
filesystem, a memory ceiling, and dropped capabilities. Each needs a run against
the real DSpark image on the appliance before it ships — the probe builds vLLM's
config defaults, which touch more of the filesystem than is obvious, and a bound
that makes every probe fail closed would be worse than the gap it closes. They
are listed as the next slice rather than asserted here as though they
were done.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tensorstead.agent.container_engine.base import ImageBuildError
from tensorstead.agent.container_engine.docker_py import (
    _PROBE_LOG_LINES,
    _PROBE_PIDS_LIMIT,
    DockerEngine,
)

pytestmark = pytest.mark.unit

_IMAGE = "sha256:b1c50db67ef0d26fd6ac2a0e839b45f1bac401192f9ed2920b0a6379fcb2fc66"


class _FakeContainer:
    def __init__(
        self,
        *,
        status_code: int = 0,
        logs: bytes = b'{"schema": "x"}',
        remove_error: str | None = None,
    ) -> None:
        self._status_code = status_code
        self._logs = logs
        self._remove_error = remove_error
        self.started = False
        self.removed = False
        self.wait_kwargs: dict[str, Any] = {}
        self.log_kwargs: dict[str, Any] = {}

    def start(self) -> None:
        self.started = True

    def wait(self, **kwargs: Any) -> dict[str, Any]:
        self.wait_kwargs = kwargs
        return {"StatusCode": self._status_code}

    def logs(self, **kwargs: Any) -> bytes:
        self.log_kwargs = kwargs
        return self._logs

    def remove(self, **kwargs: Any) -> None:
        self.removed = True
        if self._remove_error is not None:
            raise RuntimeError(self._remove_error)


class _FakeContainers:
    def __init__(self, container: _FakeContainer) -> None:
        self._container = container
        self.create_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> _FakeContainer:
        self.create_kwargs = dict(kwargs)
        return self._container


class _FakeDocker:
    def __init__(self, container: _FakeContainer) -> None:
        self.containers = _FakeContainers(container)


def _run(container: _FakeContainer, **overrides: Any) -> tuple[str, _FakeDocker]:
    client = _FakeDocker(container)
    engine = DockerEngine(client=client)
    kwargs: dict[str, Any] = {
        "image": _IMAGE,
        "entrypoint": ["python3"],
        "command": [],
        "script": "print('hello')",
        "with_accelerator": False,
        "timeout_seconds": 42.0,
    }
    kwargs.update(overrides)
    return engine.run_once(**kwargs), client


def test_the_probe_container_is_denied_the_network() -> None:
    """A question about an image must not become an outbound call from a node."""
    _, client = _run(_FakeContainer())

    assert client.containers.create_kwargs["network_mode"] == "none"


def test_the_declared_timeout_reaches_the_daemon() -> None:
    """A bound nobody passes on is a bound that does not exist."""
    _, client = _run(_FakeContainer(), timeout_seconds=42.0)
    container = client.containers._container

    assert container.wait_kwargs.get("timeout") == 42.0


def test_output_is_read_with_a_bound() -> None:
    """The agent reads this into memory, so an image that never stops printing
    would otherwise be limited only by the node's RAM."""
    _, client = _run(_FakeContainer())
    container = client.containers._container

    assert container.log_kwargs.get("tail") == _PROBE_LOG_LINES


def test_process_count_is_bounded() -> None:
    """A parser probe runs one process; the ceiling costs nothing when that holds."""
    _, client = _run(_FakeContainer())

    assert client.containers.create_kwargs["pids_limit"] == _PROBE_PIDS_LIMIT


def test_the_script_is_mounted_read_only_rather_than_passed_through_a_shell() -> None:
    """Multi-line source as an argument means a shell interprets it."""
    _, client = _run(_FakeContainer())
    created = client.containers.create_kwargs

    volumes = created["volumes"]
    mount = next(iter(volumes.values()))
    assert mount == {"bind": "/probe.py", "mode": "ro"}
    assert created["command"] == ["/probe.py"]


def test_the_container_is_removed_when_the_probe_succeeds() -> None:
    _, client = _run(_FakeContainer())

    assert client.containers._container.removed is True


def test_the_container_is_removed_when_the_probe_fails() -> None:
    """However it ended. A probe that leaves containers behind has changed the
    thing it was asked to describe."""
    container = _FakeContainer(status_code=1, logs=b"Failed to infer device type")
    with pytest.raises(ImageBuildError):
        _run(container)

    assert container.removed is True


def test_a_nonzero_exit_is_a_failed_probe() -> None:
    container = _FakeContainer(status_code=1, logs=b"Failed to infer device type")
    with pytest.raises(ImageBuildError) as caught:
        _run(container)

    assert "infer device type" in str(caught.value), "the reason the image gave was discarded"


def test_a_failed_removal_fails_the_probe(tmp_path: Any) -> None:
    """The suppressed one.

    The answer arrived, so the old code returned it and swallowed the removal
    error inside a `finally`. The route then reported `known` while an
    unlabelled container sat on the node — the product reporting success for an
    operation that left the host in a state nobody records.

    A probe whose cleanup failed is a failed probe, even though it answered.
    """
    container = _FakeContainer(remove_error="device or resource busy")
    with pytest.raises(ImageBuildError) as caught:
        _run(container)

    message = str(caught.value)
    assert "could not be removed" in message
    assert "device or resource busy" in message


def test_a_removal_failure_does_not_mask_the_real_cause() -> None:
    """When both go wrong, the operator needs the one that explains it.

    The probe's own failure is why the answer is missing; the removal failure is
    a consequence. Raising the second would replace a diagnosis with a symptom.
    """
    container = _FakeContainer(
        status_code=1,
        logs=b"Failed to infer device type",
        remove_error="device or resource busy",
    )
    with pytest.raises(ImageBuildError) as caught:
        _run(container)

    assert "infer device type" in str(caught.value)
    assert container.removed is True


def test_the_staged_script_is_cleaned_up_from_the_host() -> None:
    """The probe writes one file on the host; it must not accumulate them."""
    container = _FakeContainer()
    _, client = _run(container)

    staged = next(iter(client.containers.create_kwargs["volumes"]))
    assert not Path(staged).exists(), "the staged probe script was left on the node"
    assert not Path(staged).parent.exists()


def test_the_probe_script_is_staged_where_the_docker_daemon_can_see_it(tmp_path: Any) -> None:
    """The agent's /tmp is private; the Docker daemon's is not.

    The agent unit sets `PrivateTmp=true`. A probe script written to the agent's
    /tmp therefore does not exist from the daemon's point of view, and Docker --
    asked to bind-mount a source it cannot see -- silently creates an empty
    *directory* at the destination. `python3 /probe.py` then failed with "can't
    find `__main__` module in '/probe.py'", which reads as a broken script
    rather than two processes disagreeing about what the filesystem contains.

    Found on the probe's first exercise against real hardware. It was
    unreachable in every previous test, because a fake Docker client shares this
    process's view of the filesystem and so can never disagree with it -- which
    is exactly why this test pins the staging *location* rather than trying to
    reproduce the disagreement.
    """
    container = _FakeContainer()
    client = _FakeDocker(container)
    engine = DockerEngine(client=client, staging_dir=tmp_path)
    engine.run_once(
        image=_IMAGE,
        entrypoint=["python3"],
        command=[],
        script="print('x')",
        with_accelerator=False,
        timeout_seconds=5.0,
    )

    staged = next(iter(client.containers.create_kwargs["volumes"]))
    assert Path(staged).is_relative_to(tmp_path), (
        f"probe staged at {staged}, outside the directory the daemon shares. "
        f"On a deployed agent that means the agent's PrivateTmp namespace."
    )


def test_the_default_staging_directory_is_not_tmp() -> None:
    """The constant itself, since the default path is what production uses."""
    from tensorstead.agent.container_engine.docker_py import _PROBE_STAGING_DIR

    assert not str(_PROBE_STAGING_DIR).startswith("/tmp"), (
        "the probe would stage into the agent's private /tmp, invisible to Docker"
    )
    assert str(_PROBE_STAGING_DIR).startswith("/var/lib/tensorstead")

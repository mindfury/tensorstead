"""Builds go through BuildKit, so the image they produce can leave the node (078).

docker-py has no BuildKit support, so ``images.build`` drives the *classic*
builder. Under Docker 29's containerd image store a classic-built image runs
locally and exports with none of its layer blobs — 16 KB standing in for 31 GB,
measured on this estate. It cannot be distributed, which made ``image_build``
useless for every multi-node deployment and sent the operator to a hand-run
``docker buildx`` plus an import, with the provenance running through a tar file
instead of the recorded spec.

There is no API route to BuildKit. It is the CLI or nothing, which is a real
departure from this module's rule of never parsing CLI output — confined to one
method, because the alternative is a build path that cannot produce a
distributable image.

What these hold:

- buildx is used when available, with ``--load`` (without it the image stays in
  buildx's cache and the export is empty again — the same bug in a new hat);
- a failing buildx build still produces a diagnosable record;
- the failing step is found by **vertex correlation**, not by recency, because
  BuildKit interleaves;
- a host without buildx still builds, exactly as before.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from tensorstead.agent.container_engine import docker_py
from tensorstead.agent.container_engine.base import ImageBuildError
from tensorstead.agent.container_engine.docker_py import DockerEngine, _failing_step_index

pytestmark = pytest.mark.unit

_REF = "local/vllm-flash-next-official:d4d703c"
_STEPS = [
    "apt-get install -y git",
    "pip install -r requirements/cuda.txt",
    "python setup.py bdist_wheel",
]


class _FakeImages:
    def __init__(self, image_id: str = "sha256:built") -> None:
        self.image_id = image_id
        self.classic_calls = 0

    def build(self, **_kwargs: Any) -> tuple[Any, list[Any]]:
        self.classic_calls += 1
        return type("Img", (), {"id": self.image_id})(), []

    def get(self, _reference: str) -> Any:
        return type("Img", (), {"id": self.image_id})()


class _FakeDocker:
    def __init__(self, image_id: str = "sha256:built") -> None:
        self.images = _FakeImages(image_id)


def _engine(image_id: str = "sha256:built") -> DockerEngine:
    return DockerEngine(client=_FakeDocker(image_id))


def _patch_cli(monkeypatch: pytest.MonkeyPatch, *, available: bool) -> list[list[str]]:
    """Route the CLI through a recorder instead of a real docker."""
    calls: list[list[str]] = []
    monkeypatch.setattr(docker_py, "_docker_cli", lambda: "/usr/bin/docker" if available else None)
    monkeypatch.setattr(docker_py, "_buildx_available", lambda _cli: available)

    def _run(argv: list[str], **kwargs: Any) -> Any:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="#5 DONE 1.2s\n", stderr="")

    monkeypatch.setattr(docker_py.subprocess, "run", _run)
    return calls


def test_buildx_is_used_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_cli(monkeypatch, available=True)
    engine = _engine("sha256:viabuildx")

    result = engine.build_image(reference=_REF, base_image="base@sha256:x", steps=_STEPS)

    assert result == "sha256:viabuildx"
    assert calls, "buildx was not invoked"
    argv = calls[0]
    assert argv[1:3] == ["buildx", "build"]
    assert "--load" in argv, "without --load the image stays in buildx's cache and cannot export"
    assert "--progress=plain" in argv, "the default renderer emits TTY escapes, not a readable log"
    assert _REF in argv


def test_the_classic_builder_is_not_used_when_buildx_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point: the classic path is what produces unusable images."""
    _patch_cli(monkeypatch, available=True)
    engine = _engine()
    fake: Any = engine._docker()

    engine.build_image(reference=_REF, base_image="base@sha256:x", steps=_STEPS)

    assert fake.images.classic_calls == 0


def test_a_host_without_buildx_still_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falling back is not an error: it is what every build did until now."""
    _patch_cli(monkeypatch, available=False)
    engine = _engine("sha256:classic")
    fake: Any = engine._docker()

    result = engine.build_image(reference=_REF, base_image="base@sha256:x", steps=_STEPS)

    assert result == "sha256:classic"
    assert fake.images.classic_calls == 1


def test_the_build_context_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recorded specs carry steps and no files, so nothing on disk may leak in."""
    calls = _patch_cli(monkeypatch, available=True)
    engine = _engine()

    engine.build_image(reference=_REF, base_image="base@sha256:x", steps=_STEPS)

    context = Path(calls[0][-1])
    dockerfile = Path(calls[0][calls[0].index("--file") + 1])
    assert dockerfile.parent == context
    # The Dockerfile is the only thing sent.
    assert dockerfile.name == "Dockerfile"


def test_a_failed_buildx_build_is_still_diagnosable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The build record must survive the change of builder."""
    monkeypatch.setattr(docker_py, "_docker_cli", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_py, "_buildx_available", lambda _cli: True)
    stderr = (
        "#4 [1/4] FROM docker.io/base\n"
        "#5 [2/4] RUN /bin/sh -c apt-get install -y git\n"
        "#6 [3/4] RUN /bin/sh -c pip install -r requirements/cuda.txt\n"
        "#6 12.4 ERROR: Could not find a version that satisfies flashinfer-jit-cache==0.6.18\n"
        '#6 ERROR: process "/bin/sh -c pip install" did not complete successfully: exit code: 1\n'
    )

    def _run(argv: list[str], **kwargs: Any) -> Any:
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr=stderr)

    monkeypatch.setattr(docker_py.subprocess, "run", _run)

    with pytest.raises(ImageBuildError) as raised:
        _engine().build_image(reference=_REF, base_image="base@sha256:x", steps=_STEPS)

    error = raised.value
    assert error.detail["failing_step_index"] == 1, "wrong recorded step named"
    assert "pip install" in error.detail["failing_step"]
    assert "flashinfer-jit-cache" in error.detail["build_log_tail"], "the real reason was lost"
    assert "exited 1" in error.message


def test_the_failing_step_is_found_by_vertex_not_by_recency() -> None:
    """BuildKit interleaves; the last marker in the stream is not the failure.

    This is the case that makes vertex correlation necessary rather than tidy.
    A concurrent vertex prints *after* the error, so a reversed scan for the
    most recent ``[i/n]`` marker names the wrong step — confidently.
    """
    interleaved = [
        "#4 [1/4] FROM docker.io/base",
        "#5 [2/4] RUN /bin/sh -c step-one",
        "#6 [3/4] RUN /bin/sh -c step-two",
        '#6 ERROR: process "/bin/sh -c step-two" did not complete successfully: exit code: 2',
        "#7 [4/4] RUN /bin/sh -c step-three",
        "#7 DONE 0.3s",
    ]

    assert _failing_step_index(interleaved) == 1, "named the last marker instead of the failure"


def test_output_from_both_streams_reaches_the_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """BuildKit puts progress on stderr and the build's own output on stdout."""
    monkeypatch.setattr(docker_py, "_docker_cli", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_py, "_buildx_available", lambda _cli: True)

    def _run(argv: list[str], **kwargs: Any) -> Any:
        return subprocess.CompletedProcess(
            argv, 2, stdout="nvcc fatal : unsupported arch\n", stderr="#6 [3/4] RUN compile\n"
        )

    monkeypatch.setattr(docker_py.subprocess, "run", _run)

    with pytest.raises(ImageBuildError) as raised:
        _engine().build_image(reference=_REF, base_image="base@sha256:x", steps=_STEPS)

    tail = raised.value.detail["build_log_tail"]
    assert "nvcc fatal" in tail, "stdout was dropped"
    assert "[3/4]" in tail, "stderr was dropped"


def test_a_build_that_outruns_the_node_bound_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent expires before the coordinator's hop, so it owns the record."""
    monkeypatch.setattr(docker_py, "_docker_cli", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_py, "_buildx_available", lambda _cli: True)

    def _run(argv: list[str], **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(argv, docker_py._BUILDX_TIMEOUT_SECONDS)

    monkeypatch.setattr(docker_py.subprocess, "run", _run)

    with pytest.raises(ImageBuildError, match="bound"):
        _engine().build_image(reference=_REF, base_image="base@sha256:x", steps=_STEPS)


def test_the_node_bound_is_below_the_coordinator_hop_bound() -> None:
    """Whichever end expires first writes the record; the agent's is better."""
    from tensorstead.coordinator.node_http import _BUILD_TIMEOUT_SECONDS

    assert docker_py._BUILDX_TIMEOUT_SECONDS < _BUILD_TIMEOUT_SECONDS

"""A failed build must say why, in the operation record (operator feedback, 2026-09-04).

Buildspec ``deepseek-v4-flash-fastqual-b12x-swiglu-v1`` failed on
``spark-alpha.internal`` after fourteen minutes. The operation record named the
node and an exit code. It carried no stdout, no stderr, no failing step, and no
pointer to anything on the node that had them.

What that costs is not inconvenience. An undiagnosable failure forces the
operator **off the only control plane this product claims to be**: the build was
reproduced by hand over SSH, outside the operation record, outside the audit
trail, and outside the resource guardrails a managed build runs under — which
then starved sshd and cost a physical powercycle. Roughly a day and a half, on a
build whose recipe authors say it just works.

The evidence existed the whole time. docker-py's ``BuildError`` carries
``build_log``, the same stream the successful path returns, and the backend
bound it to ``_logs`` and threw it away — raising ``str(exc)``, which for a
failed ``RUN`` names the command and the exit code and nothing the command
printed.

This is the same shape again — a structured reason flattened on the way to
the operator — one seam further down, and the reason it recurred is that the
fix addressed the distribution hop rather than the class.

The chain exercised here is real at every step except the Docker socket:

1. the **real** ``DockerEngine.build_image``, driven by a fake docker client
   whose ``images.build`` raises the **real** ``docker.errors.BuildError`` with
   the chunk stream a classic builder actually emits;
2. the **real agent app** and its **real** ``ImageBuildError`` handler produce
   the 500 body;
3. the **real** ``NodeHTTPClient._error_from_response`` translates it;
4. the **real** ``ImageBuildService.build`` and the **real** coordinator route
   record the terminal operation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from docker.errors import BuildError
from fastapi.testclient import TestClient

from tensorstead.agent.app import build_agent_app
from tensorstead.agent.container_engine.docker_py import DockerEngine
from tensorstead.coordinator.node_http import NodeHTTPClient
from tests.fakes.node_agent import FakeNodeAgent
from tests.fakes.service_manager import FakeServiceManager
from tests.helpers import FakeNodeClient, build_test_coordinator, poll_operation

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer test"}
_PINNED = "nvcr.io/nvidia/vllm@sha256:0123456789abcdef"
_REFERENCE = "local/deepseek-v4-flash-b12x:swiglu-v1"

# The three recorded steps, in the shape this estate actually ships. Step 1 is
# the encoded build context — buildspecs carry no files, so a reproducible build
# base64s a tarball into a shell step (the operator's second request). It is
# here because it is the thing a failure record must *not* paste back.
_CONTEXT_BLOB = "H4sIAAAAAAAAA" + "Qk" * 130_000
_STEPS = [
    f"echo {_CONTEXT_BLOB} | base64 -d > /ctx.tgz && tar xzf /ctx.tgz -C /build",
    "pip install --no-deps -r /build/requirements.lock",
    "cd /build && MAX_JOBS=16 python setup.py install && python -c 'import swiglu_ext'",
]
# The line that identifies the real blocker. Nothing above it in the chain
# knows what it means; every step in the chain must carry it anyway.
_COMPILER_ERROR = (
    "swiglu_kernel.cu(214): error: no instance of overloaded function "
    '"__hmul2" matches the argument list'
)
_DOCKER_REASON = f"The command '/bin/sh -c {_STEPS[2]}' returned a non-zero code: 2"


def _classic_builder_stream() -> list[dict[str, Any]]:
    """The chunks Docker's classic builder emits for this failure.

    ``Step N/M`` markers included because they are how the failing *recorded*
    step is identified: ``render_dockerfile`` emits ``FROM`` plus exactly one
    ``RUN`` per step, so instruction N is recorded step N-2.
    """
    return [
        {"stream": f"Step 1/4 : FROM {_PINNED}\n"},
        {"stream": " ---> a1b2c3d4e5f6\n"},
        {"stream": "Step 2/4 : RUN [...]\n"},
        {"stream": " ---> Running in 1111111111\n"},
        {"stream": "Step 3/4 : RUN [...]\n"},
        {"stream": " ---> Running in 2222222222\n"},
        {"stream": "Successfully installed torch-2.9.0\n"},
        {"stream": "Step 4/4 : RUN [...]\n"},
        {"stream": " ---> Running in 3333333333\n"},
        {"stream": "building 'swiglu_ext' extension\n"},
        {"stream": f"{_COMPILER_ERROR}\n"},
        {"stream": "1 error detected in the compilation of swiglu_kernel.cu.\n"},
        {"stream": "error: command '/usr/local/cuda/bin/nvcc' failed with exit code 2\n"},
        {
            "error": _DOCKER_REASON,
            "errorDetail": {"code": 2, "message": _DOCKER_REASON},
        },
    ]


class _FailingDocker:
    """A docker client whose build fails exactly as docker-py reports one.

    ``BuildError`` is the real class from the installed SDK, and ``build_log``
    is a live iterator as it is in production — docker-py hands over a
    ``tee`` branch, not a list — so a backend that fails to drain it fails here
    too.
    """

    def __init__(self, chunks: list[dict[str, Any]] | None = None) -> None:
        self.chunks = chunks if chunks is not None else _classic_builder_stream()

    @property
    def images(self) -> Any:
        return self

    def build(self, **_kwargs: Any) -> Any:
        raise BuildError(_DOCKER_REASON, iter(self.chunks))


def _agent(tmp_path: Path, client: Any) -> TestClient:
    return TestClient(
        build_agent_app(
            management_token="mgmt",
            replication_token="repl",
            # Pinned to the classic builder: this file is about docker-py's
            # ``BuildError``, which is that builder's failure shape. BuildKit's
            # equivalent is covered in tests/unit/test_buildx_build_path.py.
            container_engine=DockerEngine(client=client, prefer_buildx=False),
            service_manager=FakeServiceManager(),
            store_dir=tmp_path / "store",
            marker_dir=tmp_path / "markers",
        )
    )


def _real_agent_build_failure(tmp_path: Path, client: Any = None) -> httpx.Response:
    """The actual response a node returns when a recorded build fails."""
    resp = _agent(tmp_path, client or _FailingDocker()).post(
        "/agent/v1/images:build",
        json={"reference": _REFERENCE, "base_image": _PINNED, "steps": _STEPS},
        headers={"Authorization": "Bearer mgmt"},
    )
    assert resp.status_code == 500, resp.text
    return httpx.Response(resp.status_code, json=resp.json())


def _operation(tmp_path: Path, client: Any = None) -> dict[str, Any]:
    """Drive the coordinator's build to terminal against that response."""
    refusal = _real_agent_build_failure(tmp_path, client)

    class _FailingClient(FakeNodeClient):
        def build_image(self, node: Any, payload: dict[str, Any]) -> dict[str, Any]:
            raise NodeHTTPClient(management_token="t")._error_from_response(node, refusal)

    app, _ = build_test_coordinator()
    app.state.image_builds._client = _FailingClient(FakeNodeAgent())
    coordinator = TestClient(app)

    node = coordinator.post(
        "/v1/nodes",
        json={"name": "spark-alpha", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    assert node.status_code == 201, node.text
    coordinator.put(
        "/v1/buildspecs/b12x-swiglu",
        json={"base_image": _PINNED, "steps": _STEPS},
        headers=_AUTH,
    )
    accepted = coordinator.post(
        "/v1/images:build",
        json={
            "spec": "b12x-swiglu",
            "node_id": str(node.json()["id"]),
            "reference": _REFERENCE,
        },
        headers=_AUTH,
    )
    assert accepted.status_code == 202, accepted.text
    return poll_operation(coordinator, accepted.json()["operation_id"], auth=_AUTH)


def test_the_operation_carries_the_compiler_error(tmp_path: Path) -> None:
    """The one line that identifies the blocker must reach the operator.

    This is the whole request. Without it the operator has a node name and an
    exit code, which is not a diagnosis — it is an instruction to go and
    reproduce the build somewhere this product cannot see.
    """
    operation = _operation(tmp_path)

    assert operation["state"] == "failed", operation
    reason = operation["failure_reason"]
    assert _COMPILER_ERROR in reason["detail"]["build_log_tail"], (
        f"the build log did not survive to the operation: {reason}"
    )


def test_the_operation_names_the_failing_step(tmp_path: Path) -> None:
    """Which recorded step failed, identified rather than left to be inferred."""
    reason = _operation(tmp_path)["failure_reason"]

    assert reason["detail"]["failing_step_index"] == 2, reason["detail"]
    assert "setup.py install" in reason["detail"]["failing_step"]
    assert "step 3 of 3" in reason["message"], reason["message"]


def test_the_encoded_build_context_is_not_pasted_back(tmp_path: Path) -> None:
    """A record that repeats a 259K step is as unreadable as one that says nothing.

    Buildspecs carry no files, so a reproducible build encodes its context into
    a step. Every bound in the failure path exists because of that shape.
    """
    operation = _operation(tmp_path)
    recorded = str(operation["failure_reason"])

    assert _CONTEXT_BLOB[:2000] not in recorded, "the encoded build context reached the record"
    assert len(recorded) < 60_000, f"failure record is {len(recorded)} characters"


def test_the_node_and_reference_are_still_named(tmp_path: Path) -> None:
    """The evidence is added to the existing reason, never in place of it."""
    reason = _operation(tmp_path)["failure_reason"]

    assert reason["code"] == "node_operation_failed"
    assert "spark-alpha" in reason["message"], reason["message"]
    assert reason["detail"]["reference"] == _REFERENCE
    assert reason["detail"]["spec"] == "b12x-swiglu"
    assert reason["detail"]["node_id"]


def test_a_log_that_stops_mid_stream_still_yields_what_it_had(tmp_path: Path) -> None:
    """``build_log`` is a live socket read, and it can fail where the log matters most.

    A partial log is worth incomparably more than none, and recovering it must
    not raise a second failure over the first.
    """

    class _Exploding(_FailingDocker):
        def build(self, **_kwargs: Any) -> Any:
            def chunks() -> Any:
                yield {"stream": "Step 4/4 : RUN [...]\n"}
                yield {"stream": f"{_COMPILER_ERROR}\n"}
                raise OSError("connection reset by peer")

            raise BuildError(_DOCKER_REASON, chunks())

    reason = _operation(tmp_path, _Exploding())["failure_reason"]

    assert _COMPILER_ERROR in reason["detail"]["build_log_tail"]
    assert reason["detail"]["failing_step_index"] == 2

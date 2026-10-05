"""The published port must be the runtime's own, not vLLM's.

``create_container`` mapped the deployment's endpoint to container port 8000 for
every runtime, with a comment naming it as vLLM's default. ``llama-server``
listens on 8080, and the shipped llama.cpp adapter neither declared a port nor
could be given one — ``port`` is a forbidden ``extra_args`` key, correctly, since
the endpoint is declared on the deployment.

So a llama.cpp deployment would start, report running, pass a container-liveness
check, and map its recorded endpoint to a container port with nothing behind it.
The record and the reality disagree, and nothing compared them — the same shape
as every other finding in this batch.

The runtime-agnostic agent held one runtime's fact. That is exactly what the
adapter seam exists to prevent, and it is the third time this specific mistake
has been found before.
"""

from __future__ import annotations

from typing import Any

import pytest

from tensorstead.adapters.runtimes.exllama import ExLlamaAdapter
from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter
from tensorstead.adapters.runtimes.sglang import SGLangAdapter
from tensorstead.adapters.runtimes.trtllm import TRTLLMAdapter
from tensorstead.adapters.runtimes.vllm import VLLMAdapter
from tensorstead.agent.container_engine.docker_py import DockerEngine
from tensorstead.ports.runtime_adapter import ContainerRequirements

pytestmark = pytest.mark.unit


def test_vllm_declares_its_own_listener() -> None:
    assert VLLMAdapter().container_requirements({}).api_port == 8000


def test_llamacpp_declares_its_own_listener() -> None:
    """8080, as this project's own CLI documentation already stated."""
    assert LlamaCppAdapter().container_requirements({}).api_port == 8080


def test_every_shipped_adapter_declares_a_port() -> None:
    """A guardrail, so "forgot to declare" is unreachable for a shipped runtime.

    The fallback in the engine exists only for a call that supplies no
    requirements at all. If a shipped adapter ever relies on it, that adapter
    silently inherits vLLM's port — which is the defect this file is about.
    """
    for adapter in (
        VLLMAdapter(),
        LlamaCppAdapter(),
        SGLangAdapter(),
        ExLlamaAdapter(),
        TRTLLMAdapter(),
    ):
        declared = adapter.container_requirements({}).api_port
        assert declared, f"{type(adapter).__name__} declares no in-container API port"


def _created(requirements: ContainerRequirements | None) -> dict[str, Any]:
    """Create a container against a fake docker client, returning the kwargs."""
    captured: dict[str, Any] = {}

    class _Containers:
        def create(self, image: str, **kwargs: Any) -> Any:
            captured.update(kwargs)
            captured["image"] = image
            return type("C", (), {"id": "cid"})()

    class _Client:
        containers = _Containers()

    DockerEngine(client=_Client()).create_container(
        name="tensorstead-d1",
        image="repo/runtime:tag",
        endpoint="10.0.0.11:9001",
        model_path="/srv/models/m",
        command_args=["--model", "/srv/models/m"],
        requirements=requirements,
    )
    return captured


def test_the_declared_port_is_the_one_published() -> None:
    """The wiring, not just the declaration — a fix not called is not a fix."""
    created = _created(LlamaCppAdapter().container_requirements({}))

    assert created["ports"] == {"8080/tcp": 9001}


def test_vllms_mapping_is_unchanged() -> None:
    """Running vLLM deployments must be untouched by this change."""
    created = _created(VLLMAdapter().container_requirements({}))

    assert created["ports"] == {"8000/tcp": 9001}


def test_the_host_side_port_still_comes_from_the_endpoint() -> None:
    """Only the container side moved; the endpoint still selects the host port."""
    created = _created(ContainerRequirements(api_port=5000))

    assert created["ports"] == {"5000/tcp": 9001}


def test_a_call_with_no_requirements_behaves_as_before() -> None:
    """The pre-spec shape keeps its historical port rather than failing."""
    assert _created(None)["ports"] == {"8000/tcp": 9001}

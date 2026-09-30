"""The managed inference credential wins the environment merge, not loses it.


``create_container`` used to set ``environment`` (the credential the agent
route resolved) *before* ``_apply_requirements`` folded in the runtime's own
``environment`` (``host_config.environment``, an operator-declared,
unvalidated passthrough) -- and that fold overwrote on any key collision.
The schema now refuses a deployment that names the credential's own variable
there (``test_container_requirements.py``); this is the second, independent
layer: even a value that reached this call some other way must still lose
to the credential, not win against it.
"""

from __future__ import annotations

from typing import Any

import pytest

from tensorstead.agent.container_engine.docker_py import DockerEngine
from tensorstead.ports.runtime_adapter import ContainerRequirements

pytestmark = pytest.mark.unit


def _created(*, environment: dict[str, str] | None, requirements: ContainerRequirements) -> dict:
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
        environment=environment,
        requirements=requirements,
    )
    return captured


def test_the_credential_wins_a_collision_with_requirements_environment() -> None:
    """The core proof: even if a colliding value reaches this call, it loses."""
    created = _created(
        environment={"VLLM_API_KEY": "the-real-managed-secret"},
        requirements=ContainerRequirements(environment={"VLLM_API_KEY": "attacker-chosen"}),
    )

    assert created["environment"]["VLLM_API_KEY"] == "the-real-managed-secret"


def test_non_colliding_requirements_environment_still_applies() -> None:
    """The merge is real, not the credential simply replacing everything."""
    created = _created(
        environment={"VLLM_API_KEY": "the-real-managed-secret"},
        requirements=ContainerRequirements(environment={"VLLM_CACHE_ROOT": "/somewhere"}),
    )

    assert created["environment"] == {
        "VLLM_API_KEY": "the-real-managed-secret",
        "VLLM_CACHE_ROOT": "/somewhere",
    }


def test_no_credential_leaves_requirements_environment_untouched() -> None:
    """A deployment with no bound credential must not lose its own environment."""
    created = _created(
        environment=None,
        requirements=ContainerRequirements(environment={"VLLM_CACHE_ROOT": "/somewhere"}),
    )

    assert created["environment"] == {"VLLM_CACHE_ROOT": "/somewhere"}

"""Where a deployment's model appears inside its container."""

from __future__ import annotations

import pytest

from tensorstead.agent.container_engine.docker_py import _model_mount_target
from tensorstead.ports.runtime_adapter import ContainerRequirements

pytestmark = pytest.mark.unit

_STORE_PATH = "/var/lib/tensorstead/models/aHVnZ2luZ2ZhY2U6UXdlbi9Rd2VuMy0xLjdC"


def test_the_model_mounts_at_its_host_path_by_default() -> None:
    assert _model_mount_target(_STORE_PATH, None) == _STORE_PATH
    assert _model_mount_target(_STORE_PATH, ContainerRequirements()) == _STORE_PATH


def test_an_adapter_may_rename_the_destination() -> None:
    requirements = ContainerRequirements(model_mount_target="/models/qwen3")
    assert _model_mount_target(_STORE_PATH, requirements) == "/models/qwen3"


@pytest.mark.parametrize("target", ["models/qwen3", "/models/../etc", "/"])
def test_a_destination_must_be_absolute_and_below_root(target: str) -> None:
    with pytest.raises(ValueError):
        _model_mount_target(_STORE_PATH, ContainerRequirements(model_mount_target=target))

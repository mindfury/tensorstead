"""An image copied from a peer must start and be removable, not merely "found".

``docker save``/``load`` drops a ``name@sha256:`` reference, so a peer-copied,
digest-pinned image is held only by its id (measured on the Sparks,
2026-10-16). The first fix made the agent *find* it by that id and then still
created the container from the unresolvable name -- which on a real daemon fails
with "no such image". These pin that every daemon call uses what the image is
actually held as.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tensorstead.agent.app import build_agent_app
from tests.fakes.container_engine import FakeContainerEngine
from tests.fakes.service_manager import FakeServiceManager
from tests.helpers import make_vllm_model_dir

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer mgmt"}
_DIGEST = "sha256:" + "2e" * 32
_PINNED = f"repo/vllm@{_DIGEST}"


def _agent(engine: Any, tmp_path: Path) -> TestClient:
    return TestClient(
        build_agent_app(
            management_token="mgmt",
            container_engine=engine,
            service_manager=FakeServiceManager(),
            store_dir=tmp_path / "store",
            marker_dir=tmp_path / "markers",
        )
    )


def _copied_engine() -> FakeContainerEngine:
    engine = FakeContainerEngine()
    engine.local_images[_DIGEST] = _DIGEST  # held by id only, as after docker load
    engine.unpullable.add(_PINNED)  # a pull would be the bug, so make it fail
    return engine


def test_a_peer_copied_image_starts_from_the_id_it_is_held_by(tmp_path: Path) -> None:
    engine = _copied_engine()
    resp = _agent(engine, tmp_path).post(
        "/agent/v1/deployments",
        json={
            "deployment_id": "01J00000000000000000000000",
            "revision": 1,
            "runtime_type": "vllm",
            "image_reference": _PINNED,
            "runtime_config": {"tensor_parallel_size": 1},
            "model_path": make_vllm_model_dir(tmp_path),
            "endpoint": "0.0.0.0:8000",
        },
        headers=_AUTH,
    )

    assert resp.status_code == 200, resp.text
    assert engine.pulled == []
    (container,) = engine.containers.values()
    assert container.image_reference == _DIGEST, "created from the name the daemon cannot resolve"


def test_a_peer_copied_image_is_actually_removed(tmp_path: Path) -> None:
    """Not reported "not held" while its bytes stay -- that orphaned them on the Sparks."""
    engine = _copied_engine()
    resp = _agent(engine, tmp_path).post(
        "/agent/v1/images:remove", json={"reference": _PINNED}, headers=_AUTH
    )

    assert resp.status_code == 200
    assert resp.json()["removed"] is True
    assert engine.removed_images == [_DIGEST]
    assert _DIGEST not in engine.local_images

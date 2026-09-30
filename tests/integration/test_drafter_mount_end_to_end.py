"""A drafter reaches the container, through the whole create path.

The unit tests pin the two halves separately: the adapter names where a drafter
was configured, and the acquisition service decides whether a named thing is
something this node owns. Neither proves they are wired to each other, and the
wiring is exactly where this failed on hardware -- the adapter had always been
able to describe its needs, and the agent had always mounted precisely one model
directory, and nothing connected the two.

So these go through the real agent route: POST a deployment, then look at what
the engine was told to mount.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from httpx2 import Response

pytestmark = pytest.mark.integration


def _agent(tmp_path: Path) -> tuple[FastAPI, TestClient]:
    from fastapi.testclient import TestClient

    from tensorstead.agent.app import build_agent_app
    from tests.fakes.container_engine import FakeContainerEngine
    from tests.fakes.service_manager import FakeServiceManager

    app = build_agent_app(
        management_token="test",
        container_engine=FakeContainerEngine(),
        service_manager=FakeServiceManager(),
        store_dir=tmp_path / "store",
        marker_dir=tmp_path / "markers",
    )
    return app, TestClient(app)


def _start(
    client: TestClient, deployment_id: str, speculative: dict | None, model_path: str
) -> Response:
    config: dict = {"tensor_parallel_size": 1}
    if speculative is not None:
        config["speculative_config"] = speculative
    return client.post(
        "/agent/v1/deployments",
        json={
            "deployment_id": deployment_id,
            "revision": 1,
            "runtime_type": "vllm",
            "image_reference": "local/vllm:test",
            "runtime_config": config,
            "model_path": model_path,
            "endpoint": "0.0.0.0:8000",
        },
        headers={"Authorization": "Bearer test"},
    )


def test_an_acquired_drafter_is_mounted(tmp_path: Path) -> None:
    """The capability, end to end.

    Before this, a drafter acquired onto the node could not be referenced by
    path at all: vLLM rejected it with "Invalid repository ID or local directory
    specified", because the directory existed on the host and not in the
    container.
    """
    from tests.helpers import make_vllm_model_dir

    app, client = _agent(tmp_path)
    model_path = make_vllm_model_dir(tmp_path)
    drafter = tmp_path / "store" / "aHVnZ2luZ2ZhY2U6RHJhZnRlcg"
    drafter.mkdir(parents=True)

    response = _start(
        client,
        "01J00000000000000000000011",
        {"method": "dspark", "model": str(drafter), "num_speculative_tokens": 7},
        model_path,
    )
    assert response.status_code == 200, response.text

    container = app.state.container_engine.containers["tensorstead-01J00000000000000000000011"]
    assert str(drafter.resolve()) in container.extra_model_paths


def test_a_repo_id_drafter_mounts_nothing(tmp_path: Path) -> None:
    """The pre-existing behaviour is untouched.

    A HuggingFace id is not a path, so nothing is mounted and the runtime
    fetches it itself -- which is how every DSpark deployment ran before this
    existed. Had this regressed, those deployments would break on upgrade.
    """
    from tests.helpers import make_vllm_model_dir

    app, client = _agent(tmp_path)
    model_path = make_vllm_model_dir(tmp_path)

    response = _start(
        client,
        "01J00000000000000000000012",
        {"method": "dspark", "model": "Org/Drafter", "num_speculative_tokens": 7},
        model_path,
    )
    assert response.status_code == 200, response.text

    container = app.state.container_engine.containers["tensorstead-01J00000000000000000000012"]
    assert container.extra_model_paths == []


def test_a_drafter_outside_the_store_mounts_nothing(tmp_path: Path) -> None:
    """The security property, through the route that would grant it.

    An operator naming a host directory in ``speculative_config.model`` must not
    thereby mount it. The deployment still starts -- refusing would turn a
    configuration mistake into an outage -- and the runtime is left to fail on
    its own terms, which it does, loudly, at startup.
    """
    from tests.helpers import make_vllm_model_dir

    app, client = _agent(tmp_path)
    model_path = make_vllm_model_dir(tmp_path)
    outside = tmp_path / "not-the-store"
    outside.mkdir()

    response = _start(
        client,
        "01J00000000000000000000013",
        {"method": "dspark", "model": str(outside), "num_speculative_tokens": 7},
        model_path,
    )
    assert response.status_code == 200, response.text

    container = app.state.container_engine.containers["tensorstead-01J00000000000000000000013"]
    assert container.extra_model_paths == []


def test_mtp_mounts_nothing_extra(tmp_path: Path) -> None:
    """The common case stays exactly as it was.

    MTP's head ships inside the target checkpoint and shares its embeddings and
    lm_head, so there are no second weights to mount -- and a deployment that
    named no drafter must not acquire a mount because this feature exists.
    """
    from tests.helpers import make_vllm_model_dir

    app, client = _agent(tmp_path)
    model_path = make_vllm_model_dir(tmp_path)

    response = _start(
        client,
        "01J00000000000000000000014",
        {"method": "mtp", "num_speculative_tokens": 5},
        model_path,
    )
    assert response.status_code == 200, response.text

    container = app.state.container_engine.containers["tensorstead-01J00000000000000000000014"]
    assert container.extra_model_paths == []

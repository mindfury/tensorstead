"""The managed archive-import route refuses a mismatched image and cleans up
after itself.

``POST /agent/v1/images:import`` shared the same defect as peer distribution:
``import_image`` loads the archive into the daemon before its identity is
compared against what the caller declared. On mismatch the route already
refused to record the image but left the wrongly-loaded image
sitting in the daemon under whatever id/tags the archive itself carried --
this is what closes that half, and there was no existing contract coverage
of this route's mismatch path at all before this file.
"""

from __future__ import annotations

from pathlib import Path

import httpx2
import pytest
from fastapi.testclient import TestClient

from tensorstead.agent.app import build_agent_app
from tests.fakes.container_engine import FakeContainerEngine

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer mgmt"}


@pytest.fixture
def engine() -> FakeContainerEngine:
    return FakeContainerEngine()


@pytest.fixture
def client(tmp_path: Path, engine: FakeContainerEngine) -> TestClient:
    store = tmp_path / "images"
    store.mkdir()
    (store / "archive.tar").write_bytes(b"not a real archive, the fake engine ignores it")
    app = build_agent_app(
        management_token="mgmt",
        container_engine=engine,
        image_store_path=str(store),
    )
    return TestClient(app)


def _import(client: TestClient, expected_image_id: str | None) -> httpx2.Response:
    # ``httpx2``, not ``httpx``: both majors are installed here, and Starlette's
    # TestClient subclasses ``httpx2.Client``, so this is the response type it
    # actually returns. Annotating it ``object`` -- which is what this helper
    # said before -- typechecks but makes every ``resp.status_code`` below an
    # error, which is how this file failed ``make check``.

    return client.post(
        "/agent/v1/images:import",
        json={
            "archive_name": "archive.tar",
            "reference": "local/v:1",
            "expected_image_id": expected_image_id,
        },
        headers=_AUTH,
    )


def test_a_matching_archive_is_accepted_and_nothing_is_removed(
    client: TestClient, engine: FakeContainerEngine
) -> None:
    engine.import_image_returns = "sha256:right"

    resp = _import(client, "sha256:right")

    assert resp.status_code == 200, resp.text
    assert resp.json()["image_id"] == "sha256:right"
    assert engine.removed_images == []


def test_a_mismatched_archive_is_refused_and_removed_from_the_daemon(
    client: TestClient, engine: FakeContainerEngine
) -> None:
    """The core proof: rejection must not leave the wrong image behind."""
    engine.import_image_returns = "sha256:wrong"

    resp = _import(client, "sha256:right")

    assert resp.status_code != 200, resp.text
    message = resp.json()["message"]
    assert "sha256:wrong" in message
    assert "sha256:right" in message
    assert engine.removed_images == ["sha256:wrong"], (
        "a rejected import must be removed from the daemon, not merely unrecorded"
    )


def test_no_expected_id_means_no_verification_and_no_removal(
    client: TestClient, engine: FakeContainerEngine
) -> None:
    """Unchanged prior behaviour: verification is opt-in via expected_image_id."""
    engine.import_image_returns = "sha256:whatever"

    resp = _import(client, None)

    assert resp.status_code == 200, resp.text
    assert engine.removed_images == []


def test_an_unmaterializable_archive_is_refused_and_removed(
    client: TestClient, engine: FakeContainerEngine
) -> None:
    """Loading is not availability.

    The id matches, so every check this route made before now was satisfied.
    What the archive did not bring is the config content, which nothing
    discovers until a container is created -- previously two minutes later, on
    the deployment route, as an unattributed 500.
    """
    engine.import_image_returns = "sha256:right"
    engine.unmaterializable.add("sha256:right")

    resp = _import(client, "sha256:right")

    assert resp.status_code != 200, resp.text
    assert "no container can be created" in resp.json()["message"]
    assert engine.removed_images == ["sha256:right"], (
        "an image that cannot be materialized poisons every later load of the "
        "same id, so it must be removed rather than merely unrecorded"
    )


def test_verification_runs_even_when_no_expected_id_is_given(
    client: TestClient, engine: FakeContainerEngine
) -> None:
    """Integrity is not opt-in the way identity is.

    ``expected_image_id`` is a question about *which* image the caller wanted,
    which only the caller can answer. Whether the image works at all is not a
    matter of caller opinion, so it is checked on every import.
    """
    engine.import_image_returns = "sha256:whatever"
    engine.unmaterializable.add("sha256:whatever")

    resp = _import(client, None)

    assert resp.status_code != 200, resp.text
    assert engine.removed_images == ["sha256:whatever"]

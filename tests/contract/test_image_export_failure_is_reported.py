"""A holder that cannot export an image says so, rather than denying it has it.

Issue #61: copying NVIDIA's multi-arch TensorRT-LLM images to a second node
failed with ``404 image_not_present`` from the holder, straight after the holder
had reported pulling them. ``serve_image_content`` turned *every* export
failure into that 404, so the daemon's own reason was discarded and the record
pointed at an absence that was not there.

The route now separates the two: an image this node does not hold is 404
``image_not_present``; one it holds but cannot export is 500
``image_export_failed`` carrying the export's message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tensorstead.agent.app import build_agent_app
from tensorstead.agent.container_engine.base import ImageBuildError
from tests.fakes.container_engine import FakeContainerEngine
from tests.fakes.service_manager import FakeServiceManager

pytestmark = pytest.mark.contract

_REPLICATION = "replication-token"
_REFERENCE = "nvcr.io/nvidia/tensorrt-llm/release@sha256:" + "1" * 64
_CONTENT = f"/agent/v1/images/{_REFERENCE}/content"
_DAEMON_REASON = "content digest sha256:abc: not found"


def _holder(tmp_path: Path, *, export: Any = None) -> tuple[TestClient, Path]:
    engine = FakeContainerEngine()
    engine.local_images[_REFERENCE] = "sha256:" + "1" * 64
    if export is not None:
        engine.export_image = export  # type: ignore[attr-defined]
    app = build_agent_app(
        management_token="management-token",
        replication_token=_REPLICATION,
        container_engine=engine,
        service_manager=FakeServiceManager(),
        store_dir=tmp_path / "store",
        marker_dir=tmp_path / "markers",
    )
    images = tmp_path / "images"
    app.state.image_store_path = str(images)
    return TestClient(app), images


def _failing_export(*, reference: str, archive_path: str) -> str:
    Path(archive_path).write_bytes(b"partial")
    raise ImageBuildError(_DAEMON_REASON, reference=reference)


def _get(client: TestClient) -> Any:
    return client.get(_CONTENT, headers={"Authorization": f"Bearer {_REPLICATION}"})


def test_a_held_image_that_cannot_be_exported_is_a_server_error(tmp_path: Path) -> None:
    client, _ = _holder(tmp_path, export=_failing_export)

    response = _get(client)

    assert response.status_code == 500, response.text
    assert response.json()["detail"]["code"] == "image_export_failed"


def test_the_export_failure_carries_the_daemons_reason(tmp_path: Path) -> None:
    client, _ = _holder(tmp_path, export=_failing_export)

    message = _get(client).json()["detail"]["message"]

    assert _DAEMON_REASON in message, (
        f"the export's own reason was dropped again; the peer would see only {message!r}"
    )


def test_a_failed_export_leaves_no_partial_archive(tmp_path: Path) -> None:
    client, images = _holder(tmp_path, export=_failing_export)

    _get(client)

    assert not list(images.glob(".serve-*")), "a partial export archive was left behind"


def test_an_image_this_node_does_not_hold_is_still_not_present(tmp_path: Path) -> None:
    client, _ = _holder(tmp_path)

    response = client.get(
        "/agent/v1/images/local/never-pulled:1/content",
        headers={"Authorization": f"Bearer {_REPLICATION}"},
    )

    assert response.status_code == 404, response.text
    assert response.json()["detail"]["code"] == "image_not_present"

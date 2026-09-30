"""Unit test unpinned rendering.

``revision_pinned: false`` must be accompanied by the prose note, so a reader
cannot walk away believing a guarantee the source never made. This is a pure
projection test — no repository or I/O (ExportService is a pure projection).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from tensorstead.domain.identity import new_ulid
from tensorstead.domain.models import Deployment, DeploymentRevision
from tensorstead.service.export import _UNPINNED_NOTE, ExportService

pytestmark = pytest.mark.unit


@pytest.fixture
def service() -> ExportService:
    # ExportService is a pure projection; node-name resolution degrades to the
    # id when no node is registered (unit scope needs no persisted store).
    class _EmptyRepo:
        def get_node(self, node_id: str):  # type: ignore[no-untyped-def]
            return None

        # A double should answer everything the port declares, or it stops
        # being a faithful stand-in for it. An estate with no
        # produced images is the ordinary case, not a missing method.
        def list_images(self):  # type: ignore[no-untyped-def]
            return []

        # Same rule: the export reads the model to recover its file selection,
        # which is part of identity. An estate where the row
        # is absent answers None rather than raising.
        def get_model(self, model_id: str):  # type: ignore[no-untyped-def]
            return None

        def get_build_spec(self, name: str):  # type: ignore[no-untyped-def]
            return None

    return ExportService(repository=_EmptyRepo())


@pytest.fixture
def deployment() -> Deployment:
    return Deployment(
        id=new_ulid(),
        name="llama-70b",
        desired_state="stopped",
        current_revision=1,
    )


@pytest.fixture
def unpinned_revision(deployment: Deployment) -> DeploymentRevision:
    return DeploymentRevision(
        deployment_id=deployment.id,
        revision=1,
        model_id=new_ulid(),
        model_source_id="huggingface",
        source_model_id="org/unpinned",
        resolved_revision=None,  # no stable revision identifier
        revision_pinned=False,
        runtime_type="vllm",
        runtime_version="0.6.0",
        image_reference="repo/vllm:tag",
        image_digest="",
        runtime_config={},
        participating_nodes=("01J-node",),
        endpoint="10.0.0.11:8000",
        origin_platform_facts={},
        created_at=datetime.now().astimezone(),
    )


def test_unpinned_exports_with_prose_note(
    service: ExportService, deployment: Deployment, unpinned_revision: DeploymentRevision
) -> None:
    """The explicit unpinned flag travels with the prose note."""
    body = service.as_dict(deployment, unpinned_revision)
    model = body["model"]
    assert model["revision_pinned"] is False
    assert model["revision"] is None
    assert model["note"] == _UNPINNED_NOTE


def test_unpinned_note_is_in_yaml(
    service: ExportService, deployment: Deployment, unpinned_revision: DeploymentRevision
) -> None:
    """The rendered YAML document carries the note, not only the dict."""
    text = service.export(deployment, unpinned_revision)
    assert "revision_pinned: false" in text
    assert "stable revision identifier" in text


def test_pinned_has_no_note(service: ExportService, deployment: Deployment) -> None:
    """A pinned revision renders without the unpinned prose note."""
    pinned = DeploymentRevision(
        deployment_id=deployment.id,
        revision=1,
        model_id=new_ulid(),
        model_source_id="huggingface",
        source_model_id="org/pinned",
        resolved_revision="e1f2a3b",
        revision_pinned=True,
        runtime_type="vllm",
        runtime_version="0.6.0",
        image_reference="repo/vllm:tag",
        image_digest="",
        runtime_config={},
        participating_nodes=("01J-node",),
        endpoint="10.0.0.11:8000",
        origin_platform_facts={},
        created_at=datetime.now().astimezone(),
    )
    model = service.as_dict(deployment, pinned)["model"]
    assert model["revision_pinned"] is True
    assert "note" not in model
    text = service.export(deployment, pinned)
    assert "stable revision identifier" not in text

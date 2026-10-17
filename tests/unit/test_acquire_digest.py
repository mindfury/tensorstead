"""The agent records a content digest even when the source reports none.

The real Hugging Face adapter never reported one; only the test fake did. With
no digest the coordinator had nothing to verify a peer copy against, skipped
replication, and every further node downloaded the model from the Internet
again (165 GiB twice for GLM-5.3, 2026-10-16).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorstead.agent.acquisition import ModelAcquisitionService
from tensorstead.agent.replication import directory_digest
from tensorstead.ports.model_source import AcquireProgress

pytestmark = pytest.mark.unit


class _Source:
    supports_revision_pinning = True
    requires_credential = False

    def __init__(self, reported: str | None) -> None:
        self._reported = reported

    def source_id(self) -> str:
        return "test"

    def resolve(self, model_id: str, revision: str | None) -> str:
        return revision or "rev1"

    def acquire(
        self,
        model_id: str,
        revision: str,
        destination_dir: str,
        credential: str | None,
        progress: AcquireProgress | None = None,
        file_selector: tuple[str, ...] = (),
    ) -> object:
        root = Path(destination_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(json.dumps({"arch": "test"}))
        (root / "model.safetensors").write_bytes(b"weights")
        return type("R", (), {"size_bytes": 14, "content_digest": self._reported})()


def _acquire(tmp_path: Path, reported: str | None) -> tuple[ModelAcquisitionService, object]:
    service = ModelAcquisitionService(store_dir=tmp_path / "models", marker_dir=tmp_path / "state")
    replica = service.acquire(
        _Source(reported),
        model_id="test:org/model",
        source_model_id="org/model",
        revision="rev1",
        credential=None,
    )
    return service, replica


def test_a_digest_is_computed_when_the_source_reports_none(tmp_path: Path) -> None:
    service, replica = _acquire(tmp_path, None)
    expected = directory_digest(service.local_path("test:org/model"))
    assert replica.content_digest == expected  # type: ignore[attr-defined]
    assert expected.startswith("tsdir2-sha256:")


def test_a_digest_the_source_reports_is_kept(tmp_path: Path) -> None:
    _, replica = _acquire(tmp_path, "sha256:" + "ab" * 32)
    assert replica.content_digest == "sha256:" + "ab" * 32  # type: ignore[attr-defined]

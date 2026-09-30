"""Agent acquisition reports the immutable revision it actually downloaded."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tensorstead.adapters.sources.huggingface import HuggingFaceSource
from tensorstead.agent.app import build_agent_app
from tests.fakes.hf_hub import FakeHuggingFaceHub

pytestmark = pytest.mark.contract


def test_agent_acquire_reports_and_persists_resolved_revision(tmp_path: Path) -> None:
    resolved = "0123456789abcdef0123456789abcdef01234567"
    hub = FakeHuggingFaceHub()
    hub.set_revision("org/model", resolved)
    app = build_agent_app(
        management_token="test",
        model_source=HuggingFaceSource(hub=hub),
        store_dir=tmp_path / "models",
        marker_dir=tmp_path / "state",
    )

    response = TestClient(app).post(
        "/agent/v1/models:acquire",
        json={"source_id": "huggingface", "source_model_id": "org/model"},
        headers={"Authorization": "Bearer test"},
    )

    assert response.status_code == 200
    assert response.json()["resolved_revision"] == resolved
    assert response.json()["revision_pinned"] is True
    replica = app.state.acquisition.get_replica("huggingface:org/model")
    assert replica is not None
    assert replica.resolved_revision == resolved


def test_agent_acquire_reports_upstream_failure_and_removes_partial_staging(tmp_path: Path) -> None:
    """A failed upstream fetch must be actionable and leave no retry-blocking bytes."""

    class _FailingSource:
        def resolve(self, _model_id: str, _revision: str | None) -> str:
            return "0123456789abcdef0123456789abcdef01234567"

        def acquire(
            self,
            _model_id: str,
            _revision: str,
            destination: str,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            Path(destination, "partial.gguf").write_text("incomplete")
            raise RuntimeError("upstream responded 403: repository access denied")

    app = build_agent_app(
        management_token="test",
        model_source=_FailingSource(),
        store_dir=tmp_path / "models",
        marker_dir=tmp_path / "state",
    )

    response = TestClient(app).post(
        "/agent/v1/models:acquire",
        json={"source_id": "huggingface", "source_model_id": "org/gated-model"},
        headers={"Authorization": "Bearer test"},
    )

    assert response.status_code == 502, response.text
    assert response.json()["code"] == "model_acquire_failed"
    assert "403: repository access denied" in response.json()["message"]
    assert not list((tmp_path / "models").glob("*.staging.*"))

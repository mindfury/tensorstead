"""A failed acquire must not erase the record of the replica that is still there.

Marking a failure rewrote the whole marker, so a re-acquire at a new revision
that failed upstream left the node describing itself wrongly: the previously
promoted tree was untouched on disk and still good, while the marker said
``failed`` with an empty ``local_path``. Two consumers read that marker and
believed it -- ``GET /agent/v1/models``, which an operator reads, and
``agent/replication``, which refuses to serve anything not ``available`` and so
declined to hand a peer bytes that were fine.

The transfer failing is an event. What is present on the node is a state. The
marker records the state; the event reaches the operator as the raised
``ModelAcquireError`` and the 502 it becomes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tensorstead.agent.acquisition import ModelAcquireError, ModelAcquisitionService

pytestmark = pytest.mark.unit


class _Source:
    """Succeeds or fails on demand, writing a partial tree when it fails."""

    def __init__(self, *, fails: bool) -> None:
        self._fails = fails

    def resolve(self, _model_id: str, revision: str | None) -> str:
        return revision or "main"

    def acquire(
        self, _model_id: str, _revision: str, destination: str, *_args: object, **_kwargs: object
    ) -> object:
        if self._fails:
            Path(destination, "partial.safetensors").write_text("incomplete")
            raise RuntimeError("upstream responded 403 for token hf_SECRET: access denied")
        Path(destination, "config.json").write_text("{}")

        class _Result:
            size_bytes = 10
            content_digest = "sha256:abc"

        return _Result()


def _service(tmp_path: Path) -> ModelAcquisitionService:
    return ModelAcquisitionService(store_dir=tmp_path / "models", marker_dir=tmp_path / "state")


def test_failed_reacquire_leaves_the_prior_available_replica_described(tmp_path: Path) -> None:
    """The good replica of the old revision keeps its marker when a new one fails."""
    service = _service(tmp_path)
    good = service.acquire(
        _Source(fails=False),
        model_id="test:org/model",
        source_model_id="org/model",
        revision="rev1",
        credential=None,
    )
    assert good.state == "available"

    with pytest.raises(ModelAcquireError):
        service.acquire(
            _Source(fails=True),
            model_id="test:org/model",
            source_model_id="org/model",
            revision="rev2",
            credential=None,
        )

    replica = service.get_replica("test:org/model")
    assert replica is not None
    assert replica.state == "available", (
        "a failed attempt at rev2 installed nothing, so rev1 is still what is present"
    )
    assert replica.resolved_revision == "rev1"
    assert replica.local_path and Path(replica.local_path).exists()


def test_failed_first_acquire_still_records_failed(tmp_path: Path) -> None:
    """With no intact replica to describe, ``failed`` is the truthful record."""
    service = _service(tmp_path)

    with pytest.raises(ModelAcquireError):
        service.acquire(
            _Source(fails=True),
            model_id="test:org/model",
            source_model_id="org/model",
            revision="rev1",
            credential=None,
        )

    replica = service.get_replica("test:org/model")
    assert replica is not None
    assert replica.state == "failed"


def test_failed_acquire_keeps_the_credential_out_of_the_marker(tmp_path: Path) -> None:
    """The detail is persisted to disk, so the same redaction rule applies to it."""
    service = _service(tmp_path)

    with pytest.raises(ModelAcquireError) as raised:
        service.acquire(
            _Source(fails=True),
            model_id="test:org/model",
            source_model_id="org/model",
            revision="rev1",
            credential="hf_SECRET",
        )

    assert "hf_SECRET" not in str(raised.value)
    on_disk = "\n".join(
        path.read_text() for path in (tmp_path / "state").rglob("*") if path.is_file()
    )
    assert "hf_SECRET" not in on_disk
    assert "[REDACTED]" in on_disk


def test_failed_acquire_leaves_no_staging_tree(tmp_path: Path) -> None:
    """Partial bytes are not reusable, so they are not kept."""
    service = _service(tmp_path)

    with pytest.raises(ModelAcquireError):
        service.acquire(
            _Source(fails=True),
            model_id="test:org/model",
            source_model_id="org/model",
            revision="rev1",
            credential=None,
        )

    assert not list((tmp_path / "models").glob("*.staging.*"))

"""Per-key mutual exclusion on model_acquire.

The defect was two concurrent ``model_acquire`` calls for the same (source, model,
revision) racing into one on-disk target directory. ``ModelService.acquire``
held no lock, so both drove the agent to download; the coordinator's reuse
check raced past an empty store. The fix is a per-(source, model, requested
revision) lock at the coordinator and a per-model lock plus a re-check after
resolve at the agent, so a concurrent second acquire reuses the first's
verified replica instead of duplicating it.

These tests pin the serialization with real threads: the coordinator test
asserts one node download for two concurrent same-revision acquires, and the
agent test asserts one available replica for two concurrent acquires of the
same model.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from tensorstead.agent.acquisition import ModelAcquisitionService
from tensorstead.ports.model_source import AcquireProgress
from tensorstead.service.models_ import ModelService
from tests.helpers import build_test_coordinator, make_node

pytestmark = pytest.mark.unit


# ----------------------------------------------------- coordinator serialization


def test_acquire_serializes_concurrent_same_revision() -> None:
    """Two concurrent same-revision acquires do one node download.

    The per-key lock makes the second acquire reuse the first's verified
    replica instead of racing it. ``FakeNodeAgent`` counts every
    ``acquire_model`` call, so one download reads as ``acquisition_calls == 1``
    even though two callers asked for the model.
    """
    app, repo = build_test_coordinator()
    agent = app.state.fake_agent
    service: ModelService = app.state.models_service
    node = make_node()
    repo.save_node(node)

    barrier = threading.Barrier(2)
    results: list[object] = []
    errors: list[BaseException] = []

    def _go() -> None:
        try:
            barrier.wait()
            results.append(
                service.acquire(
                    source_id="huggingface",
                    source_model_id="org/model",
                    revision="rev1",
                    nodes=[node.id],
                )
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_go) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"acquire raised: {errors}"
    assert len(results) == 2
    assert agent.acquisition_calls == 1, (
        "the per-key lock must serialize same-revision acquires so the second "
        "reuses the first's replica instead of downloading again (D1)"
    )


# --------------------------------------------------------- agent serialization


class _RecordingSource:
    """A source whose ``acquire`` is observable.

    ``acquire_calls`` counts downloads. Under the lock two concurrent acquires
    serialize, so only the first reaches ``acquire``; the second's re-check
    after resolve finds the first's available replica and returns without
    downloading. No artificial barrier is needed -- and none could fire, since
    the lock guarantees only one caller is ever inside ``acquire`` at a time.
    """

    supports_revision_pinning = True
    requires_credential = False

    def __init__(self) -> None:
        self.acquire_calls = 0
        self._guard = threading.Lock()

    def source_id(self) -> str:
        return "test"

    def resolve(self, model_id: str, revision: str | None) -> str:
        return revision or "main"

    def acquire(
        self,
        model_id: str,
        revision: str,
        destination_dir: str,
        credential: str | None,
        progress: AcquireProgress | None = None,
        file_selector: tuple[str, ...] = (),
    ) -> object:
        with self._guard:
            self.acquire_calls += 1
        Path(destination_dir).mkdir(parents=True, exist_ok=True)
        (Path(destination_dir) / "config.json").write_text("{}")
        return type("R", (), {"size_bytes": 1, "content_digest": None})()

    def verify(self, destination_dir: Path) -> None:
        pass


def test_concurrent_acquire_reuses_available_replica(tmp_path: Path) -> None:
    """Two concurrent agent acquires of one model produce one available replica.

    The per-model lock plus the re-check after resolve means the second acquire
    sees the first's available replica (same resolved revision) and returns it
    without downloading. One download, one available replica -- not two
    overlapping downloads racing one promote target.
    """
    service = ModelAcquisitionService(store_dir=tmp_path / "models", marker_dir=tmp_path / "state")
    source = _RecordingSource()

    start = threading.Barrier(2)
    results: list[object] = []
    errors: list[BaseException] = []

    def _go() -> None:
        try:
            start.wait()
            results.append(
                service.acquire(
                    source,
                    model_id="test:org/model",
                    source_model_id="org/model",
                    revision=None,
                    credential=None,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_go) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"acquire raised: {errors}"
    assert len(results) == 2
    # The re-check after resolve lets one caller download and the other reuse,
    # so exactly one download occurs even though both callers reached acquire.
    assert source.acquire_calls == 1, (
        "the second acquire must reuse the first's available replica rather "
        "than download again (D1)"
    )
    replica = service.get_replica("test:org/model")
    assert replica is not None
    assert replica.state == "available"

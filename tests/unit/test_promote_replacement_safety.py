"""A failed replacement must never destroy the prior good replica.

``promote()`` used to delete the existing final directory unconditionally
before attempting to install the replacement. A rename or copy failure after
that point left no usable copy on disk while the marker still claimed the old
one was ``available`` -- a re-acquire that could turn a working model into a
missing one on nothing worse than a transient filesystem error.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from tensorstead.agent.acquisition import ModelAcquisitionService

pytestmark = pytest.mark.unit


def _service(tmp_path: Path) -> ModelAcquisitionService:
    return ModelAcquisitionService(store_dir=tmp_path / "store", marker_dir=tmp_path / "markers")


def _stage(service: ModelAcquisitionService, model_id: str, content: str) -> Path:
    staging = service.staging_path(model_id)
    staging.mkdir(parents=True)
    (staging / "weights.bin").write_text(content, encoding="utf-8")
    return staging


def test_first_promote_is_unaffected(tmp_path: Path) -> None:
    """No prior replica: behaves exactly as before this fix."""
    service = _service(tmp_path)
    staging = _stage(service, "m", "weights-v1")

    replica = service.promote("m", staging, resolved_revision="v1")

    assert replica.state == "available"
    final_dir = service.local_path("m")
    assert (final_dir / "weights.bin").read_text(encoding="utf-8") == "weights-v1"


def test_a_successful_replacement_installs_the_new_replica_and_leaves_no_backup(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    service.promote("m", _stage(service, "m", "weights-v1"), resolved_revision="v1")

    service.promote("m", _stage(service, "m", "weights-v2"), resolved_revision="v2")

    final_dir = service.local_path("m")
    assert (final_dir / "weights.bin").read_text(encoding="utf-8") == "weights-v2"
    leftovers = list(final_dir.parent.glob(f"{final_dir.name}.rollback.*"))
    assert leftovers == [], f"a rollback backup survived a successful replacement: {leftovers}"

    replica = service.get_replica("m")
    assert replica is not None
    assert replica.resolved_revision == "v2"


def test_a_failed_replacement_restores_the_prior_replica_and_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core defect: install the old replica, fail the replacement outright."""
    service = _service(tmp_path)
    service.promote("m", _stage(service, "m", "weights-v1"), resolved_revision="v1")
    final_dir = service.local_path("m")

    new_staging = _stage(service, "m", "weights-v2")
    real_rename = os.rename

    def _failing_rename(src: Any, dst: Any) -> None:
        if str(src) == str(new_staging):
            raise OSError("simulated: cannot install the replacement")
        real_rename(src, dst)

    def _failing_copytree(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("simulated: the fallback copy fails too")

    monkeypatch.setattr(os, "rename", _failing_rename)
    monkeypatch.setattr(shutil, "copytree", _failing_copytree)

    with pytest.raises(OSError, match="fallback copy fails too"):
        service.promote("m", new_staging, resolved_revision="v2")

    monkeypatch.undo()  # restore real os.rename/shutil.copytree for assertions below

    # The old replica is exactly as it was -- not gone, not partially
    # overwritten with the failed replacement's content.
    assert (final_dir / "weights.bin").read_text(encoding="utf-8") == "weights-v1"
    assert not list(final_dir.parent.glob(f"{final_dir.name}.rollback.*")), (
        "a rollback directory survived instead of being renamed back"
    )

    # And the marker still describes that same, still-good replica -- not a
    # stale "available" pointing at content that is now gone (the finding's
    # reproduced failure) and not silently left in some other state either.
    replica = service.get_replica("m")
    assert replica is not None
    assert replica.state == "available"
    assert replica.resolved_revision == "v1"


def test_a_marker_write_failure_also_rolls_back_the_directory_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Required correction: restore on *every* failed step, not just the swap.

    The directory swap can succeed while the marker write that must follow it
    fails -- disk full, permissions. The new content must not be left in place
    describing itself as the old marker (or no marker at all).
    """
    service = _service(tmp_path)
    service.promote("m", _stage(service, "m", "weights-v1"), resolved_revision="v1")
    final_dir = service.local_path("m")

    def _failing_replace(_src: Any, _dst: Any) -> None:
        raise OSError("simulated: cannot write the marker")

    monkeypatch.setattr(os, "replace", _failing_replace)

    with pytest.raises(OSError, match="cannot write the marker"):
        service.promote("m", _stage(service, "m", "weights-v2"), resolved_revision="v2")

    monkeypatch.undo()

    assert (final_dir / "weights.bin").read_text(encoding="utf-8") == "weights-v1"
    replica = service.get_replica("m")
    assert replica is not None and replica.resolved_revision == "v1"


def test_marker_writes_leave_no_temp_file_behind_and_are_valid_json(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.promote("m", _stage(service, "m", "weights-v1"), resolved_revision="v1")

    marker_dir = tmp_path / "markers"
    entries = list(marker_dir.iterdir())
    assert all(p.suffix == ".json" for p in entries), f"leftover temp marker files: {entries}"
    (marker_json,) = entries
    record = json.loads(marker_json.read_text(encoding="utf-8"))
    assert record["state"] == "available"
    assert record["resolved_revision"] == "v1"

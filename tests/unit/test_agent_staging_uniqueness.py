"""Per-acquire unique staging directories.

The defect was two concurrent ``model_acquire`` calls for the same model sharing one
staging directory: ``staging_path`` was a pure function of ``model_id``, so the
second download clobbered the first's staging tree mid-flight, and both then
renamed into the same final dir. Making the staging name carry a random suffix
gives each acquire its own staging dir, so no acquire can corrupt another's
staged bytes. Promotion stays deterministic (the final dir is still one name),
so the rename is serialized by the lock, not by the staging name.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorstead.agent.acquisition import ModelAcquisitionService, _safe_token

pytestmark = pytest.mark.unit


def test_staging_path_is_unique_per_call(tmp_path: Path) -> None:
    """Each call gets a distinct staging dir for the same model."""
    service = ModelAcquisitionService(store_dir=tmp_path / "models", marker_dir=tmp_path / "state")
    model_id = "huggingface:nvidia/Qwen3.6-27B-NVFP4"

    first = service.staging_path(model_id)
    second = service.staging_path(model_id)

    assert first != second, "concurrent acquires must not share a staging dir"
    # Both live under the model store and stay Docker-safe (no colon in the name).
    assert first.parent == tmp_path / "models"
    assert second.parent == tmp_path / "models"
    assert ":" not in first.name and ":" not in second.name
    # The deterministic model token is still the prefix, so a replica's final
    # dir (which uses the bare token) remains discoverable and stable.
    token = _safe_token(model_id)
    assert first.name.startswith(f"{token}.staging.")
    assert second.name.startswith(f"{token}.staging.")


def test_promote_renames_only_its_own_staging(tmp_path: Path) -> None:
    """Two unique staging dirs do not clobber each other on promote."""
    service = ModelAcquisitionService(store_dir=tmp_path / "models", marker_dir=tmp_path / "state")
    model_id = "huggingface:org/model"

    first = service.staging_path(model_id)
    second = service.staging_path(model_id)
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "config.json").write_text(json.dumps({"who": "first"}))
    (second / "config.json").write_text(json.dumps({"who": "second"}))

    replica = service.promote(model_id, first, resolved_revision="r1")
    assert replica.state == "available"

    # The promoted tree is the first one; the second staging dir is untouched.
    final = service.local_path(model_id)
    assert (final / "config.json").read_text() == json.dumps({"who": "first"})
    assert second.exists(), "a sibling staging dir must not be clobbered"
    assert (second / "config.json").read_text() == json.dumps({"who": "second"})

"""Model-store path tests for Docker-compatible model mounts."""

from __future__ import annotations

import json
from pathlib import Path

from tensorstead.agent.acquisition import ModelAcquisitionService


def test_model_store_uses_a_docker_safe_directory_name(tmp_path: Path) -> None:
    service = ModelAcquisitionService(store_dir=tmp_path / "models", marker_dir=tmp_path / "state")

    path = service.local_path("huggingface:nvidia/Qwen3.6-27B-NVFP4")

    assert path.parent == tmp_path / "models"
    assert ":" not in path.name


def test_runtime_model_path_migrates_a_legacy_colon_path_without_redownload(tmp_path: Path) -> None:
    store = tmp_path / "models"
    service = ModelAcquisitionService(store_dir=store, marker_dir=tmp_path / "state")
    model_id = "huggingface:nvidia/Qwen3.6-27B-NVFP4"
    legacy = store / model_id
    legacy.mkdir(parents=True)
    (legacy / "weights.bin").write_text("already downloaded")
    # A genuine legacy store has the replica marker the old agent wrote; a
    # migration is only ever of a replica this agent recorded (CodeGuard CG-02).
    markers = tmp_path / "state"
    markers.mkdir(parents=True, exist_ok=True)
    (markers / "legacy.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "resolved_revision": "a" * 40,
                "state": "available",
                "local_path": str(legacy),
                "verified_at": None,
                "size_bytes": 18,
                "content_digest": None,
            }
        )
    )

    runtime_path = service.runtime_model_path(str(legacy))

    assert ":" not in runtime_path
    assert not legacy.exists()
    assert (service.local_path(model_id) / "weights.bin").read_text() == "already downloaded"


def test_runtime_model_path_rebases_a_path_recorded_under_another_root(tmp_path: Path) -> None:
    # An install upgraded in place keeps its old store (TENSORSTEAD_MODEL_STORE_PATH),
    # while the coordinator records every new acquisition under its default root.
    store = tmp_path / "old-store" / "models"
    service = ModelAcquisitionService(store_dir=store, marker_dir=tmp_path / "state")
    model_id = "huggingface:Qwen/Qwen3-1.7B"
    service.local_path(model_id).mkdir(parents=True)

    runtime_path = service.runtime_model_path(f"/var/lib/tensorstead/models/{model_id}")

    assert runtime_path == str(service.local_path(model_id))
    assert service.require_model_path(runtime_path) == service.local_path(model_id).resolve()


def test_runtime_model_path_rebases_a_file_selected_model(tmp_path: Path) -> None:
    store = tmp_path / "models"
    service = ModelAcquisitionService(store_dir=store, marker_dir=tmp_path / "state")
    model_id = "huggingface:unsloth/Qwen3-8B-GGUF#0123456789ab"
    service.local_path(model_id).mkdir(parents=True)

    runtime_path = service.runtime_model_path(f"/var/lib/tensorstead/models/{model_id}")

    assert runtime_path == str(service.local_path(model_id))


def test_runtime_model_path_leaves_a_foreign_path_it_does_not_hold(tmp_path: Path) -> None:
    service = ModelAcquisitionService(store_dir=tmp_path / "models", marker_dir=tmp_path / "state")
    recorded = "/var/lib/tensorstead/models/huggingface:Qwen/Qwen3-1.7B"

    assert service.runtime_model_path(recorded) == recorded


def test_rebased_path_cannot_escape_the_store(tmp_path: Path) -> None:
    store = tmp_path / "models"
    service = ModelAcquisitionService(store_dir=store, marker_dir=tmp_path / "state")
    outside = tmp_path / "huggingface:evil" / "x"
    outside.mkdir(parents=True)

    runtime_path = service.runtime_model_path(str(outside))

    # The id is recovered and looked up inside the store only; the real directory
    # outside it is neither returned nor touched.
    assert runtime_path == str(outside)
    assert outside.is_dir()
    try:
        service.require_model_path(runtime_path)
    except ValueError:
        pass
    else:
        raise AssertionError("a directory outside the store was accepted")

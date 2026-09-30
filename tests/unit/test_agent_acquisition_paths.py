"""Model-store path tests for Docker-compatible model mounts."""

from __future__ import annotations

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

    runtime_path = service.runtime_model_path(str(legacy))

    assert ":" not in runtime_path
    assert not legacy.exists()
    assert (service.local_path(model_id) / "weights.bin").read_text() == "already downloaded"

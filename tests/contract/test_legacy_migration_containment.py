"""Legacy model-path migration never moves anything from outside the store.

CodeGuard CG-02: ``runtime_model_path`` checked containment lexically
(``relative_to``) and renamed a colon-bearing legacy path *before* the resolved
containment check. ``<store>/../outside:secret`` is lexically under the store, so
the rename pulled a directory from outside the store into it, after which
``require_model_path`` accepted it. The code-grant check runs the same helper
first, so a grant did not prevent it.

Every case asserts the outside directory and its sentinel are untouched and that
no container was created; the last proves a genuine legacy store still migrates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tensorstead.agent.acquisition import ModelAcquisitionService
from tensorstead.agent.app import build_agent_app
from tensorstead.domain.approvals import fingerprint
from tests.fakes.container_engine import FakeContainerEngine
from tests.fakes.service_manager import FakeServiceManager

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer mgmt"}
_REVISION = "fab0aecb760cec45227f6656abcaafa11abca87a"


def _model_tree(d: Path) -> None:
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"arch": "qwen3"}))
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}})
    )
    (d / "model-00001-of-00001.safetensors").write_bytes(b"shard")
    (d / "SENTINEL").write_text("must stay put")


def _marker(tmp_path: Path, model_id: str, local_path: Path) -> None:
    markers = tmp_path / "markers"
    markers.mkdir(parents=True, exist_ok=True)
    (markers / f"{abs(hash(model_id))}.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "resolved_revision": _REVISION,
                "state": "available",
                "local_path": str(local_path),
                "verified_at": None,
                "size_bytes": 5,
                "content_digest": None,
            }
        )
    )


@pytest.fixture
def engine() -> FakeContainerEngine:
    return FakeContainerEngine()


@pytest.fixture
def agent(tmp_path: Path, engine: FakeContainerEngine) -> TestClient:
    (tmp_path / "store").mkdir()
    return TestClient(
        build_agent_app(
            management_token="mgmt",
            container_engine=engine,
            service_manager=FakeServiceManager(),
            store_dir=tmp_path / "store",
            marker_dir=tmp_path / "markers",
        )
    )


def _create(agent: TestClient, model_path: str, *, grant: bool = False) -> Any:
    body: dict[str, Any] = {
        "deployment_id": "01J00000000000000000000000",
        "revision": 1,
        "runtime_type": "vllm",
        "image_reference": "repo/vllm:tag",
        "runtime_config": {"tensor_parallel_size": 1},
        "model_path": model_path,
        "endpoint": "0.0.0.0:8000",
    }
    if grant:
        digest = agent.post(
            "/agent/v1/images:pull", json={"reference": "repo/vllm:tag"}, headers=_AUTH
        ).json()["digest"]
        g = {
            "approval_id": "01J00000000000000000000001",
            "option": "trust_remote_code",
            "runtime_type": "vllm",
            "model_source_id": "huggingface",
            "source_model_id": "outside",
            "model_revision": _REVISION,
            "image_digest": digest,
            "policy_version": "2026-09-05.1",
        }
        g["fingerprint"] = fingerprint(
            **{
                k: str(g[k])
                for k in (
                    "option",
                    "runtime_type",
                    "model_source_id",
                    "source_model_id",
                    "model_revision",
                    "image_digest",
                )
            }
        )
        body["code_execution_grant"] = g
        body["runtime_config"]["trust_remote_code"] = True
    return agent.post("/agent/v1/deployments", json=body, headers=_AUTH)


def _store_entries(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in (tmp_path / "store").iterdir())


@pytest.mark.parametrize("grant", [False, True])
def test_a_traversing_legacy_path_moves_nothing(
    agent: TestClient, engine: FakeContainerEngine, tmp_path: Path, grant: bool
) -> None:
    outside = tmp_path / "outside:secret"
    _model_tree(outside)
    before = _store_entries(tmp_path)

    resp = _create(agent, str(tmp_path / "store" / ".." / "outside:secret"), grant=grant)

    assert resp.status_code >= 400
    assert (outside / "SENTINEL").read_text() == "must stay put"
    assert _store_entries(tmp_path) == before
    assert engine.containers == {}


def test_a_symlinked_legacy_directory_is_not_followed(tmp_path: Path) -> None:
    """Even with a replica recorded under that id -- the strongest claim to migrate."""
    store = tmp_path / "store"
    store.mkdir()
    outside = tmp_path / "victim"
    _model_tree(outside)
    model_id = "huggingface:evil"
    (store / model_id).symlink_to(outside, target_is_directory=True)
    _marker(tmp_path, model_id, store / model_id)
    service = ModelAcquisitionService(store_dir=store, marker_dir=tmp_path / "markers")

    service.runtime_model_path(str(store / model_id))

    assert (outside / "SENTINEL").exists()
    assert not service.local_path(model_id).exists()


def test_a_symlinked_parent_is_not_followed(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    outside = tmp_path / "victim-parent"
    _model_tree(outside / "model")
    model_id = "huggingface:org/model"
    (store / "huggingface:org").symlink_to(outside, target_is_directory=True)
    _marker(tmp_path, model_id, store / model_id)
    service = ModelAcquisitionService(store_dir=store, marker_dir=tmp_path / "markers")

    service.runtime_model_path(str(store / model_id))

    assert (outside / "model" / "SENTINEL").exists()
    assert not service.local_path(model_id).exists()


def test_a_legacy_path_without_a_recorded_replica_is_not_migrated(tmp_path: Path) -> None:
    store = tmp_path / "store"
    legacy = store / "huggingface:unknown"
    _model_tree(legacy)
    service = ModelAcquisitionService(store_dir=store, marker_dir=tmp_path / "markers")

    assert service.runtime_model_path(str(legacy)) == str(legacy)
    assert (legacy / "SENTINEL").exists()


def test_a_genuine_legacy_store_still_migrates_and_starts(
    agent: TestClient, engine: FakeContainerEngine, tmp_path: Path
) -> None:
    model_id = "huggingface:nvidia/Qwen3.8-Flash-Next-NVFP4"
    legacy = tmp_path / "store" / model_id
    _model_tree(legacy)
    _marker(tmp_path, model_id, legacy)

    resp = _create(agent, str(legacy))

    assert resp.status_code == 200, resp.text
    assert not legacy.exists()
    assert len(engine.containers) == 1

"""Unit test config merge vs. replace on modify.

``deployment modify --config`` used to replace the whole ``runtime_config`` map,
so one ``--config tool_call_parser=...`` silently dropped ``gpu_memory_utilization``
and its neighbours. The fix: ``runtime_config`` merges by default, and whole-map
replacement requires an explicit ``replace_config`` switch.

These tests hold the service layer to both halves of that and to the third — the
modify outcome carries the configuration actually recorded, so a drop is
visible at the moment it happens rather than only at the next ``deployment show``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tensorstead.adapters.runtimes.vllm import VLLMAdapter
from tensorstead.adapters.sqlite.connection import connect
from tensorstead.adapters.sqlite.migrations import migrate
from tensorstead.adapters.sqlite.repository import SQLiteRepository
from tensorstead.domain.identity import new_ulid
from tensorstead.service.deployments import DeploymentService

pytestmark = pytest.mark.unit

_MIGRATIONS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "tensorstead"
    / "adapters"
    / "sqlite"
    / "migrations"
)


def _make_service() -> tuple[DeploymentService, SQLiteRepository, str]:
    """Return a service, its repo, and a model id usable by ``create``."""
    conn = connect(":memory:")
    migrate(conn, _MIGRATIONS)
    repo = SQLiteRepository(conn)
    deployments = DeploymentService(repo, {"vllm": VLLMAdapter()})

    from datetime import datetime

    from tensorstead.domain.models import Model, ModelSource, Node

    repo.save_model_source(
        ModelSource(id="huggingface", supports_revision_pinning=True, requires_credential=True)
    )
    node = Node(
        id=new_ulid(),
        name="spark-01",
        agent_endpoint="https://10.0.0.11:8443",
        agent_contract_version="1.0",
        agent_cert_fingerprint="AA:BB:CC",
        platform_facts={"cpu_arch": "aarch64", "os_family": "linux"},
        registered_at=datetime.now().astimezone(),
    )
    repo.save_node(node)
    model = Model(
        id=new_ulid(),
        source_id="huggingface",
        source_model_id="org/model",
        resolved_revision="e1f2a3b",
        revision_pinned=True,
    )
    repo.save_model(model)
    return deployments, repo, model.id


def _create(deployments: DeploymentService, model_id: str, node_id: str) -> str:
    deployment = deployments.create(
        name="qwen36-27b",
        model_id=model_id,
        runtime_type="vllm",
        runtime_version="0.6.0",
        image_reference="repo/vllm:tag",
        # Three settings: the one we will touch, and two it must not silently lose.
        runtime_config={
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.4,
            "tool_call_parser": "hermes",
        },
        participating_nodes=[node_id],
        endpoint="10.0.0.11:8000",
    )
    return deployment.id


def test_modify_config_merges_and_keeps_untouched_keys() -> None:
    """A one-key modify patches that key and keeps every other."""
    deployments, repo, model_id = _make_service()
    node_id = repo.list_nodes()[0].id
    dep_id = _create(deployments, model_id, node_id)

    result = deployments.modify(dep_id, runtime_config={"tool_call_parser": "qwen3_xml"})

    recorded = deployments.get_revision(dep_id).runtime_config
    assert recorded["tool_call_parser"] == "qwen3_xml"  # the patch landed
    assert recorded["gpu_memory_utilization"] == 0.4  # ...and the neighbour stayed
    assert recorded["tensor_parallel_size"] == 1
    # The outcome says what was recorded, so a drop is visible here too.
    assert result["runtime_config"] == dict(recorded)


def test_modify_config_replace_drops_untouched_keys() -> None:
    """``replace_config`` replaces the whole map; keys not named are dropped."""
    deployments, repo, model_id = _make_service()
    node_id = repo.list_nodes()[0].id
    dep_id = _create(deployments, model_id, node_id)

    result = deployments.modify(
        dep_id,
        runtime_config={"tool_call_parser": "qwen3_xml"},
        replace_config=True,
    )

    recorded = deployments.get_revision(dep_id).runtime_config
    assert recorded["tool_call_parser"] == "qwen3_xml"
    # gpu_memory_utilization was not named, so a replace drops it. This is the
    # destructive reading: reachable only through the explicit switch, never by
    # accident.
    assert "gpu_memory_utilization" not in recorded
    assert result["runtime_config"] == dict(recorded)


def test_modify_config_merge_overrides_a_key_that_was_already_set() -> None:
    """A merge overrides the named key and leaves the rest at their prior values."""
    deployments, repo, model_id = _make_service()
    node_id = repo.list_nodes()[0].id
    dep_id = _create(deployments, model_id, node_id)

    deployments.modify(dep_id, runtime_config={"gpu_memory_utilization": 0.5})
    recorded = deployments.get_revision(dep_id).runtime_config
    assert recorded["gpu_memory_utilization"] == 0.5
    assert recorded["tool_call_parser"] == "hermes"  # untouched


def test_modify_without_config_leaves_config_unchanged() -> None:
    """A modify that names no config does not touch the config (PATCH semantics)."""
    deployments, repo, model_id = _make_service()
    node_id = repo.list_nodes()[0].id
    dep_id = _create(deployments, model_id, node_id)

    deployments.modify(dep_id, runtime_version="0.7.0")
    recorded = deployments.get_revision(dep_id).runtime_config
    assert recorded["gpu_memory_utilization"] == 0.4
    assert recorded["tool_call_parser"] == "hermes"
    assert recorded["tensor_parallel_size"] == 1

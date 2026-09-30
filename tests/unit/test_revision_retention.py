"""Unit test revision retention.

After any sequence of modifications, every prior revision remains individually
inspectable and exportable, and the running revision is reported.
Modification never issues an UPDATE against the revision table and never
restarts a running deployment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tensorstead.adapters.runtimes.vllm import VLLMAdapter
from tensorstead.adapters.sqlite.connection import connect
from tensorstead.adapters.sqlite.migrations import migrate
from tensorstead.adapters.sqlite.repository import SQLiteRepository
from tensorstead.domain.identity import new_ulid
from tensorstead.service.deployments import DeploymentService
from tensorstead.service.export import ExportService

pytestmark = pytest.mark.unit

_MIGRATIONS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "tensorstead"
    / "adapters"
    / "sqlite"
    / "migrations"
)


def _make_service() -> tuple[DeploymentService, ExportService, SQLiteRepository, str]:
    """Return services, repo, and a model id usable by ``create``."""
    conn = connect(":memory:")
    migrate(conn, _MIGRATIONS)
    repo = SQLiteRepository(conn)
    deployments = DeploymentService(repo, {"vllm": VLLMAdapter()})
    export = ExportService(repo)

    # Seed a node and a logical model so ``create`` is valid.
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
        registered_at=__import__("datetime").datetime.now().astimezone(),
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
    return deployments, export, repo, model.id


def _create_default(deployments: DeploymentService, model_id: str, node_id: str) -> Any:
    return deployments.create(
        name="llama-70b",
        model_id=model_id,
        runtime_type="vllm",
        runtime_version="0.6.0",
        image_reference="repo/vllm:tag",
        runtime_config={"tensor_parallel_size": 1},
        participating_nodes=[node_id],
        endpoint="10.0.0.11:8000",
    )


def test_revisions_retained_and_exportable_after_modifications() -> None:
    """Every prior revision stays inspectable and exportable."""
    deployments, export, repo, model_id = _make_service()
    node_id = repo.list_nodes()[0].id

    deployment = _create_default(deployments, model_id, node_id)
    dep_id = deployment.id

    # Modify twice — each acceptance inserts a new numbered revision.
    r2 = deployments.modify(dep_id, runtime_config={"tensor_parallel_size": 2})
    r3 = deployments.modify(dep_id, runtime_version="0.7.0")
    assert r2["revision"] == 2
    assert r3["revision"] == 3
    assert r2["applied"] is False  # host untouched
    assert r3["applied"] is False

    # Every revision is individually retained and exportable.
    revisions = deployments.list_revisions(dep_id)
    assert [r.revision for r in revisions] == [1, 2, 3]
    current = deployments.get_revision(dep_id)  # default current
    assert current.revision == 3
    for n in (1, 2, 3):
        rev = deployments.get_revision(dep_id, n)
        body = export.as_dict(deployment, rev)
        assert body["exported_from_revision"] == n, f"revision {n} not exportable"


def test_modification_reports_restart_required_only_when_running() -> None:
    """A running deployment reports restart_required; a stopped one does not."""
    deployments, _, repo, model_id = _make_service()
    node_id = repo.list_nodes()[0].id

    deployment = _create_default(deployments, model_id, node_id)
    dep_id = deployment.id

    # Stopped — no restart required.
    stopped = deployments.modify(dep_id, runtime_version="0.7.0")
    assert stopped["restart_required"] is False

    # Simulate a running deployment by recording a running revision.
    from dataclasses import replace

    current = repo.get_deployment(dep_id)
    assert current is not None
    running = replace(current, running_revision=2)
    repo.save_deployment(running)
    running_mod = deployments.modify(dep_id, runtime_version="0.8.0")
    assert running_mod["restart_required"] is True
    assert running_mod["applied"] is False  # not restarted as an implicit consequence


def test_running_revision_reported() -> None:
    """Record which revision a running deployment was started from."""
    deployments, _, repo, model_id = _make_service()
    node_id = repo.list_nodes()[0].id

    deployment = _create_default(deployments, model_id, node_id)
    dep_id = deployment.id
    deployments.modify(dep_id, runtime_version="0.7.0")

    from dataclasses import replace

    current = repo.get_deployment(dep_id)
    assert current is not None
    running = replace(current, running_revision=2)
    repo.save_deployment(running)
    fetched = repo.get_deployment(dep_id)
    assert fetched is not None
    assert fetched.current_revision == 2
    assert fetched.running_revision == 2

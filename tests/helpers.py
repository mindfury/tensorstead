"""Test helpers for Phase 3 contract/integration tests.

Builds a coordinator FastAPI app wired to a fake node agent and an in-memory
SQLite repository, so the coordinator service layer runs end to end with no
external anything (tier 1). A small adapter maps the ``FakeNodeAgent``
onto the node-client port the coordinator services call.
"""

from __future__ import annotations

import contextlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from tensorstead.adapters.credentials.local_file import LocalFileCredentialProvider
from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter
from tensorstead.adapters.runtimes.vllm import VLLMAdapter
from tensorstead.adapters.sqlite.connection import connect
from tensorstead.adapters.sqlite.migrations import migrate
from tensorstead.adapters.sqlite.repository import SQLiteRepository
from tensorstead.coordinator.app import build_coordinator_app
from tensorstead.domain.errors import AuthorizationRefusedError
from tensorstead.domain.identity import new_ulid
from tensorstead.domain.models import Node
from tensorstead.ports.node_client import AgentCallError
from tests.fakes.node_agent import FakeNodeAgent

_MIGRATIONS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "tensorstead"
    / "adapters"
    / "sqlite"
    / "migrations"
)


# The agent request model each coordinator call must satisfy.
#
# Agent models are ``extra="forbid"`` on purpose: an agent that
# silently dropped a field it did not understand would report success on a host
# left in a state nobody asked for. The consequence is that a coordinator
# sending an unrecognised key does not degrade -- it fails the whole operation.
#
# That has now happened twice. ``node_position`` went into the create payload
# with no matching agent field and no contract bump; it did not
# bite only because both agents were current. ``entrypoint`` went into the
# build payload and not into ``ImageBuildRequest``, and the coordinator stored
# it, sent it, and the agent rejected the build -- found by an operator running
# the first real build, not by this suite.
#
# Validating here rather than in a dedicated test is the point: the payloads are
# built in the service layer and a source scan for them was brittle enough to
# find nothing. This way *every* existing test that drives a deployment or a
# build through the fake checks the hop, without any of them knowing they do.
def _agent_request_models() -> dict[str, Any]:
    from tensorstead.agent.routes.deployments import DeploymentCreateRequest
    from tensorstead.agent.routes.images import ImageBuildRequest, ImageDistributeRequest

    return {
        "create_deployment": DeploymentCreateRequest,
        "build_image": ImageBuildRequest,
        "distribute_image": ImageDistributeRequest,
    }


def _reject_what_a_real_agent_would(operation: str, payload: dict[str, Any]) -> None:
    """Fail exactly where a real agent would, and say which key did it."""
    model = _agent_request_models().get(operation)
    if model is None:
        return
    try:
        model(**payload)
    except Exception as exc:  # pydantic.ValidationError
        raise AssertionError(
            f"the coordinator's {operation} payload would be refused by a real "
            f"agent: {exc}. Agent request models forbid unknown fields "
            ", so a field added on the coordinator side must be"
            "added on the agent side too, with a contract bump so the skew is "
            "visible during a rolling upgrade."
        ) from exc


class FakeNodeClient:
    """Adapter mapping ``FakeNodeAgent`` onto the node-client port.

    The coordinator's service layer calls the node client (``get_info``,
    ``acquire_model``, ``create_deployment``, …). This adapter routes those onto
    the fake agent's methods so the service layer sees a real-shaped client
    while the fake stands in for the host.
    """

    def __init__(self, agent: FakeNodeAgent) -> None:
        self.agent = agent
        # Per-node agents, so a two-node test can make one node fail while
        # the other succeeds. Unset nodes share the default agent, which keeps
        # every existing single-node test behaving exactly as before.
        self.agents: dict[str, FakeNodeAgent] = {}

    def agent_for(self, node_id: str) -> FakeNodeAgent:
        """The agent standing in for ``node_id`` (the default unless assigned)."""
        return self.agents.get(node_id, self.agent)

    def _for(self, node: Node) -> FakeNodeAgent:
        return self.agent_for(node.id)

    def get_info(self, node: Node) -> dict[str, Any]:
        try:
            return self._for(node).get_info()
        except (ConnectionError, OSError) as exc:
            raise AgentCallError(
                code="agent_unreachable",
                message=str(exc),
                node_id=node.id,
            ) from exc

    def get_resources(self, node: Node) -> dict[str, Any]:
        try:
            return self._for(node).get_resources()
        except (ConnectionError, OSError) as exc:
            raise AgentCallError(
                code="agent_unreachable",
                message=str(exc),
                node_id=node.id,
            ) from exc

    def acquire_model(
        self,
        node: Node,
        *,
        source_id: str,
        source_model_id: str,
        revision: str | None,
        credential: str | None = None,
        progress: Any = None,
        file_selector: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Pass the resolved credential value through for this one call.

        An upstream refusal surfaces as the agent contract's
        ``authorization_refused`` failure, exactly as the real transport maps a
        403 body from the agent.
        """
        try:
            return self._for(node).acquire_model(
                source_id=source_id,
                source_model_id=source_model_id,
                revision=revision,
                credential=credential,
            )
        except AuthorizationRefusedError as exc:
            raise AgentCallError(
                code=exc.code,
                message=exc.message,
                node_id=node.id,
                detail=exc.detail,
            ) from exc

    def replicate_model(
        self,
        node: Node,
        *,
        model: dict[str, Any],
        source_node: Node,
    ) -> dict[str, Any]:
        """Route a peer replication instruction onto the destination's agent.

        The instruction goes to the *destination* and names the source; the
        transfer itself is modelled inside the destination agent. What this
        adapter asserts by construction is that no artifact bytes come back
        through the coordinator.
        """
        try:
            return self._for(node).replicate_model(
                source_id=model["source_id"],
                source_model_id=model["source_model_id"],
                resolved_revision=model.get("resolved_revision"),
                content_digest=model.get("content_digest"),
                source_node_id=source_node.id,
                source_agent_endpoint=source_node.agent_endpoint,
            )
        except (ConnectionError, OSError, ValueError) as exc:
            raise AgentCallError(
                code="replication_failed",
                message=str(exc),
                node_id=node.id,
            ) from exc

    def build_image(self, node: Node, payload: dict[str, Any]) -> dict[str, Any]:
        """Produce a deterministic identifier for a build.

        Derived from the reference so a test can assert which image a
        deployment refers to, and so building the same reference twice looks
        the same -- a real build of the same pinned spec should too.
        """
        _reject_what_a_real_agent_would("build_image", payload)
        reference = str(payload.get("reference", ""))
        return {"image_id": f"sha256:built-{reference}", "reference": reference, "origin": "built"}

    def distribute_image(self, node: Node, payload: dict[str, Any]) -> dict[str, Any]:
        """Pull a produced image from a peer and return the identifier that arrived.

        Returns the *expected* id, which is the behaviour that matters: an
        image copied between nodes must keep its identifier, and the caller
        refuses the result when it does not. The fake had no such method at all
        until this test needed it, which is why `build_and_distribute` had unit
        tests against a purpose-built stub and had never once run through the
        real wiring.
        """
        _reject_what_a_real_agent_would("distribute_image", payload)
        return {
            "image_id": str(payload.get("expected_image_id", "")),
            "reference": str(payload.get("reference", "")),
            "origin": "distributed",
        }

    def import_image(self, node: Node, payload: dict[str, Any]) -> dict[str, Any]:
        """Load an archive by name from the node's managed store."""
        reference = str(payload.get("reference", ""))
        return {
            "image_id": f"sha256:imported-{payload.get('archive_name', '')}",
            "reference": reference,
            "origin": "imported",
        }

    def create_deployment(self, node: Node, payload: dict[str, Any]) -> dict[str, Any]:
        _reject_what_a_real_agent_would("create_deployment", payload)
        try:
            self._for(node).create_deployment(
                payload["deployment_id"], endpoint=payload["endpoint"]
            )
        except (ConnectionError, OSError) as exc:
            raise AgentCallError(
                code="agent_unreachable",
                message=str(exc),
                node_id=node.id,
            ) from exc
        return {"status": "created", "deployment_id": payload["deployment_id"]}

    def get_observed(self, node: Node, deployment_id: str) -> dict[str, Any]:
        try:
            return self._for(node).get_observed(deployment_id)
        except (ConnectionError, OSError) as exc:
            raise AgentCallError(
                code="agent_unreachable",
                message=str(exc),
                node_id=node.id,
            ) from exc

    def get_runtime(self, node: Node, deployment_id: str, *, tail: int = 500) -> dict[str, Any]:
        try:
            return self._for(node).get_runtime(deployment_id, tail=tail)
        except (ConnectionError, OSError) as exc:
            raise AgentCallError(
                code="agent_unreachable",
                message=str(exc),
                node_id=node.id,
            ) from exc

    def stop_deployment(self, node: Node, deployment_id: str) -> dict[str, Any]:
        try:
            return self._for(node).stop_deployment(deployment_id)
        except (ConnectionError, OSError) as exc:
            raise AgentCallError(
                code="agent_unreachable",
                message=str(exc),
                node_id=node.id,
            ) from exc

    def reconcile_deployment(self, node: Node, deployment_id: str) -> dict[str, Any]:
        try:
            return self._for(node).reconcile_deployment(deployment_id)
        except (ConnectionError, OSError) as exc:
            raise AgentCallError(
                code="agent_unreachable",
                message=str(exc),
                node_id=node.id,
            ) from exc

    def remove_deployment(self, node: Node, deployment_id: str) -> dict[str, Any]:
        try:
            return self._for(node).remove_deployment(deployment_id)
        except (ConnectionError, OSError) as exc:
            raise AgentCallError(
                code="agent_unreachable",
                message=str(exc),
                node_id=node.id,
            ) from exc

    def delete_model(self, node: Node, model_id: str) -> dict[str, Any]:
        try:
            return self._for(node).delete_model(model_id)
        except (ConnectionError, OSError) as exc:
            raise AgentCallError(
                code="agent_unreachable",
                message=str(exc),
                node_id=node.id,
            ) from exc


# Connections opened by the helpers below, closed after each test by the
# autouse fixture in conftest.
#
# Not fastidiousness. CPython raises ResourceWarning for an unclosed sqlite
# connection at whatever moment the collector happens to run, which is almost
# never the test that opened it -- the suite was reporting them against
# `test_revision_is_frozen` and `typing.py`, neither of which has ever opened a
# database. A hundred and sixty warnings blaming innocent tests is worse than
# none: it is noise that a real warning would have to be spotted inside, which
# is the same failure as a log tail buried under access logging.
_OPEN_CONNECTIONS: list[Any] = []


def close_test_connections() -> None:
    """Close every connection the helpers opened. Called between tests."""
    while _OPEN_CONNECTIONS:
        connection = _OPEN_CONNECTIONS.pop()
        with contextlib.suppress(Exception):
            connection.close()


def build_test_coordinator(
    *, agent: FakeNodeAgent | None = None
) -> tuple[FastAPI, SQLiteRepository]:
    """Build a coordinator app wired to a fake agent and in-memory store.

    Returns ``(app, repository)`` so tests can inspect declared state.
    """
    agent = agent or FakeNodeAgent()
    conn = connect(":memory:")
    _OPEN_CONNECTIONS.append(conn)
    migrate(conn, _MIGRATIONS)
    repository = SQLiteRepository(conn)
    node_client = FakeNodeClient(agent)
    # A per-test credential store under a temp dir, so a test never writes a
    # secret into a real location and tests never share one.
    credential_provider = LocalFileCredentialProvider(
        root=Path(tempfile.mkdtemp(prefix="tensorstead-test-credentials-"))
    )
    app = build_coordinator_app(
        management_token="test",
        repository=repository,
        node_client=node_client,
        # Both v1 runtimes, so the distributed/non-distributed seam is exercised
        # by real adapters rather than by a stub.
        runtime_adapters={"vllm": VLLMAdapter(), "llamacpp": LlamaCppAdapter()},
        credential_provider=credential_provider,
    )
    app.state.fake_agent = agent  # exposed for tests that drive the agent
    return app, repository


def make_node() -> Node:
    """A registered Node record for tests that need one pre-seeded."""
    return Node(
        id=new_ulid(),
        name="spark-01",
        agent_endpoint="https://10.0.0.11:8443",
        agent_contract_version="1.0",
        agent_cert_fingerprint="AA:BB:CC",
        platform_facts={"cpu_arch": "aarch64", "os_family": "linux", "memory_is_unified": True},
        registered_at=datetime.now().astimezone(),
    )


def make_vllm_model_dir(tmp_path: Path) -> str:
    """A real on-disk vLLM model tree the agent pre-flight will accept.

    Before the pre-flight the agent-deployment tests pointed ``model_path`` at
    ``/var/lib/...`` -- a path that does not exist on the test host -- which is
    exactly what let the missing-config.json incident hide: the agent launched a
    container against a directory it never inspected. The pre-flight now
    refuses that, so tests that drive the real agent route build a tree here.
    Returns the path string to pass as ``model_path``.

    Built under ``tmp_path / "store"`` because every current caller configures
    the agent with ``store_dir=tmp_path / "store"``, and the primary model path
    must now resolve inside it or the agent
    refuses the start before this tree is ever inspected for shape.
    """
    import json

    d = tmp_path / "store" / "model"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"arch": "qwen3"}))
    (d / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (d / "model-00001-of-00002.safetensors").write_bytes(b"shard1")
    (d / "model-00002-of-00002.safetensors").write_bytes(b"shard2")
    return str(d)


def make_llamacpp_model_dir(tmp_path: Path) -> str:
    """A real on-disk llama.cpp model tree the agent pre-flight accepts."""
    d = tmp_path / "gguf-model"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.gguf").write_bytes(b"gguf-bytes")
    return str(d)


def poll_operation(
    client: Any,
    operation_id: str,
    *,
    timeout: float = 5.0,
    interval: float = 0.02,
    auth: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Poll ``GET /v1/operations/{id}`` until terminal; return the final record.

    ``model_acquire`` returns 202 immediately and runs on a background thread, so a
    test that posts acquire and then reads ``/v1/models`` races the background
    work: the model row may not be written yet. The fake agent is instant, so
    this resolves in 1-3 polls; the 5s bound makes it never-flaky. The GIL
    schedules the test thread before the background work, so the first poll is
    not guaranteed terminal — loop until it is.
    """
    import time

    headers = auth if auth is not None else {"Authorization": "Bearer test"}
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        resp = client.get(f"/v1/operations/{operation_id}", headers=headers)
        assert resp.status_code == 200, resp.text
        last = dict(resp.json())
        if last.get("state") in ("succeeded", "failed"):
            return last
        time.sleep(interval)
    raise AssertionError(
        f"operation {operation_id!r} did not reach a terminal state within {timeout}s "
        f"(last state: {last.get('state')!r})"
    )

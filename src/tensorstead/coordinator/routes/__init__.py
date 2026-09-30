"""Coordinator routes — thin HTTP projection of the service layer.

The API is a *projection* of the service layer: these handlers
marshal arguments into service calls and render results, holding no management
logic of their own. Every route is management-token gated.

Routes wired in this phase: nodes, runtimes, models, deployment create, and
deployment start. The service instances are carried on ``app.state`` so tests
inject the fakes and the real app wires the real services.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials

from tensorstead.contracts.api import (
    DeploymentCreateRequest,
    DeploymentDeclared,
    DeploymentModifyRequest,
    DeploymentModifyResponse,
    DeploymentObserved,
    DeploymentObservedPerNode,
    DeploymentResponse,
    DeploymentRuntimeResponse,
    DeploymentStatusResponse,
    ModelAcquireRequest,
    ModelReplica,
    ModelResponse,
    NodeRegisterRequest,
    NodeRegisterResponse,
    NodeReserveRequest,
    NodeResourcesResponse,
    NodeRotateManagementTokenRequest,
    NodeSummary,
    OperationAcceptedResponse,
    OperationResponse,
    ReachabilityResponse,
    RuntimeInfo,
)
from tensorstead.contracts.version import CONTRACT_VERSION
from tensorstead.coordinator.auth import _bearer
from tensorstead.domain.identity import new_ulid
from tensorstead.domain.models import Node, OperationKind
from tensorstead.version import VERSION

router = APIRouter(prefix="/v1", tags=["coordinator"])


def _require_auth(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    """Management-token dependency for coordinator routes.

    Delegates to the auth object's ``require_auth`` so a single token
    configuration gates every route. ``request.app.state.auth`` is a
    ``CoordinatorAuth`` (coordinator/app.py).
    """
    request.app.state.auth.require_auth(credentials)


require_auth_dep = _require_auth


def _record_failure(
    operations: Any, operation_id: str, exc: Exception, *, default_code: str
) -> None:
    """Record a failed operation preserving the reason's structured shape.

    A typed domain failure already carries ``code``, ``node_id``, and a bounded
    ``detail``. Collapsing that to a message would throw away exactly
    the parts an operator needs afterwards — which node failed, and what the
    per-node outcomes were on a multi-node operation.
    """
    detail = dict(getattr(exc, "detail", None) or {})
    per_node = detail.get("per_node")
    operations.fail(
        operation_id,
        code=getattr(exc, "code", default_code),
        message=str(exc),
        node_id=getattr(exc, "node_id", None),
        detail=detail,
        per_node_outcomes=per_node if isinstance(per_node, dict) else None,
    )


def _nodes_service(request: Request) -> Any:
    return request.app.state.nodes_service


def _models_service(request: Request) -> Any:
    return request.app.state.models_service


def _deployments_service(request: Request) -> Any:
    return request.app.state.deployments_service


def _export_service(request: Request) -> Any:
    return request.app.state.export_service


def _lifecycle_service(request: Request) -> Any:
    return request.app.state.lifecycle_service


def _observation_service(request: Request) -> Any:
    return request.app.state.observation_service


def _node_register_response(node: Node) -> NodeRegisterResponse:
    """Shared by every route returning a full node record.

    One function rather than one inline construction per route, after
    independent copies drifted before: three
    sites once defaulted ``reserved``/``reserved_reason`` silently. The
    boolean here is the same class of field for the same reason
    -- never the token itself, just whether
    one is set.
    """
    return NodeRegisterResponse(
        id=node.id,
        name=node.name,
        agent_endpoint=node.agent_endpoint,
        agent_contract_version=node.agent_contract_version,
        platform_facts=node.platform_facts,
        registered_at=node.registered_at,
        reserved=node.reserved,
        reserved_reason=node.reserved_reason,
        has_management_token_override=bool(node.agent_management_token_ref),
    )


def _node_summary(node: Node) -> NodeSummary:
    return NodeSummary(
        id=node.id,
        name=node.name,
        agent_endpoint=node.agent_endpoint,
        registered_contract_version=node.agent_contract_version,
        registered_at=node.registered_at,
        reserved=node.reserved,
        reserved_reason=node.reserved_reason,
        has_management_token_override=bool(node.agent_management_token_ref),
    )


# -------------------------------------------------------------------- nodes
@router.post(
    "/nodes",
    response_model=NodeRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_auth_dep)],
)
def register_node(payload: NodeRegisterRequest, request: Request) -> NodeRegisterResponse:
    node = _nodes_service(request).register(
        name=payload.name,
        agent_endpoint=payload.agent_endpoint,
    )
    return _node_register_response(node)


# ------------------------------------------------- inference credentials
@router.put("/inference-credentials/{name}", dependencies=[Depends(require_auth_dep)])
def set_inference_credential(
    name: str, payload: dict[str, Any], request: Request
) -> dict[str, Any]:
    """Store a named inference credential.

    The value goes to the credential provider and the store keeps a reference.
    Accepts a value or a reference form; the MCP surface is restricted to
    references only.
    """
    service = request.app.state.inference_credentials
    return service.set(  # type: ignore[no-any-return]
        name,
        value=payload.get("value"),
        from_env=payload.get("from_env"),
        from_file=payload.get("from_file"),
    )


@router.get("/inference-credentials", dependencies=[Depends(require_auth_dep)])
def list_inference_credentials(request: Request) -> list[dict[str, Any]]:
    """Names and set times only. There is no read path."""
    return request.app.state.inference_credentials.list()  # type: ignore[no-any-return]


@router.delete("/inference-credentials/{name}", dependencies=[Depends(require_auth_dep)])
def delete_inference_credential(name: str, request: Request) -> dict[str, Any]:
    """Delete a credential; refused while a deployment binds it."""
    return request.app.state.inference_credentials.delete(name)  # type: ignore[no-any-return]


@router.put(
    "/deployments/{deployment_id}/inference-credential",
    dependencies=[Depends(require_auth_dep)],
)
def bind_inference_credential(
    deployment_id: str, payload: dict[str, Any], request: Request
) -> dict[str, Any]:
    """Bind a credential to a deployment, or clear it with a null name.

    Clearing does not disable authentication: it returns the deployment to
    whatever the node was provisioned with. Binding changes a
    definition, never host state — a restart is required for it to take
    effect, and is never performed implicitly.
    """
    service = request.app.state.inference_credentials
    return service.bind(deployment_id, payload.get("name"))  # type: ignore[no-any-return]


# --------------------------------------------- code-execution approvals
# Deliberately its own route rather than a field on deployment create or
# modify. The refusal these lift says a deployment record may not decide what
# executes "on its own", so an approval reachable through ordinary deployment
# mutation would be the same authority wearing a different name.
@router.post(
    "/code-approvals",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_auth_dep)],
)
def create_code_approval(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    """Approve one code-loading option for one exact tuple. Starts nothing."""
    service = request.app.state.code_approvals
    approval = service.create(
        option=str(payload.get("option", "")),
        runtime_type=str(payload.get("runtime_type", "")),
        model_source_id=str(payload.get("model_source_id", "")),
        source_model_id=str(payload.get("source_model_id", "")),
        model_revision=str(payload.get("model_revision", "")),
        image_digest=str(payload.get("image_digest", "")),
        reason=str(payload.get("reason", "")),
        approved_by=str(payload.get("approved_by", "")),
    )
    return _approval_json(approval)


@router.get("/code-approvals", dependencies=[Depends(require_auth_dep)])
def list_code_approvals(request: Request) -> list[dict[str, Any]]:
    """Every reviewed approval, newest first."""
    return [_approval_json(a) for a in request.app.state.code_approvals.list()]


@router.delete("/code-approvals/{approval_id}", dependencies=[Depends(require_auth_dep)])
def delete_code_approval(approval_id: str, request: Request) -> dict[str, Any]:
    """Revoke an approval. Running deployments are not disturbed; the next start fails."""
    request.app.state.code_approvals.delete(approval_id)
    return {"status": "deleted", "approval_id": approval_id}


def _approval_json(approval: Any) -> dict[str, Any]:
    return {
        "id": approval.id,
        "option": approval.option,
        "runtime_type": approval.runtime_type,
        "model_source_id": approval.model_source_id,
        "source_model_id": approval.source_model_id,
        "model_revision": approval.model_revision,
        "image_digest": approval.image_digest,
        "fingerprint": approval.fingerprint,
        "reason": approval.reason,
        "approved_by": approval.approved_by,
        "policy_version": approval.policy_version,
        "created_at": approval.created_at.isoformat(),
    }


# ------------------------------------------------ managed runtime images
@router.put("/buildspecs/{name}", dependencies=[Depends(require_auth_dep)])
def set_build_spec(name: str, payload: dict[str, Any], request: Request) -> dict[str, Any]:
    """Record a build spec. Executes nothing."""
    service = request.app.state.image_builds
    return service.record_spec(  # type: ignore[no-any-return]
        name,
        base_image=str(payload.get("base_image", "")),
        steps=list(payload.get("steps", [])),
        entrypoint=list(payload.get("entrypoint", []) or []),
    )


@router.get("/buildspecs", dependencies=[Depends(require_auth_dep)])
def list_build_specs(request: Request) -> list[dict[str, Any]]:
    """List recorded build specs."""
    return request.app.state.image_builds.list_specs()  # type: ignore[no-any-return]


@router.delete("/buildspecs/{name}", dependencies=[Depends(require_auth_dep)])
def delete_build_spec(name: str, request: Request) -> dict[str, Any]:
    """Delete a build spec; refused while referenced."""
    return request.app.state.image_builds.delete_spec(name)  # type: ignore[no-any-return]


@router.post(
    "/images:build",
    response_model=OperationAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_auth_dep)],
)
async def build_image(payload: dict[str, Any], request: Request) -> OperationAcceptedResponse:
    """Build a recorded spec on one node, or build once and distribute.

    ``nodes`` produces the image on the first and copies it to the rest.
    ``node_id`` builds on one node and is kept because it is the existing
    shape; a single-element ``nodes`` means the same thing.

    Building independently on each node yields a *different identifier for the
    same spec*, which breaks digest comparison silently and makes divergence
    detection meaningless — so a multi-node deployment on a locally produced
    image needs this path rather than two builds.

    **Accepted, then run in the background**. This was
    synchronous, and an image build is the second operation that routinely
    outlasts a client's read timeout: compiling a runtime takes many minutes,
    and the MCP client's window turned that into ``coordinator_unreachable`` --
    indistinguishable from a dead coordinator. The caller then could not tell
    whether the request had been accepted, so a retry risked a duplicate
    concurrent build or a race with an accepted build's distribution phase.

    Exactly the defect fixed for ``model_acquire``, in exactly the
    same place, found again because the fix was applied to the one operation
    that had failed rather than to the class. The two now share a shape: 202
    with an operation id, poll ``GET /v1/operations/{id}`` to terminal.
    """
    service = request.app.state.image_builds
    reference = str(payload.get("reference"))
    spec = str(payload.get("spec"))
    nodes = payload.get("nodes")
    node_ids = [str(node) for node in nodes] if isinstance(nodes, list) and nodes else None

    # Refused *before* the 202, not after. A caller must be able to tell three
    # outcomes apart, and the first of them is "not accepted,
    # safe to retry" -- which an unrecorded spec is, immediately and without
    # doing any work. Accepting it and failing the operation asynchronously
    # would turn a clean 404 into a poll, and would put a failure in the
    # operation history for work that never started.
    service.require_spec(spec)

    operations = request.app.state.operations_service
    operation = operations.begin(
        kind=OperationKind.IMAGE_BUILD,
        target_type="image",
        # No image row exists yet -- the build is what produces it -- so this is
        # the operation's own identity, as model acquire does for the same
        # reason. What is being built is recorded as progress below, so an
        # operator polling a freshly accepted operation can see which build it
        # is rather than an opaque id.
        target_id=new_ulid(),
    )
    operations.mark_running(
        operation.id,
        progress={
            "reference": reference,
            "spec": spec,
            "nodes": node_ids or [str(payload.get("node_id"))],
        },
    )
    asyncio.get_running_loop().run_in_executor(
        None, _run_build, service, operations, operation.id, spec, reference, node_ids, payload
    )
    return OperationAcceptedResponse(operation_id=operation.id)


def _run_build(
    service: Any,
    operations: Any,
    operation_id: str,
    spec: str,
    reference: str,
    node_ids: list[str] | None,
    payload: dict[str, Any],
) -> None:
    """Run an image build to terminal on the route's background thread.

    The caller already has its 202, so nothing raised here reaches a client:
    every outcome must land on the operation or it hangs in ``running`` forever.
    """
    try:
        if node_ids is not None:
            result = service.build_and_distribute(spec, nodes=node_ids, reference=reference)
        else:
            result = service.build(spec, node_id=str(payload.get("node_id")), reference=reference)
    except Exception as exc:
        _record_failure(operations, operation_id, exc, default_code="image_build_failed")
        return

    outcome = result if isinstance(result, dict) else {}
    per_node = outcome.get("per_node") if isinstance(outcome.get("per_node"), dict) else None

    # Partial success is an overall failure, and the failed nodes are named.
    # ``build_and_distribute`` reports that in its return value rather than by
    # raising, so recording ``succeeded`` merely because nothing escaped
    # would put "the build worked" in the operation history for a build that
    # reached one node out of two -- the record-and-reality gap this product
    # exists to close, written by the product itself.
    if outcome.get("status") == "failed":
        failed = ", ".join(outcome.get("failed_nodes") or []) or "an unnamed node"
        operations.fail(
            operation_id,
            code="image_build_failed",
            message=f"build of {reference!r} failed on: {failed}",
            detail={key: value for key, value in outcome.items() if key != "per_node"},
            per_node_outcomes=per_node,
        )
        return

    operations.succeed(operation_id, per_node_outcomes=per_node)


@router.post("/images:import", dependencies=[Depends(require_auth_dep)])
def import_image(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    """Import a prebuilt archive onto a node."""
    service = request.app.state.image_builds
    return service.import_archive(  # type: ignore[no-any-return]
        node_id=str(payload.get("node_id")),
        reference=str(payload.get("reference")),
        archive_name=str(payload.get("archive_name")),
        expected_image_id=payload.get("expected_image_id"),
    )


@router.get("/version", dependencies=[Depends(require_auth_dep)])
def get_version() -> dict[str, str]:
    """Coordinator and contract version.

    Behind authentication on purpose: ``/health`` answers "is the process up"
    for an unauthenticated readiness check, and disclosing the running release
    to anything that can reach the port is a different question.
    """
    return {"version": VERSION, "contract_version": CONTRACT_VERSION}


@router.get("/nodes", response_model=list[NodeSummary], dependencies=[Depends(require_auth_dep)])
def list_nodes(request: Request) -> list[NodeSummary]:
    return [_node_summary(n) for n in _nodes_service(request).list()]


@router.get(
    "/nodes/{node_id}",
    response_model=NodeRegisterResponse,
    dependencies=[Depends(require_auth_dep)],
)
def get_node(node_id: str, request: Request) -> NodeRegisterResponse:
    node = _nodes_service(request).get(node_id)
    return _node_register_response(node)


@router.delete(
    "/nodes/{node_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_auth_dep)],
)
def deregister_node(node_id: str, request: Request) -> None:
    _nodes_service(request).deregister(node_id)


@router.post(
    "/nodes/{node_id}/reserve",
    response_model=NodeRegisterResponse,
    dependencies=[Depends(require_auth_dep)],
)
def reserve_node(
    node_id: str, payload: NodeReserveRequest, request: Request
) -> NodeRegisterResponse:
    node = _nodes_service(request).reserve(node_id, payload.note)
    return _node_register_response(node)


@router.post(
    "/nodes/{node_id}/unreserve",
    response_model=NodeRegisterResponse,
    dependencies=[Depends(require_auth_dep)],
)
def unreserve_node(node_id: str, request: Request) -> NodeRegisterResponse:
    node = _nodes_service(request).unreserve(node_id)
    return _node_register_response(node)


@router.post(
    "/nodes/{node_id}/rotate-management-token",
    response_model=NodeRegisterResponse,
    dependencies=[Depends(require_auth_dep)],
)
def rotate_node_management_token(
    node_id: str, payload: NodeRotateManagementTokenRequest, request: Request
) -> NodeRegisterResponse:
    node = _nodes_service(request).rotate_agent_management_token(node_id, payload.token)
    return _node_register_response(node)


@router.post(
    "/nodes/{node_id}/clear-management-token",
    response_model=NodeRegisterResponse,
    dependencies=[Depends(require_auth_dep)],
)
def clear_node_management_token(node_id: str, request: Request) -> NodeRegisterResponse:
    node = _nodes_service(request).clear_agent_management_token(node_id)
    return _node_register_response(node)


@router.get(
    "/nodes/{node_id}/reachability",
    response_model=ReachabilityResponse,
    dependencies=[Depends(require_auth_dep)],
)
def reachability(node_id: str, request: Request) -> ReachabilityResponse:
    result = _nodes_service(request).reachability(node_id)
    return ReachabilityResponse(**result)


@router.get(
    "/nodes/{node_id}/resources",
    response_model=NodeResourcesResponse,
    dependencies=[Depends(require_auth_dep)],
)
def node_resources(node_id: str, request: Request) -> NodeResourcesResponse:
    """On-demand accelerator, memory, and managed-storage reading.

    Reported for operator judgment; never gates an operation.
    ``status`` may be ``unknown`` or ``unreachable``.
    """
    result = _nodes_service(request).resources(node_id)
    return NodeResourcesResponse(**result)


# ----------------------------------------------------------------- runtimes
@router.get(
    "/runtimes",
    response_model=list[RuntimeInfo],
    dependencies=[Depends(require_auth_dep)],
)
def list_runtimes(request: Request) -> list[RuntimeInfo]:
    return [
        RuntimeInfo(
            type=adapter.runtime_type,
            versions=[],
            supports_distributed=adapter.supports_distributed,
            suggested_images=[
                {"reference": img.reference, "note": img.note} for img in adapter.suggested_images
            ],
        )
        for adapter in request.app.state.runtime_adapters.values()
    ]


# ------------------------------------------------------------------- models
@router.post(
    "/models:acquire",
    response_model=OperationAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_auth_dep)],
)
async def acquire_model(
    payload: ModelAcquireRequest, request: Request
) -> OperationAcceptedResponse:
    operations = request.app.state.operations_service
    operation = operations.begin(
        kind=OperationKind.MODEL_ACQUIRE,
        target_type="model",
        target_id=new_ulid(),  # the Model row is created during acquire
    )
    # The work starts here, so the record says so.
    operations.mark_running(operation.id)
    # A model download is the one operation that can outlast a client's read
    # timeout: a multi-GB pull from upstream routinely takes minutes, and the
    # MCP client's 60s window turned that into ``coordinator_unreachable`` --
    # indistinguishable from a dead coordinator, which is what made a reasonable
    # retry produce the duplicate acquire. Return the operation id now and
    # run the download on a background thread; clients poll
    # ``GET /v1/operations/{id}`` to terminal, exactly as the route's 202 and
    # the MCP tool docstring already claim. The work is scheduled on the loop's
    # default executor -- the same ``ThreadPoolExecutor`` precedent
    # ``lifecycle.py`` already uses -- so it is request-driven (not an idle
    # poll the idle-polling guardrail polices) and thread-safe the same way: the
    # SQLite repo is ``check_same_thread=False`` with an ``RLock`` and
    # ``NodeHTTPClient`` builds a per-call ``httpx.Client``.
    models_service = _models_service(request)
    asyncio.get_running_loop().run_in_executor(
        None, _run_acquire, models_service, operations, operation.id, payload
    )
    return OperationAcceptedResponse(operation_id=operation.id)


def _run_acquire(
    models_service: Any,
    operations: Any,
    operation_id: str,
    payload: ModelAcquireRequest,
) -> None:
    """Run a model acquire to terminal on the route's background thread.

    The caller has already returned 202; nothing this raises reaches a client,
    so every outcome must be recorded on the operation or it hangs in
    ``running`` forever. ``_record_failure`` preserves the structured failure
    shape (code, node_id, detail) the way the sibling lifecycle routes do.
    """
    try:
        # The service call persists the model; the return value is not used.
        models_service.acquire(
            source_id=payload.source_id,
            source_model_id=payload.source_model_id,
            revision=payload.revision,
            nodes=payload.nodes,
            credential=payload.credential,
            file_selector=tuple(payload.file_selector),
        )
        per_node = {n: {"state": "succeeded"} for n in payload.nodes}
        operations.succeed(operation_id, per_node_outcomes=per_node)
    except Exception as exc:
        try:
            _record_failure(operations, operation_id, exc, default_code="model_acquire_failed")
        except Exception:
            # An async path cannot let an exception escape silently — the
            # operation would stay running forever — so fall back to a plain
            # failure record if the structured one itself threw.
            operations.fail(operation_id, code="model_acquire_failed", message=str(exc))


@router.get(
    "/models",
    response_model=list[ModelResponse],
    dependencies=[Depends(require_auth_dep)],
)
def list_models(request: Request) -> list[ModelResponse]:
    return [_model_response(request, m.id) for m in _models_service(request).list()]


@router.get(
    "/models/{model_id}",
    response_model=ModelResponse,
    dependencies=[Depends(require_auth_dep)],
)
def get_model(model_id: str, request: Request) -> ModelResponse:
    return _model_response(request, model_id)


@router.delete(
    "/models/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_auth_dep)],
)
def delete_model(model_id: str, request: Request) -> None:
    """Delete a logical model; refused while referenced."""
    _models_service(request).delete(model_id)


# -------------------------------------------------------------------- images
@router.get(
    "/images",
    response_model=list[dict],
    dependencies=[Depends(require_auth_dep)],
)
def list_images(request: Request) -> list[dict]:
    """List images present on nodes, with their origin.

    ``origin`` and ``produced_by`` are recorded for every image but were not
    surfaced here, so "100% of managed images report a digest and an
    origin" was true in the store and false on the surface. An operator
    reading `image list` could not tell a pulled image from one built from a
    named spec, which is exactly the distinction the design makes
    load-bearing: a locally built image has no registry digest, and conflating
    the two makes an export look portable when it is not.
    """
    images = _models_service(request).list_images()
    return [
        {
            "node_id": img.node_id,
            "reference": img.reference,
            "digest": img.digest,
            "origin": str(img.origin),
            "produced_by": img.produced_by,
            "pulled_at": img.pulled_at,
            "is_registry_digest": img.is_registry_digest,
        }
        for img in images
    ]


@router.delete(
    "/images/{node_id}/{digest}",
    response_model=dict,
    dependencies=[Depends(require_auth_dep)],
)
def delete_image(node_id: str, digest: str, request: Request) -> dict:
    """Delete an image from the node, then the record.

    Returns what happened rather than 204. The caller could not previously tell
    a removal from a no-op, and the no-op was what it had been doing all along.
    ``outcome`` is ``removed`` when the node freed the
    image and ``record_reaped`` when the node did not hold it.
    """
    outcome: dict = _models_service(request).delete_image(node_id, digest)
    return outcome


@router.post(
    "/images:reconcile",
    response_model=dict,
    dependencies=[Depends(require_auth_dep)],
)
def reconcile_images(request: Request, node_id: str | None = None) -> dict:
    """Compare image records against the nodes and reap orphaned records.

    Deletes records whose object the node does not hold; reports images present
    with no record, and nodes it could not reach, without acting on either.
    """
    outcome: dict = _models_service(request).reconcile_images(node_id)
    return outcome


# ---------------------------------------------------------------- operations
@router.get(
    "/operations",
    response_model=list[OperationResponse],
    dependencies=[Depends(require_auth_dep)],
)
def list_operations(
    request: Request,
    deployment_id: str | None = None,
) -> list[OperationResponse]:
    """List operations, optionally filtered by deployment."""
    ops = request.app.state.operations_service.list(deployment_id)
    return [_operation_response(op) for op in ops]


@router.get(
    "/operations/{operation_id}",
    response_model=OperationResponse,
    dependencies=[Depends(require_auth_dep)],
)
def get_operation(operation_id: str, request: Request) -> OperationResponse:
    """Progress and terminal outcome of one operation."""
    op = request.app.state.operations_service.get(operation_id)
    return _operation_response(op)


def _operation_response(op: Any) -> OperationResponse:
    from tensorstead.contracts.api import FailureReason

    failure = None
    if op.failure_reason:
        failure = FailureReason(
            code=op.failure_reason.get("code", "unknown"),
            message=op.failure_reason.get("message", ""),
            node_id=op.failure_reason.get("node_id"),
            detail=op.failure_reason.get("detail", {}),
        )
    return OperationResponse(
        id=op.id,
        kind=op.kind.value,
        state=op.state.value,
        deployment_revision=op.deployment_revision,
        per_node_outcomes=op.per_node_outcomes,
        failure_reason=failure,
        progress=op.progress,
        started_at=op.started_at,
        finished_at=op.finished_at,
    )


def _model_response(request: Request, model_id: str) -> ModelResponse:
    model = _models_service(request).get(model_id)
    replicas = request.app.state.repository.list_replicas(model_id)
    return ModelResponse(
        id=model.id,
        source_id=model.source_id,
        source_model_id=model.source_model_id,
        resolved_revision=model.resolved_revision,
        revision_pinned=model.revision_pinned,
        size_bytes=model.size_bytes,
        file_selector=list(model.file_selector),
        replicas=[
            ModelReplica(
                node_id=r.node_id,
                state=r.state.value,
                verified_at=r.verified_at if r.verified_at else None,
            )
            for r in replicas
        ],
    )


# -------------------------------------------------------------- deployments
@router.post(
    "/deployments",
    response_model=OperationAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_auth_dep)],
)
def create_deployment(
    payload: DeploymentCreateRequest, request: Request
) -> OperationAcceptedResponse:
    operations = request.app.state.operations_service
    operation = operations.begin(
        kind=OperationKind.DEPLOYMENT_CREATE,
        target_type="deployment",
        target_id=new_ulid(),  # the Deployment row is created during create
    )
    # The work starts here, so the record says so.
    operations.mark_running(operation.id)
    try:
        deployment = _deployments_service(request).create(
            name=payload.name,
            model_id=payload.model_id,
            runtime_type=payload.runtime_type,
            runtime_version=payload.runtime_version,
            image_reference=payload.image_reference,
            runtime_config=payload.runtime_config,
            participating_nodes=payload.participating_nodes,
            endpoint=payload.endpoint,
            restore_on_boot=payload.restore_on_boot,
        )
        # Bind the operation to the deployment it just created, so the
        # creation appears in that deployment's own history.
        operations.retarget(operation.id, deployment.id)
        # Advisory pre-flight, performed *after* creation so it can
        # never gate one. The agent has implemented this endpoint since early on
        # and nothing called it until now.
        service = _deployments_service(request)
        advisory = service.endpoint_preflight(payload.participating_nodes, payload.endpoint)
        # Runtime-specific notes about a configuration that is valid but whose
        # consequences are easy to miss -- speculative decoding quietly
        # reshaping the runtime's own per-step token budget.
        advisory += service.config_advisories(
            payload.runtime_type, payload.runtime_config, payload.participating_nodes
        )
        operations.succeed(operation.id)
        return OperationAcceptedResponse(operation_id=operation.id, warnings=advisory)
    except Exception as exc:
        operations.fail(
            operation.id,
            code=getattr(exc, "code", "deployment_create_failed"),
            message=str(exc),
        )
        raise


@router.get(
    "/deployments",
    response_model=list[DeploymentResponse],
    dependencies=[Depends(require_auth_dep)],
)
def list_deployments(request: Request) -> list[DeploymentResponse]:
    deployments = _deployments_service(request).list()
    return [_deployment_response(request, d.id) for d in deployments]


@router.get(
    "/deployments/{deployment_id}",
    response_model=DeploymentResponse,
    dependencies=[Depends(require_auth_dep)],
)
def get_deployment(deployment_id: str, request: Request) -> DeploymentResponse:
    return _deployment_response(request, deployment_id)


@router.post(
    "/deployments/{deployment_id}:start",
    response_model=OperationAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_auth_dep)],
)
def start_deployment(deployment_id: str, request: Request) -> OperationAcceptedResponse:
    operations = request.app.state.operations_service
    operation = operations.begin(
        kind=OperationKind.START,
        target_type="deployment",
        target_id=deployment_id,
        deployment_revision=request.app.state.deployments_service.get(
            deployment_id
        ).current_revision,
    )
    # The work starts here, so the record says so.
    operations.mark_running(operation.id)
    try:
        lifecycle = _lifecycle_service(request)
        lifecycle.start(deployment_id)
        operations.succeed(operation.id)
    except Exception as exc:
        if getattr(exc, "code", None) == "already_in_state":
            operations.succeed(operation.id)
        else:
            _record_failure(operations, operation.id, exc, default_code="start_failed")
        raise
    # Same distinction as restart: started is not serving.
    return OperationAcceptedResponse(
        operation_id=operation.id, warnings=lifecycle.start_advisories()
    )


@router.post(
    "/deployments/{deployment_id}:stop",
    response_model=OperationAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_auth_dep)],
)
def stop_deployment(deployment_id: str, request: Request) -> OperationAcceptedResponse:
    operations = request.app.state.operations_service
    operation = operations.begin(
        kind=OperationKind.STOP,
        target_type="deployment",
        target_id=deployment_id,
    )
    # The work starts here, so the record says so.
    operations.mark_running(operation.id)
    try:
        _lifecycle_service(request).stop(deployment_id)
        operations.succeed(operation.id)
    except Exception as exc:
        if getattr(exc, "code", None) == "already_in_state":
            operations.succeed(operation.id)
        else:
            _record_failure(operations, operation.id, exc, default_code="stop_failed")
        raise
    return OperationAcceptedResponse(operation_id=operation.id)


@router.post(
    "/deployments/{deployment_id}:restart",
    response_model=OperationAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_auth_dep)],
)
def restart_deployment(deployment_id: str, request: Request) -> OperationAcceptedResponse:
    operations = request.app.state.operations_service
    operation = operations.begin(
        kind=OperationKind.RESTART,
        target_type="deployment",
        target_id=deployment_id,
    )
    # The work starts here, so the record says so.
    operations.mark_running(operation.id)
    try:
        lifecycle = _lifecycle_service(request)
        lifecycle.restart(deployment_id)
        operations.succeed(operation.id)
    except Exception as exc:
        _record_failure(operations, operation.id, exc, default_code="restart_failed")
        raise
    # Succeeded means the containers started, which is not the same as serving.
    # A restart returns in seconds while a large model loads for minutes.
    return OperationAcceptedResponse(
        operation_id=operation.id, warnings=lifecycle.start_advisories()
    )


@router.post(
    "/deployments/{deployment_id}:reconcile",
    response_model=OperationAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_auth_dep)],
)
def reconcile_deployment(deployment_id: str, request: Request) -> OperationAcceptedResponse:
    operations = request.app.state.operations_service
    operation = operations.begin(
        kind=OperationKind.RECONCILE,
        target_type="deployment",
        target_id=deployment_id,
    )
    # The work starts here, so the record says so.
    operations.mark_running(operation.id)
    try:
        observation = _observation_service(request)
        result = observation.reconcile(deployment_id)
        if result["status"] == "succeeded":
            operations.succeed(operation.id, per_node_outcomes=result.get("per_node"))
        else:
            operations.fail(
                operation.id,
                code="partial_failure",
                message="reconcile could not fully converge",
                per_node_outcomes=result.get("per_node"),
            )
    except Exception as exc:
        operations.fail(
            operation.id,
            code=getattr(exc, "code", "reconcile_failed"),
            message=str(exc),
        )
        raise
    return OperationAcceptedResponse(operation_id=operation.id)


@router.delete(
    "/deployments/{deployment_id}",
    response_model=OperationAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_auth_dep)],
)
def remove_deployment(deployment_id: str, request: Request) -> OperationAcceptedResponse:
    operations = request.app.state.operations_service
    operation = operations.begin(
        kind=OperationKind.REMOVE,
        target_type="deployment",
        target_id=deployment_id,
    )
    # The work starts here, so the record says so.
    operations.mark_running(operation.id)
    try:
        result = _lifecycle_service(request).remove(deployment_id)
        operations.succeed(operation.id, per_node_outcomes=result.get("per_node"))
    except Exception as exc:
        _record_failure(operations, operation.id, exc, default_code="remove_failed")
        raise
    return OperationAcceptedResponse(operation_id=operation.id)


@router.patch(
    "/deployments/{deployment_id}",
    response_model=DeploymentModifyResponse,
    dependencies=[Depends(require_auth_dep)],
)
def modify_deployment(
    deployment_id: str, payload: DeploymentModifyRequest, request: Request
) -> DeploymentModifyResponse:
    """Modify a deployment definition — new numbered revision.

    Accepts any subset; records restart_required without acting on it.
    The deployment id stays stable.
    """
    result = _deployments_service(request).modify(
        deployment_id,
        runtime_config=payload.runtime_config,
        replace_config=payload.replace_config,
        image_reference=payload.image_reference,
        runtime_version=payload.runtime_version,
        endpoint=payload.endpoint,
        participating_nodes=payload.participating_nodes,
        model_id=payload.model_id,
        restore_on_boot=payload.restore_on_boot,
        # The optimistic half. Accepted by the request model, offered by
        # the CLI and MCP, and checked by the service -- but not forwarded here,
        # so a caller that explicitly said "only if still at revision N" was
        # told it succeeded after losing the race.
        expected_revision=payload.expected_revision,
    )
    return DeploymentModifyResponse(**result)


@router.get(
    "/deployments/{deployment_id}/revisions",
    response_model=list[dict],
    dependencies=[Depends(require_auth_dep)],
)
def list_deployment_revisions(deployment_id: str, request: Request) -> list[dict]:
    """List every retained revision, oldest first."""
    revisions = _deployments_service(request).list_revisions(deployment_id)
    return [_revision_info(request, r) for r in revisions]


@router.get(
    "/deployments/{deployment_id}/revisions/{revision}",
    response_model=dict,
    dependencies=[Depends(require_auth_dep)],
)
def get_deployment_revision(deployment_id: str, revision: int, request: Request) -> dict:
    """Retrieve one retained revision by number."""
    rev = _deployments_service(request).get_revision(deployment_id, revision)
    return _revision_info(request, rev)


@router.get(
    "/deployments/{deployment_id}/export",
    response_model=dict,
    dependencies=[Depends(require_auth_dep)],
)
def export_deployment(
    deployment_id: str, request: Request, revision: int | None = None
) -> dict[str, Any]:
    """Render one revision (default: current) as the export projection."""
    deployments = _deployments_service(request)
    deployment = deployments.get(deployment_id)
    rev = deployments.get_revision(deployment_id, revision)
    operations = request.app.state.operations_service.list(deployment_id)
    return dict(_export_service(request).as_dict(deployment, rev, operations=operations))


@router.post(
    "/deployments:create-from-export",
    response_model=OperationAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_auth_dep)],
)
def create_deployment_from_export(payload: dict, request: Request) -> OperationAcceptedResponse:
    """Recreate a deployment from an export artifact."""
    export = payload.get("export", payload)
    deployments = _deployments_service(request)
    operations = request.app.state.operations_service
    operation = operations.begin(
        kind=OperationKind.DEPLOYMENT_CREATE,
        target_type="deployment",
        target_id=new_ulid(),
    )
    # The work starts here, so the record says so.
    operations.mark_running(operation.id)
    try:
        node_names = (export or {}).get("placement", {}).get("nodes", []) or []
        warnings = deployments.comparability_warnings(export or {}, node_names)
        deployments.create_from_export(export or {})
        operations.succeed(
            operation.id,
            per_node_outcomes={"warnings": warnings} if warnings else None,
        )
        return OperationAcceptedResponse(operation_id=operation.id)
    except Exception as exc:
        operations.fail(
            operation.id,
            code=getattr(exc, "code", "deployment_create_failed"),
            message=str(exc),
        )
        raise


def _revision_info(request: Request, revision: Any) -> dict:
    """Render one DeploymentRevision as a client-facing dict."""
    return {
        "revision": revision.revision,
        "model_id": revision.model_id,
        "model": {
            "source_id": revision.model_source_id,
            "source_model_id": revision.source_model_id,
            "resolved_revision": revision.resolved_revision,
            "revision_pinned": revision.revision_pinned,
        },
        "runtime_type": revision.runtime_type,
        "runtime_version": revision.runtime_version,
        "image_reference": revision.image_reference,
        "image_digest": revision.image_digest,
        "runtime_config": revision.runtime_config,
        "participating_nodes": list(revision.participating_nodes),
        "endpoint": revision.endpoint,
        "created_at": revision.created_at.isoformat(),
    }


def _deployment_response(request: Request, deployment_id: str) -> DeploymentResponse:
    """Declared + observed, never merged.

    Two labelled blocks — ``declared`` from the coordinator's authoritative
    record, ``observed`` fetched from the responsible node at request time.
    Declared is always returned in full even when observation fails.
    Divergences are a by-product of observation and nothing
    is mutated.
    """
    deployments = _deployments_service(request)
    deployment = deployments.get(deployment_id)
    revision = deployments.get_revision(deployment_id, deployment.current_revision)

    declared = DeploymentDeclared(
        id=deployment.id,
        name=deployment.name,
        desired_state=deployment.desired_state,
        current_revision=deployment.current_revision,
        running_revision=deployment.running_revision,
        revision={
            "revision": revision.revision,
            "model": {
                "source_id": revision.model_source_id,
                "source_model_id": revision.source_model_id,
                "resolved_revision": revision.resolved_revision,
                "revision_pinned": revision.revision_pinned,
            },
            "runtime_type": revision.runtime_type,
            "runtime_version": revision.runtime_version,
            "image_reference": revision.image_reference,
            "image_digest": revision.image_digest,
            "runtime_config": revision.runtime_config,
            "participating_nodes": list(revision.participating_nodes),
            "endpoint": revision.endpoint,
        },
    )

    observation = _observation_service(request)
    _, _, observed_dict, divergences = observation.observe(deployment_id)

    observed = _build_observed(observed_dict)
    return DeploymentResponse(
        declared=declared,
        observed=observed,
        divergences=[d.model_dump() for d in divergences],
    )


@router.get(
    "/deployments/{deployment_id}/status",
    response_model=DeploymentStatusResponse,
    dependencies=[Depends(require_auth_dep)],
)
def deployment_status(deployment_id: str, request: Request) -> DeploymentStatusResponse:
    """Observed state only — ``deployment.status``.

    The observed portion of a deployment response. Declared state is not
    included; it is available from ``GET /v1/deployments/{id}``.
    """
    observation = _observation_service(request)
    _, _, observed_dict, divergences = observation.observe(deployment_id)
    observed = _build_observed(observed_dict)
    return DeploymentStatusResponse(
        observed=observed,
        divergences=[d.model_dump() for d in divergences],
    )


@router.get(
    "/deployments/{deployment_id}/runtime",
    response_model=DeploymentRuntimeResponse,
    dependencies=[Depends(require_auth_dep)],
)
def deployment_runtime(
    deployment_id: str,
    request: Request,
    # Matches the agent's default, which was resized from measurement against
    # a real vLLM. Leaving this at 200 would give an API caller a
    # different answer than the CLI for the same question.
    tail: int = Query(default=500, ge=1, le=2000),
) -> DeploymentRuntimeResponse:
    """The runtime's own account of itself -- argv, restarts, output.

    The product can already report that a deployment stopped serving.
    This is what reports why. Read-only, and nothing it returns is stored:
    a runtime may log request content, so the management plane reads through to
    it and keeps none of it.
    """
    observation = _observation_service(request)
    return DeploymentRuntimeResponse(**observation.runtime(deployment_id, tail=tail))


def _build_observed(observed_dict: dict[str, Any]) -> DeploymentObserved:
    """Build the DeploymentObserved block from the service-layer dict."""
    per_node: dict[str, DeploymentObservedPerNode] = {}
    for node_id, node_obs in observed_dict.get("per_node", {}).items():
        per_node[node_id] = DeploymentObservedPerNode(
            status=node_obs.get("status", "unknown"),
            running_image_digest=node_obs.get("running_image_digest"),
            running_revision=node_obs.get("running_revision"),
            endpoint_reachable=node_obs.get("endpoint_reachable"),
            inference_ready=node_obs.get("inference_ready"),
            serves_inference=node_obs.get("serves_inference", True),
            endpoint_authenticated=node_obs.get("endpoint_authenticated"),
            detail=node_obs.get("detail"),
        )
    observed_at = observed_dict.get("observed_at")
    if isinstance(observed_at, str):
        observed_at = datetime.fromisoformat(observed_at)
    return DeploymentObserved(
        status=observed_dict.get("status", "unknown"),
        observed_at=observed_at or datetime.now().astimezone(),
        per_node=per_node,
        running_image_digest=observed_dict.get("running_image_digest"),
        endpoint_reachable=observed_dict.get("endpoint_reachable"),
        inference_ready=observed_dict.get("inference_ready"),
        detail=observed_dict.get("detail"),
    )

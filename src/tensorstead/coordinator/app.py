"""Coordinator FastAPI application.

Creates the coordinator's HTTP surface. The key concern here is mapping domain
errors to HTTP responses that *preserve the structured failure shape*
``{code, message, node_id, detail}`` rather than collapsing to
a generic 500.

The routes themselves are thin projections of the service layer and
are wired incrementally and in stages. This module provides the app, the auth
dependency, and the error mapping they build on. Routes attach
``app.state.auth.require_auth`` as a dependency so a management token gates
every route.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tensorstead.coordinator.auth import CoordinatorAuth
from tensorstead.domain import errors as domain_errors
from tensorstead.version import VERSION

_logger = logging.getLogger("tensorstead.coordinator")

# HTTP status for each typed domain error code. Exit-code parity with the CLI
# is its own concern; here the status reflects the nature of the failure.
_STATUS_BY_CODE: dict[str, int] = {
    "not_found": 404,
    "already_exists": 409,
    "already_in_state": 200,  # a satisfied request, not a failure
    "agent_unreachable": 503,
    # The node replied; the work failed. 502 rather than 500: the failure was
    # reported by something upstream of the coordinator, not caused by it.
    "node_operation_failed": 502,
    "agent_version_incompatible": 409,
    "operation_unsupported_by_agent": 409,
    "still_referenced": 409,
    "endpoint_conflict": 409,
    "runtime_not_distributed": 422,
    "node_reserved": 422,
    "invalid_agent_endpoint": 422,
    "credential_resolution_failed": 422,
    # Some nodes changed and others did not: the request could not be applied
    # consistently, which is a conflict with the state on the ground.
    "partial_failure": 409,
    "authorization_refused": 403,
    "invalid_deployment": 422,
    "concurrent_modification": 409,
}


def build_coordinator_app(
    *,
    management_token: str | None = None,
    repository: Any | None = None,
    node_client: Any | None = None,
    runtime_adapters: dict[str, Any] | None = None,
    credential_provider: Any | None = None,
    store: str | Path | None = None,
) -> FastAPI:
    """Create the coordinator application.

    ``management_token``, when given, is the token that gates every route.
    When not given, it is read from ``TENSORSTEAD_MGMT_TOKEN``.
    The auth dependency is exposed as ``app.state.auth`` so route modules attach
    ``app.state.auth.require_auth`` to every management route.

    The service seams are injectable for tests:
    ``repository``, ``node_client``, ``runtime_adapters``, and
    ``credential_provider``. When omitted, production defaults are built lazily
    (an in-memory repository is used only when no ``store`` is configured; the
    real SQLite adapter is wired by ``coordinator serve``). An injected
    ``repository`` takes precedence over ``store``.
    """
    token = management_token if management_token is not None else management_token_from_env()
    app = FastAPI(title="tensorstead coordinator", version=VERSION)
    app.state.auth = CoordinatorAuth(token)

    _wire_services(app, repository, node_client, runtime_adapters, credential_provider, store)
    _mount_routes(app)

    @app.exception_handler(domain_errors.DomainError)
    async def _handle_domain_error(
        request: Request, exc: domain_errors.DomainError
    ) -> JSONResponse:
        status = _STATUS_BY_CODE.get(exc.code, 500)
        return JSONResponse(status_code=status, content=exc.as_failure())

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # An untyped exception is still a real event: log it with a traceback so
        # the cause is diagnosable, and name its type in the response rather
        # than discarding it as an empty ``detail``. An opaque
        # ``internal_error`` on a call that actually completed is the
        # recurring defect in this codebase -- a record that
        # stopped matching reality with nothing comparing them. Surfacing the
        # exception type and a bounded message is what makes "nothing
        # comparing them" false. The message is capped at 500 chars; these
        # exceptions name tables, columns, and caller-supplied ids and paths,
        # never credentials.
        _logger.exception("unexpected internal error")
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": "unexpected internal error",
                "detail": {
                    "exception_type": type(exc).__name__,
                    "message": str(exc)[:500],
                },
            },
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        # Deliberately minimal and unauthenticated: it answers "is the process
        # up" for a readiness check. Version lives behind auth below, so a
        # reachable port does not disclose the running release.
        return {"status": "ok"}

    return app


def _wire_services(
    app: FastAPI,
    repository: Any | None,
    node_client: Any | None,
    runtime_adapters: dict[str, Any] | None,
    credential_provider: Any | None = None,
    store: str | Path | None = None,
) -> None:
    """Attach the services to ``app.state`` (injectable for tests)."""
    from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter
    from tensorstead.adapters.runtimes.sglang import SGLangAdapter
    from tensorstead.adapters.runtimes.vllm import VLLMAdapter
    from tensorstead.service.credentials import CredentialService
    from tensorstead.service.deployments import DeploymentService
    from tensorstead.service.lifecycle import LifecycleService
    from tensorstead.service.models_ import ModelService
    from tensorstead.service.nodes import NodeService
    from tensorstead.service.operations import OperationService

    # The shipped runtimes: vLLM and SGLang distribute, llama.cpp
    # does not. Shipping the counter-example is what makes the refusal
    # real, and shipping a third adapter is what keeps the seam a seam rather
    # than two implementations of one shape.
    #
    # A fourth, ComfyUI, was built, ran, and was deliberately removed --
    # it is a graph-executing workstation tool, not a request/response
    # inference server, and every seam this port defines (one deployment, one
    # model, one endpoint) had to be bent to fit it.
    adapters = (
        runtime_adapters
        if runtime_adapters is not None
        else {
            "vllm": VLLMAdapter(),
            "llamacpp": LlamaCppAdapter(),
            "sglang": SGLangAdapter(),
        }
    )
    app.state.runtime_adapters = adapters

    # A caller-provided repository is used as-is. Otherwise the explicit store
    # selected by ``coordinator serve`` wins, followed by TENSORSTEAD_STORE, with
    # an in-memory repository reserved for callers that configure neither.
    if repository is None:
        from tensorstead.adapters.sqlite.connection import connect
        from tensorstead.adapters.sqlite.migrations import migrate
        from tensorstead.adapters.sqlite.repository import SQLiteRepository

        _store = (
            Path(store)
            if store is not None
            else Path(os.environ.get("TENSORSTEAD_STORE", ":memory:"))
        )
        _conn = connect(_store)
        migrate(_conn, Path(__file__).resolve().parents[1] / "adapters" / "sqlite" / "migrations")
        repository = SQLiteRepository(_conn)

    # Credentials: the provider owns the secret values; the repository
    # holds only references. The model service resolves a reference to a value
    # at acquisition time and passes it per-request. Resolved
    # before node_client below: a per-node management token
    # is stored through this same provider, so
    # NodeHTTPClient needs it at construction, not after.
    provider = credential_provider or _default_credential_provider()
    app.state.credential_provider = provider
    app.state.credentials_service = CredentialService(repository, provider)

    if node_client is None:
        from tensorstead.coordinator.node_http import NodeHTTPClient

        node_client = NodeHTTPClient(
            management_token=os.environ.get("TENSORSTEAD_MGMT_TOKEN", ""),
            verify=os.environ.get("TENSORSTEAD_AGENT_CA_BUNDLE", True),
            credential_provider=provider,
        )

    app.state.repository = repository
    app.state.node_client = node_client

    app.state.nodes_service = NodeService(repository, node_client, provider)
    app.state.models_service = ModelService(
        repository, node_client, credentials=app.state.credentials_service
    )
    app.state.deployments_service = DeploymentService(repository, adapters, node_client)
    app.state.lifecycle_service = LifecycleService(repository, node_client, adapters)
    app.state.operations_service = OperationService(repository)

    from tensorstead.service.approvals import ApprovalService
    from tensorstead.service.credentials import InferenceCredentialService
    from tensorstead.service.models_ import ImageBuildService

    # The product owns the inference credential, storing a reference
    # and resolving it per start.
    app.state.inference_credentials = InferenceCredentialService(repository, provider)
    # The product builds and imports runtime images, so repairing a broken upstream
    # image does not require SSH.
    app.state.image_builds = ImageBuildService(repository, node_client)
    app.state.code_approvals = ApprovalService(repository)
    # Lifecycle resolves a bound credential immediately before each start.
    app.state.lifecycle_service._inference_credentials = app.state.inference_credentials

    from tensorstead.service.export import ExportService

    app.state.export_service = ExportService(repository)

    from tensorstead.service.observation import ObservationService

    app.state.observation_service = ObservationService(repository, node_client)

    # Resolve non-terminal operations left from a prior coordinator
    # run. Operations still pending or running when the coordinator
    # terminated are marked failed with outcome_unknown — no probe, no replay.
    _repo = app.state.repository
    _repo.resolve_non_terminal_operations(finished_at=datetime.now().astimezone())


def _default_credential_provider() -> Any:
    """Build the v1 local-file credential provider.

    The store's location is the adapter's own business — the coordinator knows
    nothing about any filesystem layout, which is what keeps it free of a
    colocation assumption. The directory is created on first
    write, so a coordinator that never sets a credential never creates it.
    """
    from tensorstead.adapters.credentials.local_file import LocalFileCredentialProvider

    return LocalFileCredentialProvider()


def _mount_routes(app: FastAPI) -> None:
    """Register the coordinator route modules onto the app."""
    from tensorstead.coordinator.routes import router as coordinator_router
    from tensorstead.coordinator.routes.credentials import router as credentials_router

    app.include_router(coordinator_router)
    app.include_router(credentials_router)


def management_token_from_env() -> str | None:
    """Read the management token from the environment, if configured."""
    return os.environ.get("TENSORSTEAD_MGMT_TOKEN")

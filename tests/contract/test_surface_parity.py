"""Surface parity test.

Enumerates the operation registry (``service/registry.py``) and diffs it
against the API, CLI, and MCP surfaces, failing on any asymmetry. A new
operation cannot be added to one surface alone without breaking this test.

This was marked ``xfail`` on three dimensions until each surface
arrived. **All three now assert for real**: credentials landed in phase 7
and the MCP tool surface in phase 9.

"100% of the supported management operation set is reachable through
API, CLI, and MCP" — is therefore a complete, enforced check rather than a
review item. Adding an operation to the registry without projecting it onto all
three surfaces fails here.
"""

from __future__ import annotations

import pytest

from tensorstead.service.registry import all_operations, operation_ids

pytestmark = pytest.mark.contract

# The v1 catalogue — the parity source of truth.
_EXPECTED_OPERATIONS = {
    "node.register",
    "node.list",
    "node.get",
    "node.deregister",
    "node.reachability",
    "node.resources",
    "runtime.list",
    "model.acquire",
    "model.list",
    "model.get",
    "model.delete",
    "image.list",
    "image.delete",
    "image.reconcile",
    "credential.set",
    "credential.list",
    "credential.delete",
    "deployment.create",
    "deployment.modify",
    "deployment.list",
    "deployment.get",
    "deployment.revisions",
    "deployment.status",
    "deployment.export",
    "deployment.start",
    "deployment.stop",
    "deployment.restart",
    "deployment.reconcile",
    "deployment.remove",
    "operation.get",
    "operation.list",
    # Managed runtime images. Added consciously: this set is the
    # v1 catalogue, and growing it is a product decision, not a side effect.
    "image.build",
    "image.import",
    "buildspec.set",
    "buildspec.list",
    "buildspec.delete",
    # Inference credentials. Added consciously: the registry is the
    # source of truth, and growing it obliges all three surfaces at once.
    "inferencekey.set",
    "inferencekey.list",
    "inferencekey.delete",
    "inferencekey.bind",
    # Runtime observability. Added consciously: registering it is
    # what obliges all three surfaces, and the whole point of the operation is
    # that a human *or an agent* can read a failing runtime without a shell.
    "deployment.runtime",
}


def test_registry_is_complete() -> None:
    """The registry enumerates the full v1 catalogue."""
    ids = set(operation_ids())
    assert ids == _EXPECTED_OPERATIONS, (
        f"registry missing {sorted(_EXPECTED_OPERATIONS - ids)} "
        f"or has extra {sorted(ids - _EXPECTED_OPERATIONS)}"
    )


def test_registry_ids_are_dotted_noun_verb() -> None:
    """Every operation id is ``noun.verb`` (MCP tool naming)."""
    for op in all_operations():
        parts = op.id.split(".")
        assert len(parts) == 2, f"{op.id} is not noun.verb"
        assert parts[0] == op.noun and parts[1] == op.verb


def _api_surface() -> set[str]:
    """The coordinator API surface as operation ids derived from route paths.

    Routes are wired incrementally; this helper maps the
    routes registered on the coordinator app to operation ids.

    The enumeration reads the generated OpenAPI document rather than walking
    ``app.routes``. Since Starlette 1.4 / FastAPI 0.141, ``include_router`` no
    longer flattens a router's routes into ``app.routes`` — it inserts a single
    opaque ``_IncludedRouter`` entry — so walking ``app.routes`` silently sees
    none of the management routes and reports a total parity failure even when
    every route is mounted and serving. The OpenAPI document is the surface the
    API actually publishes, which is what this check is about.
    """
    from tensorstead.coordinator.app import build_coordinator_app

    app = build_coordinator_app(management_token="test")
    surface: set[str] = set()
    for path, methods_by_verb in app.openapi()["paths"].items():
        for method in methods_by_verb:
            op = _route_to_operation(path, {method.upper()})
            if op is not None:
                surface.add(op)
    return surface


def _route_to_operation(path: str, methods: set[str]) -> str | None:
    """Best-effort mapping of an HTTP route to an operation id.

    The authoritative API surface is built by the route modules; this helper
    maps the registered coordinator routes to operation ids so the parity
    assertion can run. Unknown routes map to None (not in the operation set).
    """
    if path == "/v1/nodes" and "POST" in methods:
        return "node.register"
    if path == "/v1/nodes" and "GET" in methods:
        return "node.list"
    if path.startswith("/v1/nodes/") and path.endswith("/reachability"):
        return "node.reachability"
    if path.startswith("/v1/nodes/") and path.endswith("/resources"):
        return "node.resources"
    if path.startswith("/v1/nodes/") and "DELETE" in methods:
        return "node.deregister"
    if path.startswith("/v1/nodes/") and "GET" in methods:
        return "node.get"
    if path == "/v1/runtimes" and "GET" in methods:
        return "runtime.list"
    if path == "/v1/models:acquire" and "POST" in methods:
        return "model.acquire"
    if path == "/v1/models" and "GET" in methods:
        return "model.list"
    if path.startswith("/v1/models/") and "GET" in methods:
        return "model.get"
    if path == "/v1/deployments" and "POST" in methods:
        return "deployment.create"
    if path == "/v1/deployments" and "GET" in methods:
        return "deployment.list"
    if path.startswith("/v1/deployments/") and path.endswith(":start"):
        return "deployment.start"
    if path.startswith("/v1/deployments/") and path.endswith(":stop"):
        return "deployment.stop"
    if path.startswith("/v1/deployments/") and path.endswith(":restart"):
        return "deployment.restart"
    if path.startswith("/v1/deployments/") and path.endswith(":reconcile"):
        return "deployment.reconcile"
    if path == "/v1/deployments:create-from-export":
        return None  # re-import maps onto deployment.create
    if path.startswith("/v1/deployments/") and path.endswith("/revisions"):
        return "deployment.revisions"
    if path.startswith("/v1/deployments/") and "/export" in path:
        return "deployment.export"
    if path.startswith("/v1/deployments/") and "PATCH" in methods:
        return "deployment.modify"
    if path.startswith("/v1/deployments/") and "DELETE" in methods:
        return "deployment.remove"
    if path.startswith("/v1/deployments/") and path.endswith("/status"):
        return "deployment.status"
    if path.startswith("/v1/deployments/") and path.endswith("/runtime"):
        return "deployment.runtime"
    if path.startswith("/v1/deployments/") and "GET" in methods:
        return "deployment.get"
    if path.startswith("/v1/models/") and "DELETE" in methods:
        return "model.delete"
    if path == "/v1/inference-credentials" and "GET" in methods:
        return "inferencekey.list"
    if path.startswith("/v1/inference-credentials/") and "PUT" in methods:
        return "inferencekey.set"
    if path.startswith("/v1/inference-credentials/") and "DELETE" in methods:
        return "inferencekey.delete"
    if path.endswith("/inference-credential") and "PUT" in methods:
        return "inferencekey.bind"
    if path == "/v1/images:reconcile":
        return "image.reconcile"
    if path == "/v1/images:build":
        return "image.build"
    if path == "/v1/images:import":
        return "image.import"
    if path == "/v1/buildspecs" and "GET" in methods:
        return "buildspec.list"
    if path.startswith("/v1/buildspecs/") and "PUT" in methods:
        return "buildspec.set"
    if path.startswith("/v1/buildspecs/") and "DELETE" in methods:
        return "buildspec.delete"
    if path == "/v1/images" and "GET" in methods:
        return "image.list"
    if path.startswith("/v1/images/") and "DELETE" in methods:
        return "image.delete"
    if path == "/v1/credentials" and "GET" in methods:
        return "credential.list"
    if path.startswith("/v1/credentials/") and "PUT" in methods:
        return "credential.set"
    if path.startswith("/v1/credentials/") and "DELETE" in methods:
        return "credential.delete"
    if path == "/v1/operations" and "GET" in methods:
        return "operation.list"
    if path.startswith("/v1/operations/") and "GET" in methods:
        return "operation.get"
    return None


def test_api_surface_parity() -> None:
    """Every registry operation is reachable through the API surface."""
    registered = _api_surface()
    missing = _EXPECTED_OPERATIONS - registered
    # The registry is the source of truth and the CLI/MCP are projections of it.
    # Adding an operation to the registry without routing it fails here.
    assert missing == set(), f"registry operations missing from API surface: {missing}"


def test_cli_surface_parity() -> None:
    """Every registry operation is reachable through the CLI surface."""
    cli_operations = _cli_surface()
    missing = _EXPECTED_OPERATIONS - cli_operations
    assert missing == set(), f"registry operations missing from CLI surface: {missing}"


def test_mcp_surface_parity() -> None:
    """Every registry operation is reachable through the MCP surface."""
    mcp_operations = _mcp_surface()
    missing = _EXPECTED_OPERATIONS - mcp_operations
    assert missing == set(), f"registry operations missing from MCP surface: {missing}"


# The CLI addresses a human, so it says ``show`` where the registry says
# ``get``. That is a deliberate difference in vocabulary, not in capability, so
# the alias is declared here rather than either surface being renamed to match
# the other. Parity is about the operation being reachable, not about the word.
_CLI_VERB_ALIASES = {"show": "get"}


def _cli_surface() -> set[str]:
    """CLI command surface as operation ids.

    Enumerates the CLI command groups registered on the Typer app, mapping
    ``<noun> <verb>`` commands onto registry operation ids. The CLI is an HTTP
    thin client; every management command here is reachable.
    """
    from tensorstead.cli.main import app

    operations: set[str] = set()
    for group in app.registered_groups:
        typer_instance = group.typer_instance
        if typer_instance is None:
            continue
        # The command group's noun is on the Typer instance's info, not the
        # DefaultPlaceholder used as the registration key.
        noun = getattr(typer_instance.info, "name", "")
        if not noun or noun == "coordinator":
            continue  # coordinator init/serve are process commands, not management ops
        for command in typer_instance.registered_commands:
            verb = getattr(command, "name", "")
            operations.add(f"{noun}.{_CLI_VERB_ALIASES.get(verb, verb)}")
    return operations


def _mcp_surface() -> set[str]:
    """MCP tool surface as operation ids.

    Enumerates the tools actually registered on a built server — not the
    hand-maintained ``TOOL_NAMES`` list — so a tool that is declared but never
    registered fails here rather than passing on paperwork.

    MCP names tools ``<noun>_<verb>``; the registry uses ``<noun>.<verb>``. The
    split is on the first underscore, since verbs like ``reachability`` are one
    word but nouns never contain an underscore.
    """
    import asyncio

    from tensorstead.mcp.server import build_server

    tools = asyncio.run(build_server().list_tools())
    operations: set[str] = set()
    for tool in tools:
        noun, _, verb = tool.name.partition("_")
        if noun and verb:
            operations.add(f"{noun}.{verb}")
    return operations

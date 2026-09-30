"""Operation catalogue — parity source of truth.

``src/tensorstead/service/registry.py`` enumerates the supported management
operation set. It is the single source of truth: the API, CLI, and MCP
contracts are each a *total projection* of it.

That parity rule ("100% of the supported management operation set is reachable
through API, CLI, and MCP") is therefore a test, not a review item: enumerate
this registry and diff it against each surface, failing on any asymmetry.

Every operation is described as data, paired with the version-minimum keys in
``contracts/version.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Operation:
    """One supported management operation."""

    id: str  # dotted noun.verb id, e.g. "node.register"
    noun: str
    verb: str
    purpose: str


_OPERATIONS: list[Operation] = [
    Operation(
        "node.register", "node", "register", "Register a host whose agent is already running"
    ),
    Operation("node.list", "node", "list", "Inventory of registered nodes"),
    Operation("node.get", "node", "get", "Detail of one node"),
    Operation(
        "node.deregister",
        "node",
        "deregister",
        "Remove a node; refused while a deployment names it",
    ),
    Operation("node.reachability", "node", "reachability", "On-demand reachability check"),
    Operation(
        "node.resources",
        "node",
        "resources",
        "On-demand accelerator, memory, managed-storage reading",
    ),
    Operation("runtime.list", "runtime", "list", "Supported runtimes and their capabilities"),
    Operation(
        "model.acquire", "model", "acquire", "Acquire onto nodes — once upstream, then replicated"
    ),
    Operation("model.list", "model", "list", "Identity, source, resolved revision"),
    Operation("model.get", "model", "get", "Detail of one model"),
    Operation("model.delete", "model", "delete", "Explicit delete; refused while referenced"),
    Operation("image.list", "image", "list", "List images present on nodes"),
    Operation(
        "image.reconcile",
        "image",
        "reconcile",
        "Compare image records against the nodes; reap records whose image is gone",
    ),
    Operation("image.build", "image", "build", "Build a recorded spec into an image on a node"),
    Operation("image.import", "image", "import", "Import a prebuilt image archive onto a node"),
    Operation("buildspec.set", "buildspec", "set", "Record a build spec; executes nothing"),
    Operation("buildspec.list", "buildspec", "list", "List recorded build specs"),
    Operation(
        "buildspec.delete", "buildspec", "delete", "Delete a build spec; refused while referenced"
    ),
    Operation("image.delete", "image", "delete", "Explicit delete; refused while referenced"),
    Operation(
        "credential.set",
        "credential",
        "set",
        "Set named per-source secret material (reference forms)",
    ),
    Operation("credential.list", "credential", "list", "Names and default flag only, never values"),
    Operation("credential.delete", "credential", "delete", "Delete a named credential"),
    Operation(
        "inferencekey.set",
        "inferencekey",
        "set",
        "Store a named inference credential; reference only, no read path",
    ),
    Operation(
        "inferencekey.list", "inferencekey", "list", "Names and set times only, never values"
    ),
    Operation(
        "inferencekey.delete",
        "inferencekey",
        "delete",
        "Delete a credential; refused while a deployment binds it",
    ),
    Operation(
        "inferencekey.bind",
        "inferencekey",
        "bind",
        "Bind a credential to a deployment, or clear the binding",
    ),
    Operation("deployment.create", "deployment", "create", "Define a deployment; host untouched"),
    Operation("deployment.modify", "deployment", "modify", "New numbered revision"),
    Operation("deployment.list", "deployment", "list", "List declared deployments"),
    Operation("deployment.get", "deployment", "get", "Declared + observed, distinguishable"),
    Operation("deployment.revisions", "deployment", "revisions", "List retained revisions"),
    Operation("deployment.status", "deployment", "status", "Observed state, fetched now"),
    Operation(
        "deployment.runtime",
        "deployment",
        "runtime",
        "The runtime's own account of itself: argv, relaunches, recent output",
    ),
    Operation("deployment.export", "deployment", "export", "Human-readable export of any revision"),
    Operation("deployment.start", "deployment", "start", "Start lifecycle"),
    Operation("deployment.stop", "deployment", "stop", "Stop lifecycle"),
    Operation("deployment.restart", "deployment", "restart", "Restart lifecycle"),
    Operation("deployment.reconcile", "deployment", "reconcile", "Converge toward declared state"),
    Operation("deployment.remove", "deployment", "remove", "Remove deployment + instance only"),
    Operation(
        "operation.get", "operation", "get", "Progress and terminal outcome of one operation"
    ),
    Operation("operation.list", "operation", "list", "List operations"),
]


def all_operations() -> list[Operation]:
    """Return the full catalogue (a copy, so callers cannot mutate it)."""
    return list(_OPERATIONS)


def operation_ids() -> list[str]:
    """Return the dotted ids of every operation in catalogue order."""
    return [op.id for op in _OPERATIONS]


def operation(id: str) -> Operation | None:
    """Return the operation with ``id``, or ``None`` if not in the catalogue."""
    for op in _OPERATIONS:
        if op.id == id:
            return op
    return None

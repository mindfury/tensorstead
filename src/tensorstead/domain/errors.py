"""Typed domain failures.

Every failure carries a structured ``code``, ``message``, and the originating
``node_id`` when there is one, so "actionable reason" is a shape rather than a
prose convention. The coordinator maps these to HTTP errors that preserve the
structured failure shape; the operation store records them.

The ``detail`` field holds concise, bounded diagnostic context only — never a
log stream and never secret material.
"""

from __future__ import annotations

from typing import Any

from .identity import assert_valid_ulid


class DomainError(Exception):
    """Base class for all typed domain failures.

    Subclasses set ``code``; instances bind a message and an optional node id.
    """

    code: str = "domain_error"

    def __init__(
        self,
        message: str,
        *,
        node_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.node_id = node_id
        self.detail = detail or {}

    def as_failure(self) -> dict[str, Any]:
        """Render the structured failure shape."""
        failure: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
        }
        if self.node_id is not None:
            failure["node_id"] = self.node_id
        return failure


class NotFoundError(DomainError):
    """A referenced entity does not exist."""

    code = "not_found"

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)


class AlreadyExistsError(DomainError):
    """An entity that must be unique already exists (e.g. a node name)."""

    code = "already_exists"


class AlreadyInStateError(DomainError):
    """A requested lifecycle state is already held.

    This is a *satisfied request*, not a failure — the CLI exits 0 on it
    rather than reporting an error.
    """

    code = "already_in_state"


class AgentUnreachableError(DomainError):
    """A node agent could not be reached."""

    code = "agent_unreachable"

    def __init__(self, message: str, node_id: str, **kwargs: Any) -> None:
        assert_valid_ulid(node_id, what="node_id")
        super().__init__(message, node_id=node_id, **kwargs)


class NodeOperationFailedError(DomainError):
    """A node answered and reported that the work itself failed.

    Deliberately distinct from ``AgentUnreachableError``: the agent was
    reached and replied. Collapsing the two sends an operator to check the
    network when the real cause was, for instance, a build step that exited
    non-zero. Before this existed, an image build that failed on the node
    escaped as an unhandled error and reached the operator as
    ``internal_error: unexpected internal error`` — a terminal outcome that
    named nothing, which is the failure this error exists to prevent.
    """

    code = "node_operation_failed"


class AgentVersionIncompatibleError(DomainError):
    """Major contract-version mismatch on a coordinator->agent call."""

    code = "agent_version_incompatible"


class OperationUnsupportedByAgentError(DomainError):
    """Agent below the operation's minimum contract version."""

    code = "operation_unsupported_by_agent"


class StillReferencedError(DomainError):
    """Removal refused while a retained reference names the target."""

    code = "still_referenced"


class EndpointConflictError(DomainError):
    """Another managed deployment already holds the same node+endpoint."""

    code = "endpoint_conflict"


class RuntimeNotDistributedError(DomainError):
    """Multi-node requested for a runtime that cannot distribute."""

    code = "runtime_not_distributed"


class NodeReservedError(DomainError):
    """A create, modify, or start named a node an operator declared reserved."""

    code = "node_reserved"


class CredentialResolutionError(DomainError):
    """A deployment has a bound credential that could not be resolved.

    Distinct from "nothing bound" (a legitimate ``None``, meaning the node's
    own provisioning applies): this means a binding exists and is broken --
    its record is missing, or the provider could not produce its value.
    Raised rather than silently degrading to unauthenticated
    access: a broken binding is never treated as no binding.
    """

    code = "credential_resolution_failed"


class InvalidAgentEndpointError(DomainError):
    """A registration's agent_endpoint is not a canonical https origin.

    Checked before any call is made to it:
    registration itself sends the estate's management token to whatever
    ``agent_endpoint`` names, so anything the URL shape could smuggle --
    userinfo, a path, a query, a fragment -- reaches that request too.
    """

    code = "invalid_agent_endpoint"


class AuthorizationRefusedError(DomainError):
    """Upstream refused credentials/access."""

    code = "authorization_refused"


class InvalidDeploymentError(DomainError):
    """A deployment definition is invalid (validation rejection, exit code 2)."""

    code = "invalid_deployment"


class PartialFailureError(DomainError):
    """Some participating nodes succeeded and others did not.

    Deliberately **not** reported as success. A multi-node operation that
    half-worked leaves the deployment in a state no one asked for, so it is an
    overall failure carrying per-node outcomes, and the coordinator does not
    record the desired state as changed. The nodes that did change surface as
    divergence on the next observation rather than being silently rolled back —
    reported, never acted on behind the operator's back.
    """

    code = "partial_failure"


class ConcurrentModificationError(DomainError):
    """A concurrent conflict lost an optimistic revision check."""

    code = "concurrent_modification"

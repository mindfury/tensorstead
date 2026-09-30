"""Node-client port (structural, not a named seam).

The transport over which the coordinator drives a node agent. Implemented by
``coordinator/node_http.py`` over httpx. This port is *structural*:
node transport is not one of the six named areas of variability in this
design, so it is a Protocol the coordinator depends on, not a registered seam.

Every coordinator→agent call travels this path, is TLS-protected with a
management token, and pins the agent's certificate fingerprint.
It performs version negotiation on every call and surfaces a
structured failure shape.
"""

from __future__ import annotations

from typing import Any, Protocol

from tensorstead.domain.models import Node


class AgentCallError(Exception):
    """A coordinator→agent call failed at the transport or HTTP level.

    Carries a structured ``code`` (one of the failure codes the agent contract
    defines) and the agent's structured reason when available.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        node_id: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.node_id = node_id
        self.detail = detail or {}


class NodeClient(Protocol):
    """The coordinator's view of a node agent's closed operation set."""

    def get_info(self, node: Node) -> dict[str, Any]:
        """``GET /agent/v1/info`` — version + platform facts + engine facts."""

    def get_resources(self, node: Node) -> dict[str, Any]:
        """``GET /agent/v1/resources`` — on-demand accelerator/storage reading."""

    def get_observed(self, node: Node, deployment_id: str) -> dict[str, Any]:
        """``GET /agent/v1/deployments/{id}/observed`` — read-only observation."""

    def get_runtime(self, node: Node, deployment_id: str, *, tail: int = 500) -> dict[str, Any]:
        """Read the runtime's argv, restart count, and recent output.

        Read-through only. Nothing the runtime wrote is persisted by the
        coordinator any more than by the agent: the value is fetched when an
        operator asks and is gone when the response ends.
        """

    def check_endpoint(self, node: Node, port: int) -> dict[str, Any]:
        """``GET /agent/v1/endpoint-check`` — advisory pre-flight port check."""

    # The remaining calls raise CompatibilityRefusal when the
    # agent's contract version does not support the operation.
    def acquire_model(
        self,
        node: Node,
        *,
        source_id: str,
        source_model_id: str,
        revision: str | None,
        credential: str | None,
        progress: Any = None,
        file_selector: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """``POST /agent/v1/models:acquire`` — from upstream.

        ``file_selector`` names which files of the repository to retrieve;
        empty means all of them.
        """

    def replicate_model(
        self,
        node: Node,
        *,
        model: dict[str, Any],
        source_node: Node,
    ) -> dict[str, Any]:
        """``POST /agent/v1/models:replicate`` — pull from a peer.

        Instructs ``node`` (the destination) to make ``model`` locally available
        by pulling from ``source_node``. The destination agent owns the
        operation; the coordinator only names the peer, and **no model byte
        transits the coordinator**.
        """

    def build_image(self, node: Node, payload: dict[str, Any]) -> dict[str, Any]:
        """``POST /agent/v1/images:build`` — produce an image from a spec."""

    def distribute_image(self, node: Node, payload: dict[str, Any]) -> dict[str, Any]:
        """``POST /agent/v1/images:distribute`` — pull an image from a peer."""

    def import_image(self, node: Node, payload: dict[str, Any]) -> dict[str, Any]:
        """``POST /agent/v1/images:import`` — load a prebuilt archive."""

    def remove_image(self, node: Node, *, reference: str) -> dict[str, Any]:
        """``POST /agent/v1/images:remove`` — remove an image from the node.

        Returns ``{"reference", "removed", "digest"}``. ``removed: false`` means
        the daemon did not hold it, which is a success the coordinator acts on
        by reaping its own record. An image a container still holds raises
        ``AgentCallError`` with code ``image_in_use``, and the record survives.
        """

    def list_images(self, node: Node) -> list[dict[str, Any]]:
        """``GET /agent/v1/images`` — what the node's daemon actually holds.

        The read that lets the coordinator compare its records against reality
        instead of trusting them.
        """

    def create_deployment(self, node: Node, payload: dict[str, Any]) -> dict[str, Any]:
        """``POST /agent/v1/deployments`` — materialize and run."""

    def stop_deployment(self, node: Node, deployment_id: str) -> dict[str, Any]:
        """``POST /agent/v1/deployments/{id}:stop``."""

    def reconcile_deployment(self, node: Node, deployment_id: str) -> dict[str, Any]:
        """``POST /agent/v1/deployments/{id}:reconcile``."""

    def remove_deployment(self, node: Node, deployment_id: str) -> dict[str, Any]:
        """``DELETE /agent/v1/deployments/{id}``."""
        ...

    def delete_model(self, node: Node, model_id: str) -> dict[str, Any]:
        """``DELETE /agent/v1/models/{id}`` — unconditional at this layer."""
        ...

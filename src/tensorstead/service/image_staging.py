"""Put a registry image on several nodes, fetching it from the Internet once.

Before this, a multi-node ``deployment start`` let every agent resolve its image
alone: local first, otherwise pull. Two nodes without the image meant two
registry pulls of the same bytes -- 9.7 GB each for the GLM-5.3 runtime.

So the image is staged first: a node that already holds it is the holder;
otherwise the first declared node pulls it, and every other node copies it from
the holder over the existing agent-to-agent distribution (stage, verify by id,
load). No image byte passes through the coordinator.

Failure is reported, never papered over. A node the copy fails on fails the
staging, naming it; nothing then pulls the image from the registry a second time
behind the operator's back. The nodes that did receive it keep it -- the product
does not reverse host work it was not asked to reverse.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

from tensorstead.domain.errors import ImageStagingError, NotFoundError
from tensorstead.domain.models import ImageRecord, Node, pinned_digest
from tensorstead.ports.node_client import AgentCallError
from tensorstead.ports.repository import Repository


def held_id(rows: list[dict[str, Any]], reference: str) -> str | None:
    """The local id under which a node's image list holds ``reference``, if it does.

    By reference, or -- for a digest-pinned reference -- by content address: a
    copied image is held only by id, and on the containerd store that id is the
    pinned manifest digest (the agent resolves it the same way at start).
    """
    pinned = pinned_digest(reference)
    for row in rows:
        if row.get("reference") == reference:
            return str(row.get("digest") or "") or None
    if pinned is not None:
        for row in rows:
            if row.get("digest") == pinned:
                return pinned
    return None


class ImageStager:
    """Stage one image reference across nodes; one registry fetch at most."""

    def __init__(self, repository: Repository, node_client: Any) -> None:
        self._repo = repository
        self._client = node_client
        # Per reference, so two starts sharing an image cannot both decide no
        # node has it and both pull: the second finds the first's copy.
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, reference: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(reference, threading.Lock())

    def _node(self, node_id: str) -> Node:
        node = self._repo.get_node(node_id)
        if node is None:
            raise NotFoundError(f"node {node_id!r} is not registered", node_id=node_id)
        return node

    def stage(self, reference: str, node_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Make ``reference`` present on every node; return what happened per node.

        Each outcome is ``present`` (already held), ``pulled`` (fetched from
        the registry -- at most one node), or ``distributed`` (copied from the
        holder).
        """
        with self._lock_for(reference):
            nodes = [self._node(n) for n in node_ids]
            outcomes: dict[str, dict[str, Any]] = {}
            held: dict[str, str] = {}
            reachable: list[Node] = []
            for node in nodes:
                try:
                    rows = self._client.list_images(node)
                except AgentCallError as exc:
                    # Not this step's failure to report: the start that follows
                    # names an unreachable node exactly as it always has.
                    outcomes[node.id] = {"status": "unreachable", "detail": str(exc)}
                    continue
                reachable.append(node)
                image_id = held_id(rows, reference)
                if image_id:
                    held[node.id] = image_id
                    outcomes[node.id] = {"status": "present", "image_id": image_id}
            if not reachable:
                return outcomes

            if not held:
                head = reachable[0]
                try:
                    pulled = str(self._client.pull_image(head, reference=reference)["digest"])
                except AgentCallError as exc:
                    raise ImageStagingError(
                        f"pulling {reference!r} on {head.name} failed: {exc}",
                        node_id=head.id,
                        detail={"reference": reference, "per_node": outcomes},
                    ) from exc
                held[head.id] = pulled
                outcomes[head.id] = {"status": "pulled", "image_id": pulled}
                self._record(head.id, reference, pulled)

            holder_id, image_id = next(iter(held.items()))
            holder = next(n for n in nodes if n.id == holder_id)
            failed: list[str] = []
            for node in reachable:
                if node.id in held:
                    continue
                try:
                    response = self._client.distribute_image(
                        node,
                        {
                            "reference": reference,
                            "source_endpoint": holder.agent_endpoint,
                            "expected_image_id": image_id,
                        },
                    )
                except AgentCallError as exc:
                    failed.append(node.name)
                    outcomes[node.id] = {
                        "status": "failed",
                        "code": getattr(exc, "code", "image_distribution_failed"),
                        "detail": str(exc),
                    }
                    continue
                arrived = str((response or {}).get("image_id", ""))
                if arrived != image_id:
                    failed.append(node.name)
                    outcomes[node.id] = {"status": "failed", "detail": "identifier mismatch"}
                    continue
                outcomes[node.id] = {"status": "distributed", "image_id": arrived}
                self._record(node.id, reference, arrived)

            if failed:
                raise ImageStagingError(
                    f"copying {reference!r} from {holder.name} failed on "
                    f"{', '.join(failed)}; it was not pulled from the registry again",
                    detail={"reference": reference, "holder": holder.id, "per_node": outcomes},
                )
            return outcomes

    def _record(self, node_id: str, reference: str, digest: str) -> None:
        self._repo.save_image(
            ImageRecord(
                node_id=node_id,
                reference=reference,
                digest=digest,
                pulled_at=datetime.now().astimezone(),
            )
        )

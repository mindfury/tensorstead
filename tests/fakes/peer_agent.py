"""Fake peer agent.

Models the agent-to-agent artifact replication hop (the agent
contract). The destination agent owns the operation: pull from the
source agent, stage, verify against ``content_digest``, then atomically promote
An interrupted or unverifiable transfer must never be presented as an
``available`` replica.

The fake models the four-step pull so the conformance/parity harness can verify
the stage→verify→promote ordering and that a staging copy never propagates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class FakePeerReplica:
    model_id: str
    state: str  # staging | available | failed
    content_digest: str | None = None
    verified_at: datetime | None = None


class FakePeerAgent:
    """A peer node agent that can serve and pull model artifacts."""

    def __init__(self, *, content_digest: str | None = "sha256:peer-digest") -> None:
        self.replicas: dict[str, FakePeerReplica] = {}
        self._default_digest = content_digest
        self.pull_count = 0

    def serve_replica(self, model_id: str) -> FakePeerReplica | None:
        """The source-side view: a peer may only serve an ``available`` replica.

        Refusing to serve a staging copy is what stops a staging artifact from
        propagating (content endpoint).
        """
        replica = self.replicas.get(model_id)
        if replica is None or replica.state != "available":
            return None
        return replica

    def pull(self, model_id: str, *, content_digest: str | None) -> FakePeerReplica:
        """Destination-agent pull: stage → verify → atomic promote."""
        self.pull_count += 1
        # Stage first.
        replica = FakePeerReplica(model_id=model_id, state="staging")
        self.replicas[model_id] = replica
        # Verify against the expected digest.
        expected = content_digest or self._default_digest
        if expected != self._default_digest:
            replica.state = "failed"
            return replica
        # Atomic promote.
        replica.state = "available"
        replica.content_digest = expected
        replica.verified_at = datetime.now()
        return replica

    def get_state(self, model_id: str) -> str:
        replica = self.replicas.get(model_id)
        return replica.state if replica else "absent"

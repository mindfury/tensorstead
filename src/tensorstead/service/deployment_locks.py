"""One lock per deployment, shared by every service that changes one.

The lifecycle service used to own its locks while definition changes
(``modify``, ``rename``) took none, so a modify accepted during a start was
overwritten by the start's stale snapshot (CodeGuard CG-08). Both now take the
same lock. It is in-process only; the repository's compare-and-swap
(``advance_revision``) and column-scoped writes are what hold across
processes.

Re-entrant, so a service that already holds a deployment's lock can call
another that takes it without deadlocking.
"""

from __future__ import annotations

import threading


class DeploymentLocks:
    """Per-deployment re-entrant locks, created on first use."""

    def __init__(self) -> None:
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.Lock()

    def for_deployment(self, deployment_id: str) -> threading.RLock:
        with self._guard:
            lock = self._locks.get(deployment_id)
            if lock is None:
                lock = self._locks[deployment_id] = threading.RLock()
            return lock

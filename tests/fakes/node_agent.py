"""Fake node agent.

Models the coordinator→agent contract as a
stateful object the coordinator service layer calls through the node-client
port. It models the *behaviour* that matters — version reporting, staged
replicas, deployment lifecycle state — rather than asserting on calls, so the
agent conformance suite can run it against the real agent and the two must agree.

The fake is used in place of a live agent so coordinator tests run with no
external anything (tier 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from tensorstead.contracts.version import CONTRACT_VERSION
from tests.fakes.hf_hub import FakeHuggingFaceHub


@dataclass
class FakeReplica:
    model_id: str
    local_path: str
    state: str  # staging | available | failed
    verified_at: datetime | None = None


@dataclass
class FakeDeployment:
    id: str
    revision: int = 1
    desired_state: str = "stopped"
    running: bool = False
    running_revision: int | None = None
    endpoint_reachable: bool = False
    # Separate from reachability: a listening socket is not a serving runtime
    # ``None`` means the probe could not establish the fact.
    inference_ready: bool | None = None
    image_digest: str | None = None
    unexpected_instance: bool = False
    # What the runtime itself would report.
    argv: list[str] = field(default_factory=list)
    restart_count: int = 0
    runtime_log: str = ""


class FakeNodeAgent:
    """A stateful in-memory node agent implementing the agent contract.

    Behavioural guarantees the conformance suite asserts:

    - Reports the in-band contract version.
    - A staged replica is never presented as ``available``.
    - ``deployments/{id}/observed`` is read-only — it mutates nothing.
    """

    def __init__(self, *, contract_version: str = CONTRACT_VERSION) -> None:
        self.contract_version = contract_version
        self.platform_facts: dict[str, Any] = {
            "cpu_arch": "aarch64",
            "os_family": "linux",
            "os_version": "DGX OS 7.x",
            "memory_is_unified": True,
        }
        self.service_manager = "systemd"
        # Which build this agent is. Empty models an agent too old
        # to report it, which is not the same as one that could not identify
        # its own build.
        self.build: dict[str, Any] = {}
        self.container_engine = {"name": "docker", "version": "0.0.0"}
        self.replicas: dict[str, FakeReplica] = {}
        self.deployments: dict[str, FakeDeployment] = {}
        # Containers in the ``tensorstead-`` namespace that this agent did not
        # create. Empty by default: a node is clean unless a test dirties it.
        self.unmanaged_containers: list[dict[str, Any]] = []
        # An agent below contract 1.3 cannot enumerate its namespace and omits
        # the field entirely. Distinct from reporting an empty list, which
        # claims the node was looked at and found clean.
        self.enumerates_containers = True
        self.acquisition_calls = 0
        self._available_digest: dict[str, str] = {}
        # The agent's model source. Acquisition runs the *real*
        # HuggingFaceSource against this fake hub, so the refusal mapping under
        # test is the one the product ships, not a second copy of it.
        self.hub = FakeHuggingFaceHub()
        # Credential values this agent was handed, in call order. The agent
        # never persists one — this list is the test's window on
        # the per-request hop, not agent state the product can read.
        self.credentials_seen: list[str | None] = []
        # Stays empty forever: nothing in the product can append to it, which is
        # what this rule asserts (access terms are never accepted on the
        # operator's behalf).
        self.access_terms_accepted: list[str] = []
        self._staging: str | None = None
        # Peer replication. ``replication_pulls`` is the test's evidence
        # that a transfer went agent-to-agent rather than upstream twice.
        self.replication_pulls: list[dict[str, Any]] = []
        self.replication_unavailable = False
        self._expected_digest = "sha256:fake-digest"
        # An unreachable agent simulates a network failure so the
        # coordinator's observation degrades to ``unreachable``.
        self.unreachable = False
        # Configurable resource readings. Defaults match
        # the prior fixed return so existing tests stay unchanged.
        self._resources: dict[str, Any] | None = None

    # --- info ---
    def get_info(self) -> dict[str, Any]:
        if self.unreachable:
            raise ConnectionError("simulated agent unreachable")
        return {
            "contract_version": self.contract_version,
            "agent_version": "0.1.0",
            "build": dict(self.build),
            "platform_facts": self.platform_facts,
            "service_manager": self.service_manager,
            "container_engine": self.container_engine,
        }

    def get_resources(self) -> dict[str, Any]:
        if self.unreachable:
            raise ConnectionError("simulated agent unreachable")
        if self._resources is not None:
            return self._resources
        return {
            "status": "ok",
            "observed_at": datetime.now().astimezone().isoformat(),
            "accelerator_utilization_pct": 0.0,
            "accelerator_memory_used": 0,
            "accelerator_memory_total": 0,
            "memory_is_unified": True,
            "storage": [],
        }

    def set_resources(self, resources: dict[str, Any]) -> None:
        """Configure the resource readings for tests."""
        self._resources = resources

    # --- acquisition ---
    def gate_model(self, source_model_id: str, *, requires_access_terms: bool = False) -> None:
        """Make a model gated upstream, optionally on access terms.

        ``requires_access_terms`` models the refusal a credential cannot clear:
        only the operator can accept terms, with the provider.
        """
        if requires_access_terms:
            self.hub.require_access_terms(source_model_id)
        else:
            self.hub.gate(source_model_id)

    def _acquire_upstream(
        self,
        source_id: str,
        source_model_id: str,
        revision: str | None,
        credential: str | None,
    ) -> str | None:
        """Run the real source adapter against the fake hub.

        Only the refusal behaviour is taken from the adapter; the recorded
        replica values below stay the fake's own, so this adds the gated-model
        path without disturbing what other tests already assert.
        """
        if source_id != "huggingface":
            return revision
        import tempfile

        from tensorstead.adapters.sources.huggingface import HuggingFaceSource

        if self._staging is None:
            self._staging = tempfile.mkdtemp(prefix="tensorstead-fake-agent-")
        source = HuggingFaceSource(hub=self.hub)
        resolved = source.resolve(source_model_id, revision)
        source.acquire(source_model_id, resolved, self._staging, credential)
        return resolved

    def acquire_model(
        self,
        *,
        source_id: str,
        source_model_id: str,
        revision: str | None,
        credential: str | None = None,
        file_selector: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Stage then atomically promote a model."""
        self.acquisition_calls += 1
        self.credentials_seen.append(credential)
        resolved = self._acquire_upstream(source_id, source_model_id, revision, credential)
        model_id = f"{source_id}:{source_model_id}"
        # Simulate an interruption: a partial transfer must never be available.
        staging = FakeReplica(
            model_id=model_id,
            local_path=f"/var/lib/tensorstead/models/{model_id}",
            state="staging",
        )
        self.replicas[model_id] = staging
        # Promote to available atomically after verification.
        staging.state = "available"
        staging.verified_at = datetime.now()
        self._available_digest[model_id] = "sha256:fake-digest"
        return {
            "resolved_revision": resolved,
            "revision_pinned": resolved is not None,
            "size_bytes": 1234,
            "content_digest": "sha256:fake-digest",
        }

    # --- peer replication ---
    def replicate_model(
        self,
        *,
        source_id: str,
        source_model_id: str,
        resolved_revision: str | None,
        content_digest: str | None,
        source_node_id: str,
        source_agent_endpoint: str,
    ) -> dict[str, Any]:
        """Pull from a peer: stage → verify → atomically promote.

        Records the pull so a test can prove the transfer was agent-to-agent
        and that upstream was not hit a second time.
        """
        if self.replication_unavailable:
            raise ConnectionError("simulated peer unreachable")
        model_id = f"{source_id}:{source_model_id}"
        self.replication_pulls.append(
            {
                "model_id": model_id,
                "from_node": source_node_id,
                "from_endpoint": source_agent_endpoint,
                "content_digest": content_digest,
            }
        )
        staging = FakeReplica(
            model_id=model_id,
            local_path=f"/var/lib/tensorstead/models/{model_id}",
            state="staging",
        )
        self.replicas[model_id] = staging
        # Verification: a mismatched digest never becomes available.
        if content_digest is not None and content_digest != self._expected_digest:
            staging.state = "failed"
            raise ValueError(
                f"digest mismatch replicating {model_id!r}: "
                f"expected {self._expected_digest}, got {content_digest}"
            )
        staging.state = "available"
        staging.verified_at = datetime.now()
        self._available_digest[model_id] = content_digest or self._expected_digest
        return {
            "model_id": model_id,
            "state": "available",
            "content_digest": content_digest or self._expected_digest,
            "size_bytes": 1234,
        }

    def get_model_status(self, model_id: str) -> str:
        replica = self.replicas.get(model_id)
        return replica.state if replica else "absent"

    def get_available_digest(self, model_id: str) -> str | None:
        return self._available_digest.get(model_id)

    # --- deployments ---
    def create_deployment(self, deployment_id: str, *, endpoint: str) -> dict[str, Any]:
        # An unreachable host cannot be made to run anything. Modelled here so
        # a multi-node test can fail exactly one node.
        if self.unreachable:
            raise ConnectionError("simulated agent unreachable")
        self.deployments[deployment_id] = FakeDeployment(
            id=deployment_id,
            running=True,
            running_revision=1,
            endpoint_reachable=True,
            inference_ready=True,
        )
        return {"status": "created", "deployment_id": deployment_id}

    def start_deployment(self, deployment_id: str) -> dict[str, Any]:
        if self.unreachable:
            raise ConnectionError("simulated agent unreachable")
        dep = self.deployments[deployment_id]
        dep.running = True
        dep.running_revision = dep.revision
        dep.desired_state = "running"
        return {"status": "running", "deployment_id": deployment_id}

    def stop_deployment(self, deployment_id: str) -> dict[str, Any]:
        if self.unreachable:
            raise ConnectionError("simulated agent unreachable")
        dep = self.deployments[deployment_id]
        dep.running = False
        dep.desired_state = "stopped"
        return {"status": "stopped", "deployment_id": deployment_id}

    def get_observed(self, deployment_id: str) -> dict[str, Any]:
        """Read-only observation; mutates nothing.

        Reports ``unexpected_instance`` when a container in the managed
        namespace exists that the coordinator did not record.
        Reported, never killed.
        """
        if self.unreachable:
            raise ConnectionError("simulated agent unreachable")
        dep = self.deployments.get(deployment_id)
        if dep is None:
            return {
                "status": "unknown",
                "observed_at": datetime.now().astimezone().isoformat(),
            }
        observed: dict[str, Any] = {
            "status": "running" if dep.running else "not_running",
            "observed_at": datetime.now().astimezone().isoformat(),
            "running_image_digest": dep.image_digest,
            "running_revision": dep.running_revision,
            "endpoint_reachable": dep.endpoint_reachable,
            "inference_ready": dep.inference_ready,
            "unexpected_instance": dep.unexpected_instance,
        }
        if self.enumerates_containers:
            observed["managed_containers"] = self._managed_containers()
        return observed

    def _managed_containers(self) -> list[dict[str, Any]]:
        """The ``tensorstead-`` namespace this node would report.

        Derived from the fake's own deployments so it stays self-consistent,
        plus anything an operator put there behind the product's back. The
        agent enumerates; the *coordinator* decides what is unexpected, because
        only the coordinator holds the deployment records to compare against.
        """
        containers = [
            {
                "name": f"tensorstead-{deployment_id}",
                "deployment_id": deployment_id,
                "running": dep.running,
            }
            for deployment_id, dep in sorted(self.deployments.items())
        ]
        return containers + list(self.unmanaged_containers)

    def add_unmanaged_container(self, name: str, *, running: bool = True) -> None:
        """Put a container in the managed namespace that no deployment records.

        The case this covers: an operator's own ``docker run``, or a container that
        outlived the deployment it belonged to. It carries no
        ``tensorstead.deployment_id`` label, exactly as a hand-started one would
        not, so the coordinator has to fall back to the name to identify it.
        """
        self.unmanaged_containers.append({"name": name, "deployment_id": None, "running": running})

    def get_runtime(self, deployment_id: str, *, tail: int = 500) -> dict[str, Any]:
        """The runtime's account of itself. Read-only; stores nothing."""
        if self.unreachable:
            raise ConnectionError("simulated agent unreachable")
        dep = self.deployments.get(deployment_id)
        if dep is None:
            return {
                "observed_at": datetime.now().astimezone().isoformat(),
                "lines_requested": tail,
                "detail": "no container exists for this deployment",
            }
        lines = dep.runtime_log.splitlines()
        return {
            "observed_at": datetime.now().astimezone().isoformat(),
            "argv": list(dep.argv),
            "restart_count": dep.restart_count,
            "exit_code": None if dep.running else 1,
            "running": dep.running,
            "log_tail": "\n".join(lines[-max(1, tail) :]),
            "lines_requested": tail,
            "detail": None,
        }

    def reconcile_deployment(self, deployment_id: str) -> dict[str, Any]:
        """Reconcile toward declared state.

        The only agent endpoint permitted to mutate in response to
        divergence, and only because it was explicitly called. Reports
        what it changed and what it could not.
        """
        if self.unreachable:
            raise ConnectionError("simulated agent unreachable")
        dep = self.deployments.get(deployment_id)
        if dep is None:
            return {
                "status": "reconciled",
                "deployment_id": deployment_id,
                "changed": [],
                "remaining": [],
            }
        # In the fake, convergence means: if the deployment desires running
        # but is not running, restart it. Report what changed.
        changed: list[dict[str, Any]] = []
        remaining: list[dict[str, Any]] = []
        if dep.desired_state == "running" and not dep.running:
            # Attempt to restart — in the fake this succeeds.
            dep.running = True
            dep.running_revision = dep.revision
            changed.append({"action": "started", "node_id": "self"})
        elif dep.desired_state == "stopped" and dep.running:
            dep.running = False
            changed.append({"action": "stopped", "node_id": "self"})
        return {
            "status": "reconciled",
            "deployment_id": deployment_id,
            "changed": changed,
            "remaining": remaining,
        }

    def remove_deployment(self, deployment_id: str) -> dict[str, Any]:
        """Remove the container and unit; retain model artifacts and image."""
        if self.unreachable:
            raise ConnectionError("simulated agent unreachable")
        self.deployments.pop(deployment_id, None)
        return {
            "status": "removed",
            "deployment_id": deployment_id,
            "retained": True,
            "retained_what": ["model_artifacts", "image"],
            "removed_what": ["container", "systemd_unit"],
        }

    def delete_model(self, model_id: str) -> dict[str, Any]:
        """Delete a model replica from this host.

        Unconditional at this layer — the coordinator's *referenced* check
        governs whether deletion is permitted.
        """
        if self.unreachable:
            raise ConnectionError("simulated agent unreachable")
        self.replicas.pop(model_id, None)
        return {"status": "deleted", "model_id": model_id}

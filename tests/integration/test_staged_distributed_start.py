"""A distributed group starts its head first.

This product shipped without launch ordering, deliberately and with the reasoning
recorded: most distributed backends rendezvous with retry, in which case
ordering is convention, and designing around an unwatched guess had already cost
this project two restarts. The requirement said to attempt it concurrently and let the
log tail decide.

The first live TP=2 start decided it. The worker on rank 1 reached Gloo
initialization before the head was accepting, exhausted its retry budget in four
attempts, and exited — while the start operation reported success, because
"containers started" is all a start establishes.

So the ordering is now built, from measurement rather than from the recipe's
convention. Worth noting the recipe's stated order was *worker first, then head*
the hardware contradicted it, and a head-first barrier is
more robust than either order without one.

**How this reconciles with concurrent dispatch** rather than quietly undoing it:

- every nominated node is still attempted when the head comes up;
- the workers are still dispatched concurrently *with each other*, so a slow
  worker still cannot hide the rest;
- only when the head itself fails are the workers deliberately not started —
  because a worker dispatched against a dead head cannot join anything, and
  would sit holding accelerator memory for a group that will never exist.

Staging is opt-in by declaration: an adapter that declares no rendezvous port
keeps the concurrent dispatch every deployment has always had.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from tensorstead.domain.errors import PartialFailureError
from tensorstead.domain.models import Node
from tensorstead.ports.node_client import AgentCallError
from tensorstead.ports.runtime_adapter import ContainerRequirements
from tensorstead.service.lifecycle import LifecycleService

pytestmark = pytest.mark.integration


class _Staged:
    """A runtime whose ranks must be started in order."""

    @staticmethod
    def container_requirements(_config: dict, _position: Any) -> ContainerRequirements:
        return ContainerRequirements(rendezvous_port=29500)


class _Concurrent:
    """A runtime whose ranks may start together — the ordinary case."""

    @staticmethod
    def container_requirements(_config: dict, _position: Any) -> ContainerRequirements:
        return ContainerRequirements()


class _Revision:
    runtime_type = "vllm"
    runtime_config: dict[str, Any] = {"tensor_parallel_size": 2}

    def __init__(self, node_ids: list[str]) -> None:
        self.participating_nodes = tuple(node_ids)


class _Repo:
    def __init__(self, nodes: dict[str, Node]) -> None:
        self._nodes = nodes

    def get_node(self, node_id: str) -> Node | None:
        return self._nodes.get(node_id)


# Real ULIDs: ``Node`` validates its id, and a readable placeholder is not one.
_IDS = [f"01KZJ5JJESQ48BX3JNR3ZR4Q{c}{c}" for c in "ABCDE"]


def _nodes(count: int) -> dict[str, Node]:
    from datetime import datetime

    return {
        _IDS[i]: Node(
            id=_IDS[i],
            name=f"spark-{i}",
            agent_endpoint=f"https://10.0.0.1{i}:8443",
            agent_cert_fingerprint="",
            agent_contract_version="1.11",
            platform_facts={},
            registered_at=datetime.now().astimezone(),
        )
        for i in range(count)
    }


def _service(adapter: Any, count: int = 2) -> tuple[LifecycleService, _Revision]:
    nodes = _nodes(count)
    service = LifecycleService(_Repo(nodes), None, {"vllm": adapter})  # type: ignore[arg-type]
    return service, _Revision(list(nodes))


def test_every_worker_is_started_before_the_head() -> None:
    """Workers first, head last -- the order of the recipe that works.

    A worker is *held* while the assertion runs, so this cannot pass by luck.
    The equivalent test for the previous order made exactly that mistake in an
    early draft: it recorded call order and let the head return immediately,
    and under concurrent dispatch a no-op head usually wins that race anyway.
    A regression test the defect can pass is not one.

    This was the inverse until recently -- head dispatched alone and
    awaited, then workers. Head-first was never proven wrong; the argument for
    changing it is that the reference recipe is the only sequence known to
    work, and a first hardware attempt is a bad place to test two hypotheses.
    """
    service, revision = _service(_Staged(), count=3)
    order: list[str] = []
    worker_entered = threading.Event()
    release_worker = threading.Event()

    def call(node: Node, _revision: Any, _peers: list[Node]) -> None:
        order.append(node.id)
        if node.id == _IDS[1]:
            worker_entered.set()
            assert release_worker.wait(timeout=5.0), "the test never released the worker"

    driver = threading.Thread(target=lambda: service._drive_nodes(revision, "start", call))
    driver.start()
    try:
        assert worker_entered.wait(timeout=5.0), "no worker was dispatched"
        # A worker is still inside its call. The head must not have been
        # dispatched: it is started only once every worker has returned.
        assert _IDS[0] not in order, f"the head was dispatched while a worker ran: {order}"
    finally:
        release_worker.set()
        driver.join(timeout=5.0)

    assert order[-1] == _IDS[0], f"the head was not last: {order}"
    assert set(order) == set(_IDS[:3]), "not every node was attempted"


def test_the_workers_are_still_dispatched_concurrently() -> None:
    """Concurrent dispatch among the workers: a slow one must not hide the others.

    Observable only with three nodes -- with two there is a single worker and
    nothing to be concurrent with, which is why this does not use the pair.
    """
    service, revision = _service(_Staged(), count=3)
    both_in_flight = threading.Barrier(2, timeout=5.0)

    def call(node: Node, _revision: Any, _peers: list[Node]) -> None:
        if node.id != _IDS[0]:
            both_in_flight.wait()

    # Times out and raises BrokenBarrierError if the workers were serialized.
    service._drive_nodes(revision, "start", call)


def test_the_head_is_not_started_when_a_worker_fails() -> None:
    """A head brought up for an incomplete group waits for a rank never coming.

    It would hold accelerator memory doing it, which is the same waste the
    previous order guarded against from the other direction.
    """
    service, revision = _service(_Staged(), count=3)
    attempted: list[str] = []

    def call(node: Node, _revision: Any, _peers: list[Node]) -> None:
        attempted.append(node.id)
        if node.id == _IDS[1]:
            raise AgentCallError("agent_unreachable", "connection refused", node_id=node.id)

    with pytest.raises(Exception) as caught:
        service._drive_nodes(revision, "start", call)

    assert _IDS[0] not in attempted, f"the head was started after a worker failed: {attempted}"
    # Every *worker* is still attempted -- concurrent dispatch is not weakened by ordering.
    assert set(attempted) == {_IDS[1], _IDS[2]}, attempted
    per_node = getattr(caught.value, "detail", {}).get("per_node", {})
    assert per_node[_IDS[0]]["code"] == "group_workers_unavailable"
    assert "cannot form" in per_node[_IDS[0]]["message"]


def test_the_head_failure_names_the_head() -> None:
    """The reported reason must point at the rank that actually failed."""
    service, revision = _service(_Staged())

    def call(node: Node, _revision: Any, _peers: list[Node]) -> None:
        if node.id == _IDS[0]:
            raise AgentCallError("rendezvous_unavailable", "head never accepted", node_id=node.id)

    with pytest.raises(Exception) as caught:
        service._drive_nodes(revision, "start", call)

    per_node = getattr(caught.value, "detail", {}).get("per_node", {})
    assert per_node[_IDS[0]]["code"] == "rendezvous_unavailable"


def test_every_rank_sees_the_whole_group() -> None:
    """Positions derive from the full declared group, not from who is dispatched.

    Narrowing the node list to "whoever is left to call" would have given the
    workers a short peer list and therefore the wrong rank and node count --
    a silent corruption of exactly the facts this staging exists to get right.
    """
    service, revision = _service(_Staged(), count=3)
    seen: dict[str, int] = {}

    def call(node: Node, _revision: Any, peers: list[Node]) -> None:
        seen[node.id] = len(peers)

    service._drive_nodes(revision, "start", call)

    assert seen == dict.fromkeys(_IDS[:3], 3), seen


def test_an_unstaged_runtime_keeps_concurrent_dispatch() -> None:
    """Opt-in by declaration: everything else behaves exactly as before."""
    service, revision = _service(_Concurrent(), count=2)
    both_in_flight = threading.Barrier(2, timeout=5.0)

    def call(_node: Node, _revision: Any, _peers: list[Node]) -> None:
        both_in_flight.wait()

    service._drive_nodes(revision, "start", call)


def test_a_single_node_deployment_is_never_staged() -> None:
    """One node is not a group; it must take the path it always took."""
    service, revision = _service(_Staged(), count=1)
    calls: list[str] = []

    service._drive_nodes(revision, "start", lambda n, _r, _p: calls.append(n.id))

    assert calls == [_IDS[0]]


@pytest.mark.parametrize("action", ["stop", "remove"])
def test_only_start_is_staged(action: str) -> None:
    """Stopping a group has no rendezvous to wait for.

    Serializing a stop behind an unreachable head would make a partly-running
    group harder to clear up, which is the opposite of what this is for.
    """
    service, revision = _service(_Staged(), count=2)
    both_in_flight = threading.Barrier(2, timeout=5.0)

    def call(_node: Node, _revision: Any, _peers: list[Node]) -> None:
        both_in_flight.wait()

    service._drive_nodes(revision, action, call)


def test_a_service_without_adapters_does_not_stage() -> None:
    """The optional constructor argument must mean "behave as before"."""
    nodes = _nodes(2)
    service = LifecycleService(_Repo(nodes), None)  # type: ignore[arg-type]
    both_in_flight = threading.Barrier(2, timeout=5.0)

    def call(_node: Node, _revision: Any, _peers: list[Node]) -> None:
        both_in_flight.wait()

    service._drive_nodes(_Revision(list(nodes)), "start", call)


# ------------------------------------------------- unwinding a failed group
#
# The live TP=2 retry left the head exited and the worker running, logging
# broken pipes and holding accelerator memory, until an operator stopped it by
# hand. That is the evidence that settled a question this
# file previously left open: a half-formed runtime *group* is not the same case
# as a partial image build, where the product deliberately keeps what it made.


class _RecordingClient:
    """Captures the cleanup calls a failed group makes."""

    def __init__(self) -> None:
        self.stopped: list[str] = []
        self.evidence_for: list[str] = []
        self.runtime: dict[str, Any] = {"log_tail": "Gloo connectFullMesh failed"}

    def get_runtime(self, node: Node, _deployment_id: str, *, tail: int = 500) -> dict[str, Any]:
        self.evidence_for.append(node.id)
        return self.runtime

    def stop_deployment(self, node: Node, _deployment_id: str) -> dict[str, Any]:
        self.stopped.append(node.id)
        return {"status": "stopped"}


def _failing_group(adapter: Any, client: Any, count: int = 2) -> tuple[LifecycleService, Any]:
    nodes = _nodes(count)
    service = LifecycleService(_Repo(nodes), client, {"vllm": adapter})  # type: ignore[arg-type]
    revision = _Revision(list(nodes))
    revision.deployment_id = "01KZWP1YTK21JSZTJJ19G9XBSS"  # type: ignore[attr-defined]
    return service, revision


def _one_worker_fails(node: Node, _revision: Any, _peers: list[Node]) -> None:
    """Second worker fails; the first started cleanly and is left orphaned.

    Three nodes, not two. With workers dispatched first, a two-node group whose
    only worker fails leaves nothing running -- so the unwind would have nothing
    to do and these tests would pass vacuously. The orphan under this order is a
    worker that succeeded while a sibling did not.
    """
    if node.id == _IDS[2]:
        raise AgentCallError("invalid_runtime", "rank exited", node_id=node.id)


def test_a_rank_left_by_a_failed_group_is_stopped() -> None:
    """The state an operator had to clear by hand must not recur."""
    client = _RecordingClient()
    service, revision = _failing_group(_Staged(), client, count=3)

    with pytest.raises(PartialFailureError):
        service._drive_nodes(revision, "start", _one_worker_fails)

    assert client.stopped == [_IDS[1]], (
        f"the started worker was left running after the group failed: {client.stopped}"
    )


def test_the_evidence_is_captured_before_the_stop() -> None:
    """Stopping destroys the logs; a cleanup that erased the cause is no better."""
    client = _RecordingClient()
    service, revision = _failing_group(_Staged(), client, count=3)

    with pytest.raises(Exception) as caught:
        service._drive_nodes(revision, "start", _one_worker_fails)

    assert client.evidence_for == [_IDS[1]]
    per_node = getattr(caught.value, "detail", {}).get("per_node", {})
    assert "Gloo" in per_node[_IDS[1]]["runtime_evidence"]["log_tail"]


def test_the_stopped_rank_says_why_it_was_stopped() -> None:
    """A rank reported as merely "stopped" would look like an operator action."""
    client = _RecordingClient()
    service, revision = _failing_group(_Staged(), client, count=3)

    with pytest.raises(Exception) as caught:
        service._drive_nodes(revision, "start", _one_worker_fails)

    outcome = getattr(caught.value, "detail", {}).get("per_node", {})[_IDS[1]]
    assert outcome["state"] == "stopped"
    assert outcome["code"] == "group_did_not_form"


def test_a_cleanup_that_fails_is_reported_not_swallowed() -> None:
    """A rank that could not be stopped is exactly what must reach an operator."""

    class _UnstoppableClient(_RecordingClient):
        def stop_deployment(self, node: Node, _deployment_id: str) -> dict[str, Any]:
            raise AgentCallError("agent_unreachable", "no route", node_id=node.id)

    service, revision = _failing_group(_Staged(), _UnstoppableClient(), count=3)

    with pytest.raises(Exception) as caught:
        service._drive_nodes(revision, "start", _one_worker_fails)

    outcome = getattr(caught.value, "detail", {}).get("per_node", {})[_IDS[1]]
    assert "cleanup_failed" in outcome, outcome


def test_an_unstaged_partial_failure_is_left_alone() -> None:
    """The standing rule is unchanged for everything that is not a group.

    A distributed image build keeps the image it produced; an ordinary
    multi-node operation leaves what succeeded to surface as divergence. Only a
    runtime group unwinds, because only a lone rank is useless by construction.
    """
    client = _RecordingClient()
    service, revision = _failing_group(_Concurrent(), client, count=3)

    with pytest.raises(PartialFailureError):
        service._drive_nodes(revision, "start", _one_worker_fails)

    assert client.stopped == [], "an unstaged partial failure was rolled back"
